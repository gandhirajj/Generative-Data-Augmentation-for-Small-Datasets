# app.py
# --------------------------------------------------
# GenAI Data Augmentation – Single-File Streamlit App
# Features
# - Tabular augmentation via SDV CTGAN (preferred) with graceful fallback to GaussianCopula or SMOTE-like bootstrapping.
# - Text augmentation via PEGASUS / backtranslation / WordNet synonym swap.
# - Image augmentation via torchvision/PIL, with optional Stable Diffusion (Diffusers).
# - Class balancing controls, validation charts, and ZIP export of results.
#
# How to run:
#   pip install -U streamlit pandas numpy pillow scikit-learn matplotlib nltk
#   pip install -U sdv ctgan ydata-synthetic
#   pip install -U transformers sentencepiece torch torchvision
#   pip install -U diffusers accelerate safetensors
#   python -m nltk.downloader wordnet omw-1.4
#   streamlit run app.py
#
# Heavy deps are imported lazily with graceful fallbacks.

import io
import os
import zipfile
import random
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------

def warn(msg: str):
    st.warning(msg, icon="⚠️")

def info(msg: str):
    st.info(msg, icon="ℹ️")

def success(msg: str):
    st.success(msg, icon="✅")

# -----------------------------
# Validation helpers (tabular)
# -----------------------------

def compare_distributions(real: pd.DataFrame, synth: pd.DataFrame) -> plt.Figure:
    num_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in real.columns if c not in num_cols]

    n_plots = len(num_cols) + len(cat_cols)
    n_plots = max(n_plots, 1)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3*n_plots))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    for c in num_cols:
        ax = axes[idx]; idx += 1
        ax.hist(real[c].dropna().values, bins=30, alpha=0.5, label='real', density=True)
        ax.hist(synth[c].dropna().values, bins=30, alpha=0.5, label='synthetic', density=True)
        ax.set_title(f"Numeric: {c}"); ax.legend()

    for c in cat_cols:
        ax = axes[idx]; idx += 1
        real_counts = real[c].astype(str).value_counts(normalize=True)
        synth_counts = synth[c].astype(str).value_counts(normalize=True)
        all_idx = sorted(set(real_counts.index).union(set(synth_counts.index)))
        real_vals = [real_counts.get(k, 0) for k in all_idx]
        synth_vals = [synth_counts.get(k, 0) for k in all_idx]
        x = np.arange(len(all_idx)); width = 0.35
        ax.bar(x - width/2, real_vals, width, label='real')
        ax.bar(x + width/2, synth_vals, width, label='synthetic')
        ax.set_xticks(x); ax.set_xticklabels(all_idx, rotation=45, ha='right')
        ax.set_title(f"Categorical: {c}"); ax.legend()

    fig.tight_layout()
    return fig

# -----------------------------
# TABULAR AUGMENTATION
# -----------------------------

def reduce_cardinality(df: pd.DataFrame, cardinality_limit: int = 1000, top_k: int = 100) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == object or pd.api.types.is_string_dtype(df2[col]):
            nunique = df2[col].nunique(dropna=True)
            if nunique > cardinality_limit:
                top = df2[col].value_counts().nlargest(top_k).index
                df2[col] = df2[col].where(df2[col].isin(top), other="<<OTHER>>")
                info(f"Column '{col}' reduced from {nunique} → top {top_k}+OTHER.")
    return df2

def synthesize_tabular(df: pd.DataFrame, n_samples: int, target: Optional[str]=None,
                       max_train_rows: int = 5000, cardinality_limit: int = 1000, top_k: int = 100) -> Tuple[pd.DataFrame, str]:
    df = df.reset_index(drop=True)
    if df.empty: raise ValueError("Input dataframe is empty.")

    safe_df = reduce_cardinality(df, cardinality_limit=cardinality_limit, top_k=top_k)

    train_df = safe_df
    if len(safe_df) > max_train_rows:
        train_df = safe_df.sample(max_train_rows, random_state=42).reset_index(drop=True)
        info(f"Training on subsample {len(train_df)} / {len(safe_df)} rows.")

    try:
        from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        meta = SingleTableMetadata(); meta.detect_from_dataframe(train_df)
        if target and target in train_df.columns:
            try: meta.update_column(column_name=target, sdtype='categorical')
            except: pass
        try:
            synth = CTGANSynthesizer(meta); synth.fit(train_df)
            out = synth.sample(n_samples)
            return out, "SDV_CTGAN"
        except Exception as e_ct:
            warn(f"CTGAN failed: {e_ct} → trying GaussianCopula.")
            try:
                synth = GaussianCopulaSynthesizer(meta); synth.fit(train_df)
                out = synth.sample(n_samples)
                return out, "SDV_GaussianCopula"
            except Exception as e_gc:
                warn(f"GaussianCopula failed: {e_gc} → fallback.")
    except Exception as e:
        info(f"SDV not available: {e}")

    try:
        from sdv.tabular import CTGAN
        model = CTGAN(); model.fit(train_df)
        out = model.sample(n_samples)
        return out, "SDV_tabular_CTGAN"
    except Exception as e:
        info(f"Legacy CTGAN failed: {e}")

    # Bootstrap fallback
    real = df.copy(); synth_rows = []
    num_cols = real.select_dtypes(include=[np.number]).columns
    for _ in range(n_samples):
        row = real.sample(1, replace=True).iloc[0].to_dict()
        for c in num_cols:
            val = row[c]
            if pd.notnull(val):
                noise = np.random.normal(0, 0.01 * (real[c].std() or 1.0))
                row[c] = val + noise
        synth_rows.append(row)
    out = pd.DataFrame(synth_rows)
    return out, "Bootstrap+Jitter"

def balance_by_target(df: pd.DataFrame, target: str, per_class: int,
                      max_train_rows: int = 5000, cardinality_limit: int = 1000, top_k: int = 100) -> pd.DataFrame:
    balanced_parts = []
    for cls, grp in df.groupby(target):
        need = max(0, per_class - len(grp))
        if need == 0:
            balanced_parts.append(grp); continue
        synth, method = synthesize_tabular(grp, need,
                                           max_train_rows=max_train_rows,
                                           cardinality_limit=cardinality_limit,
                                           top_k=top_k)
        synth[target] = cls
        balanced_parts.append(pd.concat([grp, synth], ignore_index=True))
    return pd.concat(balanced_parts, ignore_index=True)

# -----------------------------
# TEXT AUGMENTATION
# -----------------------------

def paraphrase_pegasus(texts: List[str], num_return_sequences=3, max_length=128) -> List[List[str]]:
    try:
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        model_name = 'tuner007/pegasus_paraphrase'
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
        outputs = []
        for t in texts:
            batch = tokenizer([t], truncation=True, padding='longest', return_tensors='pt')
            gen = model.generate(**batch, max_length=max_length, num_beams=5,
                                 num_return_sequences=num_return_sequences, temperature=1.2)
            outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
            outputs.append(list(dict.fromkeys(outs)))
        return outputs
    except Exception:
        return [[] for _ in texts]

def backtranslate_marian(texts: List[str]) -> List[str]:
    try:
        from transformers import MarianMTModel, MarianTokenizer
        def translate(batch, src, tgt):
            model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
            tok = MarianTokenizer.from_pretrained(model_name)
            m = MarianMTModel.from_pretrained(model_name)
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True)
            gen = m.generate(**enc, max_length=256)
            return tok.batch_decode(gen, skip_special_tokens=True)
        de = translate(texts, 'en', 'de'); en = translate(de, 'de', 'en')
        return en
    except Exception:
        return [""] * len(texts)

def synonym_augment_wordnet(text: str, n_swaps: int = 2) -> str:
    try:
        import nltk; from nltk.corpus import wordnet as wn
    except Exception:
        return text
    tokens = text.split(); idxs = list(range(len(tokens))); random.shuffle(idxs)
    swaps = 0
    for i in idxs:
        word = tokens[i]; syns = wn.synsets(word)
        lemmas = set(l.name().replace('_',' ') for s in syns for l in s.lemmas())
        lemmas = [l for l in lemmas if l.lower() != word.lower() and l.isalpha()]
        if lemmas: tokens[i] = random.choice(lemmas); swaps += 1
        if swaps >= n_swaps: break
    return ' '.join(tokens)

def augment_texts(texts: List[str], k: int = 3) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    pegasus = paraphrase_pegasus(texts, num_return_sequences=k)
    bt = backtranslate_marian(texts)
    for i, t in enumerate(texts):
        variants = []
        if pegasus[i]: variants.extend(pegasus[i][:k])
        if len(variants) < k and bt[i]: variants.append(bt[i])
        while len(variants) < k: variants.append(synonym_augment_wordnet(t))
        seen = []
        for v in variants:
            if v not in seen: seen.append(v)
        out[t] = seen[:k]
    return out

# -----------------------------
# IMAGE AUGMENTATION
# -----------------------------
from torchvision import transforms

def torchvision_pipeline():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    ])

def augment_image_pil(img: Image.Image, n: int = 5) -> List[Image.Image]:
    pipe = torchvision_pipeline(); return [pipe(img) for _ in range(n)]

def generate_with_sd(prompt: str, n: int = 4, guidance_scale: float = 7.5, steps: int = 30):
    try:
        from diffusers import StableDiffusionPipeline; import torch
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        if torch.cuda.is_available(): pipe = pipe.to('cuda')
        images = pipe([prompt]*n, guidance_scale=guidance_scale,
                      num_inference_steps=steps).images
        return images
    except Exception as e:
        warn(f"Stable Diffusion not available: {e}"); return []

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="GenAI Data Augmentation", layout="wide")
st.title("🧪 GenAI Data Augmentation Toolkit")
st.caption("Augment tabular, text, and image datasets with generative + classical techniques.")

with st.sidebar:
    st.header("Controls")
    section = st.radio("Choose task", ["Tabular", "Text", "Images"], index=0)
    seed = st.number_input("Random seed", value=42, step=1)
    random.seed(seed); np.random.seed(seed)

    # row limit settings
    max_rows_allowed = st.number_input("Max rows allowed in uploaded CSV", value=20000, step=1000)
    max_train_rows = st.slider("Max rows for training synthesizer", 1000, 20000, 5000, step=500)
    cardinality_limit = st.number_input("Cardinality threshold", min_value=50, max_value=50000, value=1000, step=50)
    top_k_keep = st.number_input("Top-K categories to keep", min_value=5, max_value=1000, value=100, step=5)

if section == "Tabular":
    st.subheader("📊 Tabular Augmentation")
    file = st.file_uploader("Upload CSV", type=["csv"])
    col1, col2, col3 = st.columns([1,1,1])
    n_samples = col1.number_input("# synthetic rows", min_value=10, max_value=100000, value=500, step=10)
    target_col = col2.text_input("Target column (optional)")
    per_class = col3.number_input("Per-class min rows", min_value=0, max_value=100000, value=0, step=10)

    if file is not None:
        df = pd.read_csv(file)
        if len(df) > max_rows_allowed:
            warn(f"Uploaded file has {len(df)} rows, which exceeds the limit ({max_rows_allowed}). Please reduce size.")
        else:
            st.write("Preview real data:"); st.dataframe(df.head())
            if st.button("Generate Synthetic"):
                with st.spinner("Synthesizing..."):
                    if target_col and per_class > 0 and target_col in df.columns:
                        synth_df = balance_by_target(df, target_col, int(per_class),
                                                     max_train_rows=max_train_rows,
                                                     cardinality_limit=cardinality_limit,
                                                     top_k=top_k_keep)
                        method = "Balanced per class"
                    else:
                        synth_df, method = synthesize_tabular(df, int(n_samples),
                                                              max_train_rows=max_train_rows,
                                                              cardinality_limit=cardinality_limit,
                                                              top_k=top_k_keep)
                success(f"Done with method: {method}. Synthetic rows: {len(synth_df)}")
                st.dataframe(synth_df.head())
                try:
                    fig = compare_distributions(df, synth_df); st.pyplot(fig)
                except Exception as e:
                    warn(f"Validation plot failed: {e}")
                buf = io.BytesIO(); synth_df.to_csv(buf, index=False)
                st.download_button("Download synthetic CSV", data=buf.getvalue(),
                                   file_name="synthetic.csv", mime="text/csv")

elif section == "Text":
    st.subheader("📝 Text Augmentation")
    sample = "Machine learning models often overfit on small datasets.\nData augmentation can improve generalization."
    text_input = st.text_area("Input lines", value=sample, height=160)
    k = st.slider("Variants per line", 1, 10, 3)
    if st.button("Generate Paraphrases"):
        lines = [l.strip() for l in text_input.splitlines() if l.strip()]
        with st.spinner("Generating..."):
            aug = augment_texts(lines, k=k)
        for src, variants in aug.items():
            st.markdown(f"**Source:** {src}")
            for j, v in enumerate(variants, 1): st.write(f"{j}. {v}")
            st.divider()
        rows = [{"source": src, "augmented": v} for src, vs in aug.items() for v in vs]
        df_out = pd.DataFrame(rows)
        buf = io.BytesIO(); df_out.to_csv(buf, index=False)
        st.download_button("Download CSV", data=buf.getvalue(),
                           file_name="text_augmentations.csv", mime="text/csv")

else:  # Images
    st.subheader("🖼️ Image Augmentation")
    imgs = st.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    n_variants = st.slider("Variants per image", 1, 20, 5)
    colA, colB = st.columns(2)
    prompt = colA.text_input("(Optional) SD prompt")
    sd_count = colB.slider("# SD images", 0, 12, 0)

    if st.button("Run Image Augmentation"):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for f in imgs or []:
                try: img = Image.open(f).convert('RGB')
                except Exception as e: warn(f"Failed {f.name}: {e}"); continue
                zf.writestr(f"original/{f.name}", f.getvalue())
                augd = augment_image_pil(img, n=n_variants)
                st.write(f"**{f.name}**"); prev_cols = st.columns(min(5, len(augd)))
                for i, im in enumerate(augd):
                    out_name = f"aug/{os.path.splitext(f.name)[0]}__v{i+1}.png"
                    b = io.BytesIO(); im.save(b, format='PNG'); zf.writestr(out_name, b.getvalue())
                    if i < len(prev_cols): prev_cols[i].image(im, caption=out_name.split('/')[-1])
            if prompt and sd_count > 0:
                sd_images = generate_with_sd(prompt, n=sd_count)
                for i, im in enumerate(sd_images):
                    out_name = f"sd/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.png"
                    b = io.BytesIO(); im.save(b, format='PNG'); zf.writestr(out_name, b.getvalue())
                    st.image(im, caption=out_name)
        zip_buf.seek(0)
        st.download_button("Download all (ZIP)", data=zip_buf.getvalue(),
                           file_name="image_augmentations.zip", mime="application/zip")
