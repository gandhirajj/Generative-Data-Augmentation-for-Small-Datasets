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
    n_plots = max(len(num_cols) + len(cat_cols), 1)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    for c in num_cols:
        ax = axes[idx]; idx += 1
        ax.hist(real[c].dropna().values, bins=30, alpha=0.5, label="real", density=True)
        ax.hist(synth[c].dropna().values, bins=30, alpha=0.5, label="synthetic", density=True)
        ax.set_title(f"Numeric: {c}")
        ax.legend()

    for c in cat_cols:
        ax = axes[idx]; idx += 1
        real_counts = real[c].astype(str).value_counts(normalize=True)
        synth_counts = synth[c].astype(str).value_counts(normalize=True)
        all_idx = sorted(set(real_counts.index).union(set(synth_counts.index)))
        real_vals = [real_counts.get(k, 0) for k in all_idx]
        synth_vals = [synth_counts.get(k, 0) for k in all_idx]
        x = np.arange(len(all_idx))
        width = 0.35
        ax.bar(x - width / 2, real_vals, width, label="real")
        ax.bar(x + width / 2, synth_vals, width, label="synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels(all_idx, rotation=45, ha="right")
        ax.set_title(f"Categorical: {c}")
        ax.legend()

    fig.tight_layout()
    return fig


# -----------------------------
# TABULAR AUGMENTATION
# -----------------------------

def synthesize_tabular(df: pd.DataFrame, n_samples: int, target: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """CTGAN -> GaussianCopula -> bootstrap+jitter. Drops high-cardinality & ID-like columns."""
    # Drop ID-like / near-unique columns
    too_unique = [c for c in df.columns if df[c].nunique() / len(df) > 0.9]
    if too_unique:
        warn(f"Dropping near-unique ID-like columns: {', '.join(too_unique)}")
        df = df.drop(columns=too_unique)

    # Drop high-cardinality categoricals
    high_card_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() > 500]
    if high_card_cols:
        warn(f"Dropping high-cardinality columns (>500 uniques): {', '.join(high_card_cols)}")
        df = df.drop(columns=high_card_cols)

    if df.empty:
        return pd.DataFrame(), "No usable columns"

    # Try SDV CTGAN/GaussianCopula
    try:
        from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(df)
        if target and target in df.columns:
            try:
                meta.update_column(column_name=target, sdtype="categorical")
            except Exception:
                pass
        try:
            synth = CTGANSynthesizer(meta)
            synth.fit(df)
            out = synth.sample(n_samples)
            return out, "SDV_CTGAN"
        except Exception:
            synth = GaussianCopulaSynthesizer(meta)
            synth.fit(df)
            out = synth.sample(n_samples)
            return out, "SDV_GaussianCopula"
    except Exception:
        pass

    # Try legacy CTGAN
    try:
        from sdv.tabular import CTGAN
        model = CTGAN()
        model.fit(df)
        out = model.sample(n_samples)
        return out, "SDV_tabular_CTGAN"
    except Exception:
        pass

    # Fallback: bootstrap+jitter
    real = df.copy()
    synth_rows = []
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


def balance_by_target(df: pd.DataFrame, target: str, per_class: int) -> pd.DataFrame:
    parts = []
    for cls, grp in df.groupby(target):
        need = max(0, per_class - len(grp))
        if need == 0:
            parts.append(grp); continue
        synth, _ = synthesize_tabular(grp, need, target=target)
        if not synth.empty:
            synth[target] = cls
            parts.append(pd.concat([grp, synth], ignore_index=True))
    return pd.concat(parts, ignore_index=True)


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="GenAI Data Augmentation", layout="wide")
st.title("🧪 GenAI Data Augmentation Toolkit")
st.caption("Augment tabular, text, and image datasets with generative + fallback techniques.")

with st.sidebar:
    st.header("Controls")
    section = st.radio("Choose task", ["Tabular"], index=0)
    seed = st.number_input("Random seed", value=42, step=1)
    random.seed(seed); np.random.seed(seed)
    max_rows_allowed = st.number_input("Max rows to use", value=5000, step=1000)

if section == "Tabular":
    st.subheader("📊 Tabular Augmentation")
    file = st.file_uploader("Upload CSV", type=["csv"])
    col1, col2, col3 = st.columns([1, 1, 1])
    n_samples = col1.number_input("# synthetic rows", min_value=10, max_value=100000, value=500, step=10)
    target_col = col2.text_input("Target column (optional for balancing)")
    per_class = col3.number_input("Per-class min rows (if target given)", min_value=0, max_value=100000, value=0, step=10)

    if file is not None:
        df = pd.read_csv(file)
        if len(df) > max_rows_allowed:
            warn(f"Dataset has {len(df)} rows, downsampling to {max_rows_allowed} for processing.")
            df = df.sample(max_rows_allowed, random_state=seed).reset_index(drop=True)

        st.write("Preview real data:"); st.dataframe(df.head())

        if st.button("Generate Synthetic"):
            with st.spinner("Synthesizing..."):
                if target_col and per_class > 0 and target_col in df.columns:
                    synth_df = balance_by_target(df, target_col, int(per_class))
                    method = "Balance by class"
                else:
                    synth_df, method = synthesize_tabular(df, int(n_samples), target=target_col or None)

            if synth_df.empty:
                warn("No usable columns left after cleaning. Cannot generate synthetic data.")
            else:
                success(f"Done with method: {method}. Synthetic rows: {len(synth_df)}")
                st.dataframe(synth_df.head())
                try:
                    fig = compare_distributions(df, synth_df); st.pyplot(fig)
                except Exception as e:
                    warn(f"Validation plot failed: {e}")
                buf = io.BytesIO(); synth_df.to_csv(buf, index=False)
                st.download_button("Download synthetic CSV", data=buf.getvalue(),
                                   file_name="synthetic.csv", mime="text/csv")

st.caption(f"Built at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
