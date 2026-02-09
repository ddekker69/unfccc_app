# utils/diagnostics.py

import pandas as pd


def compute_extraction_diagnostics(df):
    total_docs = len(df)
    empty_docs = df['text'].apply(lambda x: not isinstance(x, str) or len(x.strip()) == 0).sum()
    non_empty_docs = total_docs - empty_docs
    empty_ratio = empty_docs / total_docs * 100

    return {
        "total": total_docs,
        "non_empty": non_empty_docs,
        "empty": empty_docs,
        "empty_ratio": empty_ratio
    }
