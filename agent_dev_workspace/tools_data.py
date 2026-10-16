import os
from typing import Dict, Any, List, Optional, Literal
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

# Ensure nltk punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and directories in a repository folder.

        Args:
            subdirectory (str): Subdirectory relative to workspace root (default ".").

        Returns:
            Dict[str, Any]: Dict containing result_id, directory, files, and directories.
        """
        target_dir = os.path.abspath(subdirectory)
        
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            items = os.listdir(target_dir)
            files = sorted([f for f in items if os.path.isfile(os.path.join(target_dir, f)) and not f.startswith(".")])
            directories = sorted([d for d in items if os.path.isdir(os.path.join(target_dir, d)) and not d.startswith(".") and d != "__pycache__"])
        else:
            files = []
            directories = []

        res_id = session.next_result_id("list_files")
        summary = {
            "result_id": res_id,
            "directory": subdirectory,
            "files": files,
            "directories": directories
        }
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes each document's text into unigram tokens (lowercased).

        Returns:
            Dict[str, Any]: Summary containing result_id, document count, total tokens, and sample tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        def _tokenize(val):
            if pd.isna(val) or not str(val).strip():
                return []
            text_str = str(val).lower()
            return nltk.word_tokenize(text_str)

        tokens_series = df["text"].apply(_tokenize)
        session.dataframe["unigrams"] = tokens_series
        session.artifacts["unigrams"] = tokens_series.tolist()

        total_tokens = int(tokens_series.apply(len).sum())
        doc_count = len(df)

        res_id = session.next_result_id("tokenize")
        summary = {
            "result_id": res_id,
            "n_documents": doc_count,
            "total_tokens": total_tokens,
            "sample_unigrams": tokens_series.iloc[0] if doc_count > 0 else []
        }
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing or empty text values in the working dataset.

        Returns:
            Dict[str, Any]: Summary containing result_id, missing_count, total_rows, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        missing_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_indices = [int(idx) for idx in df[missing_mask].index.tolist()]
        missing_count = len(missing_indices)
        total_rows = len(df)

        res_id = session.next_result_id("check_missing")
        summary = {
            "result_id": res_id,
            "missing_count": missing_count,
            "total_rows": total_rows,
            "missing_indices": missing_indices
        }
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """Checks for and optionally drops duplicate documents in the working dataset.

        When dropping duplicates (drop=True), uses keep=False to drop all copies of duplicate rows,
        matching the pipeline specification.

        Args:
            drop (bool): Whether to drop duplicate rows from session.dataframe (default False).
            subset (Optional[List[str]]): List of column names to check for duplicates. Defaults to ['text'].

        Returns:
            Dict[str, Any]: Summary containing result_id, duplicate_count, duplicate_indices, dropped, and remaining_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        cols_to_check = subset if subset is not None else (["text"] if "text" in df.columns else None)

        dup_mask = df.duplicated(subset=cols_to_check, keep="first")
        dup_indices = [int(idx) for idx in df[dup_mask].index.tolist()]
        duplicate_count = len(dup_indices)

        if drop and duplicate_count > 0:
            session.dataframe.drop_duplicates(subset=cols_to_check, keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        remaining_rows = len(session.dataframe)
        res_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": res_id,
            "duplicate_count": duplicate_count,
            "duplicate_indices": dup_indices,
            "dropped": drop,
            "remaining_rows": remaining_rows
        }
        session.store_result("check_duplicates_tool", {"drop": drop, "subset": subset}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        replace: bool = False
    ) -> Dict[str, Any]:
        """Subsamples rows from the working dataset.

        Args:
            n (Optional[int]): Number of items to sample.
            frac (Optional[float]): Fraction of items to sample.
            random_state (Optional[int]): Random seed for reproducible sampling.
            replace (bool): Whether to sample with replacement (default False).

        Returns:
            Dict[str, Any]: Summary containing result_id, n_samples, and sampled_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if df.empty:
            return {"error": "Working dataset is empty."}

        if n is None and frac is None:
            n = min(5, len(df))

        sampled_df = df.sample(n=n, frac=frac, replace=replace, random_state=random_state)
        sampled_indices = [int(idx) for idx in sampled_df.index.tolist()]

        res_id = session.next_result_id("sample_data")
        summary = {
            "result_id": res_id,
            "n_samples": len(sampled_df),
            "sampled_indices": sampled_indices
        }
        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state, "replace": replace},
            summary
        )
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes text length statistics and creates a boxplot grouped by category.

        Returns:
            Dict[str, Any]: Summary containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        temp_df = df.copy()
        temp_df["text_length"] = temp_df["text"].fillna("").astype(str).apply(len)

        overall_stats = temp_df["text_length"].describe().to_dict()

        cat_col = "category_name" if "category_name" in temp_df.columns else ("category" if "category" in temp_df.columns else None)
        per_category_stats = {}

        if cat_col is not None:
            for cat, group in temp_df.groupby(cat_col):
                per_category_stats[str(cat)] = group["text_length"].describe().to_dict()

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=temp_df, x=cat_col, y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length (characters)")
            plt.tight_layout()

            session.pending_figure = fig

        res_id = session.next_result_id("describe_data")
        summary = {
            "result_id": res_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }
        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        position: Literal["head", "tail", "random"] = "head"
    ) -> Dict[str, Any]:
        """Peeks at a specified slice of rows (head, tail, or random) in the working dataset.

        Args:
            n (int): Number of rows to inspect (default 5).
            position (Literal["head", "tail", "random"]): Slice position: 'head', 'tail', or 'random' (default 'head').

        Returns:
            Dict[str, Any]: Summary containing result_id, position, n_rows, shape, columns, and rows.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        if df.empty:
            return {"error": "Working dataset is empty."}

        if position == "head":
            sliced_df = df.head(n)
        elif position == "tail":
            sliced_df = df.tail(n)
        elif position == "random":
            sliced_df = df.sample(n=min(n, len(df)))
        else:
            sliced_df = df.head(n)

        rows = sliced_df.to_dict(orient="records")
        res_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": res_id,
            "position": position,
            "n_rows": len(sliced_df),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "rows": rows
        }
        session.store_result("inspect_data_tool", {"n": n, "position": position}, summary)
        return summary

    return [
        list_files_tool,
        tokenize_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        inspect_data_tool
    ]
