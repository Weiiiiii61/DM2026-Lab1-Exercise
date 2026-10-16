from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0,
    ) -> Dict[str, Any]:
        """Builds the global document-term matrix (DTM) from the text column in session.dataframe.

        Fits a CountVectorizer on the loaded dataset text using default stop_words=None, 
        populating session.feature_matrix and session.feature_names. Stores the fitted 
        vectorizer at session.artifacts["count_vectorizer"] and updates document category labels.

        Args:
            ngram_range: Range of n-gram sizes to extract, e.g. [1, 1] for unigrams or [1, 2] for unigrams + bigrams.
            min_df: Minimum document frequency threshold for terms.
            max_df: Maximum document frequency threshold or ratio for terms.

        Returns:
            A summary dictionary containing result_id, n_documents, n_terms, non_zero,
            total_elements, and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded or 'text' column missing in session.dataframe."}

        if len(ngram_range) != 2:
            return {"status": "error", "message": "ngram_range must contain exactly two integers [min_n, max_n]."}

        try:
            min_n, max_n = int(ngram_range[0]), int(ngram_range[1])
        except (ValueError, TypeError):
            return {"status": "error", "message": "ngram_range elements must be integers."}

        if min_n <= 0 or min_n > max_n:
            return {"status": "error", "message": f"Invalid ngram_range [{min_n}, {max_n}]. Expected min_n > 0 and min_n <= max_n."}

        vectorizer = CountVectorizer(
            ngram_range=(min_n, max_n),
            min_df=min_df,
            max_df=max_df,
            stop_words=None,
        )

        texts = session.dataframe["text"].fillna("").astype(str)
        try:
            feature_matrix = vectorizer.fit_transform(texts)
        except ValueError as e:
            return {"status": "error", "message": f"Vocabulary error: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].tolist())

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("dtm"),
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": float(round(sparsity_pct, 4)),
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df}, summary)
        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the global document-term matrix.

        Returns:
            Dict[str, Any]: Summary containing result_id, n_terms, total_term_count, and term_frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Please run build_dtm_tool first."}

        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten().tolist()
        term_frequencies = {name: int(count) for name, count in zip(session.feature_names, counts)}
        total_term_count = sum(term_frequencies.values())

        res_id = session.next_result_id("term_freq")
        summary = {
            "result_id": res_id,
            "n_terms": len(term_frequencies),
            "total_term_count": total_term_count,
            "term_frequencies": term_frequencies
        }
        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Renders a heatmap of a positional slice of the raw global document-term matrix.

        Args:
            n_terms (int): Number of terms (columns) to include in existing vocabulary order (default 20).
            n_documents (int): Number of documents (rows) to include in existing row order (default 20).

        Returns:
            Dict[str, Any]: Summary containing result_id, n_terms, n_documents, terms, and matrix slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Please run build_dtm_tool first."}

        actual_docs, actual_terms = session.feature_matrix.shape
        slice_docs = min(n_documents, actual_docs)
        slice_terms = min(n_terms, actual_terms)

        matrix_slice = session.feature_matrix[0:slice_docs, 0:slice_terms].toarray()
        terms_slice = session.feature_names[0:slice_terms]

        # Build Seaborn Heatmap
        fig, ax = plt.subplots(figsize=(max(6, slice_terms * 0.4), max(4, slice_docs * 0.3)))
        sns.heatmap(
            matrix_slice,
            xticklabels=terms_slice,
            yticklabels=[f"Doc {i}" for i in range(slice_docs)],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            ax=ax
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Attach to session.pending_figure per plotting requirement
        session.pending_figure = fig

        res_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": res_id,
            "n_terms": slice_terms,
            "n_documents": slice_docs,
            "terms": terms_slice,
            "matrix": matrix_slice.tolist()
        }
        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
