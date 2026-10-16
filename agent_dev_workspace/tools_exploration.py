from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int
    ) -> Dict[str, Any]:
        """Computes the cosine similarity between two documents in session.feature_matrix.

        Args:
            doc_index_1 (int): Zero-based row index of the first document.
            doc_index_2 (int): Zero-based row index of the second document.

        Returns:
            Dict[str, Any]: Summary containing result_id, doc_index_1, doc_index_2, similarity, and document previews.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix built in session. Please run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if not (0 <= doc_index_1 < n_docs) or not (0 <= doc_index_2 < n_docs):
            return {"error": f"Document indices must be between 0 and {n_docs - 1}."}

        vec1 = session.feature_matrix[doc_index_1]
        vec2 = session.feature_matrix[doc_index_2]

        sim_val = float(cosine_similarity(vec1, vec2)[0][0])
        if np.isnan(sim_val):
            sim_val = 0.0
        else:
            sim_val = float(round(sim_val, 4))

        doc_1_text = ""
        doc_2_text = ""
        if session.dataframe is not None and "text" in session.dataframe.columns:
            if doc_index_1 < len(session.dataframe):
                doc_1_text = str(session.dataframe.iloc[doc_index_1]["text"])
            if doc_index_2 < len(session.dataframe):
                doc_2_text = str(session.dataframe.iloc[doc_index_2]["text"])

        res_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": res_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": sim_val,
            "doc_1_text": doc_1_text,
            "doc_2_text": doc_2_text
        }

        session.store_result("cosine_similarity_tool", {
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2
        }, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Computes Pearson correlation matrix for top 20 terms by variance and generates a heatmap.

        Returns:
            Dict[str, Any]: Summary containing result_id, top_terms, n_terms, correlation_matrix_preview, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Please run build_dtm_tool first."}

        X_dense = session.feature_matrix.toarray()
        feature_names = list(session.feature_names)

        variances = np.var(X_dense, axis=0)

        # Take top 20 terms by variance
        top_indices = np.argsort(variances)[::-1][:20]
        top_terms = [feature_names[i] for i in top_indices]
        X_top = X_dense[:, top_indices]

        # Compute Pearson correlation matrix
        corr_matrix = np.corrcoef(X_top, rowvar=False)

        # Handle NaNs or scalar returns
        if corr_matrix.ndim == 0 or np.isnan(corr_matrix).all():
            corr_matrix_list = [[1.0]]
        else:
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            corr_matrix_list = [[float(round(val, 4)) for val in row] for row in corr_matrix.tolist()]

        session.artifacts["feature_correlation_matrix"] = corr_matrix_list
        preview_matrix = [row[:5] for row in corr_matrix_list[:5]]

        # Generate Heatmap figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix_list,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Top Terms Pearson Correlation Heatmap")
        plt.tight_layout()

        session.pending_figure = fig

        res_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": res_id,
            "top_terms": top_terms,
            "n_terms": len(top_terms),
            "correlation_matrix_preview": preview_matrix,
            "correlation_matrix": corr_matrix_list
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
