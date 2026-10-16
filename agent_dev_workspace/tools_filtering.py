from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Flags low-variance terms in the global document-term matrix for removal.

        Args:
            threshold (float): Variance threshold below which terms are filtered out (default 0.0).

        Returns:
            Dict[str, Any]: Summary containing result_id, threshold, total_features,
            kept_count, filtered_count, kept_features, and filtered_features.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Please run build_dtm_tool first."}

        matrix_dense = session.feature_matrix.toarray()
        variances = np.var(matrix_dense, axis=0, ddof=0)

        report_df = pd.DataFrame({
            "feature": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_mask = report_df["variance"] >= threshold
        kept_df = report_df[kept_mask]
        filtered_df = report_df[~kept_mask]

        kept_features = kept_df["feature"].tolist()
        filtered_features = filtered_df["feature"].tolist()

        res_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": res_id,
            "threshold": float(threshold),
            "total_features": len(session.feature_names),
            "kept_count": len(kept_features),
            "filtered_count": len(filtered_features),
            "kept_features": kept_features,
            "filtered_features": filtered_features
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Computes Pearson correlation between each term's count and a target class vector.

        Args:
            target_class (str): Target category name to compute correlation against (e.g. 'catB').
            threshold (float): Pearson correlation absolute threshold below which terms are filtered (default 0.0).

        Returns:
            Dict[str, Any]: Summary containing result_id, target_class, threshold, total_features,
            kept_count, filtered_count, kept_features, and filtered_features.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Please run build_dtm_tool first."}

        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        if cat_col is None:
            return {"error": "DataFrame missing category/category_name column."}

        y = (df[cat_col].astype(str) == str(target_class)).astype(float).values
        matrix_dense = session.feature_matrix.toarray()

        pearson_rs = []
        for j in range(matrix_dense.shape[1]):
            x_j = matrix_dense[:, j]
            if np.std(x_j) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r_matrix = np.corrcoef(x_j, y)
                r = float(r_matrix[0, 1])
                if np.isnan(r):
                    r = 0.0
            pearson_rs.append(r)

        report_df = pd.DataFrame({
            "feature": session.feature_names,
            "pearson_r": pearson_rs
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        kept_mask = report_df["pearson_r"].abs() >= threshold
        kept_df = report_df[kept_mask]
        filtered_df = report_df[~kept_mask]

        kept_features = kept_df["feature"].tolist()
        filtered_features = filtered_df["feature"].tolist()

        res_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "threshold": float(threshold),
            "total_features": len(session.feature_names),
            "kept_count": len(kept_features),
            "filtered_count": len(filtered_features),
            "kept_features": kept_features,
            "filtered_features": filtered_features
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Computes Spearman rank correlation between each term's count and a target class vector.

        Args:
            target_class (str): Target category name to compute correlation against (e.g. 'catB').
            threshold (float): Spearman correlation absolute threshold below which terms are filtered (default 0.0).

        Returns:
            Dict[str, Any]: Summary containing result_id, target_class, threshold, total_features,
            kept_count, filtered_count, kept_features, and filtered_features.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Please run build_dtm_tool first."}

        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        if cat_col is None:
            return {"error": "DataFrame missing category/category_name column."}

        y = (df[cat_col].astype(str) == str(target_class)).astype(float).values
        matrix_dense = session.feature_matrix.toarray()

        spearman_rs = []
        for j in range(matrix_dense.shape[1]):
            x_j = matrix_dense[:, j]
            if np.std(x_j) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                res_corr = spearmanr(x_j, y)
                r = float(res_corr.statistic if hasattr(res_corr, 'statistic') else res_corr[0])
                if np.isnan(r):
                    r = 0.0
            spearman_rs.append(r)

        report_df = pd.DataFrame({
            "feature": session.feature_names,
            "spearman_r": spearman_rs
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        kept_mask = report_df["spearman_r"].abs() >= threshold
        kept_df = report_df[kept_mask]
        filtered_df = report_df[~kept_mask]

        kept_features = kept_df["feature"].tolist()
        filtered_features = filtered_df["feature"].tolist()

        res_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "threshold": float(threshold),
            "total_features": len(session.feature_names),
            "kept_count": len(kept_features),
            "filtered_count": len(filtered_features),
            "kept_features": kept_features,
            "filtered_features": filtered_features
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
