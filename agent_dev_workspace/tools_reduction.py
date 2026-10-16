from typing import List, Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """Reduces the dimensionality of session.feature_matrix to 2D and generates a scatter plot.

        Args:
            method (Literal["pca", "tsne", "umap"]): Reduction algorithm to apply (default 'pca').
            n_components (int): Target number of dimensions, usually 2 (default 2).
            random_state (Optional[int]): Random seed for reproducible output.
            perplexity (float): Perplexity parameter for t-SNE (default 30.0).
            n_neighbors (int): Number of neighbors parameter for UMAP (default 15).

        Returns:
            Dict[str, Any]: Summary containing result_id, method, n_components, n_documents,
            a preview of coordinates (first 5 documents), and explained_variance_ratio (if PCA).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built in session. Please run build_dtm_tool first."}

        X_dense = session.feature_matrix.toarray()
        n_samples = X_dense.shape[0]

        if n_samples < 2:
            return {"error": "Cannot reduce dimensions with fewer than 2 documents."}

        explained_variance_ratio = None

        if method == "pca":
            pca = PCA(n_components=n_components)
            coords = pca.fit_transform(X_dense)
            explained_variance_ratio = [float(round(v, 4)) for v in pca.explained_variance_ratio_.tolist()]
        elif method == "tsne":
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state
            )
            coords = tsne.fit_transform(X_dense)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X_dense)
        else:
            return {"error": f"Unsupported reduction method '{method}'."}

        coords_list = [[float(round(c, 4)) for c in row] for row in coords.tolist()]
        session.artifacts["reduced_coordinates"] = coords_list

        # Mandatory Plotting Requirement: Scatter plot colored by category on session.pending_figure
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].tolist()
        if labels is None and session.labels is not None:
            labels = list(session.labels)
        if labels is None:
            labels = ["document"] * n_samples

        plot_df = pd.DataFrame({
            "Dim 1": [c[0] for c in coords_list],
            "Dim 2": [c[1] for c in coords_list],
            "Category": labels
        })

        sns.scatterplot(data=plot_df, x="Dim 1", y="Dim 2", hue="Category", style="Category", s=100, ax=ax)
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        plt.tight_layout()

        # Set figure on session.pending_figure per assignment contract
        session.pending_figure = fig

        res_id = session.next_result_id("reduce_dims")
        summary = {
            "result_id": res_id,
            "method": method,
            "n_components": n_components,
            "n_documents": n_samples,
            "coordinates_preview": coords_list[:5],
            "coordinates": coords_list
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors
        }, summary)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarizes/one-hot encodes document category labels in the working dataset.

        Returns:
            Dict[str, Any]: Summary containing result_id, n_documents, categories, and one_hot_labels preview.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        if cat_col is None:
            return {"error": "DataFrame missing category/category_name column."}

        one_hot_df = pd.get_dummies(df[cat_col], dtype=int)
        categories = list(one_hot_df.columns)
        records = one_hot_df.to_dict(orient="records")

        session.artifacts["binarized_labels"] = records

        res_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": res_id,
            "n_documents": len(df),
            "categories": categories,
            "one_hot_labels": records
        }
        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
