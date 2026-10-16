import os
import tempfile
from typing import Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI mining modules
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 10
    ) -> Dict[str, Any]:
        """Mines frequent term patterns for a single category using PAMI algorithms.

        Args:
            category_name (str): Category to mine documents for.
            filtering_method (Literal["variance", "tfidf", "term_frequency"]): Per-category vocabulary filter method.
            algorithm (Literal["fpgrowth", "topk", "maxfpgrowth"]): PAMI mining algorithm (default 'fpgrowth').
            min_sup (int): Minimum support threshold for fpgrowth/maxfpgrowth (default 1).
            k (int): Number of top patterns to return for topk (default 10).

        Returns:
            Dict[str, Any]: Summary containing result_id, category_name, filtering_method,
            algorithm, patterns_found_count, and patterns list.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        if cat_col is None:
            return {"error": "DataFrame missing category/category_name column."}

        # Step 1: Filter to target category and build fresh, category-local DTM
        cat_df = df[df[cat_col].astype(str) == str(category_name)].copy()
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        texts = cat_df["text"].fillna("").astype(str).tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X_cat = vectorizer.fit_transform(texts)
        except ValueError:
            # Empty vocabulary after stopword removal
            res_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns_found_count": 0,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        feature_names = np.array(vectorizer.get_feature_names_out())
        X_dense = X_cat.toarray()

        if X_dense.shape[1] == 0:
            res_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns_found_count": 0,
                "patterns": [],
                "note": "no terms survived initial vectorization"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 2: Apply chosen vocabulary filtering method
        if filtering_method in ["variance", "term_frequency"]:
            if filtering_method == "variance":
                scores = np.var(X_dense, axis=0, ddof=0)
            else:
                scores = np.sum(X_dense, axis=0)

            if len(scores) <= 2:
                # 5th-95th percentile across <= 2 terms excludes all terms
                surviving_indices = []
            else:
                p5 = np.percentile(scores, 5)
                p95 = np.percentile(scores, 95)
                surviving_indices = np.where((scores >= p5) & (scores <= p95))[0].tolist()

        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(X_cat).toarray()
            mean_tfidf = np.mean(tfidf, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            surviving_indices = np.where(mean_tfidf >= p20)[0].tolist()
        else:
            surviving_indices = list(range(len(feature_names)))

        # Step 3: Handle zero terms surviving
        if len(surviving_indices) == 0:
            res_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns_found_count": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        surviving_terms = feature_names[surviving_indices]
        X_surviving = X_dense[:, surviving_indices]

        # Step 4: Convert surviving terms to transactional database DataFrame
        tx_df = pd.DataFrame(X_surviving, columns=surviving_terms)

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "transactional.db")
            
            # Convert DF2DB passing required condition and thresholdValue
            converter = DF2DB.DF2DB(tx_df)
            converter.convert2TransactionalDatabase(db_file, condition=">=", thresholdValue=1)

            # Step 5: Run chosen PAMI mining algorithm
            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE.FAE(iFile=db_file, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
            else:
                miner = FPGrowth.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

        patterns = []
        if patterns_df is not None and not patterns_df.empty:
            for _, row in patterns_df.iterrows():
                pat_str = str(row.iloc[0]).strip()
                sup_val = int(row.iloc[1])
                patterns.append({
                    "pattern": pat_str,
                    "support": sup_val
                })

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        res_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns_found_count": len(patterns),
            "patterns": patterns
        }

        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary)
        return summary

    return [mine_patterns_tool]
