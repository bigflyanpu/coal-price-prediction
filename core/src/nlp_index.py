from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


@dataclass
class NLPConfig:
    policy_dims: int = 12
    lda_topics: int = 6
    bert_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    use_bert: bool = True
    bert_local_files_only: bool = False


class PolicySentimentIndexer:
    def __init__(self, cfg: NLPConfig = NLPConfig()) -> None:
        self.cfg = cfg
        self.policy_count_vectorizer = CountVectorizer(max_features=2000, min_df=2)
        self.lda = LatentDirichletAllocation(n_components=cfg.lda_topics, random_state=42, learning_method="batch")
        self.policy_tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)

    def _bert_embed(self, texts: list[str]) -> np.ndarray:
        if not self.cfg.use_bert:
            tfidf = self.policy_tfidf.fit_transform(texts).toarray()
            return tfidf

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.bert_model_name,
                local_files_only=self.cfg.bert_local_files_only,
            )
            model = AutoModel.from_pretrained(
                self.cfg.bert_model_name,
                local_files_only=self.cfg.bert_local_files_only,
            )
            model.eval()

            embs = []
            for i in range(0, len(texts), 32):
                batch = texts[i : i + 32]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    out = model(**encoded)
                pooled = out.last_hidden_state.mean(dim=1).cpu().numpy()
                embs.append(pooled)
            return np.vstack(embs)
        except Exception:
            # Fallback to TF-IDF vectors when model download/runtime is unavailable.
            tfidf = self.policy_tfidf.fit_transform(texts).toarray()
            return tfidf

    def build_policy_index(self, policy_df: pd.DataFrame) -> pd.DataFrame:
        if policy_df.empty:
            return pd.DataFrame(columns=["date"] + [f"policy_index_{i+1}" for i in range(self.cfg.policy_dims)])

        docs = (policy_df["title"].fillna("") + " " + policy_df["body"].fillna("")).tolist()
        bow = self.policy_count_vectorizer.fit_transform(docs)
        topic_dist = self.lda.fit_transform(bow)
        semantic = self._bert_embed(docs)
        semantic = np.nan_to_num(semantic, nan=0.0, posinf=0.0, neginf=0.0)
        semantic = np.clip(semantic, -1e3, 1e3)

        # In fast mode we avoid unstable heavy decomposition and expand topic-based indices directly.
        if not self.cfg.use_bert:
            topic_df = pd.DataFrame(topic_dist, columns=[f"policy_topic_{i+1}" for i in range(topic_dist.shape[1])])
            topic_df["policy_doc_len"] = np.array([len(t) for t in docs], dtype=float)
            topic_df["policy_unique_ratio"] = np.array(
                [len(set(t.split())) / (len(t.split()) + 1e-6) for t in docs], dtype=float
            )
            topic_df["policy_keyword_boost"] = np.array(
                [sum(k in t for k in ["保供", "稳价", "调控", "长协", "安全"]) for t in docs], dtype=float
            )
            merged = topic_df.to_numpy(dtype=float)
            scaler = StandardScaler()
            merged = scaler.fit_transform(merged)
            comp = min(self.cfg.policy_dims, merged.shape[1])
            reduced = merged[:, :comp]
        else:
            # For full mode use SVD with defensive fallbacks.
            if semantic.ndim == 1:
                semantic = semantic.reshape(-1, 1)
            if semantic.shape[1] > 256:
                semantic = semantic[:, :256]
            merged = np.hstack([topic_dist, semantic])
            merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
            merged = np.clip(merged, -1e3, 1e3)
            scaler = StandardScaler()
            merged = scaler.fit_transform(merged)
            comp = min(self.cfg.policy_dims, merged.shape[1], merged.shape[0] - 1)
            if comp <= 0:
                comp = 1
            try:
                svd = TruncatedSVD(n_components=comp, random_state=42)
                reduced = svd.fit_transform(merged)
            except Exception:
                reduced = merged[:, :comp]

        # If comp < target dims, zero pad for stable schema.
        if comp < self.cfg.policy_dims:
            pad = np.zeros((reduced.shape[0], self.cfg.policy_dims - comp))
            reduced = np.hstack([reduced, pad])

        out = pd.DataFrame(reduced[:, : self.cfg.policy_dims], columns=[f"policy_index_{i+1}" for i in range(self.cfg.policy_dims)])
        out["date"] = pd.to_datetime(policy_df["date"]).to_numpy()

        daily = out.groupby("date").mean().reset_index()
        daily["policy_strength"] = daily[[c for c in daily.columns if c.startswith("policy_index_")]].abs().mean(axis=1)
        daily["policy_uncertainty"] = daily["policy_strength"].rolling(7, min_periods=1).std().fillna(0)
        return daily

    def build_sentiment_index(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        if sentiment_df.empty:
            return pd.DataFrame(columns=["date", "sentiment_score", "sentiment_heat", "sentiment_volatility"])

        pos_words = ["上涨", "增长", "改善", "利好", "稳定", "回暖"]
        neg_words = ["下跌", "紧张", "下滑", "风险", "波动", "承压"]

        texts = (sentiment_df["title"].fillna("") + " " + sentiment_df["body"].fillna("")).tolist()
        score = []
        for t in texts:
            p = sum(w in t for w in pos_words)
            n = sum(w in t for w in neg_words)
            score.append((p - n) / (p + n + 1.0))

        temp = sentiment_df.copy()
        temp["date"] = pd.to_datetime(temp["date"])
        temp["doc_score"] = score

        daily = temp.groupby("date").agg(sentiment_score=("doc_score", "mean"), sentiment_heat=("news_id", "count"))
        daily = daily.reset_index()
        daily["sentiment_volatility"] = daily["sentiment_score"].rolling(14, min_periods=2).std().fillna(0)
        return daily

    def build_indices(self, policy_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.build_policy_index(policy_df), self.build_sentiment_index(sentiment_df)
