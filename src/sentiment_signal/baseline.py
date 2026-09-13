"""Baselines: TF-IDF features (also the gate G4 fallback) and unfitted VADER scores."""

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentiment_signal.model import stock_day_scores


def tfidf_featurize(
    train: pl.DataFrame, evals: list[pl.DataFrame], params: dict, test_year: int
) -> tuple[object, list[object]]:
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, min_df=5, max_features=50_000, ngram_range=tuple(params["ngram"])
    )
    x_train = vectorizer.fit_transform(train["headline"].to_list())
    return x_train, [vectorizer.transform(e["headline"].to_list()) for e in evals]


def vader_scores(panel: pl.DataFrame) -> pl.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    compound = np.array(
        [analyzer.polarity_scores(h)["compound"] for h in panel["headline"].to_list()]
    )
    return stock_day_scores(panel, compound)
