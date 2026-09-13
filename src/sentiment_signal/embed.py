"""ChronoBERT headline embeddings: one model version per test year, cached per (version, year)."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
import torch

from sentiment_signal import config


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def vintage_for(test_year: int) -> str:
    """Latest version whose pretraining text ends before the test year starts."""
    return config.CHRONOBERT.format(year=test_year - 1)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class ChronoEmbedder:
    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        max_tokens: int = config.MAX_TOKENS,
        batch_size: int = config.EMBED_BATCH,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa")
        self.model.to(self.device).eval()
        self.max_tokens, self.batch_size = max_tokens, batch_size

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            pooled = mean_pool(hidden, batch["attention_mask"])
            chunks.append(pooled.float().cpu().numpy().astype(np.float16))
        return np.concatenate(chunks) if chunks else np.zeros((0, config.EMBED_DIM), np.float16)


def _cache_dir(cache_root: Path, model_id: str, year: int) -> Path:
    return cache_root / model_id.replace("/", "__") / str(year)


def embed_rows(
    rows: pl.DataFrame,
    model_id: str,
    cache_root: Path,
    embedder_factory: Callable[[str], Embedder],
    dim: int = config.EMBED_DIM,
) -> np.ndarray:
    """float16 matrix aligned to `rows`; unique headlines embedded once per year and cached."""
    indexed = rows.with_row_index("_pos").with_columns(_year=pl.col("signal_date").dt.year())
    out = np.zeros((rows.height, dim), dtype=np.float16)
    embedder = None
    for year in sorted(indexed["_year"].unique().to_list()):
        part = indexed.filter(pl.col("_year") == year)
        cache = _cache_dir(cache_root, model_id, year)
        if (cache / "vectors.npy").exists():
            texts = pl.read_parquet(cache / "texts.parquet")["headline"]
            vectors = np.load(cache / "vectors.npy")
        else:
            texts = part["headline"].unique().sort()
            embedder = embedder or embedder_factory(model_id)
            vectors = embedder.embed(texts.to_list())
            cache.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"headline": texts}).write_parquet(cache / "texts.parquet")
            np.save(cache / "vectors.npy", vectors)
        lookup = pl.DataFrame({"headline": texts}).with_row_index("_vec")
        matched = part.select("_pos", "headline").join(lookup, on="headline", how="left")
        if matched["_vec"].null_count():
            raise ValueError(f"embedding cache {cache} lacks headlines; delete it and re-run")
        out[matched["_pos"].to_numpy()] = vectors[matched["_vec"].to_numpy()]
    return out


def make_chrono_featurize(
    panel: pl.DataFrame,
    cache_root: Path,
    embedder_factory: Callable[[str], Embedder] = ChronoEmbedder,
    train_years: int = config.TRAIN_YEARS,
    dim: int = config.EMBED_DIM,
):
    """Featurizer for model.walk_forward. Embeds the whole (test_year - train_years .. test_year)
    block once with vintage_for(test_year); keeps one block in memory (8 GB machine)."""
    memo: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def featurize(train: pl.DataFrame, evals: list[pl.DataFrame], params: dict, test_year: int):
        if test_year not in memo:
            memo.clear()
            block = panel.filter(
                pl.col("signal_date").dt.year().is_between(test_year - train_years, test_year)
            )
            vectors = embed_rows(block, vintage_for(test_year), cache_root, embedder_factory, dim)
            position = np.full(int(panel["row_id"].max()) + 1, -1, dtype=np.int64)
            position[block["row_id"].to_numpy()] = np.arange(block.height)
            memo[test_year] = (vectors, position)
        vectors, position = memo[test_year]

        def pick(rows: pl.DataFrame) -> np.ndarray:
            idx = position[rows["row_id"].to_numpy()]
            if (idx < 0).any():
                raise ValueError(f"rows outside the embedded block for test year {test_year}")
            return vectors[idx].astype(np.float32)

        return pick(train), [pick(e) for e in evals]

    return featurize
