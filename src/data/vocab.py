"""Event token vocabulary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"


class Vocabulary:
    def __init__(self):
        self.token_to_id: dict[str, int] = {
            PAD_TOKEN: 0,
            CLS_TOKEN: 1,
            UNK_TOKEN: 2,
            MASK_TOKEN: 3,
        }
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[CLS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[MASK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def fit(self, tokens: Iterable[str], min_freq: int = 2) -> "Vocabulary":
        counts = Counter(tokens)
        next_id = len(self.token_to_id)
        for token, freq in sorted(counts.items()):
            if freq >= min_freq and token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
        self._min_freq = min_freq
        self._fitted_counts = dict(counts)
        return self

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.unk_id
        vocab = self.token_to_id
        min_freq = getattr(self, "_min_freq", 2)
        counts = getattr(self, "_fitted_counts", {})
        ids = []
        for t in tokens:
            if t in vocab:
                ids.append(vocab[t])
            elif counts.get(t, 0) >= min_freq:
                # token seen in fit but below freq threshold at fit time
                ids.append(unk)
            else:
                ids.append(unk)
        return ids

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def to_dict(self) -> dict:
        return {
            "token_to_id": self.token_to_id,
            "min_freq": getattr(self, "_min_freq", 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        v = cls()
        v.token_to_id = dict(d["token_to_id"])
        v.id_to_token = {int(i): t for t, i in v.token_to_id.items()}
        v._min_freq = d.get("min_freq", 2)
        v._ensure_mask_token()
        return v

    def _ensure_mask_token(self) -> None:
        """Inject [MASK] into vocab loaded from older artifacts."""
        if MASK_TOKEN in self.token_to_id:
            return
        next_id = max(self.token_to_id.values()) + 1
        self.token_to_id[MASK_TOKEN] = next_id
        self.id_to_token[next_id] = MASK_TOKEN

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path) as f:
            return cls.from_dict(json.load(f))
