from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, resources_directory: Path) -> None:
        tokenizer_path = resources_directory / "tokenizer.json"
        if not tokenizer_path.exists():
            msg = f"Missing tokenizer artifact: {tokenizer_path}"
            raise FileNotFoundError(msg)

        self.resources_directory = resources_directory
        self.tokenizer_path = tokenizer_path
        self.backend = HFTokenizer.from_file(str(tokenizer_path))
        self.bos_token_id = self.backend.token_to_id("<|bos|>")
        self.eos_token_id = self.backend.token_to_id("<|eos|>")
        self.pad_token_id = self.backend.token_to_id("<|pad|>")

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.backend.decode(token_ids)

    @property
    def vocabulary_size(self) -> int:
        return self.backend.get_vocab_size()

    @property
    def fingerprint(self) -> str:
        return _file_sha256(self.tokenizer_path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_token_ids(
    token_ids: list[int],
    path: Path,
    tokenizer_fingerprint: str,
) -> None:
    tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
    )

    payload = {
        "token_ids": tensor,
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }
    torch.save(payload, path)

    print(f"Saved {tensor.numel():,} token IDs to {path}")


def load_token_ids(path: Path, tokenizer_fingerprint: str) -> list[int]:
    payload: Any = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(payload, torch.Tensor):
        raise ValueError(
            "Cached token IDs were created with an older tokenizer format. "
            "Delete the cache file and regenerate it."
        )

    if not isinstance(payload, dict):
        raise TypeError("Saved token data must be a dictionary payload.")

    saved_fingerprint = payload.get("tokenizer_fingerprint")
    if saved_fingerprint != tokenizer_fingerprint:
        raise ValueError(
            "Cached token IDs do not match the current tokenizer. "
            "Delete the cache file and regenerate it."
        )

    tensor = payload.get("token_ids")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Saved token data must contain a tensor under 'token_ids'.")

    if tensor.dtype != torch.long:
        raise TypeError("Saved token IDs must use torch.long.")

    return tensor.tolist()


def encode_parallel(
    text: str,
    tokenizer: Tokenizer,
) -> list[int]:
    lines = text.splitlines(keepends=True)
    encodings = tokenizer.backend.encode_batch_fast(lines)
    return [token_id for encoding in encodings for token_id in encoding.ids]
