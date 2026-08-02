"""FastAPI app for decoder-only Transformer inference."""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
import uvicorn

from ..config import Config
from ..generate import generate_text
from ..main import create_tokenizer
from ..model import DecoderOnlyTransformer
from ..tokenizer import Tokenizer

from .schemas import GenerateRequest, GenerateResponse

CAPACITY_ENVIRONMENT_VARIABLE = "MODEL_CAPACITY"


def resolve_capacity_name(explicit_capacity_name: str | None = None) -> str:
    """Return the requested capacity name or raise if none was provided."""

    capacity_name = explicit_capacity_name or os.getenv(CAPACITY_ENVIRONMENT_VARIABLE)
    if capacity_name is None or not capacity_name.strip():
        msg = (
            "Missing model capacity. Pass --capacity <capacity-name> when starting the API "
            f"or set {CAPACITY_ENVIRONMENT_VARIABLE}."
        )
        raise RuntimeError(msg)
    return capacity_name.strip()


def resolve_checkpoint_path(checkpoint_directory: Path, capacity_name: str) -> Path:
    """Build the checkpoint path for a named capacity."""

    return checkpoint_directory / capacity_name / "best.pt"


class InferenceRuntime:
    """Owns model assets that are shared by API requests."""

    def __init__(self, config: Config, capacity_name: str | None = None) -> None:
        self.config = config
        self.capacity_name = capacity_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = Lock()
        self.tokenizer: Tokenizer | None = None
        self.model: DecoderOnlyTransformer | None = None

    def load(self) -> None:
        self.capacity_name = resolve_capacity_name(self.capacity_name)
        tokenizer = create_tokenizer(self.config.tokenizer_resources_directory)
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_directory=self.config.checkpoint_directory,
            capacity_name=self.capacity_name,
        )
        if not checkpoint_path.exists():
            msg = f"Missing checkpoint: {checkpoint_path}"
            raise FileNotFoundError(msg)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        tokenizer_metadata = checkpoint.get("tokenizer_metadata", {})
        saved_vocabulary_size = tokenizer_metadata.get("vocabulary_size")
        if saved_vocabulary_size != tokenizer.vocabulary_size:
            msg = "checkpoint tokenizer vocabulary size does not match the current tokenizer"
            raise ValueError(msg)

        saved_fingerprint = tokenizer_metadata.get("fingerprint")
        if saved_fingerprint != tokenizer.fingerprint:
            msg = (
                "checkpoint tokenizer fingerprint does not match the current tokenizer"
            )
            raise ValueError(msg)

        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("checkpoint is missing model_config")

        model = DecoderOnlyTransformer(
            vocabulary_size=int(model_config["vocabulary_size"]),
            maximum_context_length=int(model_config["context_length"]),
            model_dimension=int(model_config["model_dimension"]),
            number_of_heads=int(model_config["number_of_heads"]),
            number_of_decoder_blocks=int(model_config["number_of_decoder_blocks"]),
            feed_forward_dimension=int(model_config["feed_forward_dimension"]),
            dropout=float(model_config["dropout"]),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

    def generate(self, prompt: str, temperature: float) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Inference runtime is not loaded")

        with self.lock:
            full_text = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                generation_length=self.config.generation_length,
                temperature=temperature,
                device=self.device,
            )

        if full_text.startswith(prompt):
            return full_text[len(prompt) :]
        return full_text


config = Config()


def create_app(capacity_name: str | None = None) -> FastAPI:
    """Create a FastAPI app that loads a specific capacity checkpoint."""

    runtime = InferenceRuntime(config=config, capacity_name=capacity_name)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.load()
        yield

    app = FastAPI(
        title="Transformer Decoder SLM API",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "capacity": runtime.capacity_name,
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            generated_text = runtime.generate(
                prompt=request.prompt,
                temperature=request.temperature,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
        )

    return app


app = create_app()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for serving the API."""

    parser = argparse.ArgumentParser(
        description="Serve the Transformer Decoder SLM API."
    )
    parser.add_argument(
        "--capacity",
        required=True,
        help="Capacity checkpoint folder to load, for example capacity-01.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    """Start the API with a required capacity checkpoint."""

    args = parse_args()
    os.environ[CAPACITY_ENVIRONMENT_VARIABLE] = args.capacity
    uvicorn.run(
        "transformer_decoder_slm.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
