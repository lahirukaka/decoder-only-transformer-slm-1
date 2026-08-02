"""Evaluate a capacity checkpoint across fixed prompts and prepare an LLM review payload."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import torch

from .api.app import resolve_checkpoint_path
from .config import Config
from .generate import GenerationStepTrace, generate_text
from .main import create_tokenizer
from .model import DecoderOnlyTransformer

SAMPLE_PROMPTS: list[str] = [
    "Tokyo is the capital of",
    "The capital of France is",
    "In mathematics, a function is",
    "During the Second World War",
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_ENVIRONMENT_VARIABLE = "OPENROUTER_MODEL"


@dataclass(slots=True)
class PromptEvaluationResult:
    """Evaluation artifacts for one prompt."""

    prompt: str
    prompt_token_ids: list[int]
    generated_text: str
    trace_steps: list[GenerationStepTrace]


@dataclass(slots=True)
class CapacityMetadata:
    """Architecture and training metadata for one evaluated checkpoint."""

    capacity_name: str
    dataset_name: str
    epoch_count: int
    best_epoch_index: int
    train_loss: float
    best_validation_loss: float
    context_length: int
    model_dimension: int
    number_of_heads: int
    feed_forward_dimension: int
    number_of_decoder_blocks: int


def load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from a local .env file into the environment."""

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key or normalized_key in os.environ:
            continue

        normalized_value = value.strip().strip('"').strip("'")
        os.environ[normalized_key] = normalized_value


def load_capacity_runtime(
    capacity_name: str, config: Config
) -> tuple[Any, DecoderOnlyTransformer, torch.device, CapacityMetadata]:
    """Load tokenizer and model for a named capacity checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = create_tokenizer(config.tokenizer_resources_directory)
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_directory=config.checkpoint_directory,
        capacity_name=capacity_name,
    )
    if not checkpoint_path.exists():
        msg = f"Missing checkpoint: {checkpoint_path}"
        raise FileNotFoundError(msg)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    tokenizer_metadata = checkpoint.get("tokenizer_metadata", {})
    if tokenizer_metadata.get("vocabulary_size") != tokenizer.vocabulary_size:
        raise ValueError(
            "checkpoint tokenizer vocabulary size does not match the current tokenizer"
        )
    if tokenizer_metadata.get("fingerprint") != tokenizer.fingerprint:
        raise ValueError(
            "checkpoint tokenizer fingerprint does not match the current tokenizer"
        )

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
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = CapacityMetadata(
        capacity_name=capacity_name,
        dataset_name="WikiText-103-raw",
        epoch_count=config.epoch_count,
        best_epoch_index=int(checkpoint["epoch"]),
        train_loss=float(checkpoint["train_loss"]),
        best_validation_loss=float(checkpoint["best_validation_loss"]),
        context_length=int(model_config["context_length"]),
        model_dimension=int(model_config["model_dimension"]),
        number_of_heads=int(model_config["number_of_heads"]),
        feed_forward_dimension=int(model_config["feed_forward_dimension"]),
        number_of_decoder_blocks=int(model_config["number_of_decoder_blocks"]),
    )
    return tokenizer, model, device, metadata


def evaluate_capacity(
    capacity_name: str,
    prompts: list[str],
    generation_length: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    """Run fixed prompts against one capacity checkpoint and collect traces."""

    config = Config()
    tokenizer, model, device, metadata = load_capacity_runtime(
        capacity_name=capacity_name, config=config
    )

    prompt_results: list[PromptEvaluationResult] = []
    for prompt in prompts:
        trace_steps: list[GenerationStepTrace] = []
        full_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_length=generation_length,
            temperature=temperature,
            device=device,
            top_k=top_k,
            trace_steps=trace_steps,
        )
        generated_text = (
            full_text[len(prompt) :] if full_text.startswith(prompt) else full_text
        )
        prompt_results.append(
            PromptEvaluationResult(
                prompt=prompt,
                prompt_token_ids=tokenizer.encode(prompt),
                generated_text=generated_text,
                trace_steps=trace_steps,
            )
        )

    return {
        "capacity": capacity_name,
        "metadata": asdict(metadata),
        "generation_length": generation_length,
        "temperature": temperature,
        "top_k": top_k,
        "prompt_results": [asdict(result) for result in prompt_results],
    }


def build_openrouter_system_prompt(evaluation_payload: dict[str, Any]) -> str:
    """Build the system prompt for the evaluator LLM."""

    metadata = evaluation_payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Evaluation payload is missing metadata")

    return (
        "You are an expert evaluator of small decoder-only Transformer language models. "
        "Your job is to analyze generation behavior from prompt traces, next-token probability distributions, and the training context. "
        "Be technical, evidence-based, and specific. "
        "Identify strengths, weaknesses, likely causes, and concrete ways the model could be improved.\n\n"
        "Model context:\n"
        f"- Capacity name: {metadata['capacity_name']}\n"
        f"- Dataset: {metadata['dataset_name']}\n"
        f"- Epoch count: {metadata['epoch_count']}\n"
        f"- Best checkpoint epoch index: {metadata['best_epoch_index']}\n"
        f"- Training loss at saved best checkpoint: {metadata['train_loss']}\n"
        f"- Best validation loss: {metadata['best_validation_loss']}\n"
        f"- Model dimensions: {metadata['model_dimension']}\n"
        f"- Context length: {metadata['context_length']}\n"
        f"- Attention heads: {metadata['number_of_heads']}\n"
        f"- Feed-forward dimensions: {metadata['feed_forward_dimension']}\n"
        f"- Decoder blocks: {metadata['number_of_decoder_blocks']}\n"
    )


def build_openrouter_user_prompt(evaluation_payload: dict[str, Any]) -> str:
    """Build the user message sent to the evaluator LLM."""

    pretty_payload = json.dumps(evaluation_payload, indent=2, ensure_ascii=False)
    return (
        "You are evaluating a decoder-only language model from its prompt traces.\n\n"
        "For each prompt, inspect:\n"
        "- the prompt text\n"
        "- the prompt token sequence\n"
        "- each generation step's input token sequence\n"
        "- the sampled next token\n"
        "- the top-k token probabilities after softmax\n\n"
        "Please analyze:\n"
        "- what the model does well\n"
        "- what its weaknesses are\n"
        "- whether it shows repetition, uncertainty, topic drift, or shallow heuristics\n"
        "- whether the top-k distributions look confident, confused, or over-concentrated\n"
        "- what architectural or training changes could improve it\n\n"
        "Evaluation payload:\n"
        f"{pretty_payload}"
    )


def send_to_openrouter(
    *,
    model_name: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Send the evaluation payload to OpenRouter."""

    body = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    http_request = request.Request(
        OPENROUTER_API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter request failed with status {exc.code}: {detail}"
        ) from exc


def extract_openrouter_text(response_payload: dict[str, Any]) -> str:
    """Extract the assistant text from an OpenRouter response payload."""

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response is missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("OpenRouter response choice must be a dictionary")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response choice is missing message")

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)

    raise ValueError("OpenRouter response does not contain assistant text content")


def write_capacity_evaluation_markdown(capacity_name: str, content: str) -> Path:
    """Write the LLM evaluation into the selected capacity checkpoint folder."""

    evaluation_path = Path("checkpoints") / capacity_name / "evaluation.md"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(content, encoding="utf-8")
    return evaluation_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a named capacity checkpoint."
    )
    parser.add_argument(
        "--capacity",
        required=True,
        help="Capacity checkpoint folder to load, for example capacity-01.",
    )
    parser.add_argument(
        "--generation-length",
        type=int,
        default=15,
        help="Number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k predictions to record for each generated token.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the evaluation payload as JSON.",
    )
    parser.add_argument(
        "--openrouter-model",
        default=None,
        help="Optional OpenRouter model name. Overrides OPENROUTER_MODEL from .env when provided.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the evaluator CLI."""

    load_dotenv()
    args = parse_args()
    evaluation_payload = evaluate_capacity(
        capacity_name=args.capacity,
        prompts=SAMPLE_PROMPTS,
        generation_length=args.generation_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    if args.output is not None:
        args.output.write_text(
            json.dumps(evaluation_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # print(json.dumps(evaluation_payload, indent=2, ensure_ascii=False))

    if args.openrouter_model is None:
        env_model_name = os.getenv(OPENROUTER_MODEL_ENVIRONMENT_VARIABLE)
        if not env_model_name:
            return
        model_name = env_model_name
    else:
        model_name = args.openrouter_model

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable")

    system_prompt = build_openrouter_system_prompt(evaluation_payload)
    user_prompt = build_openrouter_user_prompt(evaluation_payload)

    openrouter_response = send_to_openrouter(
        model_name=model_name,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    evaluation_markdown = extract_openrouter_text(openrouter_response)
    evaluation_path = write_capacity_evaluation_markdown(
        capacity_name=args.capacity,
        content=evaluation_markdown,
    )
    print(f"Wrote LLM evaluation to {evaluation_path}")


if __name__ == "__main__":
    main()
