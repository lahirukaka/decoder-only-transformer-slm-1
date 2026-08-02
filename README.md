# transformer-decoder-slm

Small decoder-only Transformer language model scaffold built with Python and PyTorch. The project handles corpus loading, fixed-context dataset windows, training, checkpointing, and uncached autoregressive generation while leaving the Transformer internals for you to implement.

## Structure

```text
transformer-decoder-slm/
|-- pyproject.toml
|-- README.md
|-- data/
|   |-- train.txt
|   |-- validation.txt
|   |-- train_token_ids.pt
|   `-- validation_token_ids.pt
|-- resources/
|   `-- tokenizer/
|       `-- tokenizer.json
|-- checkpoints/
|-- src/
|   `-- transformer_decoder_slm/
|       |-- __init__.py
|       |-- config.py
|       |-- tokenizer.py
|       |-- dataset.py
|       |-- attention.py
|       |-- block.py
|       |-- model.py
|       |-- train.py
|       |-- generate.py
|       `-- main.py
`-- .gitignore
```

## Data And Tokenizer

Place training text in `data/train.txt` and validation text in `data/validation.txt`.
Place the exported Hugging Face tokenizer artifact in `resources/tokenizer/`:

- `tokenizer.json`

This project now loads the tokenizer through the `tokenizers` library, so a byte-level BPE tokenizer exported from Hugging Face plugs in directly.

The dataset flow is:

```text
train.txt / validation.txt -> tokenizer.encode(text) -> token IDs -> overlapping windows
```

For context length `T`:

```text
input  = tokens[i : i + T]
target = tokens[i + 1 : i + T + 1]
```

## Tensor Shapes

```text
dataset item:
inputs  [T]
targets [T]

model:
token IDs [B, T]
logits    [B, T, V]

loss:
logits  [B, T, V] -> [B*T, V]
targets [B, T]    -> [B*T]
```

## Intentionally Unfinished Files

These files contain guided TODO scaffolds and are expected to fail until you implement them:

- `src/transformer_decoder_slm/attention.py`
- `src/transformer_decoder_slm/block.py`
- `src/transformer_decoder_slm/model.py`

## Training And Generation

Install dependencies, then run:

```bash
python -m transformer_decoder_slm.main
```

The generation loop currently recomputes the full context on every new token. It does not use KV caching yet.

## API

Start the FastAPI server with:

```bash
uvicorn transformer_decoder_slm.api.app:app --host 0.0.0.0 --port 8000
```

The server loads `checkpoints/best.pt` on startup and uses the existing tokenizer in `resources/tokenizer/tokenizer.json`.

Send a JSON request to `POST /generate`:

```json
{
  "prompt": "Once upon a time",
  "temperature": 0.8
}
```

Response shape:

```json
{
  "prompt": "Once upon a time",
  "generated_text": " there was ..."
}
```

Checkpoints:

- `checkpoints/latest.pt` is overwritten every epoch.
- `checkpoints/best.pt` is updated only when validation loss improves.

Changing the tokenizer resources invalidates existing cached token IDs and checkpoints because the saved tokenizer fingerprint must match the current `tokenizer.json`.
