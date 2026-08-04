# transformer-decoder-slm

Decoder-only Transformer language model built in PyTorch, using a Hugging Face tokenizer artifact, WikiText-style next-token training, checkpointing, and a small FastAPI inference layer.

This repository does not include trained `.pt` artifacts. To run inference locally, you need to train or otherwise provide checkpoints under `checkpoints/`.

## What This Project Does

- Loads a Hugging Face tokenizer artifact from `resources/tokenizer/tokenizer.json`
- Builds fixed-context training windows from raw text
- Trains a decoder-only Transformer with causal self-attention
- Saves `latest.pt` and `best.pt` checkpoints during training
- Serves generation through a FastAPI app that loads a specific capacity checkpoint

## Project Layout

```text
transformer-decoder-slm/
|-- checkpoints/
|-- data/
|-- resources/
|   `-- tokenizer/
|       `-- tokenizer.json
|-- src/
|   `-- transformer_decoder_slm/
|       |-- api/
|       |-- attention.py
|       |-- block.py
|       |-- config.py
|       |-- dataset.py
|       |-- generate.py
|       |-- main.py
|       |-- model.py
|       |-- tokenizer.py
|       `-- train.py
|-- pyproject.toml
`-- README.md
```

## Requirements

- Python `3.11+`
- A tokenizer export at `resources/tokenizer/tokenizer.json`
- Training text at `data/train.txt`
- Validation text at `data/validation.txt`

GPU is optional, but training on CPU will be very slow for large corpora.

## Local Setup

From the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.\.venv\Scripts\activate
```

Install the project:

```bash
pip install -e .
```

If you use `uv`, this is also fine:

```bash
uv sync
```

## Required Local Files

The `data/` folder is gitignored, so the dataset will not be present when someone clones this repository.

The repo expects these local inputs:

- `data/train.txt`
- `data/validation.txt`
- `resources/tokenizer/tokenizer.json`

The token-id cache files below are optional and will be generated automatically if missing:

- `data/train_token_ids.pt`
- `data/validation_token_ids.pt`

## Preparing The Dataset

This repository does not include the training corpus. You need to prepare local text files before training.

The documented experiments in this project were trained on WikiText-103, so the easiest way to reproduce the reported checkpoint behavior is:

1. Download WikiText-103 train and validation datasets separately
2. Convert the training set into `data/train.txt`
3. Convert the validation set into `data/validation.txt`

You can also train on another dataset, as long as you provide plain text files at:

- `data/train.txt`
- `data/validation.txt`

### Tokenizer Note

`resources/tokenizer/tokenizer.json` is not a tokenizer trained by this repository. It is a Hugging Face tokenizer export artifact consumed by this project at runtime.

That means:

- if you want to reproduce the current results, use WikiText-103 or a similar Wikipedia-style corpus and keep the existing tokenizer artifact
- if you switch to a substantially different dataset, the current vocabulary may no longer be a good fit
- in that case, you may want to train or export a new Hugging Face tokenizer and replace `resources/tokenizer/tokenizer.json`

After replacing the tokenizer artifact, regenerate training so the cached token IDs and checkpoints match the new tokenizer fingerprint.

## How Training Works

Training is driven by [src/transformer_decoder_slm/main.py](/D:/projects/Neural%20Network/transformer-decoder-slm/src/transformer_decoder_slm/main.py) and the defaults in [src/transformer_decoder_slm/config.py](/D:/projects/Neural%20Network/transformer-decoder-slm/src/transformer_decoder_slm/config.py).

On a run, the project will:

1. Load the Hugging Face tokenizer artifact from `resources/tokenizer/tokenizer.json`
2. Load cached token IDs if they exist and match the tokenizer fingerprint
3. Regenerate token IDs from `train.txt` and `validation.txt` if needed
4. Train the model
5. Save checkpoints to `checkpoints/latest.pt` and `checkpoints/best.pt`

Start training with:

```bash
python -m transformer_decoder_slm.main
```

## Building Checkpoints From Scratch

Because the repository does not include `.pt` checkpoint files, you need to create them locally.

### Default training output

A normal training run writes:

- `checkpoints/latest.pt`
- `checkpoints/best.pt`

If `resume_from_checkpoint=True` in `config.py`, training will try to resume from `checkpoints/best.pt`.

### Fresh run from scratch

Before a brand-new training run, either:

- remove or move any old `checkpoints/best.pt` and `checkpoints/latest.pt`
- or set `resume_from_checkpoint=False` in `config.py`

Then run:

```bash
python -m transformer_decoder_slm.main
```

### Creating named capacity checkpoints

If you want capacity-specific inference targets such as `capacity-01`, `capacity-02`, `capacity-03`, `capacity-04`, and `capacity-05`, the current codebase expects the final checkpoint layout to look like this:

```text
checkpoints/
|-- capacity-01/
|   `-- best.pt
|-- capacity-02/
|   `-- best.pt
|-- capacity-03/
|   `-- best.pt
|-- capacity-04/
|   `-- best.pt
`-- capacity-05/
    `-- best.pt
```

The training script does not automatically create those subfolders for you. A practical workflow is:

1. Edit `config.py` to the capacity you want to train.
2. Run training until `checkpoints/best.pt` is produced.
3. Create a folder such as `checkpoints/capacity-01/`.
4. Copy `checkpoints/best.pt` into that folder as `checkpoints/capacity-01/best.pt`.
5. Repeat for the next capacity variant after changing the config again.

Example PowerShell commands after training:

```powershell
New-Item -ItemType Directory -Force checkpoints\capacity-01
Copy-Item checkpoints\best.pt checkpoints\capacity-01\best.pt
```

## Suggested Capacity Workflow

The capacity comparison currently documented for this project is:

| Model | D | Heads | Blocks | FFN | Best validation loss | Comment |
|---|---:|---:|---:|---:|---:|---|
| `capacity-01` | `128` | `4` | `2` | `512` | `4.103788831794405` | Baseline smaller-capacity run |
| `capacity-02` | `256` | `8` | `2` | `1024` | `3.754026508901224` | Wider model than `capacity-01` |
| `capacity-03` | `256` | `8` | `6` | `1024` | `3.600118788115057` | Deeper model than `capacity-02` |
| `capacity-04` | `256` | `8` | `6` | `1024` | `3.534931350039296` | Same configuration as `capacity-03`, but trained with a scheduler and weight tying (`token_embedding` weights equal final classifier weights) |
| `capacity-05` | `256` | `8` | `6` | `1024` | `3.4712566825973084` | Same width/depth as `capacity-04`, but uses `pre_norm=True` and untied output projection weights |

To reproduce that kind of workflow, train one configuration at a time and archive each run's `best.pt` into its own `checkpoints/capacity-xx/` folder before starting the next one.

## Running Inference Locally

The API requires an explicit capacity name at startup. It will load:

```text
checkpoints/<capacity-name>/best.pt
```

Run the API with:

```bash
python -m transformer_decoder_slm.api.app --capacity capacity-03 --host 0.0.0.0 --port 8000
```

You can then send a request to `POST /generate`:

```json
{
  "prompt": "Once upon a time",
  "temperature": 0.8
}
```

Example response:

```json
{
  "prompt": "Once upon a time",
  "generated_text": " there was ..."
}
```

The health endpoint is:

```text
GET /health
```

It reports the selected capacity along with status.

## Sample Generations

These examples document how output quality changed across the capacity checkpoints.

### `capacity-01`

| Prompt | Generation |
|---|---|
| `Tokyo is the capital of` | `urnal for the time to form the new city of Luna , which is the first city . The city is located in the city , an ancient village of Madhaba ( the town of Yo ) . It is also located in the city` |
| `In mathematics, a function is` | ` the first step . A new is an important figure in the world , as an equator of the world and its center . The theorem , he is a physicist of a quantum theory of the world , and is not the subject of` |
| `During the Second World War` | ` . = = = Bronze Age = = = = = = = = = The Bronze Age remains relatively unknown in the early Iron Age , although the Roman Empire , as many as the " finest Ages " .` |

### `capacity-02`

| Prompt | Generation |
|---|---|
| `During the Second World War` | ` . The Battle of Britain in 1941 was a major Allied victory in World War II ; the Battle of Britain and the Second World War , the Battle of Britain and the Battle of Britain . = = Background = = In the late 1940` |
| `In mathematics, a function is` | ` to be given in this process by a computer @-@ based computer , as well as one , a computer , and a computer can be used . = = = Thermal @-@ powered computer = = = Thermal @-@ Graph` |
| `Tokyo is the capital of` | ` the city 's downtown city , in which the city is named the city of San Diego . It serves as the city 's city center of the city 's city centre and is the city 's largest city . The city 's mayor Ak` |

### `capacity-03`

| Prompt | Generation |
|---|---|
| `Tokyo is the capital of` | ` the city . It is the only known example of its modern state , and has an architectural heritage , where there is a few architectural elements . The majority of the city 's population is composed of a mixture of urban and local and urban and suburban features` |
| `In mathematics, a function is` | ` called a class of nodes . For example , the algorithm is also used to describe the class of nodes ( nodes ) , using the type of nodes ( nodes ) . As a result , the class of nodes is` |
| `During the Second World War` | ` . The ship was laid down on 2 January 1931 and launched on 26 May 1931 . She was launched on 22 June 1934 , and commissioned into the fleet on 1 January 1936 . The ship was completed on 26 May 1936 , and was commissioned on 29 June` |

### `capacity-04`

| Prompt | Generation |
|---|---|
| `Tokyo is the capital of` | ` the city , and the city is the capital of the city . The city is located in the city 's northern suburbs . The city is the home of the city 's largest city , the city of Koku . The city is the` |
| `In mathematics, a function is` | ` a function of the unit . The unit is the unit of the unit . The unit is the unit of the unit , which is the unit of the unit . = = = = = = = Other units = = = = = =` |
| `During the Second World War` | ` . = = = World War II = = = In the mid @-@ 1950s , the battalion was involved in a number of operations in the North West Pacific , including the Battle of the Somme , the Battle of Chauvel` |

### `capacity-05`

| Prompt | Generation |
|---|---|
| `Tokyo is the capital of` | ` the Tokyo District . It is located on the edge of the city of Tokyo , which was founded in 1949 . The city is a member of the Hōkaku Municipal Corporation , who owns the city and is the city 's flagship center` |
| `In mathematics, a function is` | ` , in fact , a vector unit that is used as an eigenvalue in a number of binary fields . Thus , the real numbers of the two fields are in the form of a vector unit , which is a vector unit` |
| `During the Second World War` | ` . In 1950 , the British government agreed to the US military intervention in the Vietnam War , a move that was made by the Royal Australian Navy to prevent the war from exposing the US and Australia to the United Kingdom . \n \n = = = World` |

## Conclusion

This decoder-only Transformer was built from basic PyTorch components and used a Hugging Face tokenizer artifact for tokenization. The model was trained on WikiText-103 using causal multi-head self-attention and autoregressive next-token prediction. Scaling the model from 2 to 6 decoder blocks reduced the best validation loss from approximately 4.10 to 3.60. The final model learned coherent Wikipedia-style language patterns, but its limited scale resulted in factual errors, repetition, and weak long-range topic consistency. These are model-capacity and training-objective limitations rather than evidence that the Transformer pipeline is broken.

## Notes

- `resources/tokenizer/tokenizer.json` is a Hugging Face tokenizer export artifact used by this project at runtime.
- Changing `resources/tokenizer/tokenizer.json` invalidates existing token-id caches and checkpoints because the tokenizer fingerprint must still match.
- `generate.py` performs uncached autoregressive generation, so inference recomputes the full context each token.
- The API cannot start until a checkpoint exists for the requested capacity.
