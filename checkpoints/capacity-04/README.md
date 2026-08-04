# capacity-04

This folder contains a checkpoint run based on the `capacity-03` architecture, trained with the scheduler-backed configuration now reflected in `src/transformer_decoder_slm/config.py`, plus tied token embedding and output projection weights.

## Training config used

Source of truth for the shared defaults: `src/transformer_decoder_slm/config.py`

Capacity-specific differences for this run:

- `pre_norm=False`
- Tied weights: `token_embedding.weight == output_projection.weight`

| Setting | Value |
|---|---:|
| `training_corpus_path` | `data/train.txt` |
| `validation_corpus_path` | `data/validation.txt` |
| `training_token_ids_path` | `data/train_token_ids.pt` |
| `validation_token_ids_path` | `data/validation_token_ids.pt` |
| `tokenizer_resources_directory` | `resources/tokenizer` |
| `checkpoint_directory` | `checkpoints` during training |
| `dataset_stride` | `128` |
| `context_length` | `128` |
| `batch_size` | `64` |
| `model_dimension` | `256` |
| `number_of_heads` | `8` |
| `number_of_decoder_blocks` | `6` |
| `feed_forward_dimension` | `1024` |
| `dropout` | `0.1` |
| `peak_learning_rate` | `1e-3` |
| `minimum_learning_rate` | `3e-5` |
| `warmup_steps` | `2000` |
| `weight_decay` | `1e-4` |
| `epoch_count` | `70` |
| `random_seed` | `42` |
| `gradient_clipping_norm` | `1.0` |
| `maximum_corpus_characters` | `None` |
| `resume_from_checkpoint` | `True` |

## Model + tokenizer metadata saved in these checkpoints

| Field | Value |
|---|---:|
| Vocabulary size | `16000` |
| Context length | `128` |
| Model dimension | `256` |
| Attention heads | `8` |
| Decoder blocks | `6` |
| Feed-forward dimension | `1024` |
| Dropout | `0.1` |
| Tokenizer resources | `resources/tokenizer` |
| Tokenizer fingerprint | `fe69d11d4877b0c181df12f7da4b739f9526db6e4171b61f4d81f607ae59f4fd` |
| Tied token/output weights | `True` |

## Data artifacts used

| Artifact | Size / count |
|---|---:|
| `data/train.txt` | `540,568,191` bytes |
| `data/validation.txt` | `1,146,846` bytes |
| `data/train_token_ids.pt` | `983,827,553` bytes |
| `data/validation_token_ids.pt` | `2,058,052` bytes |
| Training token count | `122,978,227` |
| Validation token count | `257,035` |
| Training windows (`context_length=128`, `stride=128`) | `960,767` |
| Validation windows (`context_length=128`, `stride=128`) | `2,008` |

## Files in this folder

### `best.pt`

- Saved epoch index: `68` (epoch 69 of 70)
- Train loss: `3.7057039737701416`
- Validation loss: `3.534931350039296`
- Best validation loss: `3.534931350039296`

### `latest.pt`

- Saved epoch index: `69` (epoch 70 of 70)
- Train loss: `3.705284833908081`
- Validation loss: `3.5352858965140417`
- Best validation loss at that point: `3.534931350039296`

### `loss.json`

- Logged epochs: `70`
- First entry learning rate: `0.0009996320754228644`
- Final entry learning rate: `3.0000330919355117e-05`

### `evaluation.md`

- Contains a qualitative evaluation of this checkpoint's generations and failure modes.

## Notes

- These checkpoints were generated with the tokenizer at `resources/tokenizer/tokenizer.json`.
- The token ID caches in `data/*.pt` use the same tokenizer fingerprint as the checkpoints.
- This run keeps the `capacity-03` dimensions but uses scheduler-based training and tied embedding/classifier weights.
- Inference code treats capacities before `capacity-05` as `pre_norm=False`.
