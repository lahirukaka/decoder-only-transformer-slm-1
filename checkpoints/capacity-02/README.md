# capacity-02

This folder contains a checkpoint run trained with the current default `Config` in `src/transformer_decoder_slm/config.py`.

## Training config used

Source of truth: `src/transformer_decoder_slm/config.py`

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
| `number_of_decoder_blocks` | `2` |
| `feed_forward_dimension` | `1024` |
| `dropout` | `0.1` |
| `learning_rate` | `3e-4` |
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
| Decoder blocks | `2` |
| Feed-forward dimension | `1024` |
| Dropout | `0.1` |
| Tokenizer resources | `resources/tokenizer` |
| Tokenizer fingerprint | `fe69d11d4877b0c181df12f7da4b739f9526db6e4171b61f4d81f607ae59f4fd` |

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

- Saved epoch index: `67` (epoch 68 of 70)
- Train loss: `3.943202257156372`
- Validation loss: `3.754026508901224`
- Best validation loss: `3.754026508901224`

### `latest.pt`

- Saved epoch index: `69` (epoch 70 of 70)
- Train loss: `3.942702293395996`
- Validation loss: `3.7551893674994843`
- Best validation loss at that point: `3.754026508901224`

## Notes

- These checkpoints were generated with the tokenizer at `resources/tokenizer/tokenizer.json`.
- The token ID caches in `data/*.pt` use the same tokenizer fingerprint as the checkpoints.
- Training code resumes from `checkpoints/best.pt` when `resume_from_checkpoint=True`.
