## Overall assessment

`capacity-02` has learned substantial **local English syntax and WikiText-style phrase continuation**, but it is weak at **fact retrieval, semantic constraint, and recovery after an incorrect token**. The traces show a strong tendency to select high-frequency corpus patterns rather than answer the prompt’s intended question.

The model is a very small decoder:

- 2 Transformer blocks
- 256-dimensional hidden state
- 8 heads
- 1024-dimensional feed-forward layer
- 128-token context

Its best validation loss of `3.754` corresponds to perplexity of roughly `42.7`, which is plausible for a small model on WikiText-103 but leaves considerable uncertainty at the semantic level. The train loss is higher than validation loss (`3.943` vs. `3.754`), possibly because training loss was measured with dropout enabled or under a different aggregation procedure. It does not by itself establish overfitting.

## What the model does well

### 1. Strong local syntactic completion

Many short grammatical patterns are modeled accurately:

- `"the city of"`
- `"the basis for the study of"`
- `"the mathematical theory of"`
- `"Air Force ("`
- `"RAAF )"`
- `"RAAF 's"`

Several distributions are highly concentrated on structurally appropriate tokens:

- After `"the study"`: `" of"` has probability `0.9985`.
- After `"Air"` in `"Royal Australian Air"`: `" Force"` has probability virtually `1.0`.
- After `"( RAAF"`: `")"` has probability `0.999999`.
- After `"RAAF '"`: `"s"` has probability `1.0`.

This indicates that the model has learned punctuation, possessive constructions, common noun phrases, and some entity-specific continuations effectively.

### 2. WikiText-style entity and phrase memorization

The World War II prompt demonstrates particularly good phrase continuation:

> `"During the Second World War , the Royal Australian Air Force ( RAAF )"`

The model reconstructs the full expansion and abbreviation with high confidence. This is likely a memorized or strongly learned WikiText phrase pattern, not merely generic syntax. It also recognizes plausible military continuations such as:

- Royal / British / United / German
- Navy / Australian / Air / Army
- Force / Corps

The fact that `"Royal Australian Air Force"` is completed correctly is a genuine strength.

### 3. Good punctuation and clause-boundary modeling

The model generally knows when punctuation or a function word is likely:

- `"government in"`
- `"world ."`
- `"study of"`
- `"theory of"`
- `") , and"`

The generated text often has superficially valid punctuation and spacing. This is important for a language model trained on raw Wikipedia text, where formatting conventions are part of the distribution.

### 4. Some calibrated ambiguity in locally uncertain contexts

Not all distributions are collapsed. For example:

- After `"the city of"` in the Tokyo prompt, `" K"` and `" the"` are almost tied at about `0.385` each.
- After `"Royal"`: `" Navy"` is `0.516`, `" Australian"` is `0.457`.
- After `"Royal Australian"`: `" Navy"` is `0.581`, `" Air"` is `0.387`.
- After `"the"` in the mathematical prompt, several continuation nouns receive meaningful probability.

These distributions reflect genuine local ambiguity. The model is not uniformly overconfident at every position.

## Major weaknesses

## 1. Failure on basic factual prompts

The clearest problem is the prompt:

> `"The capital of France is"`

The expected continuation is `" Paris"`, but `" Paris"` does not appear in the top ten. Instead, the model generates:

> `" the largest government in the world ."`

The first token distribution already reveals the problem:

- `" the"`: `0.773`
- `" a"`: `0.176`
- `" now"`: `0.014`
- `" located"`: `0.012`

The model interprets the prefix as a likely Wikipedia-style definitional sentence, but does not retrieve the fact that France’s capital is Paris. It then confidently commits to a generic template:

> `"the largest government in the world"`

This is both semantically inappropriate and factually nonsensical in context. The high probabilities after each subsequent token show that the failure is not primarily sampling noise. Once `" the"` is selected, the model strongly prefers a familiar continuation rather than reconsidering the prompt.

The Tokyo prompt is even more revealing. The natural completion is:

> `"Japan"`

Yet the model assigns:

- `" the"`: `0.9967`
- `" Japan"`: only `0.0027`

It produces:

> `"the city of Kokokokoku"`

This suggests that the model has learned a common surface pattern:

> `"X is the capital of the city of Y"`

rather than the intended fact:

> `"Tokyo is the capital of Japan"`

This is a shallow syntactic heuristic and possibly a consequence of WikiText phrase statistics overwhelming factual structure.

### Likely causes

- Very limited capacity and only two decoder blocks.
- Training objective optimized for token likelihood, not factual question answering.
- WikiText contains facts in article prose, but does not necessarily provide enough repeated, simple declarative examples for robust retrieval.
- Tokenization and vocabulary may make entity names less accessible.
- The model has learned frequent continuations such as `"the city of"` more strongly than rare but correct entity completions.
- No instruction or supervised factual fine-tuning.

## 2. Repetition and autoregressive self-reinforcement

The Tokyo generation shows severe repetition:

> `"Kokokokoku"`

The trace is especially diagnostic:

1. `" K"`: nearly tied between `" K"` and `" the"`.
2. `"ok"` after `" K"`: `0.443`.
3. `"ok"` after `" Kok"` is sampled despite `"om"` being top at `0.381`.
4. `"ok"` after `" Kokok"`: sampled with probability `0.086`.
5. `"ok"` again after `" Kokokok"`: top probability `0.452`.
6. `"u"` is sampled after `" Kokokokok"` even though `"ok"` is top at `0.817`.

The model is operating over subword fragments, so repeated `"K"` plus `"ok"` tokens can create a pseudo-word. Once this malformed sequence exists, the model continues predicting fragments that are locally compatible with Japanese-looking names:

- `"ok"`
- `"awa"`
- `"u"`
- `"om"`
- `"uk"`
- `"ai"`

This is a classic subword-level degeneration loop. It does not have a mechanism for recognizing that `"Kokokokoku"` is not a plausible completion in this context.

The math prompt also shows semantic repetition:

> `"the mathematical theory of mathematics . The study of"`

The output repeatedly reuses:

- `"mathematics"`
- `"mathematical"`
- `"study"`
- `"theory"`

The model remains within the topic but fails to add information. This is less a literal token loop than a **semantic recurrence loop**.

The France prompt develops a formatting loop:

> `"\n \n = = = = = ="`

After two newlines, the model assigns:

- `" ="`: `0.9933`
- then `" ="`: `1.0`
- subsequent equals signs remain around `0.96–0.99`

This is highly over-concentrated degeneration. It appears to have recognized a WikiText heading or section-format pattern such as repeated equals signs and then entered a deterministic formatting continuation.

### Improvements

At inference time:

- Apply repetition penalties or frequency penalties.
- Penalize repeated n-grams, especially repeated punctuation and heading markers.
- Add stopping rules for repeated newline/equal-sign sequences.
- Use nucleus sampling rather than only top-k.
- Increase temperature modestly; `0.4` strongly suppresses alternatives.

However, these would only mitigate the symptoms. The underlying problem is model instability after an incorrect or low-quality branch.

## 3. Excessive confidence and poor calibration

The distributions frequently assign nearly all probability to one token even when the continuation is wrong:

- `" the"` after `"Tokyo is the capital of"`: `0.9967`
- `" city"` after `"the"`: `0.9890`
- `" K"` after `"the city of"` later in the sequence: `0.9963`
- `" ="` after two newlines: `0.9933`, then exactly `1.0`
- `" Force"` after `"Royal Australian Air"`: effectively `1.0`

High confidence is appropriate for `"Air Force"` or a closing parenthesis, but not for `"the"` instead of `"Japan"` or `"K"` in a malformed continuation. The model appears substantially **overconfident on common corpus templates**.

The temperature of `0.4` exacerbates this appearance and behavior. If the logged probabilities are already temperature-adjusted, the low temperature directly sharpens the distribution. If they are base softmax probabilities and sampling applies temperature separately, then the actual sampling distribution is even more concentrated. Either way, results should also be evaluated at temperature `1.0` and with greedy decoding to separate model uncertainty from decoding effects.

The France trace is an extreme example of confidence collapse into a formatting mode. Once the context ends in `"\n \n"`, almost all mass moves to `"="`, with essentially no recovery path.

## 4. Weak semantic coherence despite good grammar

The mathematical generation is grammatical but circular:

> `"In mathematics, a function is the basis for the study of the mathematical theory of mathematics ."`

The first few choices are plausible at a local level:

- `"the"`: `0.588`
- `"basis"`: `0.753`
- `"for"`: `0.657`
- `"the"`: `0.815`

But the continuation does not define a function. A stronger answer would likely involve:

- `"a relation that assigns"`
- `"a mapping"`
- `"a set"`
- `"each element of a domain"`

Instead, the model chooses generic academic prose and repeatedly recycles the prompt’s own vocabulary. This indicates that it captures **register and topical words** better than concepts.

The model does not drift completely away from mathematics, but it exhibits a shallow form of topic maintenance: it stays near “mathematics” while failing to progress propositionally.

## 5. Inconsistent local continuation in the World War II example

The World War II trace is the best overall generation, but it also exposes uncertainty:

- After `"During the Second World War"` the model prefers `"."` at `0.530` over `","` at `0.429`, yet samples the comma.
- After `"the Royal"` it spreads probability over first, British, Royal, United, ship, battalion, German, etc.
- After `"Royal Australian"` it prefers `"Navy"` at `0.581` over the correct `"Air"` at `0.387`, but samples `"Air"`.

The sampled path happens to recover the correct phrase. This is a useful example of why low-temperature sampling can still produce either good or bad branches: the correct token is not always the top token, but it remains in the candidate set.

After the abbreviation is completed, however, the model generates:

> `"and the RAAF 's"`

This is syntactically possible but repetitive and weakly informative. It appears to be following a memorized article phrase or a generic coordination pattern, not building a new coherent sentence.

## Prompt-specific analysis

### Tokyo

**Strengths**

- Recognizes that `"capital of"` is associated with cities, countries, and geographic entities.
- Produces grammatical function-word sequences such as `"the city of"`.
- Can complete subword fragments with plausible Japanese-looking endings.

**Weaknesses**

- Almost entirely misses the intended fact `"Japan"`.
- Repeats `"city of"` and invents `"Kokokokoku"`.
- Shows self-reinforcing subword degeneration.
- Uses near-deterministic confidence for incorrect continuations.

This is the strongest evidence for shallow heuristics and poor factual grounding.

### France

**Strengths**

- Produces fluent Wikipedia-like prose and coherent punctuation.
- Correctly recognizes sentence and paragraph boundaries.
- Learns section-heading formatting patterns from WikiText.

**Weaknesses**

- Does not produce `"Paris"` or even a likely direct answer.
- Generates semantically invalid text with very high confidence.
- Enters a repeated equals-sign loop after newlines.
- Demonstrates vulnerability to corpus formatting artifacts.

This prompt shows both factual weakness and formatting-mode collapse.

### Mathematics

**Strengths**

- Maintains the mathematical topic.
- Generates grammatical academic phrasing.
- Has strong confidence on syntactic dependencies such as `"basis for"`, `"study of"`, and `"theory of"`.

**Weaknesses**

- Does not provide a definition.
- Repeats prompt terms and related nouns.
- The distribution broadens at noun-selection points, but the chosen path remains generic.
- Shows semantic stagnation rather than useful progression.

This is a language-style success but a knowledge/content failure.

### Second World War

**Strengths**

- Best entity completion among the examples.
- Correctly reconstructs `"Royal Australian Air Force (RAAF)"`.
- Handles parentheses, abbreviations, possessive morphology, and punctuation well.
- Maintains topic and local grammatical structure.

**Weaknesses**

- Considerable uncertainty among related military entities.
- Repeats `"RAAF"` unnecessarily.
- The generated continuation does not develop into a clear factual statement.
- Correctness partly depends on sampling a non-top alternative (`"Air"` over `"Navy"`).

## Architectural and training recommendations

### 1. Increase depth first

Two decoder blocks are a major limitation for compositional dependency tracking and factual retrieval. A more capable version should use at least:

- 4–8 decoder blocks
- 512 or larger hidden dimension if compute allows
- appropriately scaled feed-forward layers
- modern residual and normalization configuration

Depth is likely more important than simply increasing the vocabulary or feed-forward width, because the model needs more transformation steps to combine the prompt relation with entity knowledge.

### 2. Improve training quality and schedule

The model was trained for 70 epochs, with the best checkpoint at 67. Recommended checks:

- Evaluate train and validation loss in identical `eval()` mode.
- Verify that validation is not easier because of preprocessing or document overlap.
- Inspect loss by token type: entities, punctuation, headings, and ordinary prose.
- Use learning-rate warmup and cosine decay.
- Consider AdamW with tuned weight decay.
- Use gradient clipping and monitor instability.
- Compare validation loss across epochs rather than relying only on the global minimum.

The lower validation loss than training loss should be investigated before drawing conclusions about generalization.

### 3. Address document formatting artifacts

WikiText-103-raw contains headings, newlines, markup-like formatting, and repeated equals signs. The model has clearly learned these patterns. Options include:

- Preserve formatting if the goal is WikiText continuation, but evaluate separately on clean prose.
- Normalize or downweight pathological heading patterns for general-purpose generation.
- Add generation-time handling for repeated section markers and newline loops.
- Report separate metrics for prose, headings, and entity-heavy text.

### 4. Use better tokenization or vocabulary diagnostics

The `"K" + "ok" + "ok"` behavior suggests that subword tokenization permits awkward fragments to form plausible-looking pseudo-words. Inspect:

- Whether `"Paris"`, `"Japan"`, `"Tokyo"`, and common capitals are single tokens or fragmented.
- Token frequencies for country and city names.
- Whether leading-space variants are consistently represented.
- The rate of rare subword generation.

A vocabulary with better coverage of common entities, or at least more balanced entity exposure, could improve factual completions.

### 5. Add entity and factual training signals

Plain next-token training on WikiText is not sufficient for reliable short factual prompts. Useful additions include:

- More entity-dense examples.
- Synthetic cloze prompts such as `"The capital of France is Paris."`
- Knowledge-focused continued pretraining.
- Span corruption or entity completion objectives.
- Retrieval augmentation for factual queries.
- Supervised instruction fine-tuning if question-answer behavior is desired.

For a model this small, retrieval augmentation may be more effective than trying to store all facts parametrically.

### 6. Improve decoding and stopping

For this model, decoding should be evaluated systematically:

- Greedy decoding
- Temperature 0.7–1.0
- Top-p sampling
- Repetition penalty
- No-repeat n-gram constraints
- Explicit stopping on repeated formatting patterns

A temperature of `0.4` is too sharp for diagnosing intrinsic uncertainty and makes already overconfident distributions more brittle. It may improve short phrase completion but worsens recovery and diversity.

## Bottom line

The model is a competent **local phrase and syntax predictor** with notable memorization of WikiText-style entities and formatting. Its strongest behavior is the reconstruction of `"Royal Australian Air Force (RAAF)"` and other short grammatical sequences.

Its central weakness is that it often mistakes **high-frequency surface continuation for semantic completion**. It fails easy capital-city facts, produces generic circular mathematics prose, and can enter subword or formatting loops. The probability traces show that these failures are usually accompanied by excessive confidence, not merely broad uncertainty. More depth and capacity, better entity-focused training, careful preprocessing, and repetition-aware decoding would be the most direct improvements.