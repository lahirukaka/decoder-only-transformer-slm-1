## Overall assessment

`capacity-05` is a small, locally fluent language model with strong next-token pattern completion, but weak semantic control and factual grounding. Its validation loss of **3.47** is consistent with a model that has learned common WikiText phrase templates and syntax, while still producing substantial errors on short factual prompts. The traces show a marked distinction between:

- **High confidence in local continuations**, often above 0.9 or 0.99.
- **Poor confidence calibration at the semantic level**: the model can be nearly certain about a continuation that makes the preceding statement false or nonsensical.
- **Severe degeneracy under self-generated context**, especially on the mathematics prompt.

The use of pre-norm is likely helping optimization and gradient flow, but nothing in these traces suggests that normalization placement is the primary limitation. The dominant issues are capacity, data/objective limitations, calibration, and decoding behavior.

---

## What the model does well

### 1. Strong local syntactic modeling

The model is often very effective at completing common phrase patterns:

- `"The capital of France is the largest city in the world , with a population of about 1 @,@ 000"`
- `"the Royal Australian Air Force ("`
- `"the largest ship in the world , and"`

Many grammatical transitions have very concentrated distributions:

- `"population"` → `" of"`: **0.99977**
- `"Air"` → `" Force"`: **0.9999998**
- `"in the"` → `"world"`: **0.958**
- `"largest"` → `"city"`: **0.940**
- `"city"` → `"in"`: **0.994**

This indicates that the model has learned common lexical and syntactic collocations effectively.

### 2. It can follow WikiText-style prose

The France and World War prompts produce continuations resembling encyclopedic prose rather than conversational text. The model uses:

- article-like noun phrases,
- numerical population formatting,
- military organization terminology,
- parenthetical continuations,
- punctuation and conjunction patterns.

The `@,@` token is especially informative: the model has learned the WikiText-103 preprocessing convention for comma-separated numbers. Its continuation:

> `about 1 @,@ 000`

is not arbitrary language-model noise; it reflects memorization of corpus-specific formatting.

### 3. Good phrase memorization and entity continuation

The World War trace has strong entity-level continuation once it reaches a familiar phrase:

> `Royal Australian Air Force`

The transitions are highly confident:

- `"the"` → `"Royal"`: **0.752**
- `"Royal"` → `"Australian"`: **0.948**
- `"Air"` → `"Force"`: **0.9999998**
- `"Force"` → `"("`: **0.9992**

This suggests useful memorization of proper names and article fragments. At this level of capacity, that kind of phrase knowledge is a genuine strength.

### 4. Pre-norm likely supports stable training

The six-block, 256-dimensional decoder is relatively small, but pre-norm is generally a sound choice for stable optimization, particularly when stacking multiple Transformer blocks. The reported losses do not indicate obvious numerical instability or optimization collapse during training. The model is producing coherent probability distributions rather than pathological logits.

---

## Major weaknesses

## 1. It fails simple factual prompts because local syntax dominates semantics

The clearest example is:

> `Tokyo is the capital of`

The model assigns:

- `" the"`: **0.9976**
- `" Japan"`: **0.00177**

After choosing `"the"`, it produces:

> `Tokyo is the capital of the United States .`

This is a major semantic failure. The correct completion is simply `" Japan"`, but the model strongly prefers a common continuation pattern of the form:

> `the capital of the United States`

The model appears to condition primarily on the local suffix `"capital of"` and insufficiently on the subject `"Tokyo"`.

The top-k distribution is not merely uncertain; it is **confidently wrong**. `"Japan"` is present as the second-ranked alternative at the first step, which shows that some relevant association exists, but it is suppressed by a stronger generic phrase template.

This is evidence of a shallow heuristic:

> `capital of` → `the ...`

rather than robust subject-object reasoning:

> `Tokyo` + `capital` → `Japan`.

The subsequent `"United"` → `"States"` transition is extremely confident (**0.948**), showing that once the wrong branch is selected, the model rapidly commits to a memorized phrase.

### Topic drift after the false statement

Following the generated sentence, the model continues:

> `The ship is the largest ship in the world , and`

This is grammatically fluent but semantically unrelated to the initial Tokyo prompt. The likely cause is retrieval of a high-probability WikiText article fragment after generating `"The"`, rather than maintaining a representation of the original topic.

The generation is therefore not random topic drift. It is **template-driven drift**: the model enters a highly familiar article-like continuation and follows it confidently.

---

## 2. The France prompt is fluent but factually and pragmatically poor

For:

> `The capital of France is`

the model generates:

> `the largest city in the world , with a population of about 1 @,@ 000`

The continuation does not state `"Paris"`. It instead produces a generic superlative/population template. The sentence is syntactically acceptable, but `"the largest city in the world"` is false or at least badly misleading in this context.

The first distribution is already revealing:

- `" the"`: **0.840**
- `" a"`: **0.099**
- `" located"`: **0.039**
- `" in"`: **0.012**

The model does not assign a meaningful probability to `"Paris"` in the displayed top-k list. This suggests that its representation of the prompt does not retrieve the fact directly. It is completing an article-style phrase associated with `"capital of France is"` rather than answering the factual relation.

Once `"the largest"` is generated, the remainder is extremely predictable:

- `"largest"` → `"city"`: **0.940**
- `"city"` → `"in"`: **0.994**
- `"in the"` → `"world"`: **0.933**

This is another case where high local confidence masks poor global correctness.

### Punctuation uncertainty

At:

> `The capital of France is the largest city in the world`

the top probabilities are:

- `" ."`: **0.697**
- `" ,"`: **0.297**

The sampled token is the comma, despite the period being the top prediction. This is plausible under temperature sampling, but it makes the generation continue into a longer population claim. At the next step, the distribution is nearly tied:

- `"and"`: **0.510**
- `"with"`: **0.478**

This is a genuinely uncertain discourse transition. The model is choosing between two common templates rather than deciding based on a well-formed semantic plan.

---

## 3. The mathematics prompt exhibits severe repetition collapse

The worst generation is:

> `In mathematics, a function is a function that is the function of the function of the function`

The trace shows a clear self-reinforcing loop:

1. `"a"` → `"function"`: **0.885**
2. `"function"` → `"that"`: **0.205**, while `"of"` is top at **0.779**
3. `"that"` → `"is"`: **0.927**
4. `"is"` produces a broad distribution over `"a"`, `"the"`, `"not"`, `"defined"`, etc.
5. `"the"` → `"function"` becomes increasingly likely
6. `"function"` → `"of"`: **0.999**
7. `"of"` → `"the"`: **0.786**, later **0.984**
8. `"the"` → `"function"`: eventually **0.984**

This is not ordinary repetition caused only by low temperature. The model’s conditional distribution itself has entered a high-probability attractor:

> `function` → `of` → `the` → `function`

The prompt starts in a mathematical domain, but the model fails to produce a definition such as “a relation that assigns…” or “a mapping from…”. It instead repeats the most salient noun and associated preposition.

### Why this likely happens

Several factors may contribute:

- **Insufficient capacity** to preserve a useful semantic state over the generated sequence.
- **WikiText-only training**, which contains definitions but does not explicitly train instruction-style answers.
- **Exposure bias**: at training time, the model sees correct continuations; at inference time, one poor choice creates an unusual context and the model has no robust recovery mechanism.
- **Low temperature**, which suppresses escape from the loop.
- **No repetition-aware decoding** or repetition penalty.
- Possible overfitting to short local patterns such as `function of the`.

The probability concentration grows as the loop continues. This is a signature of **autoregressive attractor behavior**, not merely general uncertainty.

---

## 4. The model is highly overconfident in familiar local contexts

Many distributions are excessively concentrated:

- `"in"` → `"the"`: **0.99966**
- `"population"` → `"of"`: **0.99977**
- `"battalions of"` → `"the"`: **0.999991**
- `"Air"` → `"Force"`: **0.9999998**
- `"function of the"` → `"function"`: **0.984**

Some concentration is appropriate for fixed collocations, especially `"Air Force"` and `"population of"`. However, the same sharpness appears after the model has already entered an incorrect or degraded trajectory.

This indicates poor calibration. The model distinguishes common token-level patterns strongly, but it does not represent uncertainty about:

- whether the current topic is still appropriate,
- whether a generated sentence is factually compatible with the prompt,
- whether it should terminate,
- whether the current continuation is a repetitive loop.

The top-k lists are therefore often **locally confident but globally confused**.

---

## 5. It has weak discourse planning and stopping behavior

None of the generations reaches a natural stopping point within 15 tokens. This is not necessarily a defect by itself because the evaluation fixes generation length, but the model repeatedly extends claims rather than closing them:

- Tokyo prompt becomes a ship description.
- France prompt becomes an unsupported population statistic.
- Mathematics prompt continues indefinitely through a loop.
- World War prompt ends at an opening parenthesis: `"Force ("`.

The model seems optimized for continuation of corpus text, not for producing a complete answer. WikiText-103 is a next-token corpus, so this behavior is expected to some extent. But for prompt-based evaluation, it means the model lacks a strong notion of answer boundaries.

---

## Prompt-specific summary

### Tokyo

**Strengths**
- Strong knowledge of the phrase `"United States"`.
- Correctly models punctuation after a noun phrase.
- Produces grammatical prose after the initial error.

**Weaknesses**
- Fails the basic fact that Tokyo is in Japan.
- Assigns only **0.00177** to `"Japan"` versus **0.9976** to `"the"`.
- Drifts into an unrelated ship article.
- Demonstrates confident continuation of a false premise.

**Interpretation**
- Strong lexical/template completion, weak entity-sensitive factual conditioning.

### France

**Strengths**
- Coherent article-like continuation.
- Excellent syntactic transitions and numerical formatting.
- Good handling of population-related phrase structure.

**Weaknesses**
- Does not answer `"Paris"`.
- Produces a false superlative claim.
- At several points, alternatives are syntactically ambiguous rather than semantically selected.

**Interpretation**
- Better discourse coherence than Tokyo, but still relies on memorized generic Wikipedia templates rather than fact retrieval.

### Mathematics

**Strengths**
- Recognizes the domain and common definition-like vocabulary.
- Initial distribution includes plausible tokens such as `"defined"`, `"called"`, `"given"`, and `"that"`.

**Weaknesses**
- Repeats `"function"` and `"of the"` almost immediately.
- Probability becomes increasingly concentrated in the repetition loop.
- No semantic definition or topic development.

**Interpretation**
- Severe autoregressive degeneration and weak semantic continuation under open-ended generation.

### Second World War

**Strengths**
- Best example of entity and phrase continuation.
- Maintains the military/WikiText topic.
- Recovers a plausible named organization: `"Royal Australian Air Force"`.

**Weaknesses**
- The beginning is uncertain and somewhat awkward:
  - period **0.590** vs comma **0.377**, but comma sampled;
  - `"Royal"` **0.295**, `"Second"` **0.172**, `"British"` **0.135**, `"first"` **0.095**.
- At `"the first of the"`, number selection is diffuse:
  - `"first"` **0.287**, `"three"` **0.241**, `"two"` **0.235**, `"four"` **0.123**.
- The continuation may be a memorized fragment whose factual compatibility with the prompt is not verified.

**Interpretation**
- Strongest memorization and topical consistency, but still largely phrase retrieval.

---

## Role of sampling and temperature

The temperature is **0.4**, so generation is already relatively conservative. This helps expose the model’s preferred modes:

- It does not merely make occasional random errors.
- It deterministically enters highly probable incorrect templates and repetition loops.

Some sampled tokens differ from the top prediction:

- Tokyo: `"ship"` sampled although `"city"` has **0.983**.
- France punctuation: comma sampled although period has **0.697**.
- World War initial punctuation: comma sampled although period has **0.590**.
- `"battalions"` sampled although `"@-@"` is top at **0.364**.

These choices are plausible under sampling, but they show that top-k sampling can amplify a weak branch. For diagnostic evaluation, greedy decoding, multiple seeds, and temperatures such as 0.7–1.0 should also be compared. The mathematical loop, however, is unlikely to disappear entirely with sampling because the model assigns very high probability to the recurrent transitions.

---

## Architectural and training improvements

### 1. Increase model capacity

The current model is small:

- dimension: **256**
- layers: **6**
- heads: **8**
- FFN dimension: **1024**
- context: **128**

Increasing hidden size and depth would likely improve:

- subject-sensitive conditioning,
- long-range prompt retention,
- discourse continuity,
- resistance to repetition attractors.

A larger FFN and more layers may help, but increasing width and total parameter count is likely more valuable than simply adding heads. Eight heads at dimension 256 gives only 32 dimensions per head, which is adequate but constrained.

### 2. Use a longer context window

A context length of 128 is sufficient for these short prompts, but longer context can improve article-level coherence and reduce immediate dependence on the last few tokens. More importantly, training with longer contiguous sequences may encourage broader discourse representations instead of isolated local n-gram behavior.

### 3. Improve tokenization and corpus preprocessing

The tokenizer appears to use WikiText/GPT-2-style tokens, including fragments such as:

- `" U"`
- `" K"`
- `"@,@"`
- `"@-@"`
- `"bu"`

This is workable, but a better subword vocabulary or tokenizer evaluation could improve:

- proper-name handling,
- country and city names,
- morphological regularity,
- numerical sequences.

The Tokyo failure is not solely a tokenization issue, but robust tokenization of entities such as `"Tokyo"`, `"Japan"`, and `"France"` can help entity associations.

### 4. Add data or objectives that reward factual and semantic consistency

WikiText-103 is useful for language modeling but is not designed for reliable question answering. To improve these prompts, training could include:

- curated factual statements,
- entity-relation examples,
- synthetic cloze tasks such as `Tokyo is the capital of ___`,
- contrastive examples penalizing factually incompatible continuations,
- instruction-tuning or supervised answer completion,
- retrieval-augmented generation for factual queries.

Without such data, the model learns that a continuation is probable, not that it is true.

### 5. Use regularization and monitor overfitting

The train loss (**3.60**) being higher than validation loss (**3.47**) is unusual if both are measured comparably, though dropout, evaluation mode, data splits, or metric conventions may explain it. This should be investigated before drawing overfitting conclusions.

Useful checks include:

- train/validation loss under identical evaluation mode,
- validation loss by document/topic,
- calibration error,
- repetition rate,
- entity prediction accuracy,
- likelihood of correct versus generic completions.

If the model is overfitting phrase templates, dropout, weight decay, data deduplication, and more diverse training text may help. If it is under-capacity, regularization alone will not solve the semantic failures.

### 6. Add decoding safeguards

For deployment, decoding changes could substantially reduce visible failures:

- repetition penalty,
- no-repeat n-gram constraints,
- detection of repeated cycles such as `function of the function`,
- adaptive temperature,
- nucleus sampling,
- minimum/maximum repetition controls,
- EOS-aware generation,
- stopping after a complete factual answer.

These are mitigation strategies rather than cures. They would reduce the mathematics loop but would not make the model know that Tokyo’s capital relationship is Japan.

### 7. Improve calibration

The distributions suggest overconfidence. Temperature scaling on a held-out calibration set, label smoothing during training, or post-hoc logit calibration may make uncertainty more realistic. Calibration should be evaluated separately for:

- syntactic transitions,
- entity completions,
- factual cloze prompts,
- repetitive contexts.

The goal is not simply to flatten all distributions. Fixed phrases such as `"Air Force"` should remain confident, while semantically ambiguous or unsupported continuations should retain meaningful alternatives.

---

## Bottom line

This model has learned a useful amount of WikiText-style syntax, punctuation, numerical formatting, and named-entity phrase continuation. Its strongest behavior is **memorized local prose completion**, particularly in the World War prompt.

Its central weakness is that it treats language modeling as local pattern continuation rather than globally constrained semantic generation. The Tokyo and France prompts show that it can confidently produce false or irrelevant continuations, while the mathematics prompt exposes a severe self-repetition attractor. Pre-norm is a sensible architectural choice and likely contributes to stable training, but the observed failures primarily call for more capacity, better semantic/factual training data, longer-context training, calibration, and repetition-aware decoding.