## Overall assessment

This is a small, partially competent WikiText continuation model, but not a reliable question-answering model. It has learned:

- strong local syntax and punctuation transitions,
- common WikiText-style discourse patterns,
- some multi-token names and phrases,
- reasonable short-range historical prose patterns.

However, it frequently follows a high-probability corpus continuation rather than the semantic intent of the prompt. It also becomes sharply overconfident after entering an incorrect trajectory, producing repetition and document-format artifacts.

The model is a 6-layer, 256-dimensional decoder with 8 heads and a 128-token context. Given the relatively high validation loss of 3.60, the behavior is consistent with a modest-capacity language model that has learned local n-gram-like regularities but has limited semantic grounding and weak long-range control.

---

## 1. “Tokyo is the capital of”

### Observed behavior

The output is:

> “the Kōchiha clan. The Kōchiha clan is the”

This is a major semantic failure. The expected completion would be “Japan,” but the model first emits:

> `the` with probability 0.9983

and then selects `K`, leading to the subword sequence:

> `K` → `ō` → `chi` → `ha` → `clan`

The tokenization is important here. The model is not initially predicting the complete erroneous phrase “Kōchiha clan.” It enters it through a series of locally plausible subword decisions. After `Kō` the distribution is nearly split between `ok` and `ash`, with `ō` only third at 0.113. Sampling at temperature 0.4 nevertheless selects `ō`. Once `Kōchiha` has been formed, the model becomes extremely confident:

- `chi` after `Kō`: 0.999999
- `ha` after `Kōchi`: 0.999985
- `clan` after `Kōchiha`: 0.999785

This is characteristic of memorized multi-token phrase completion. The model has likely seen “Kōchiha clan” or similar WikiText material and has a strong continuation representation for it, but it does not appropriately resolve the initial semantic constraint “capital of.”

### Strengths

- It handles rare Unicode/subword material surprisingly well once the phrase is entered.
- It produces a grammatically valid noun phrase and then a plausible continuation:
  > “The Kōchiha clan is the …”
- Sentence punctuation is well modeled:
  - period after `clan`: 0.772
  - sentence-initial “The”: 0.493 versus newline: 0.456

### Weaknesses

The first-step distribution reveals a strong syntactic heuristic:

> `the`: 0.9983, `Japan`: 0.00057

The model has learned that “capital of” is often followed by “the” in corpus text, but it does not sufficiently model that Tokyo is itself the subject of a factual relation whose answer is Japan. It then allows an unrelated named entity to take over.

The distribution after “the” is less concentrated:

- `K`: 0.413
- `city`: 0.329
- `I`: 0.060
- `state`: 0.034
- `Kingdom`: 0.032

This indicates genuine uncertainty at the point where semantic selection is required. The sampling decision into `K` is therefore not an isolated deterministic error; it is a low-temperature sample from a confused distribution.

After that, confidence becomes pathological. The model is highly certain about its own generated prefix rather than about the original prompt. This is an example of **self-reinforcing trajectory lock-in**.

### Repetition and topic drift

The phrase is repeated almost immediately:

> “Kōchiha clan. The Kōchiha clan…”

This is not just ordinary discourse repetition. It indicates that the model has entered a memorized phrase loop and is using the generated entity as a new topical anchor. The topic drifts from Japanese geography to a fictional clan without any semantic bridge.

---

## 2. “The capital of France is”

### Observed behavior

The output is:

> “the most widely used country in the world.  
>  
> = = = =”

This is grammatical at the token level but factually nonsensical. The expected answer “Paris” is absent from the top-10 list at the first step.

The first distribution is:

- `the`: 0.573
- `located`: 0.340
- `a`: 0.054
- `now`: 0.014

The model has learned continuations such as:

> “The capital of France is the …”

or

> “The capital of France is located …”

but apparently has not learned the direct factual completion strongly enough. More importantly, `Paris` is not visible among the top candidates, suggesting either weak factual memorization, poor tokenization for the answer, or a strong mismatch between the prompt form and training contexts.

### Local fluency

Once it selects `the`, the next continuation is coherent in form:

- after “the”: `largest` 0.934
- after “the most”: `important` 0.829
- after “the most widely”: `used` 0.797
- after “used”: broad uncertainty, with `city` 0.248, `of` 0.157, `area` 0.146, `country` 0.123
- after “country”: `in` 0.919
- after “in the”: `world` 0.778
- after “world”: period 0.931

This shows good short-range phrase modeling. The sequence “the most widely used country in the world” is syntactically well-formed, even though it is semantically inappropriate in context.

### Format contamination

After the period, the model predicts:

- newline: 0.621
- `The`: 0.287
- `It`: 0.066

After two newlines:

- `=`: 0.9986

Then repeated equals signs become nearly deterministic:

- after `=`, another `=`: 1.0
- after `= =`, another `=`: 0.952
- after `= = =`, another `=`: 0.878

This is classic WikiText structural-format learning. Equals signs are commonly used in section headings, and the model has learned the formatting pattern without understanding whether a heading is appropriate here. The prompt has been converted into a spurious article-like continuation.

### Interpretation

This example demonstrates a distinction between:

- **high local likelihood**, and
- **semantic correctness**.

The model is confident because it recognizes common WikiText phrase templates. Its confidence is not calibrated to factual validity. Low temperature makes the generated continuation more deterministic and therefore makes this kind of coherent nonsense more likely to persist.

---

## 3. “In mathematics, a function is”

### Observed behavior

The output is:

> “given to the number of applications in the class of the class of the class”

This is the clearest repetition failure.

At the first step the distribution is relatively diffuse:

- `the`: 0.267
- `a`: 0.206
- `required`: 0.119
- `defined`: 0.096
- `to`: 0.092
- `called`: 0.060
- `given`: 0.060

The sampled token `given` is only rank 7. This indicates substantial uncertainty about how to complete the definition. “A function is defined …” would be a natural continuation, but the model does not strongly prefer it.

After choosing “given,” the model becomes confident in the generic construction:

> “given to the number of …”

The next distributions are highly uneven:

- `to`: 0.769
- `the`: 0.959
- `of` after “number”: 0.99983

At “the number of applications,” the model is uncertain:

- `applications`: 0.415
- `students`: 0.170
- `other`: 0.082
- `mathematical`: 0.078
- `classes`: 0.058

This suggests the model is associating “number of” and “applications” with educational or generic explanatory prose rather than producing a mathematical definition.

### Repetition dynamics

The continuation then enters a highly deterministic loop:

> `the class` → `of the class` → `of the class of the class`

Representative probabilities include:

- after “class of the”: `class` 0.989
- after “class of the class”: `of` 0.955
- after “class of the class of”: `the` 0.809
- after the final “the”: `class` 0.999

This is a strong example of **degenerate attractor behavior**. The model has learned the local transitions among “class,” “of,” and “the,” but lacks a mechanism to recognize that the resulting phrase is semantically or syntactically deteriorating.

### Topic drift and shallow heuristics

The prompt establishes a mathematics context, but the model moves toward:

- applications,
- students,
- classes,
- physics,
- general educational prose.

The top-k candidates after “the number of” are not mathematical concepts such as “inputs,” “outputs,” “elements,” or “values.” This suggests that the model’s domain representation is weak and that it relies heavily on frequent lexical associations.

This is not merely uncertainty: it is **domain-conditioned shallow continuation**. The word “mathematics” activates nearby corpus vocabulary, but not a robust definition of function.

---

## 4. “During the Second World War”

### Observed behavior

The output is:

> “. In the early 1920s, the British began a series of military campaigns”

This is locally fluent historical prose but temporally inconsistent. “Second World War” is associated with the 1939–1945 period, while “early 1920s” is unrelated.

The first decision is punctuation:

- period: 0.576
- comma: 0.401

The sampled period is plausible as a fragment continuation, but it does not establish a coherent sentence. After the period, the model strongly prefers a new discourse segment:

- newline: 0.772
- `The`: 0.122
- `In`: 0.062
- `He`: 0.033

The sampled `In` is only rank 3. This is another example where low-temperature sampling still permits non-top-1 choices when the distribution is not fully concentrated.

### Temporal and narrative drift

The sequence:

> “In the early 1920s, the British began …”

is generated through plausible local transitions:

- after “In the”: `early` 0.463
- after “early”: `1940` 0.459, `years` 0.245, `1920` 0.088
- sampled `1920` is rank 3
- after “1920”: `s` is essentially deterministic
- after “the”: the distribution is broad:
  - `United`: 0.225
  - `ship`: 0.198
  - `British`: 0.188
  - `battalion`: 0.107
  - `Royal`: 0.097
- after “the British”: `government` 0.662, `Army` 0.284
- sampled `began` is rank 9 with probability 0.0013

The `began` choice is especially notable. It is a low-probability sample from a distribution that strongly prefers a noun phrase such as “government” or “Army.” This creates an awkward but still grammatical continuation:

> “the British began …”

The subsequent phrase is much more stable:

- after “began”: `to` 0.982, sampled `a` is rank 2
- after “a”: `series` 0.871
- after “series”: `of` is effectively 1.0
- after “series of”: broad military vocabulary, with `attacks`, `raids`, `anti`, `military`, `major`, and `operations` all near 0.12–0.16
- after “military”: `campaigns` 0.578

### Strengths

This is the strongest example for prose continuation:

- punctuation is mostly natural,
- the historical register is appropriate,
- “the British began a series of military campaigns” is fluent,
- the model maintains a plausible subject and action over several tokens.

### Weaknesses

The model does not preserve the temporal constraint from the prompt. It appears to retrieve or synthesize a generic historical passage based on “British,” “war,” and “military,” rather than continue the specific WWII context.

This is **topic retention without event- or time-consistency**.

---

## Distributional and calibration analysis

### Strong local concentration

Many steps are extremely concentrated:

- Tokyo initial `the`: 0.998
- Tokyo continuation of Kōchiha: approximately 1.0
- France after “country in”: `the`: 0.989
- mathematics after “number”: `of`: 0.9998
- war after “series”: `of`: approximately 1.0

These indicate strong modeling of local collocations, suffix completion, punctuation, and memorized phrases.

### Over-concentration after self-generated text

The most severe concentration occurs after the model has already made a questionable choice. For example:

- `Kōchiha` → `clan`: 0.9998
- `class of the` → `class`: 0.989
- newline newline → `=`: 0.9986
- `=` → `=`: 1.0

This suggests poor calibration under distribution shift from the original prompt. The model is confident in the continuation of its own generated prefix even when that prefix is semantically invalid.

### Confused states

Several important decision points are broad and multimodal:

- Tokyo after “the”: `K` 0.413 vs `city` 0.329
- mathematics initial completion: no candidate above 0.267
- mathematics after “number of”: `applications` 0.415 vs `students` 0.170
- war after “In the early”: `early` itself is 0.463, with many temporal alternatives
- war after “the”: several competing military nouns
- war after “series of military”: many plausible continuations in the 0.04–0.16 range

These are the points where the model lacks a strong semantic representation and relies on sampling among generic corpus continuations.

### Temperature effect

At temperature 0.4, the model is already being sampled conservatively. Nevertheless, several selected tokens are not top-1:

- Tokyo: `ō` is rank 3
- France: `most` is rank 3
- mathematics: `given` is rank 7
- war: `In` is rank 3
- war: `1920` is rank 3
- war: `began` is rank 9
- war: `a` is rank 2
- war: `military` is rank 4

Thus, the poor outputs are not solely caused by excessive temperature. The underlying distributions often place the correct or more sensible continuation too low, and sampling can move the generation into a bad basin. The low temperature then makes the subsequent basin highly persistent.

---

## Main failure modes

1. **Semantic prompt neglect**  
   The model often treats factual prompts as generic corpus prefixes.

2. **Weak factual recall**  
   “Tokyo … Japan” and “France … Paris” are not selected. For France, “Paris” is not even in the top 10.

3. **Memorized phrase hijacking**  
   The Tokyo prompt is captured by “Kōchiha clan,” a likely memorized WikiText entity sequence.

4. **Temporal inconsistency**  
   “Second World War” drifts to “early 1920s.”

5. **Repetition and attractor loops**  
   The mathematics example repeats “class” in a degenerate cycle.

6. **Document-format contamination**  
   Repeated `=` tokens reveal WikiText heading structure overriding prompt relevance.

7. **Poor uncertainty calibration**  
   The model is uncertain at semantic branching points but nearly certain after an incorrect branch.

8. **Limited domain abstraction**  
   The mathematics prompt activates educational vocabulary but not the conceptual structure of a function.

---

## Recommended improvements

### Training and data

- Increase model capacity, particularly width and depth. A 256-dimensional, 6-block model is small for WikiText-103 semantic modeling.
- Train longer only if validation loss continues improving; the current best epoch being 69 does not by itself indicate undertraining, but the loss remains high.
- Use better document boundary handling and explicitly preserve or mark section structure. This could reduce inappropriate heading continuation.
- Evaluate on held-out factual and semantic prompts separately from perplexity. WikiText likelihood alone will reward fluent but irrelevant continuations.
- Consider mixing in curated factual, definitional, and temporally coherent text if these behaviors are important.
- Add instruction or supervised fine-tuning if the intended behavior is answering “capital of” questions rather than merely continuing WikiText.

### Tokenization

- Inspect the tokenizer carefully. The subword path for `Kōchiha` is handled well, but factual answers such as “Paris” should be tested directly.
- A tokenizer with more balanced treatment of names, Unicode, numbers, and common factual entities may improve recall and reduce awkward partial-word uncertainty.
- Add or verify special handling for document separators, headings, and newline patterns.

### Optimization and regularization

- Use validation-based calibration checks, not just loss.
- Apply label smoothing or post-hoc temperature calibration to reduce extreme probabilities.
- Check whether the reported train loss being higher than validation loss is caused by dropout, data augmentation, or different evaluation modes. It is not necessarily problematic, but it should be verified.
- Use curriculum or sampling strategies that expose the model to longer coherent continuations and cross-sentence dependencies.

### Decoding safeguards

For deployment, decoding changes could mitigate but not solve the underlying problem:

- use a slightly higher temperature only when diversity is desired, but not as a fix for factuality;
- apply repetition penalties or no-repeat n-gram constraints;
- stop or downweight repeated heading markers such as `=`;
- use constrained decoding or retrieval for factual prompts;
- detect sharp probability collapse after repeated tokens and trigger regeneration;
- use contrastive decoding to favor continuations consistent with the prompt representation.

### Architecture

- Increase context length beyond 128 to improve document-level continuity.
- Add more layers and/or dimensions, especially if the goal is semantic and factual retention.
- Consider modern positional encodings such as RoPE if extending context.
- A larger model may improve semantic separation, but retrieval augmentation or supervised factual training is likely more effective for direct factual questions than scaling alone.

## Bottom line

The model is competent at short-range WikiText-style syntax, punctuation, phrase completion, and some historical prose patterns. Its top-k distributions show strong local confidence and good subword completion, but also severe overconfidence after an erroneous semantic branch. The central limitation is not general grammaticality; it is failure to preserve the prompt’s factual, topical, and temporal constraints. The most urgent improvements are stronger semantic/factual training, better calibration, larger capacity, and decoding controls against repetition and document-format attractors.