## Overall assessment

`capacity-01` has learned a substantial amount of **local English syntax, punctuation, and WikiText-style article continuation**, but it performs poorly on semantic completion and factual retrieval. The generations are often grammatical at the token level while being globally incoherent or factually wrong.

The main signature is:

- strong next-token preferences for common local continuations;
- severe overconfidence on punctuation, function words, and corpus-format tokens;
- weak conditioning on the actual topic or factual relation in the prompt;
- rapid transition into generic WikiText-like prose;
- repetition and self-reinforcing continuation loops.

The low model capacity is consistent with this behavior: two decoder blocks, width 128, and only 4 heads are sufficient for shallow local statistics but limited for robust entity tracking and factual composition. The validation loss of 4.10 corresponds to a perplexity of roughly 60, so the model is still a relatively weak language model despite 70 epochs.

---

## Prompt 1: “Tokyo is the capital of”

### What it does well

The initial completion is locally plausible:

> “Tokyo is the capital of the city…”

The first-step distribution is extremely concentrated:

- `the`: 0.9946
- `a`: 0.0046

This indicates that the model has learned a strong syntactic pattern after “capital of”: an article is likely. It also produces locally grammatical transitions such as:

> “The city’s population is the largest city”

The model handles:

- article and noun selection;
- punctuation after a noun phrase;
- possessive formation using the separate tokens `"'”` and `"s"`;
- common factual-article templates such as “the city’s population is…”

The possessive step is especially deterministic:

- after `"The city '"`, `"s"` has probability essentially 1.0.

That is a reasonable tokenization-level prediction and shows strong memorization of local orthographic patterns.

### Weaknesses

The completion fails immediately at the semantic level. The natural factual continuation is “Japan” or a related formulation, but the model generates:

> “the city of the city”

and then repeats “city” several times.

The sequence:

> “the city of the city . The city ’s population…”

is syntactically shaped but semantically defective. It appears to be following a memorized “city article” template rather than representing the relation:

> Tokyo → capital → Japan

The model does not preserve the distinction between:

- Tokyo as a city;
- Japan as the country;
- “capital of Japan” as the relevant relation.

### Repetition and distribution behavior

There is clear repetition:

> “the city of the city”

and later:

> “the largest city”

The model repeatedly returns to `city` with very high probability:

- after “the”: `city` = 0.9976;
- after “The”: `city` = 0.9989;
- after “largest”: `city` = 0.9938.

This is not merely sampling noise. These are highly concentrated distributions caused by a self-reinforcing local context. Once the model chooses “city,” the following contexts strongly activate a common memorized continuation.

There is some uncertainty at structurally ambiguous positions:

- after “city”: `.` = 0.476, `of` = 0.402, `,` = 0.089;
- after “population is”: `the` = 0.476, `now` = 0.102, `estimated` = 0.098, `approximately` = 0.095.

However, this uncertainty is mostly over ordinary syntactic alternatives, not over the central factual answer. The model is uncertain about how to continue a generic city description, rather than considering the correct entity-level completion.

---

## Prompt 2: “The capital of France is”

### What it does well

The model produces a fluent sequence of common English and article-style phrases:

> “a major factor in the creation of the city of France. In the late…”

Many individual transitions are very confident and grammatically appropriate:

- after “major factor,” `in` = 0.9931;
- after “factor in,” `the` = 0.9983;
- after “creation,” `of` ≈ 1.0;
- after “creation of,” `the` = 0.8712.

This demonstrates strong knowledge of collocations such as:

- “a major factor in…”
- “the creation of…”
- “in the late…”

The model also produces reasonable sentence-boundary behavior, including periods, paragraph-style capitalization, and possible newline transitions.

### Weaknesses

This is the clearest factual failure. The expected completion is:

> “Paris”

but the model begins with:

> “a major factor…”

It never recovers the intended fact. Instead, it interprets the prompt as the beginning of a generic historical or geographic article and eventually generates:

> “the city of France”

This is semantically incorrect. France is a country, not a city, and the phrase is likely assembled from corpus fragments involving “the city of France,” “the creation of the city,” and other common article constructions.

At the step after “The capital of France is,” the model’s distribution is:

- `the`: 0.694
- `a`: 0.246
- `now`: 0.019
- `also`: 0.012
- `located`: 0.0045

Notably, `Paris` is absent from the top ten. This is evidence of a missing or weak factual association, not simply unfortunate sampling. At temperature 0.4, the sampled `a` is less likely than `the`, but both are already semantically wrong.

### Topic drift

The generation drifts through several weakly connected topics:

1. generic “major factor” phrase;
2. “creation” and “city”;
3. countries and national adjectives;
4. “France” repeated as a noun;
5. a historical time phrase: “In the late…”

The top-k distribution after:

> “the creation of the”

is particularly diffuse:

- `French`: 0.293
- `city`: 0.184
- `country`: 0.117
- `United`: 0.100
- `European`: 0.088
- `German`: 0.065
- `British`: 0.056

This is a confused topic distribution. It recognizes a broad geographic/historical context but does not identify the specific answer.

The model also gives high probability to an invalid or awkward self-referential path:

> “the city of France”

After “city of,” the probability of `France` is 0.156, while `the` is 0.798 and `Paris` only 0.0118. This indicates broad lexical association with France, but poor semantic role assignment.

---

## Prompt 3: “In mathematics, a function is”

### What it does well

This prompt produces the most coherent-looking continuation:

> “an example of the number of mathematical applications. The first two are also used”

The output maintains a mathematics-related vocabulary for several steps:

- `example`
- `number`
- `mathematical`
- `applications`

It also shows strong local syntactic control:

- after “example,” `of` = 0.9997;
- after “number,” `of` = 0.99998;
- after “applications,” period = 0.7798;
- after “The,” `first` = 0.494.

The model has learned common expository structures such as:

> “The first two are also used…”

The transition after “first two” is less certain but still syntactically reasonable:

- `@-@`: 0.449
- `are`: 0.332
- `of`: 0.118
- `is`: 0.050

The sampled `are` is a sensible continuation.

### Weaknesses

The output does not define a function. A mathematically meaningful completion would mention a mapping from inputs to outputs, a domain and codomain, or assignment of exactly one output to each input. Instead, the model produces:

> “an example of the number of mathematical applications”

This is a generic mathematical-sounding phrase, but it is conceptually empty and awkward. The model appears to use topical vocabulary rather than construct a definition.

There is also an early article error. After:

> “In mathematics, a function is”

the model samples `an`, even though the highest-probability token is `a`:

- `a`: 0.788
- `the`: 0.129
- `not`: 0.036
- `an`: 0.034

At temperature 0.4, sampling should be relatively conservative, but non-top-1 sampling still occurs. This particular choice makes the continuation grammatically acceptable (“an example”), but it reflects stochastic selection rather than a strong semantic plan.

### Shallow heuristic behavior

The model seems to associate “mathematics” and “function” with phrases such as:

- “an example of…”
- “the number of…”
- “mathematical applications”
- “the first two…”

This is a topical continuation heuristic. It can stay in the right broad domain, but it does not know what information the prompt requests.

The distribution after “the” is highly uncertain:

- quote: 0.548
- number: 0.105
- first: 0.067
- mathematical: 0.061
- same: 0.050
- theory: 0.046
- fact: 0.044

This suggests the model has several possible WikiText-style continuation modes, including quotation, definition, and generic exposition. It selects “number,” but there is no evidence of a well-formed mathematical representation.

---

## Prompt 4: “During the Second World War”

### What it does well

The model recognizes that this resembles WikiText article prose. It generates:

> “In the aftermath of the war…”

The latter portion is reasonably coherent and historically styled:

> “In the aftermath of the war, the…”

The transitions are locally strong:

- after `In`: `the` = 0.9297;
- after `aftermath`: `of` = 0.788;
- after `of the`: `war` = 0.983;
- after `war,`: `the` = 0.9979.

This is good local language modeling of common historical prose.

### Major weakness: corpus-format contamination

The generation first becomes:

> “. \n \n = = = = In…”

The repeated equals signs are characteristic of WikiText article heading markup. The model has learned the dataset’s formatting distribution, but it does not know when that formatting is appropriate for the prompt.

The repetition is extreme:

> `=` → `=` → `=` → `=`

with probabilities:

- after the first equals sign: `=` = 1.0;
- after the second: `=` = 0.9975;
- after the third: `=` = 0.8945.

This is a strong mode collapse into a section-heading pattern. The prompt itself does not request a heading, and the model fails to suppress a high-frequency WikiText structural template.

The initial punctuation is also questionable:

- after “During the Second World War,” `.` = 0.669 and `,` = 0.274.

A continuation such as “During the Second World War, …” is more natural in many contexts, but the model strongly prefers ending the phrase as a standalone fragment, presumably because it has seen it in article/title contexts.

### Recovery after format drift

At the fourth equals sign, the distribution becomes more mixed:

- `=`: 0.624
- newline: 0.109
- `Second`: 0.058
- `First`: 0.043
- `The`: 0.035
- `In`: 0.034

The sampled `In` allows the model to escape the loop. This is a useful sign: it is not permanently trapped, but it requires stochastic deviation from a very high-probability formatting pattern.

The continuation afterward is semantically generic:

> “In the aftermath of the war…”

It is plausible but not specifically grounded in any event, country, battle, or consequence. The distribution after “In the” is broad:

- `United`: 0.342
- `early`: 0.280
- `aftermath`: 0.112
- `late`: 0.089
- years and historical labels in the remainder.

This shows uncertainty about the historical continuation and a tendency to select broad temporal or geopolitical phrases.

---

## Distributional diagnosis

### Over-concentration

Many distributions are dramatically over-concentrated:

- `of` after “creation” ≈ 1.0;
- `of` after “number” ≈ 1.0;
- `the` after “factor in” = 0.998;
- `city` after “The city” = 0.999;
- repeated `=` tokens ≈ 1.0;
- `s` after an apostrophe = 1.0.

This is useful for deterministic local syntax, but harmful when the model is wrong. Once it enters an incorrect phrase, the high confidence prevents recovery.

### Confusion

The model is uncertain at content-bearing positions:

- “creation of the”: many national and geographic adjectives;
- “In the”: United / early / aftermath / late / dates;
- “the number of mathematical”: mathematical / the / elements / applications / objects;
- “major”: tourist / factor / part / city / source.

The uncertainty is usually broad but shallow: alternatives are semantically related words or corpus templates, not well-structured competing interpretations.

### Calibration

The model appears poorly calibrated. It assigns extreme confidence to many continuations that are only locally plausible and globally wrong. The probabilities are therefore better interpreted as confidence in a learned phrase pattern than confidence in a correct completion.

The temperature of 0.4 further sharpens this behavior. It makes the model more likely to remain in high-probability repetition and markup loops. For diagnosis, generation at temperature 1.0, greedy decoding, and nucleus sampling should also be compared. The underlying overconfidence, however, is visible in the raw distributions and is not caused only by the sampling temperature.

---

## Common weaknesses

1. **Poor factual retrieval**
   - Fails to produce “Japan” for Tokyo.
   - Fails to produce “Paris” for France.
   - Does not provide a mathematical definition.

2. **Weak entity and relation tracking**
   - Treats “capital of France” as generic geography instead of a specific fact.
   - Generates “city of France,” showing lexical association without semantic role control.

3. **Template continuation over semantic completion**
   - Follows “city article,” “major factor,” “mathematical applications,” and historical prose templates.
   - The prompt topic influences vocabulary, but not the intended proposition.

4. **Repetition and attractor states**
   - Repeated `city`.
   - Repeated equals signs.
   - Recurrent “the … of …” structures.

5. **Dataset-format leakage**
   - WikiText section markers and blank lines are reproduced as content.
   - Article continuation priors dominate natural prompt completion.

6. **Limited long-range planning**
   - The model is good at the next few syntactic tokens but does not maintain a coherent answer objective over 15 tokens.

---

## Recommended improvements

### Model architecture

- Increase model width substantially beyond 128.
- Use more decoder blocks than 2; depth is especially important for composition and entity relations.
- Increase the number of attention heads only alongside sufficient model width. More heads alone are unlikely to solve the issue.
- Use a modern stable architecture such as pre-normalization, learned or rotary positional embeddings, and tied input/output embeddings if not already present.
- Consider gated feed-forward layers such as SwiGLU rather than a plain 512-dimensional feed-forward block.

A larger model would likely improve factual association and reduce the tendency to fall back on generic phrase templates, although WikiText-only training still limits factual reliability.

### Training and data

- Train on more tokens or use more diverse data; 70 epochs on WikiText-103 can encourage memorization of local corpus patterns.
- Inspect the unusual metric relationship where training loss is 4.33 but validation loss is 4.10. This may reflect different preprocessing, dropout/evaluation settings, split composition, or a logging issue.
- Clean or explicitly handle WikiText markup such as heading equals signs and paragraph boundaries.
- Add document-boundary and section-boundary handling so the model learns when markup should be emitted.
- Use curriculum or mixed-domain data containing factual question-like completions and definitions if factual prompting is an evaluation goal.
- Evaluate checkpoints for calibration and repetition, not only validation cross-entropy.

### Decoding and post-processing

- Use a higher temperature or nucleus sampling for exploratory generation, though this will not fix factual failures.
- Add repetition penalties or no-repeat n-gram constraints to reduce `city` and `=` loops.
- Suppress formatting tokens such as repeated `=` after ordinary prompts if clean prose is desired.
- For factual prompts, use retrieval augmentation or constrained decoding rather than expecting this small language model to retrieve facts reliably from parametric memory.

### Evaluation additions

Useful next tests would include:

- greedy versus temperature 0.4 versus temperature 1.0;
- prompts where the answer is known to occur verbatim in WikiText;
- minimal pairs such as “The capital of France is” and “The capital of Japan is”;
- repetition rate and distinct-n-gram statistics;
- calibration metrics such as expected calibration error;
- log probability assigned to the correct tokens `Paris`, `Japan`, and a standard function definition;
- document-format prompts versus clean prose prompts.

## Bottom line

The model demonstrates competent **local syntactic modeling and WikiText-style phrase continuation**, with strong handling of collocations, punctuation, possessives, and article prose. Its main limitation is that it behaves like a shallow phrase model: it recognizes topical words and common templates but does not reliably represent factual relations or maintain a semantic objective. The very concentrated distributions make this worse by locking the model into incorrect city-description, generic exposition, or WikiText-heading loops.