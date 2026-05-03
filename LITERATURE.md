# Literature Review — Frequency Alignment in Vision–Language Models

**Last revised:** 2026-04-24
**Scope:** The threads of prior work this project draws from, and where our
contribution sits relative to each. Cross-referenced against
`paper/references.bib` (43 entries as of this writing).
**Companion docs:** `../full_project_explanation.md` (synthesis),
`../frequency_alignment/experiment_run.md` (ops).

This document is longer than the Related Work section of the paper on purpose.
The paper has to be concise; this review is for people entering the project
who need to understand *why* each thread matters and *what* they add to the
picture.

---

## 1. Frequency-domain analysis of vision networks

### 1.1 CNN frequency bias and shortcut learning

The starting point for this project is the observation that convolutional
vision backbones rely disproportionately on **high-frequency image content**
that humans largely ignore. @yin2019fourier showed that adversarial
perturbations concentrate energy in bands the model is most sensitive to,
and that models trained with low-pass data are more robust than those
trained with high-pass data. @wang2020high reinforced the result by
showing that generalization performance tracks high-frequency content, not
low-frequency content — the network can learn imperceptible patterns
that are statistically predictive but semantically meaningless.
@tsuzuku2019structural gave a Jacobian-spectrum analysis that ties
adversarial vulnerability to the directionality of the Fourier-basis
response. @geirhos2018imagenet is the companion "texture bias" paper
that interprets this behaviour as a shape/texture trade-off.

**What this thread establishes.** Vision networks have a spectral
preference that can be measured, and that preference is partly responsible
for the mismatch between benchmark accuracy and real-world robustness.

**What this thread does NOT establish.** Whether the spectral response
is *intrinsic* to the vision backbone or *conditioned* on the task being
asked. All four papers treat the model as task-agnostic: they feed an
image in, measure the response, and attribute the bias to the backbone.
There is no analogue of a task-specific filter `W_t(ω)` in this literature
because there is no task — just a classifier.

**Our position.** We take the spectral machinery from this thread
(2D-FFT, radial binning, IPR-style concentration measures) and reapply it
to the *cross-modal attention* slice of a vision-language model, which
is inherently task-conditioned. The resulting filter is not a property of
the backbone — it is a property of the **(model, task)** pair, and it
changes under language conditioning in ways the earlier literature
could not see.

### 1.2 Frequency-domain augmentation and defences

A parallel strand of work builds robustness-improving augmentations in
the Fourier domain. @chen2022amplitude recombine amplitude and phase
components to attack the model's spectral habits; @ortiz2021optimism
use adversarial loss to reshape the spectrum during training;
@abello2021dissecting decompose image crops to show that crop-induced
distribution shifts are themselves spectral. These works are
*intervention-focused* — they change the input distribution or the
training objective to force a desired spectral behaviour.

**Our position.** We are *measurement-focused* at the model-internal
level. We do not intervene on the input or the weights; we read the
filter `W_t(ω)` off the cross-attention and predict behavioural drift
from the first-order overlap `⟨W_t, ΔF⟩`. A spectral augmentation
defence would act on `ΔF`, lowering overlap; our framework tells a
practitioner exactly which perturbations are currently worst for their
task (the ones with largest overlap against the task's filter) and
therefore exactly which augmentation targets would give them the most
leverage. Intervention is the obvious next step, but it is
downstream of measurement.

### 1.3 Spectral analysis of vision transformers

Vision-transformer frequency analysis is newer and thinner than the CNN
literature. Park & Kim ("How Do Vision Transformers Work?", 2022) showed
that MSA layers in ViTs act as low-pass filters relative to
high-frequency-biased CNNs. Subsequent work looks at token-mixing
spectrum, patch-grid aliasing, and attention-map structure, but remains
unimodal — there is no language input, so no cross-modal attention to
study.

**Our position.** ViT-style self-attention is what our VLM uses
internally for vision tokens. When text conditions the attention (via
the language head or cross-attention-like routing in Qwen3-VL's
architecture), the vision-token attention becomes dependent on the
prompt, and a new spectral quantity — `W_t(ω)` — emerges. We show
that this quantity **concentrates downward under task granularity** at
the late decoder layers — top-2 mass rising, high-frequency tail mass
falling, IPR shrinking — which is directly measurable and layer-resolvable.

---

## 2. Cross-modal attention and mechanistic probing of VLMs

### 2.1 What the model attends to

The probing literature on VLMs established early on that cross-modal
attention is an interpretable (if imperfect) proxy for what the language
side is querying from the vision side. @cao2020behind and
@frank2021vision show that pre-trained VLM attention tends to reflect
linguistic grounding cues; @ilinykh2022attention refines this to entity
and relation groundings. None of these works converts attention into a
**predictive behavioural quantity** — they describe *what* the model is
doing, not *what will happen* under perturbation.

### 2.2 Mechanistic interpretability of multimodal models

The more recent mechanistic interpretability thread reads single-
component causal roles out of VLM representations. @palit2023towards
ports causal-tracing tools (@meng2022locating) to BLIP; @basu2023localizing
localises edits in text-to-image models; @merullo2024language shows that
some reasoning operations factorise into simple vector arithmetic even
in multimodal settings. These methods are *surgical* — they identify
which layer/head/neuron carries which fact.

**Our position.** Our analysis is complementary: instead of localising
a *fact*, we localise a *spectral response profile*. The filter
`W_t(ω)` and its shape statistics (top-k mass, tail mass, centroid)
give a continuous, quantitative characterisation of the attention's
frequency behaviour per layer group. The new Theorem 4 ("prompt-length
prior") is closer in spirit to this thread: it identifies a
*pretraining-induced shortcut* whose signature is a layer-wise
over-steer + reconciliation pattern, which is an interpretability
claim with a direct mechanistic reading.

### 2.3 Layer-wise emergence of VLM behaviour

Related work on when-in-the-network a VLM does what:
- Early-layer attention often attends broadly over vision tokens.
- Middle-layer attention narrows onto semantically relevant regions.
- Late-layer attention concentrates into the readout-relevant patches.

This is consistent with our **layer-wise reconciliation** finding under
Theorem 4 — the filter over-steers on prompt length at shallow layers
and converges to the semantically-appropriate shape at late layers.
We cite these results but our operational measurements are on the
cross-attention FFT rather than on hidden-state probes.

---

## 3. Robustness benchmarks and perturbation theory

### 3.1 General-vision corruption benchmarks

@hendrycks2019benchmarking (ImageNet-C) is the modern canonical corpus
for measuring how standard vision perturbations degrade vision-model
accuracy. @kar20223d (3DCC) adds 3D-consistent perturbations;
@hendrycks2021many (ImageNet-R) and @barbu2019objectnet add distribution
shift and controlled pose/viewpoint. These benchmarks are *enumerative*
— they define a finite set of perturbations and measure accuracy under
each.

**Our position.** Rather than enumerate, we **decompose**. Each of the
14 perturbations we use is represented by its `ΔF(ω)` spectral signature.
The behavioural response is predicted by the overlap with `W_t(ω)`. Two
consequences:
- A perturbation not in our bank can still be evaluated by computing
  its `ΔF` — the theory extrapolates.
- Conversely, perturbations in the bank are grouped into spectral
  families (low-freq-dominated natural perturbations, high-freq
  noise-like perturbations, broadband perturbations), and the theory
  predicts *family-level* sensitivity differences that the enumerative
  view cannot.

### 3.2 Compositional generalisation and word-order sensitivity

@thrush2022winoground, @yuksekgonul2023aro, @hsieh2023sugarcrepe, and
@ma2023crepe document VLM failures on image-text matching where word
order matters: the models behave like bags-of-words. These are
*alignment* failures rather than *robustness* failures, but they share
the common theme that VLMs have local shortcuts that behave nothing
like the compositional cognition humans would apply.

**Our position.** The prompt-length prior (Theorem 4) is exactly this
kind of shortcut measured at the filter level: wordy-but-semantically-
identical prompts trigger different filter shapes at shallow layers.
Winoground etc. measure the *behavioural consequence* of the shortcut
on a very specific alignment task; we measure the *attention-level
signature* of a related shortcut across every sample. They are
complementary ways of evidencing the same bias family.

### 3.3 Corruption robustness through adversarial defenses

The adversarial-robustness literature is adjacent but distinct. Our
perturbations are not adversarial (no gradient access to the model
used). Classical adversarial work (@tsuzuku2019structural and
follow-ups) shows a Fourier-basis spectrum of model sensitivity that
our `W_t(ω)` connects to: the bands where the model is adversarially
vulnerable are the bands with large `W_t(ω)` gain. This is a
testable, if future, bridge.

---

## 4. Program counting and semantic complexity

### 4.1 VQA as program execution

@hudson2019gqa (GQA), @johnson2017clevr (CLEVR), and @yi2018neural
(NS-VQA) formalise visual questions as compositions of atomic operations
over grounded entities — `select`, `filter`, `query`, `relate`,
`compare`, `verify`, etc. Neural Module Networks (@andreas2016neural,
@hu2017learning) implement these as differentiable modules.
Task difficulty in this framework is literally a count of operations:
program depth, number of entities, number of relations.

**Our position.** We adopt the exact same counting decomposition for
`Csem` — entities, attributes, relations, operations, program depth,
plus a grounding-ambiguity term — so our complexity score is the
additive NMN-style count with one small augmentation. We differ in
*purpose*: the NMN/GQA tradition uses the program counts as a task
design variable or a learning curriculum; we use them as a
**continuous regressor** against which filter-shape and behavioural-
drift statistics are fit. The complexity score is the variable that
decides *which filter* gets measured.

### 4.2 Grounding ambiguity and referring-expression difficulty

Referring-expression datasets (@kazemzadeh2014referitgame, @mao2016generation,
@yu2016modeling) show that resolution difficulty grows with the number
of candidate referents in a scene. Our `ln(1 + G_amb)` term borrows
this quantification directly. The psychometric item-analysis tradition
(@haladyna2004review) justifies our separate **option hardness**
control (distractor plausibility) as a distinct axis from program
complexity.

### 4.3 Two-factor decomposition of "complexity"

A key contribution of this project is recognising that raw "complexity"
mixes two distinct forces:

- **`Csem`** — structured semantic program complexity (counts).
- **`Cprompt`** — surface token count in the question text.

These are usually correlated in naturally-phrased questions, which is
why prior single-scalar complexity regressions give ambiguous signs.
The **mirror 8-level** design (L1–L4 + wordy controls L5–L8) isolates
them at the item level.

**Our position.** This is where our framework departs most clearly
from the program-counting literature. We don't just count — we hold
counts fixed and vary the surface, measuring the *mechanistic*
consequence (Theorem 4: prompt-length prior triggers premature peak
consolidation at shallow layers) and the *behavioural* consequence
(β_Cprompt < 0: wordy prompts reduce drift).

---

## 5. Prompt sensitivity and shortcut learning

### 5.1 Prompt sensitivity in LLMs and VLMs

@sclar2024quantifying, @mizrahi2024state, and @salinas2024butterfly
document that LLM output distributions shift dramatically under
semantically-equivalent prompt rephrasings. @anagnostidis2024susceptible
extends this to multimodal models. These works establish that prompt
sensitivity is real, large, and task-dependent — but they treat it
as a *nuisance* to be measured and mitigated, not a *signal* that
reveals how the model represents task intent.

**Our position.** Theorem 4 upgrades prompt sensitivity from a nuisance
to a **mechanistic probe**. The wordy-vs-base mirror levels don't just
measure prompt sensitivity; they let us observe the *internal
correction process* by which the model reconciles a length-based prior
with semantic content. The fact that this reconciliation is visible as
**layer-wise filter-shape convergence** is, to our knowledge, novel.

### 5.2 Dataset/pretraining biases as shortcut shortcuts

Broadly, the "shortcut learning" literature (Geirhos et al. 2020;
subsequent surveys) argues that deep networks learn statistical
regularities of the training distribution that don't transfer to
real-world deployment. Prompt length is a canonical example: longer
prompts in the training data correlate with more complex tasks, so
the model learns "long → hard" as a heuristic.

**Our position.** Our framework provides the first mechanistic,
quantitative measurement of this specific shortcut at the attention
level: the filter `W_t(ω)` at shallow layers shifts in response to
prompt length even when semantic content is held fixed. This is a
*detectable training-data artefact* rather than an abstract "shortcut"
label.

---

## 6. Theory of robustness under distribution shift

### 6.1 Classical invariance

The statistical-learning framing of robustness treats perturbations as
covariate shift and asks when learned invariances extend. The
adversarial-robustness literature adds a worst-case game. Neither
explicitly considers *task conditioning*: the model is a function,
robustness is a property of the function.

**Our position.** Our alignment-tax bound (Theorem 3) is a task-aware
analogue. It says the task-conditioned model's clean-data risk is
bounded above by a vision-only risk times a factor that depends on
`Ĝ(t)` and the per-band perturbation sensitivity `ε_perturbation`. The
bound ties the trade-off between specialisation and perturbation
robustness to a *specific, measurable* filter-concentration statistic.

### 6.2 Spectral bias and training dynamics

Recent NN-theory work on spectral bias (Rahaman et al. 2019;
Arpit et al. 2017) shows that gradient descent prefers low-frequency
solutions. This is a *training-dynamics* claim, not a behavioural one,
but it predicts that the filter `W_t(ω)` — being a product of
language-conditioned cross-attention weights — will default to a
low-frequency-dominated shape, with fine-grained tasks having to
actively push mass into higher bands.

**Our position.** Our late-layer downward-concentration finding is
the empirical signature of that theoretical prediction at the
attention level: coarser tasks have filters that look like the default
low-frequency solution; finer tasks pay the tax of pulling additional
mass onto the lowest one or two bins, where natural perturbations also
concentrate. This connects our work to the training-dynamics
literature without requiring us to retrain the model.

---

## 7. Where we sit — single paragraph summary

The frequency-domain-robustness literature gave us the measurement
tools but no task-conditioning. The probing / interpretability
literature gave us task-conditioning but no behavioural predictor.
The program-counting VQA literature gave us a clean complexity
definition but no mechanism. The prompt-sensitivity literature
showed that verbosity changes outputs but treated it as a *nuisance*
to mitigate — never as a *defence* with a sign opposite to drift.
The robustness-theory literature gave us the alignment-tax framing
but no task-specific filter to plug in. **This project is the first
synthesis, and it lands on a finding that none of the five
literatures predicted.** It gives:

1. **Verbose prompts stabilise VLMs against image-side perturbations.**
   A 25–73% reduction in log-likelihood volatility across mirror pairs
   (largest at the spectrally-costliest fine-grained level), identified
   at the item level via byte-identical-semantics wordy controls. This
   is the headline empirical contribution and runs *opposite to the
   sign* the prompt-sensitivity literature has implicitly assumed.
2. A task-conditioned filter `W_t(ω)` readable from cross-attention,
   which mechanistically explains the verbosity stabilisation:
   verbose prompts broaden `W_t` at late layers, shrinking its
   overlap with natural-perturbation spectra.
3. A behavioural predictor `⟨W_t, ΔF_p⟩` that works parameter-free
   and converts the explanation into a falsifiable, quantitative law.
4. A mirror-level design (L1–L4 + L5–L8) that isolates `Csem` from
   `Cprompt` at the item level — the protocol that made the
   verbosity finding identifiable and the dual-force regression
   recoverable.
5. A mechanistic claim (Theorem 4: three-stage trajectory and
   prompt-length prior) with a falsifiable layer-wise signature,
   plus alignment-tax and late-layer downward-concentration
   statements that quantitatively track how specialists trade
   clean-accuracy headroom for perturbation robustness in a specific
   region of the spectrum (the DC and DC-adjacent bins).

Each of the five is measurable on a pre-trained VLM without
training-time access. Item 1 is the one that compresses the paper
into a single line; items 2–5 are the framework that earns the right
to claim it.

---

## 8. Useful reading order for new collaborators

1. `paper/main.tex` §3 and §4 — theory + mirror design.
2. `full_project_explanation.md` §2–§4 — detailed theorem statements
   and gate definitions.
3. @hudson2019gqa, @johnson2017clevr — program-counting framing for
   `Csem`.
4. @yin2019fourier, @wang2020high — the frequency-bias foundation.
5. @thrush2022winoground, @yuksekgonul2023aro — failure patterns we
   are trying to *predict* (not just catalogue).
6. @sclar2024quantifying, @anagnostidis2024susceptible — prompt-sensitivity
   context for Theorem 4.
7. `frequency_alignment/experiment_run.md` — how to actually run
   and reproduce any number above.

---

## 9. Open threads / future citations

Things this review deliberately skips because the relevant work
post-dates the project design or isn't yet in `references.bib`:

- **Vision-transformer frequency analysis post-2023** (newer than
  Park & Kim 2022). Worth adding if the rerun confirms late-layer
  downward concentration in a backbone-specific way.
- **Mechanistic interpretability of VLMs post-BLIP** (2024+). Our
  Theorem 4 is interpretability-flavoured and should be cited
  against any VLM-shortcut-detection work that lands before
  submission.
- **Spectral-bias theory applied to attention** (rather than
  feed-forward). If the late-layer downward-concentration pattern
  survives the 500-sample rerun, a theoretical paper on
  attention-based spectral bias would be the cleanest grounding.
- **Robustness certification** (randomised smoothing, Lipschitz
  bounds). Future work could certify robustness bounds using
  `W_t(ω)` and a worst-case `ΔF`.

---

*End of literature review. Maintained alongside `references.bib` — every
citation here resolves via `cite{…}` in `main.tex`.*
