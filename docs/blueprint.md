# Research and proof-of-concept plan: optimizer-aware audited low-rank backpropagation

**Research date:** 14 October 2026  
**Target execution environment:** one MacBook Pro with an M3 Max; actual memory, operating system and storage must be discovered.  
**Deliverable type:** standalone research and execution handoff, not a report of a trained new method.  
**Working method name:** OAR — **optimizer-aware residual auditing**. This is a descriptive project label, not a claim of originality.

## 1. Executive assessment

### Recommendation

Investigate a sharply defined refinement of the brainstorm's self-verifying backpropagation idea: **compute selected linear-layer weight gradients in a reused low-rank subspace, preserve exact input-gradient propagation, and decide whether to use the approximation by auditing the actual optimizer displacement.** Do not initially combine gradient prediction, approximate incoming adjoints, sample selection and a new architecture.

The most useful refinement is geometric. An orthogonal-projection gradient approximation has error entirely outside its retained subspace. Consequently, only the part of the optimizer's proposed displacement **outside that subspace** can interact with the missing gradient. Audit that interaction, rather than requiring a uniformly accurate reconstruction of every gradient entry. The core acceptance bound is

\[
\langle g,d\rangle\ \geq\ \langle\widehat g,d\rangle-
\sum_{\ell\in\mathcal A} b_\ell\left\|(I-U_\ell U_\ell^\top)d_\ell\right\|_F,
\]

where \(d=\theta-\theta_{\rm proposed}\) is the **actual** AdamW displacement, \(\mathcal A\) contains approximated weight matrices, and independent matrix-free probes provide residual-norm bounds \(b_\ell\). All quantities, implementation conventions and qualifications appear below.

This is a plausible research contribution **only if** it improves the measured quality–cost frontier over strong fixed-rank, relative-error, sampling and dense-training alternatives. The algebra is elementary projection geometry plus inexact optimization; randomized residual estimation and adaptive compression already exist. The present search does **not** establish methodological novelty. A valuable outcome could instead be a new empirical result about when optimizer-aware auditing helps, a reproducible negative result, or a well-characterized implementation improvement.

### What research changed

The broad brainstorm is substantially anticipated. INSTANT, low-rank backpropagation, randomized linear-layer gradients, VCAS, synthetic gradients, gradient-subspace reuse and adaptive gradient compression cover much of its mechanism space [R1], [R2], [R3], [R4], [R5], [R6], [R7], [R8], [R9], [R10], [R11], [R12]. Particularly close numerical-linear-algebra work now includes an October 2026 preprint on the information cost of low-rank certification and independent-family probe reuse [R13]. Therefore, neither “verify compressed gradients” nor “adapt rank with probes” is an adequate novelty claim.

There is an important systems warning. INSTANT's reported large backward arithmetic reductions did not give a training-time improvement in its V100 EfficientFormer/CIFAR-10 timing example: the displayed full-fine-tuning total is 110.8 seconds for vanilla and 116.6–120.2 seconds for the listed INSTANT variants. A separate CPU example does benefit. These are different hardware/workload conditions, not contradictory findings [R1] (appendix timing tables and figures). A Mac result must be measured, not inferred from either FLOPs or those timings.

### Scope and ambition

The core claim concerns **small-model training from scratch**, first in isolated linear/residual networks and then a decoder-only language model. This retains the brainstorm's difficult representation-learning objective rather than silently substituting easy adapter fine-tuning. A vision workload supplies a second mechanism test; it is not a claim to vision state of the art. Fine-tuning is an evidence-triggered separate branch, not a source of unearned pretraining claims.

The initial method avoids only weight-gradient work. In the illustrative dense-layer model with one unit each for forward, input-gradient and weight-gradient multiplication, eliminating all weight-gradient work gives at most **1.5× step throughput**, before overhead. Eliminating all backward work gives at most 3×. Making all backward work 100× faster gives \(3/(1+2/100)=2.941176\times\), correcting the brainstorm's approximate 2.96×. These are calculations under a simplified model, not hardware measurements.

A 10×–100× **time-to-quality** result remains an ambition requiring additional learning-efficiency, forward-computation or architectural changes. This PoC cannot support that claim merely by succeeding. Its purpose is to identify a trustworthy, useful component and discover whether the information redundancy needed for larger advances actually exists.

### Findings established during this assignment

Source inspection, several implementation-file reads and CPU numerical checks were performed. No M3 Max experiment, model-training comparison, measured training speedup or patentability review was performed. The checks verified the residual identities, an orthogonal-complement refinement, accumulation behavior and an unbiased-correction variance formula. They also constructed an Adam example in which gradient cosine similarity exceeds 0.999 but the actual proposed update is uphill. Section 5 records the environment and numerical results; Appendix A supplies reproducible reference code.

**First decisive experiment:** on actual development checkpoints, compare an oracle version of the proposed acceptance margin against a norm-only margin and a fixed-rank policy, while separately measuring the cheapest feasible implementation cost. Stop building an online controller if even the oracle adds no useful discrimination or no possible implementation has measured headroom.

## 2. Problem, scope and causal hypotheses

### 2.1 Problem statement

Beneficiaries are researchers and practitioners who train neural networks under limited accelerator time. The workflow being improved is forward evaluation, loss computation, reverse-mode differentiation, optimizer transformation and parameter update. The proposed bottleneck is repeated dense weight-gradient matrix multiplication; whether it dominates enough of an M3 Max training step is an **unknown**, to be measured before extensive infrastructure.

For a batch objective \(L_B(\theta)\), ordinary training obtains \(g=\nabla L_B(\theta)\) and applies an optimizer-dependent displacement. The project asks:

> Can a cheaper, partly projected gradient produce a useful optimizer update, and can inexpensive evidence identify the unsafe or unhelpful cases without computing the full missing gradient?

“Exact gradient” below means the gradient of the specified finite-precision forward computation evaluated by the matched dense backward implementation; mathematical identities are separately stated over real arithmetic. “Audit” means a measured diagnostic with an explicit statistical/numerical model. It does **not** mean formal verification of GPU machine code. “Learning improvement” means improved held-out quality under charged end-to-end time, or less charged time to a predeclared quality target, with regressions disclosed.

The loss gradient is a minibatch quantity, not the population gradient. A certificate about the current minibatch says nothing by itself about a rare subgroup or future generalization. Parameter gradients, gradients with respect to layer inputs, optimizer updates, communication compression and activation storage are distinct objects throughout this plan.

### 2.2 Established ingredients versus assumptions

| Category | Statement | Consequence |
|---|---|---|
| Established algebra | For a linear layer, \(G=D^\top X\) and \(dX=DW\). | One can replace the weight-gradient branch without changing the input-gradient branch. |
| Established methods | Randomized matrix products, subspace methods, residual probing and adaptive sampling have extensive prior art [R1], [R2], [R3], [R4], [R5], [R6], [R7], [R8], [R9], [R10], [R11], [R12], [R13]. | No novelty credit for those ingredients alone. |
| Derived here | Projection error interacts only with the optimizer displacement's orthogonal complement. | A potentially less conservative audit; test its extra arithmetic cost. |
| Assumption | A past-only subspace captures enough useful current gradient structure. | Test across early, middle, late and shifted training, not just adjacent batches. |
| Assumption | Residual audits are cheaper than avoided dense GEMMs on this machine. | Kernel and complete-step break-even gates precede long runs. |
| Assumption | The margin predicts consequential optimization harm beyond simpler diagnostics. | Frozen-checkpoint counterfactual tests precede learning claims. |
| Assumption | Helpful approximation does not destroy representation discovery. | Rare-pattern, depth and distribution-shift guardrails are mandatory. |
| Speculation | Gradient dictionaries or a learning-specific architecture could remove most backward work. | Separate gated branches, not ingredients needed to make this PoC look good. |

### 2.3 Central hypotheses and discriminating observations

**H1 — Reusable projected information.** A basis constructed solely from previous exact gradients allows a rank regime with useful present-batch updates and lower measured cost. Supporting context comes from low-rank training and subspace-reuse studies, not a general theorem [R1], [R3], [R6], [R8], [R11]. Alternative explanations include small-batch rank ceilings, easy late-stage adaptation, repeated data, or retaining high-energy noise rather than learning signal. H1 weakens if a future-held-out oracle needs nearly full rank, frequent refresh, or fails on early training and novel patterns. A failure at a dimension where every candidate GEMM is launch-bound is a hardware-regime result, not a universal structural falsification.

**H2 — Optimizer-aware geometry matters.** At matched proposer and cost, the complement-aware margin distinguishes damaging approximate AdamW updates better than gradient energy, cosine, a relative-Frobenius test and the unrefined Cauchy bound. The CPU counterexample demonstrates logical need, not prevalence. H2 fails locally if these events are negligible, the margin rejects harmless updates indiscriminately, or a tuned constant schedule predicts equally well. Under plain projected SGD without momentum/decay, descent is already automatic; the auditor must not be credited for it.

**H3 — Auditing pays for itself.** Savings after reconstruction, optimizer preview, complement projection, probe generation, synchronization, refresh and replay exceed the entire audit overhead. H3 is a wall-clock claim. FLOP reductions, smaller backward graphs or excluded calibration time do not establish it.

**H4 — Learning value survives.** A frozen policy improves held-out loss at a fixed charged time against the strongest executable comparison, with acceptable subgroup and stability outcomes. Faster steps but worse time-to-quality falsify the practical claim, even when H1–H3 hold individually.

**H5 — The audit is calibrated within its declared model.** Exact-gradient spot checks agree with the matrix-free statistic; independent-probe confidence behavior is consistent with the proved real-arithmetic inequality; floating-point discrepancies are separately bounded or explicitly treated as empirical guardbands. An unvalidated numerical guardband cannot support a formal run-wide probability claim.

### 2.4 Strongest positive and negative cases

The strongest positive case is a wide, sufficiently batched layer whose incoming adjoint is available anyway; its useful updates change slowly; AdamW introduces only modest off-subspace displacement; and native GEMMs make small-rank products efficient. Here, a substantial residual can be tolerated because it barely affects the chosen step. The method need not reconstruct the full gradient accurately.

The strongest negative case is equally credible: native dense GEMMs are already excellent, the approximation creates several small GEMMs and memory passes, AdamW remains dense, and the worst-case residual bound is conservative. A failed audit then adds an entire replay. Even a valid descent margin may reward noisy short-term batch progress while suppressing representation discovery. Stronger dense optimization, smaller batches or a fixed projection may obtain the same improvement more simply. These possibilities define the stage gates, not excuses to dismiss unfavorable results.

## 3. Research landscape and novelty assessment

### 3.1 Closest prior-art comparison

“Local” means implementable using permitted local operations after the compatibility gates, not already reproduced on an M3 Max. Published hardware numbers are contextual, never direct head-to-head results.

| Prior art | Problem and mechanism | Assumptions/evidence and important limitation | Implementation and computational overlap | Implication for OAR |
|---|---|---|---|---|
| INSTANT, ICLR 2026 [R1] | Compresses activations and backward signals with calibrated bases. | Fine-tuning evidence; arithmetic and wall time differ materially by hardware. | Public MIT code; inspected language operator has explicit CUDA transfers and compressed input gradients. | Closest complete low-rank BP baseline. OAR keeps input adjoints exact and tests actual update risk. |
| LBP-WHT [R3] | Low-rank backward approximation using structured transforms for vision adaptation. | Spatial/token structure and adaptation setting matter. | Specialized implementation; published GPU behavior is not a Mac result. | Compression itself is known; useful contextual/vision comparator, not mandatory transformer-scratch replication. |
| Randomized linear-layer BP [R4] | Sketches stored activations and estimates weight gradients. | Unbiasedness and variance studied; memory-saving motivation. | Matrix-product sketching with exact input-gradient branch. | Exact-adjoint, approximate-weight-gradient separation is not new. |
| VCAS, ICLR 2024 [R2] | Adapts unbiased data/token sampling to control added gradient variance. | Vision and language experiments; principal savings reported as FLOPs. | Apache-2.0 code; gather/scatter and variance-calibration costs. | Strong adaptive-compute alternative; comparison must include sampling, not only rank projection. |
| Unbiased approximate VJPs, 2026 [R5] | Randomized VJP estimates with optimality/variance analyses. | MLP, BagNet and vision-transformer experiments; preprint evidence. | Paper specifies methods; inspected version promises later code release. | “Unbiased cheap backprop” and optimal rank/sparsity allocation are not new claims. |
| Adaptive activation sketching/monitoring, 2025 [R6] | Maintains activation sketches and changes rank. | Classification/PINN and gradient-monitoring examples; monitoring/storage is not equivalent to faster training. | Adaptive sketches, reconstruction and exact error-signal handling. | Generic adaptive sketch monitoring is anticipated. |
| DropBP, NeurIPS 2024 [R7] | Omits residual-branch backward work. | Reported fine-tuning acceleration on non-Apple accelerators. | MIT code; inspected layer uses actual `detach()`, not a dense zero mask. | Simple complete-backward competitor; does not furnish an exact-adjoint audit. |
| GaLore / Fira [R8], [R25] | Low-rank optimizer states; Fira also handles residual gradient information. | LLM training under state-memory constraints. | Standard implementations obtain full gradients before optimizer compression. | Optimizer geometry and residual handling are known; memory savings are not avoided-gradient savings. |
| PowerSGD / L-GreCo [R9], [R10] | Low-rank communication compression and adaptive layerwise compression budgets. | Distributed training and network communication costs. | Usually operate after dense gradients exist. | Reused bases and error-budget allocation are not original; distributed speedups are incomparable locally. |
| Synthetic gradients / update recycling [R11], [R12] | Predict downstream gradients or reuse update subspaces. | Different tasks, dynamics and information costs. | Predictors, dictionaries and occasional exact supervision. | Ideas B/C require a specific new representation or empirical result, not the general concept. |
| Adaptive-sampling inner-product tests [R14] | Adjust gradient accuracy to obtain useful descent directions. | Stochastic optimization analyses with stated assumptions. | Gradient sampling and norm/inner-product diagnostics. | Auditing descent rather than full reconstruction has historical precedent. |
| Low-rank certification limits, October 2026 [R13] | Studies residual-query cost and validation reuse over independent approximation families. | Recent preprint; not an end-to-end neural-training result. | Gaussian residual queries and nested-path stopping. | Probe-certified adaptive rank is especially weak as a novelty claim. |
| NoProp, CoLLAs 2025 [R15] | Changes the model/training organization to block-local denoising. | Image classification; local backward differentiation still used. | Different architecture and inference procedure. | Relevant to D, not a drop-in replacement baseline for fixed-transformer training. |
| Checkpointed BP comparison, ICML 2026 [R16] | Compares BP, checkpointing, forward-mode and zeroth-order methods. | Workload-specific negative evidence against easy BP-free wins. | Strong memory-aware conventional baseline. | Always include checkpointed BP when memory is material. |

Additional searches into optimizer residuals found Fira, subset-norm/subspace-momentum optimization and DOME [R25], [R26], [R27]. They reinforce two distinctions: the optimizer is not the gradient, and dominant gradient energy need not be the most useful information. None of those observations is claimed as this project's invention.

### 3.2 What can be reused

Use native PyTorch differentiation and optimizers for dense references; public INSTANT, VCAS and DropBP algebra/code as baseline starting points; established QR/range-finding primitives; standard datasets; and ordinary experiment logging. Do not build an autodiff engine, a custom compiler, a new tokenizer library, or a learned gradient predictor before evidence requires one.

Implementation inspection found concrete portability risks:

- INSTANT `language/custom_op.py`, blob `91407c502dbfd5e0c8f04083017897bd8e3607c9`, compresses the sequence dimension, reconstructs the input gradient and explicitly calls `.cuda()` on compression tensors. Port device handling and verify the complete autograd signature and numerical behavior; do not merely replace a device string and declare reproduction [R28]. The calibration callback `language/callback.py`, blob `6dd9a22873b4092f6d53b841e0e02b347bcdcaf4`, was also read: it disables compression, performs extra forward/backward calibration passes without optimizer updates, accumulates sequence Gram matrices, and includes CUDA-specific cache/device calls. `language/SVD.py`, blob `7a48f44279b213a726ddddc934dce72ba458cf45`, scales retained vectors by an energy fraction and has boundary/indexing choices that differ from a plain orthogonal projector. Preserve and test the published implementation as one labeled variant; do not silently replace it with OAR's orthonormal basis.
- VCAS `vcas/layers/linear.py`, blob `8168b29b45f05f49086a9c15835b8fa7865126eb`, uses CUDA AMP decorators, gathers nonzero sequence rows, scatters input gradients and invokes token-sampling utilities. A faithful port must preserve these operations and the calibration controller [R29].
- DropBP `dropbp/layer.py`, blob `9d84e9c06b27ba7330702d4a8b8f4a490d45ef55`, uses Python's random generator and returns either the original tensor or a detached tensor. Both its RNG and execution path must be captured in reproduction [R30].

These are inspected file-blob identifiers, not asserted repository commit pins. At execution, archive the entire upstream revision and local patch. A failed direct container network fetch was not a code inspection; the subsequent successful GitHub connector reads were.

### 3.3 The exact contribution under investigation

The candidate contribution is **an online compute-allocation rule using the interaction between projected-gradient residuals and the actual optimizer's off-subspace displacement, with independent residual auditing, exact incoming adjoints, and fully charged transactional fallback.** It could reduce unnecessary accuracy requirements when an optimizer mostly stays within the retained subspace, while detecting cases where preconditioning/momentum moves into unobserved directions.

The discriminating experiment holds the gradient proposer, optimizer, refresh information and implementation quality fixed, and compares:

1. no audit with a tuned fixed rank or predetermined schedule;
2. relative-Frobenius and ordinary norm-based update margins;
3. the complement-aware margin;
4. schedules/fallback frequencies matched without access to current audit information.

An improvement that disappears under these controls is not evidence for the proposed contribution. A source describing the same residual–optimizer-complement criterion would invalidate methodological novelty; the project could still proceed as replication or an M3 empirical study, with that claim narrowed before confirmation. An independently proved but already-known identity is not a new method. A new name, a port or a combination alone is not a breakthrough.

### 3.4 Corrections and boundaries of the brainstorm

The randomized residual identity is correct with isotropic independent probes; “a few probes” does not imply tight reliable certification. For exact adjoints at the same parameters, the sum of positive layerwise directional inner products is a positive full-parameter first-order inner product. Simultaneous layers do not defeat this algebra; finite step size and curvature remain separate problems. Proxy-correction unbiasedness is also correct under the stated sampling conditions, but not an assurance of unbiased Adam updates or low variance.

The architecture-level option is retained rather than rejected for being unfamiliar. It changes the object of comparison and therefore requires a separate representational-capacity and learning-dynamics study. It must not be used to inflate a component result into a new learning paradigm.

## 4. Selected mechanism: full mathematical and implementation specification

### 4.1 Tensor conventions and invariants

For every eligible linear layer, flatten all batch/token axes into \(n\):

\[
X\in\mathbb R^{n\times d_i},\quad W\in\mathbb R^{d_o\times d_i},\quad
Y=XW^\top+\mathbf1b^\top,\quad D=\partial L_B/\partial Y.
\]

Then \(G=D^\top X\), input gradient \(DW\), and bias gradient \(\sum_aD_{a,:}\). Loss normalization is already included in \(D\); do **not** divide the weight gradient by tokens again.

The original forward, attention, nonlinearities, normalizations, embeddings and tied output weights stay unchanged. Initially approximate only the two untied MLP matrices in transformer blocks. Add attention projections only through an exploratory amendment after the MLP experiment. Keep biases, normalization parameters and shared/tied parameters exact. Do not approximate parameters used by multiple modules until their contributions are correctly aggregated under one shared projector and independently tested.

**Invariants:** no parameter changes during backward; exact input gradients in every primary-method linear operation; a fixed basis/rank throughout an effective optimizer batch; no audit randomness in candidate construction; no additional optimizer/scheduler/data-cursor step on rejection; identical batch and model-RNG realization on replay; only detached audit statistics retained beyond backward; one logical update per step.

### 4.2 Past-only basis construction

Maintain \(U_\ell\in\mathbb R^{d_o\times r_{\max}}\), approximately orthonormal, with ordered columns. Start with a dense update and build the first basis from that update's exact gradient for use **on the next update**. At scheduled dense refreshes, use a randomized range finder on the current exact gradient:

\[
Q=\operatorname{qr}(G\Omega),\quad C=Q^\top G,\quad
C=\widetilde U\Sigma V^\top,\quad U=Q\widetilde U_{[:,1:r_{\max}]}.
\]

\(\Omega\) is an independent Gaussian construction matrix, with width \(\min(d_i,r_{\max}+p)\). Start the oversampling parameter \(p\) at 8 as an engineering default, test sensitivity and freeze it. If small dimensions or rank deficiency prevent the requested rank, shrink it; never pretend null or duplicate columns are useful modes. No power iteration initially. Count range construction, small SVD, transfers and synchronization. Existing randomized linear algebra is being reused [R17].

An exact SVD is an **offline oracle** for auditing the best possible current rank; it is not a free runtime operation. Compare single-batch bases against a rolling past-only basis only if single-batch noise proves material. Such a change creates a distinct proposer requiring its own matched controls. Do not store hundreds of full parameter gradients merely to fit a dictionary.

### 4.3 Weight-gradient proposal with exact incoming adjoints

Use the first \(r\) columns of \(U\):

\[
H=(DU)^\top X=U^\top G,\qquad \widehat G=UH=UU^\top G.
\]

Compute the input gradient as \(DW\), exactly as in the dense reference. Reconstruct \(\widehat G\) as a dense tensor for the primary AdamW implementation; this explicitly retains dense optimizer states and parameter writes. It is not a memory-saving optimizer.

The leading candidate weight-gradient arithmetic is

\[
2nd_or+2nrd_i+2d_ord_i
\]

versus \(2nd_od_i\) for the exact product. These counts exclude input gradients, audit and optimizer work. For example, with \(n=2048,d_i=d_o=768,r=16\), the candidate weight-gradient multiplication count is about 20.2 times smaller; the **complete step is not 20.2 times faster**. This is an arithmetic example to be replaced by measured kernel timings.

### 4.4 Matrix-free residual audit

Let \(E=G-\widehat G\). Draw \(Z\in\mathbb R^{d_i\times q}\) with independent standard normal entries, independent of the current candidate conditional on training history. Compute

\[
R=D^\top(XZ)-U(HZ)=EZ,\qquad s=\|R\|_F^2/q.
\]

Then \(\mathbb E[s\mid E]=\|E\|_F^2\). There is no need to materialize \(G\). Residual probing is standard randomized numerical linear algebra, not a new identity [R13], [R17], [R18].

For a conservative implementation that does not assume a favorable residual spectrum, choose \(c\in(0,1)\) satisfying

\[
c-1-\log c=\frac{2\log(1/\delta)}q,
\qquad b_{\rm math}=\sqrt{s/c}.
\]

Over real arithmetic,

\[
\Pr\{\|E\|_F>b_{\rm math}\}\leq\delta.
\]

**Proof sufficient for implementation review.** Let \(\mu=\|E\|_F^2>0\), and let \(w_i\) be squared singular values divided by \(\mu\). They are nonnegative and sum to one. For \(t>0\),
\(\mathbb E\exp[-tq s/\mu]=\prod_i(1+2tw_i)^{-q/2}\leq(1+2t)^{-q/2}\).
Exponential Markov and \(t=(1/c-1)/2\) give
\(\Pr(s\leq c\mu)\leq[c\exp(1-c)]^{q/2}\).
The chosen equation makes the right side \(\delta\). If \(E=0\), the real-arithmetic result is exact. This is a conservative derived specialization, not an asserted optimal concentration theorem.

Use bisection in FP64 on the CPU for the scalar root. The following norm multipliers show why confidence is a feasibility issue, not decoration:

| Probes \(q\) | \(1/\sqrt c\), \(\delta=0.01\) | \(1/\sqrt c\), \(\delta=10^{-6}\) |
|---:|---:|---:|
| 8 | 2.7434 | 9.2170 |
| 16 | 1.9196 | 3.7749 |
| 32 | 1.5435 | 2.3122 |

For an indefinite run, assign the \(j\)-th scheduled residual test, with its index allocated before its probes are observed, the failure budget
\(\delta_j=\delta_{\rm run}\,6/(\pi^2j^2)\), persist \(j\) in the checkpoint and a write-ahead audit journal, and never reset it after a failure or resume. Count rejected and zero-residual tests as well. An interrupted test whose probes may have been observed consumes its allocation; a retried candidate gets fresh probes and a new index. A finite frozen maximum number of tests may instead use an equal union-bound allocation and can be less conservative. Set a declared run-wide mathematical risk budget during protocol freeze; 0.01 is a starting reliability choice, **not** an improvement threshold. Do not tune it on protected outcomes or present a per-layer per-step bound as a run-wide guarantee.

Each candidate must be independent of its audit probes. Changing the current rank after observing these same probes is not licensed by the preceding single-candidate proof. OAR initially adapts **the next step only**. Independent-family, nested-path reuse has relevant prior art [R13]; implementing that extension requires its own proof assumptions and counterfactual cost comparison, not an undocumented optimization.

### 4.5 The optimizer-aware orthogonal-complement refinement

Preview the exact proposed optimizer rule using \(\widehat g\), current states and parameters. Include gradient clipping, per-group learning rates, moment updates, bias correction, epsilon and decoupled weight decay. Define

\[
d=\theta-\theta_{\rm candidate}.
\]

This definition avoids ambiguity about optimizer signs or different learning rates. For the projected layer, \(U^\top E=0\), so

\[
\langle E,d_\ell\rangle
=\langle E,d_{\ell,\perp}\rangle,\quad
d_{\ell,\perp}=d_\ell-U(U^\top d_\ell).
\]

Consequently,

\[
a=\langle\widehat g,d\rangle,\quad
B=\sum_{\ell\in\mathcal A}b_\ell\|d_{\ell,\perp}\|_F,\quad
\langle g,d\rangle\geq a-B.
\tag{OAR}
\]

Accept the approximate proposal only if all quantities are finite, \(a>0\), and \(B\leq\tau a\), with \(0<\tau<1\). This gives a first-order minibatch margin of at least \((1-\tau)a\) under the stated bounds. Pilot values \(\tau\in\{0.25,0.5,0.75\}\) are exploration settings, not success criteria; refine only on development data and then freeze one value.

The unrefined comparator substitutes \(\|d_\ell\|_F\) for \(\|d_{\ell,\perp}\|_F\). Since the latter is no larger in exact arithmetic, the complement bound is never algebraically worse, **but computing it costs additional work**. Compute the explicit complement for correctness first. A faster formula \(\|d\|_F^2-\|U^\top d\|_F^2\) risks cancellation near zero; use it only with a validated conservative numerical treatment and measured advantage. Do not silently clamp a negative result and call the resulting smaller bound reliable.

For plain projected SGD with no momentum or decay, \(d=\eta\widehat G\) lies in the retained subspace and \(\langle G,d\rangle=\eta\|\widehat G\|_F^2\). No random audit is needed to establish descent. This is both a useful limiting case and a negative control against attributing automatic projection geometry to OAR.

A positive margin is **not** a finite-step or generalization guarantee. For an \(L_s\)-smooth batch objective,
\(L_B(\theta-d)\leq L_B(\theta)-\langle g,d\rangle+(L_s/2)\|d\|^2\).
The unknown curvature term can dominate. Measure actual finite-step losses in diagnostic counterfactuals; do not add an uncharged line search to the proposed method. Decoupled decay is included in \(d\), but \(g\) remains the data-loss gradient unless the declared objective explicitly includes regularization.

### 4.6 Finite precision and numerical claims

The orthogonality identity and confidence bound above are real-arithmetic statements. In FP32, \(U^\top U\neq I\) exactly, projection products accumulate rounding error, and subtraction in \(R\) can cancel. Therefore the primary implementation is an **empirically audited method with a real-arithmetic confidence model**, not a formally certified training system.

At the operator gate, compare FP32 MPS results to CPU FP64 references across actual shapes, spectra, scales and near-zero residuals. Monitor orthogonality and use reorthogonalization during paid refreshes. Record both an empirical guardband and its validation distribution. If rigorous numerical certification becomes a claim, derive a rounding envelope for the actual multiplication/accumulation implementation, add its contribution to the gradient-error bound and projection-orthogonality defect, and verify the reductions as well. Until then, use a conservative guardband selected on development stress tests, reject margins inside the measured numerical uncertainty interval, and label the probabilistic guarantee conditional on exact arithmetic.

Start with FP32 throughout. BF16/FP16 are later robustness/engineering branches applied symmetrically to baselines. No Gaussian-to-Rademacher substitution under the same theorem. Unsupported precision or decomposition operations must fail visibly rather than invoke a hidden CPU path in timed runs.

### 4.7 Gradient accumulation without a biased audit

For microbatches \(m\), define \(L_B=\sum_m w_mL_m\), where \(L_m\) is a valid-token mean and \(w_m\) is its valid-token count divided by the effective batch's total. Backpropagate \(w_mL_m\), so each \(D_m\) is already correctly scaled. Use the same \(U,r,Z\) for the entire effective batch and accumulate

\[
H=\sum_m(D_mU)^\top X_m,\quad
R=\sum_m\{D_m^\top(X_mZ)-U(H_mZ)\}.
\]

Then \(R=(G-\widehat G)Z\) for the **aggregated** gradient, and perform one residual-norm test per layer after accumulation. Summing independently estimated squared residual norms would generally estimate a different quantity. Store only the running \(H\), dense gradient and small \(R\), not all microbatch activations or adjoints. Shared probes within this fixed effective batch are intentional; probes must be fresh on the next candidate.

### 4.8 Transactional optimizer preview and replay

The minimal implementation uses a functional AdamW preview that returns pending moment tensors and \(d\), without changing live parameters, moments, scheduler, scaler, counters or RNGs. Match native AdamW against both one-step and multistep tests, including a zero-gradient tensor versus an absent gradient, epsilon, bias correction and parameter groups. Select `foreach`/fused native variants only after support and semantics tests; do not presume MPS equivalence [R19], [R20], [R21].

If the audit accepts, commit the pending state and parameter displacement once. If it rejects, discard pending tensors, clear candidate gradients and statistics, restore the original model RNG state, replay the same effective batch with native exact backward, and execute one exact optimizer update. A baseline exact AdamW step can itself have a negative instantaneous data-loss inner product; execute it unchanged and record `exact_fallback_no_descent_claim`, rather than secretly substituting SGD or a line search.

The data iterator supplies a replayable batch descriptor; it is not advanced twice. The scheduler and valid-token training counter advance once. A separate processed-token counter includes replayed examples. Checkpoint only at committed boundaries. Do not retain every incoming adjoint to enable late rank changes: this would invalidate the memory estimate and change the method.

### 4.9 Minimal deterministic next-step controller

Use a measured per-layer rank set: powers of two from 4 upward, plus any hardware-efficient ranks discovered by timing, restricted to ranks below \(\min(d_i,d_o)\) that have possible end-to-end savings. Include `dense` as a distinct state, not a fake full-rank compressed path. Generate \(U\) up to the largest admissible rank at paid refreshes. Parameters of the controller are frozen after the pilot.

At initialization use the cheapest rank whose **development oracle** has a nontrivial useful-update region; otherwise begin dense. At every approximate step record \(c_\ell=b_\ell\|d_{\ell,\perp}\|\), total \(B\), and slack \(B/a\). After a rejected step with \(a>0\), first choose any admissible next-rank increase whose measured cost is no greater than its current rank, preferring the largest \(c_\ell\) and breaking ties by stable layer ID. Otherwise sort approximate layers by \(c_\ell\) divided by the strictly positive measured incremental cost of their next rank, again with stable tie-breaking. Move the first eligible layer to its next rank for the next step; if it has no admissible rank that is faster than the dense path, make that layer dense. Kernel timings need not be monotone in rank; never divide by a nonpositive cost difference. If \(a\leq0\), take one dense next step instead of pretending a rank adjustment fixes optimizer momentum; retain the previous rank configuration for subsequent use and refresh bases only on the frozen schedule.

After \(h\) consecutive accepted approximate steps with \(B/a<\tau/2\), lower the rank of the eligible layer with the smallest recent error contribution per strictly positive measured saving, breaking ties by stable layer ID. Ignore decreases with no measured saving. Change only one layer; clear the success streak. Use \(h\in\{8,32,128\}\) as pilot settings, chosen to probe stability rather than as an arbitrary runtime limit. After a shrink, require a new streak before another shrink. Fixed scheduled dense refreshes rebuild all eligible bases every \(K\) logical updates; choose \(K\) from measured basis-drift lifetimes and amortization, not from the favorable setting of another paper. In the pilot bracket \(K\) geometrically around that measured lifetime.

A rejected replay does **not** automatically refresh every basis in the primary version: that would entangle the controller with extra information. It can supply a basis refresh only when the frozen schedule calls for one. A separately labeled fallback-refresh variant must give the same information to its matched control. Layers made dense can re-enter approximation at the next scheduled refresh. If no layer has a plausible fast rank, finish in dense mode and report no online advantage.

## 5. Checks actually performed before this handoff

### 5.1 Environment and scope

Executed on a Linux x86-64 CPU environment: Python 3.13.5, NumPy 2.3.5, platform `Linux-6.18.44-x86_64-with-glibc2.41`. Main seed 20261014; complement/accumulation seed 20261015. These validate algebra and numerical examples, **not** M3 compatibility, dataset learning or training performance.

| Executed check | Result | Interpretation |
|---|---:|---|
| Matrix-free residual versus materialized \(EZ\), shape \(X=97\times31,D=97\times43,U=43\times8,q=16\) | Relative error \(6.47\times10^{-16}\) | FP64 algebra implementation agrees. |
| Full orthonormal basis recovers dense gradient | Relative error \(8.42\times10^{-16}\) | Limiting case agrees. |
| Low-rank directional contraction | Relative error \(2.89\times10^{-16}\) | Correct orientation/transposes. |
| Gaussian norm estimator, rank-one residual, 50,000 replicates, \(q=16,\delta=0.05\) | Mean estimate 0.99742 for true squared norm 1; bound failure fraction 0.01048 | Consistent with a conservative bound; not a proof at smaller failure probabilities. |
| Same estimator, flat rank-eight / decaying rank-eight | Means 0.99997 / 0.99907; no failures observed | Zero observed failures does not establish zero failure probability. |
| Proxy correction, all 28 size-two subsets of eight examples | Mean error \(1.42\times10^{-16}\); measured and analytic variance both 0.3449893346 | Finite-population formula agrees in this test. |
| Complement identity for a random displacement | Absolute contraction discrepancy \(2.49\times10^{-14}\) | Projection geometry agrees numerically. |
| Complement bound / ordinary bound, a mostly in-subspace constructed displacement | 0.002044 | Synthetic illustration of potential tightness, not a measured training benefit. |
| Seven-microbatch accumulation with common probes | Residual relative error \(5.01\times10^{-16}\) | Aggregated audit identity agrees. |

For a random displacement, the refined bound was still very loose: actual residual alignment about 20.33, complement Cauchy bound about 10,844.6. This unfavorable example is retained. Geometry can help greatly or barely help; H2 must be measured on real optimizer states.

### 5.2 Optimizer and adjoint counterexamples

For a two-coordinate constructed state, take \(g=(1,0.02)\), \(\widehat g=(1,-0.02)\), old first moment zero, old second moment \((10^4,10^{-8})\), \(\beta_1=0.9\), \(\beta_2=0.999\), epsilon \(10^{-8}\), step counter 100, unit learning rate and zero decay. The current moment updates give displacement approximately
\((0.0003087203,-0.9638050377)\). Gradient cosine is 0.9992003 and raw alignment 0.9996, yet \(\langle g,d\rangle=-0.01896738\). Scaling learning rate down preserves this first-order sign. This is an extreme illustrative state, not evidence of its frequency in training; the specific approximate vector is a generic gradient-approximation example, not asserted to be an orthogonal projection of that true vector.

Separately replacing \(D\) by \(-D\) lets a verifier reconstruct its **wrong** local product with negligible residual while reversing its alignment with the true gradient. This justifies exact-adjoint scope. It does not imply approximate-adjoint methods can never work; it means this local verifier cannot certify them as written.

## 6. Evaluation design fixed before implementation optimization

### 6.1 Claims and metrics

**Primary confirmatory claim:** at a predeclared inclusive active wall-time budget \(H\), the selected OAR configuration obtains lower mean held-out negative log-likelihood (NLL, natural-log units per scored token) than (a) the strongest development-selected locally executable baseline and (b) its matched fixed-projection/no-information control. This is a learning-efficiency claim, not simply throughput.

Define `active_wall_s` from model/optimizer construction through training and the prescribed monitoring evaluations, including method-specific initialization, basis construction, preview, calibration, replays, transfers and routine checkpoints. Exclude asset download, one-time shared tokenization and external pauses; report those separately and also publish a cold-start total that includes project-specific preprocessing. On restart, include process reinitialization, checkpoint loading and recompilation in accumulated active time. Never exclude a method-specific expense because it occurs outside `loss.backward()`.

At the budget boundary, use the last **committed, saved evaluation checkpoint** whose full accumulated active cost, including its serialization, did not exceed \(H\). Score that checkpoint once on the protected endpoint set and report this scoring cost separately. Use a common elapsed-time evaluation/checkpoint cadence, determined from pilot learning-curve slopes and the desired NLL resolution and frozen for all methods. Retain the latest two checkpoints; an every-step copy of the whole optimizer state is not required. If a final step or checkpoint crosses \(H\), exclude that later state and report the retained checkpoint time, unused budget and cadence. Endpoint discretization must be smaller than the effect resolution sought; if it is not, refine the cadence during development or classify the fixed-time comparison as insufficiently resolved. Do not change the boundary rule after seeing confirmatory outcomes.

**Secondary evidence:** time to fixed NLL targets; valid unique tokens/s; processed tokens/s including replay; optimizer steps/s; total forward/backward/reconstruction/audit/preview/replay/refresh time; peak tensor and driver memory; rank profiles and dense fractions; estimator failures; actual versus predicted batch-loss change; validation and rare-slice performance at matched token counts; variance across training seeds. FLOPs are explanatory only. No “GPU-hours” comparison across different accelerator families without a separate cost model.

For time-to-quality, choose attainable targets from development baseline trajectories before confirmation. Record first target crossing as an interval between successive evaluation times; require confirmation at the next scheduled evaluation, using the time of that confirmation as the conservative crossing time. Do not interpolate a favorable crossing through noisy evaluations. If a run never crosses, report censoring and target-attainment counts. A restricted mean time \(\mathbb E[\min(T,H)]\) is allowed with an explicit horizon, but a ratio of clipped times is not an uncensored speedup claim. Training-time speedup claims require comparable attained quality and guardrails, not loss extrapolation.

### 6.2 Direct baselines and diagnostic controls

| ID | Implementation/configuration | Role and fairness requirement |
|---|---|---|
| B0 | Native dense AdamW, no unnecessary wrappers | Core semantic reference and competitive baseline. Tune learning rate, decay, schedule and effective batch. |
| B1 | Dense AdamW with activation checkpointing and hardware-efficient batch/accumulation | Required memory–time frontier when activation memory constrains B0; do not force checkpointing when it slows an already fitting baseline. |
| B2 | Native Muon for eligible hidden 2D matrices plus AdamW for embeddings, output/shared weights, biases and norms | Contemporary dense challenger if the installed release/backend supports it faithfully. Tune separately; do not copy AdamW's learning rate. Official documentation distinguishes hidden matrices from embeddings/output weights [R21]. |
| B3 | Same projected proposer and dense AdamW, fixed rank/refresh, no probes | Strong simple approximation baseline. All candidate reconstruction and refresh costs included. |
| B4 | Predetermined varying-rank/dense schedule learned on development runs | Tests whether online evidence is necessary. Schedule fixed before confirmatory data. |
| B5 | Relative-Frobenius audit; and ordinary \(b\|d\|\) update audit | Isolates optimizer/complement information from generic error control. Same candidate, confidence model and refresh information. |
| B6 | Token importance-sampling weight-gradient estimate, exact input gradients | Cheap unbiased competitor with a fixed sample count selected on development data. Formula below. Clearly label as weight-only sampling, not full VCAS. |
| B7 | Faithful portable INSTANT | Closest complete low-rank BP baseline. Preserve sequence compression and calibration, including approximate input gradients. Charge all adaptation/port overhead. |
| B8 | Faithful portable VCAS when its full controller and sampling path pass local tests | Adaptive unbiased BP competitor. Cannot substitute B6 and claim a full VCAS defeat. |
| B9 | DropBP in residual networks, complete algorithm where supported | Additional end-to-end compute-omission competitor; scratch runs are a new application, not replication of published fine-tuning results. |
| D0 | Native exact versus custom-linear exact/full-rank path | Harness/operator overhead control, never replace native B0 with the slower wrapper as the headline baseline. |
| D1 | Full-gradient oracle rank/subspace and exact residual oracle | Structural upper bound and classifier diagnostic, not a free deployable method. |
| D2 | Exact-gradient SGD and projected SGD without momentum/decay | Verifies the automatic-descent limiting case. |
| D3 | Matched dense-refresh/fallback frequencies without current audit information | Detects gains due merely to more exact work. |

Use the strongest **faithfully executable** local comparisons for direct claims. B7 is a required nearest-art reproduction attempt; B8/B9 are required feasibility attempts before freezing a broad end-to-end competitiveness claim. A blocked published implementation may remain a contextual reference, but then narrow the result to the tested methods and state the exclusion prominently. A slower Python port cannot establish superiority over the original method's best possible implementation. No CUDA/NVIDIA result is imported into a controlled M3 comparison.

GaLore and low-rank-state methods belong on the memory frontier if OAR makes a memory claim or B2 is unavailable and they are strong locally. They are not mandatory duplicate compute-avoidance baselines in a memory-unconstrained primary run. LoRA and other parameter-efficient fine-tuning methods become mandatory in a fine-tuning pivot, not scratch-training substitutes. Forward-mode/zeroth-order methods are contextual unless the claim changes to avoiding backward propagation altogether [R16].

**B5 exact specifications.** The relative-Frobenius variant accepts when all values are finite and
\[
\sqrt{\sum_{\ell\in\mathcal A}b_\ell^2}
\leq\varepsilon_F\sqrt{\sum_{\ell\in\mathcal A}\|\widehat G_\ell\|_F^2}.
\]
Use the same independent probes, confidence allocation, numerical treatment, replay semantics and scheduled refreshes as OAR; tune \(\varepsilon_F>0\) on development data under the same compute budget. When the denominator is zero, accept only if every residual bound is also zero; otherwise replay exactly. Define the normalized slack as zero in the all-zero case and infinity when the denominator is zero but a bound is nonzero. It does not require positive optimizer alignment. Its next-step rank controller uses contributions \(c_\ell=b_\ell^2\) and normalized squared slack \(\sum b_\ell^2/(\varepsilon_F^2\sum\|\widehat G_\ell\|_F^2)\); reject/shrink thresholds are 1 and 1/2, respectively, with the same rank-cost and success-streak rules. The ordinary update-audit variant uses \(a>0\) and \(\sum b_\ell\|d_\ell\|_F\leq\tau a\), and substitutes those contributions into OAR's controller. Give each variant separately tuned thresholds; do not handicap it by copying OAR's selected tolerance. These isolate reconstruction accuracy, ordinary update geometry and the proposed complement refinement. The relative-error variant need not preview AdamW or compute displacement complements: after accepting its gradient, it may use native AdamW directly. Charge only the work each comparator actually requires; a shared interface must not burden a simpler test with OAR-only operations.

**B6 exact specification.** Flatten rows \(a=1,\ldots,n\). Set scores \(s_a=\|D_a\|_2\|X_a\|_2\). If all scores are zero, return a zero weight gradient. Otherwise define \(p_a=(1-\epsilon)s_a/\sum s+\epsilon/n\), with a small pilot-selected positive uniform floor. Draw \(k\) indices independently with replacement and calculate
\[
\widetilde G=\frac1k\sum_{j=1}^k\frac{D_{a_j}^\top X_{a_j}}{p_{a_j}}.
\]
This is unbiased conditional on \(X,D\). Gather the rows and execute a genuinely smaller GEMM; zeroing rows in a full-size GEMM is not equivalent compute elimination. Include score, multinomial, gather and reconstruction cost. `k=n` with replacement is still random; use the native product for the distinct exact setting. Accumulated batches use correct loss weights. This establishes a transparent strong simple sampling control without mislabeling it as a full replication of VCAS [R2], [R5].

**B7 local algebra sufficient to validate a port.** For each sequence, use calibrated sequence-axis bases \(A\in\mathbb R^{S\times r_x}\) and \(B\in\mathbb R^{S\times r_D}\). Store \(A^\top X\); form \(\widetilde G=D^\top A(A^\top X)\); input gradient is \(B[B^\top D\,W]\). Sum weight contributions over sequences, preserve bias treatment, and reproduce upstream calibration before tuning. The inspected implementation uses these contractions [R28]. Its exact-adjoint variant is a separate ablation, not the full baseline. Calibration uses extra no-update exact passes over a declared fitting-data calibration stream at the current parameters. Average \(XX^\top\) and \(DD^\top\) over sequences and calibration batches, then use the archived upstream SVD helper. Charge all calibration passes and token exposures; they are not ordinary optimizer updates. For the inspected helper, let \(S_i\) be singular values of the Gram matrix, \(C_j=\sum_{i\leq j}S_i^2\), \(T=\sum_iS_i^2\), and use zero-based array indices in the following literal-port specification. Set `m = min(count(C <= var*T) + p, N)` for Gram dimension `N`. If `0 < m+1 < N`, retain `U[:, :m+1]` and scale it by `1/sqrt(C[m+1]/T)`; otherwise the upper-bound branch retains `U[:, :m]` and uses total energy. This reproduces the inspected nonzero-matrix helper, including its indexing, rather than silently repairing it. The matrices \(A,B\) above therefore need not have unit-length columns. Test diagonal spectra and rank-boundary cases against that source helper before timing. A zero-energy or nonfinite calibration case must use an explicitly logged dense safety path, not propagate NaNs. Also run a separately labeled paper-intent variant with the intended energy threshold/oversampling and retained-energy indexing; give both fair development tuning and report any behavior difference. Do not attribute superiority over INSTANT to an unreported off-by-one or scaling change. Use the reported calibration schedule for the initial algorithm check, then fairly tune locally; adapting BERT-classification calibration to next-token training is explicitly a new-workload implementation, not replication of its published fine-tuning score.

### 6.3 Tuning and information fairness

Perform semantic tests before timing and baseline reproduction before OAR tuning. Dense learning-rate search brackets divergence and stagnation on development data, then refines the useful interval. Give each method a comparable **charged development compute budget**, while recording configuration counts as well; equal trial counts can favor cheaper trials unfairly. A method-specific parameter is not an excuse to withhold baseline tuning. Use the same generic improvements—efficient attention, data caching, optimizer variants, precision and appropriate checkpointing—where they preserve semantics.

Start AdamW pilot bracketing at learning rates \(10^{-4},3\times10^{-4},10^{-3},3\times10^{-3}\), decay 0 or 0.1, and standard moment settings, then expand if the best lies on a boundary. These are search seeds, not claimed optimal settings. Warmup and decay must be expressed in consumed valid tokens for matched-token comparisons; allow a separately frozen time-budget-aware schedule for practical fixed-time competition. Compare matched schedules first so a schedule change is not misattributed to OAR. Resolve B2's rate convention against its installed official API before selecting a bracket.

Every exact gradient, calibration batch and oracle computation available online to OAR is charged or supplied to an information-matched control. Offline exploratory oracle costs are recorded as research costs and cannot disappear from a cold-start deployment analysis. The primary method starts from initialization, including its initial dense steps. A switch from a shared pretrained checkpoint is a separate late-stage result, with checkpoint production cost stated.

## 7. Data, architectures and leakage controls

### 7.1 Core language-model workload

Use Salesforce's WikiText raw configurations, starting with `wikitext-2-raw-v1`; use `wikitext-103-raw-v1` only after the measured scale gate [R22]. Archive the dataset revision, file SHA-256 hashes, raw card and license notices. The inspected card's metadata lists CC BY-SA 3.0/GFDL while its prose refers to CC BY-SA 4.0. Do not silently choose one as an authoritative reconciliation. Review the supplied notices and underlying provenance before use/redistribution; preserve attribution and avoid redistributing corpus excerpts or a repackaged corpus as part of the PoC. The original creator blog was not accessible during this assignment. This is an asset-compliance gate, not legal advice.

The inspected raw WikiText-103 card reports roughly 315 MB compressed and 549 MB extracted text; these are planning figures, not a promise of the next repository revision's download size [R22]. Inspect asset sizes before downloading. WikiText-2 is related to/subsampled from WikiText-103; they are **not two independent corpus replications**.

Preprocessing is deliberately fixed:

- Preserve source text and case; normalize line endings only. Reconstruct articles using top-level title lines, with fixtures for subheadings, empty lines and title punctuation. Confirm article boundaries manually on a small sample and by a parser test before creating splits.
- Hash normalized article content. Keep official train/validation/test article groups separate. Report exact cross-split duplicate hashes; exclude duplicated evaluation groups under a frozen rule and retain an exclusion manifest. Do not silently remove approximate duplicates after observing results.
- Split official training articles into fitting and development groups by a seeded hash, initially 90/10 by article, adjusting only if the development set has insufficient token count for stable decisions. Train a byte-level BPE tokenizer of target vocabulary 8192 using **fitting articles only**. Reserve explicit beginning-of-document (BOS), end-of-document (EOS) and padding tokens. Archive tokenizer bytes and hash. Byte fallback avoids unknown-token distortions.
- Use official validation only for the frozen monitoring/evaluation protocol after development selection. Keep official test sealed until final model selection and reporting. The training-article development split, not official test, chooses rank/controller settings and budgets.
- Score next-token NLL. Prepend BOS and append EOS to each article; do not score BOS, and do score EOS. Do not allow attention across unrelated articles. For training, use deterministic shuffled article chunks: each chunk contains up to \(S\) scored next-token targets with their one-token-shifted inputs, retaining the immediately preceding token across adjacent chunks. Pad and mask loss; do not selectively discard short chunks. For evaluation, use target stride \(K_{\rm eval}=\lfloor S/2\rfloor\). Each window scores at most \(K_{\rm eval}\) new target tokens, with up to \(S-K_{\rm eval}\) earlier tokens of context and shifted causal inputs; the actual input length must never exceed \(S\). Include BOS when the window reaches the beginning of an article. Mask earlier context positions from the score, score each original target once, and archive exact target positions. Use identical scored positions and context boundaries across methods.

Use a pre-normalized residual decoder: layer normalization, causal multihead self-attention, residual addition, layer normalization, GELU MLP with expansion 4, residual addition; final normalization; tied input embedding/output projection; learned positional embeddings up to the fixed context limit. Use head dimension 64 where width permits. Initial primary dropout is zero to simplify exact replay and reduce a confound, shared by all methods; later test nonzero dropout. Initialize matched models from identical parameter tensors. Keep the vocabulary/output layer exact. Use the fastest validated causal-attention implementation common to all methods.

Approximate parameter counts, ignoring small bias/norm/position contributions, are \(P\simeq Vd+12Ld^2\) with tied embedding:

| Exploratory scale | Layers \(L\) | Width \(d\) | Approximate parameters | Scientific use |
|---|---:|---:|---:|---|
| A | 6 | 256 | 6.82 million | Complete mechanism and replay tests; not assumed to be GEMM-efficient. |
| B | 8 | 384 | 17.30 million | Default first learning pilot if the timing gate supports it. |
| C | 8 | 512 | 29.36 million | Wider-kernel and rank-fraction sensitivity. |
| D | 12 | 768 | 91.23 million | Conditional scale test, not a required minimum or automatic target. |

Start context candidates at 128, 256 and 512, but select the shortest context that expresses the stated workload and produces informative timing. Do not shrink context only for OAR. Effective batch and microbatch sizes come from the memory/throughput sweep. Report \(r/d\), \(r/n\), sequence length and effective batch because a fixed rank alone hides the compression regime.

### 7.2 Secondary real-data mechanism test

Fashion-MNIST supplies a compact real image-classification workload through its official repository [R23]. Verify file checksums, notices and MIT licensing at the pinned revision. Hold out a stratified development subset from the official training set and keep the official test set for final reporting. Normalize pixel values to [0,1]; do not add augmentation to only one method.

Use a residual MLP: flatten 28×28 input; input projection; a stack of same-width pre-normalized GELU residual blocks with two linear transformations; classifier. Start depth 4 and width 512, then vary depth and width only as required to express rank turnover and useful GEMM regimes. This network is a controlled test bed, not a claim to best Fashion-MNIST accuracy. Evaluate error rate, NLL, calibration and time-to-quality. A method that wins only here but fails on language modeling has a narrower result, not a general training breakthrough.

If WikiText cannot be used under the asset gate, this dataset plus synthetic tests preserve a local neural-learning PoC, but **do not** validate the language-pretraining claim. A replacement text corpus requires a recorded amendment with verified provenance, license, splits and new baseline calibration; no silent substitution.

### 7.3 Synthetic adversarial workloads

Implement a reproducible associative-recall generator. A sequence contains sampled key–value pairs, a query key and a target value; keys and values are distinct token ranges. Sample mappings anew per sequence so a model must use context rather than memorize a global table. Use a marked minority regime that queries the first occurrence of a repeated key, while the majority queries the last occurrence; the regime marker makes the correct rule identifiable. Set minority probabilities 0.01 and 0.001 in separate training conditions. Use fixed-seed balanced evaluation sets for the two regimes, independent of training generation. Match sequence lengths and token counts so the rare slice is not simply a longer-input confound.

Add a mid-run distribution shift from majority-only to the mixed task and vary distractor count to test depth/context coordination. Exact BP must first demonstrate that the selected architecture can learn both regimes; otherwise failure of an approximate method is not informative. Report majority and minority accuracy/NLL separately, forgetting and recovery time, including binomial uncertainty for generated independent examples. These are artificial stress tests, not proxies that establish rare-language or real-world reasoning competence.

At the operator level also use flat spectra, rapidly rotating subspaces, low-rank-plus-rare-coordinate matrices, cancellation-heavy gradients, ill-conditioned optimizer states and a deliberate incorrect-adjoint injection. These tests isolate mathematical failure modes before large-model training.

## 8. Statistical and performance protocol

### 8.1 Exploratory versus confirmatory work

Development includes kernel selection, preprocessing validation, hyperparameter search, basis experiments and pilot trajectories. Maintain a visible `exploratory` label. Before protected evaluation, write and hash `protocol_frozen.yaml` containing hypotheses, primary endpoint, architecture, data/tokenizer hashes, eligible layers, winning baseline selections, method configurations, wall-time horizon, target losses, analysis method, seed list, sample size, guardrails and exclusions. Record any amendment; changed claims require fresh confirmation seeds and, where evaluation selection has leaked, a fresh protected evaluation partition.

Choose the horizon \(H\) from dense development learning curves: it must include more than initialization/warmup, appreciable feature learning and a regime in which rank drift can be evaluated. Select the smallest scientifically informative horizon, not a convenient fixed number of laptop hours. Project total study duration from measured throughput and replication requirements. A locally unaffordable precision requirement yields an inconclusive study or a narrower claim, not an invented positive result.

### 8.2 Independent unit, pairing and sample-size calibration

The primary independent experimental unit is an independently initialized **training run**, paired across methods by initialization and data-order seed. GPU timing blocks, checkpoints, minibatches and tokens are not additional independent training runs. Seeds quantify optimization variability on this fixed corpus; they do not by themselves establish generalization across tasks or datasets.

Use pilot paired differences \(D_i=\mathrm{NLL}_{\rm OAR}-\mathrm{NLL}_{\rm comparator}\) to estimate variability. Choose a scientifically useful detectable difference \(\Delta\) or desired confidence half-width \(w\) based on (i) achievable evaluation precision, (ii) the loss difference corresponding to a meaningful saving on the dense learning curve, and (iii) the decision about whether continued investment is worthwhile. Report this choice in absolute NLL units; do not insert an arbitrary “must improve by 5%” rule.

For planning, \(n\approx[(z_{1-\alpha/2}+z_{1-\beta})s_D/\Delta]^2\) gives a rough paired-test sample size; use a conservative variance estimate from the pilot and refine using a Student-t power calculation or a simulation of the planned analysis. Also calculate expected confidence width and projected runtime. Obtain additional **pilot** runs if the variance estimate is too unstable, not because a confirmatory p-value is nearly favorable. Then freeze a fixed confirmatory sample size. Do not stop when significant. If practical capacity only permits a weaker design, label the resulting detectable effect and underpowered outcome explicitly.

### 8.3 Confirmatory analysis

Report each run, paired differences, mean difference and 95% confidence interval. The prespecified primary analysis is a paired t interval/test when pilot diagnostics do not show severe outliers; otherwise freeze a paired randomization test plus a suitable paired bootstrap interval before confirmation. A sign-flip paired test requires exchangeability under the sharp null (or a symmetric paired-difference model); do not call it exact merely because the run seeds were randomized. With very few pairs, enumerate sign flips only when that assumption is defensible and acknowledge coarse attainable p-values and wide intervals. If neither the paired-mean approximation nor a justified randomization model is adequate at the affordable sample size, report an inconclusive interval/descriptive result rather than selecting the most favorable test. Bootstrap whole paired runs, not checkpoints. Publish the raw paired data so the assumptions can be checked.

Test the strongest baseline comparison and the matched-control comparison as one family using Holm's correction at familywise 0.05. Directional scientific interpretation requires a negative NLL difference, not merely a small p-value. Treat other ablations as exploratory unless independently powered and preregistered. Do not test all architectures/ranks and report only the winner as confirmed.

Article-level bootstrap intervals may separately describe evaluation-set uncertainty conditional on a trained model. Keep that uncertainty distinct from training-run uncertainty; use a nested procedure only when explicitly specified. For multiple real workloads, report each, and do not pool incomparable loss units into an unexplained grand score.

Rare-slice guardrails need prespecified absolute noninferiority margins based on the dense reference's variability and the task's meaningful error cost. When no defensible practical margin is available, report the full confidence interval and avoid claiming “no degradation.” Absence of significance is not equivalence. An average-loss win with a material rare-slice or stability regression is a trade-off, not an unconditional improvement.

### 8.4 Timing on an asynchronous laptop accelerator

Use wall-clock timing with explicit `torch.mps.synchronize()` immediately before and after a measured block. In ordinary training do not synchronize after every small operation; that changes the workload. Operator breakdowns run in separate profiling sessions, while headline time comes from the uninstrumented or minimally instrumented full path. Official MPS exposes synchronization, allocation and profiler interfaces; local compatibility still requires testing [R19].

Warm up allocations, optimizer states and representative shaders. Extend warmup until rolling block medians stabilize within the measured timing noise; archive the rule and observed sequence. Time blocks long enough to dominate timer/synchronization overhead, determined by a null block measurement. Repeat until the pilot uncertainty in the difference is smaller than the effect under investigation or declare timing unresolved. Use randomized AB/BA block ordering for microbenchmarks and interleave complete run order to reduce thermal confounding.

Run on AC power with a recorded power mode, ambient conditions when available, no competing heavy jobs and sufficient free memory. Record macOS version, thermal/power warnings, display/external-device setup and whether the system was swapping. Do not silently delete slow runs; define invalidation criteria before confirmation and retain invalidated logs with reasons. Report cold-start and steady-state costs separately. Sleep/suspend creates an explicit pause event; machine stalls caused by the method's memory use are costs, not excluded pauses.

Measure tensor allocation and Metal driver allocation separately; they are not additive independent pools. Sample process RSS and system pressure as context. Do not infer joules from FLOPs. Energy can be a secondary metric only if a supported measurement facility and explicit boundaries are verified locally.

### 8.5 Evaluation-harness validity tests

An exact-vs-exact A/A run with identical seeds must give identical or numerically explainable trajectories and no fabricated speed benefit. A deliberately slower implementation must be reported slower. An intentionally flipped gradient/rare-rule corruption must be caught by the corresponding diagnostic. Save/reload and reject/replay must match uninterrupted exact runs within the predefined numerical tolerance. Validation labels and protected scores must not be visible to policy/tuning code. Any failed harness test blocks confirmatory experiments.

## 9. M3 Max feasibility and hardware-adaptive execution

### 9.1 Hardware discovery is the first blocking gate

Do not assume a maximum-memory M3 Max. On the actual laptop, collect native architecture, physical memory, available disk, macOS version, GPU description and supported acceleration. These are existing macOS shell commands, not project scripts:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
df -k .
vm_stat
pmset -g custom
system_profiler SPHardwareDataType SPDisplaysDataType -json
```

Retain only needed hardware fields from `system_profiler`; omit serial numbers and personal device identifiers from published artifacts. Check AC power and record the power mode. Use `torch.backends.mps.is_built()` and `is_available()` in the installed environment. The PyTorch MPS documentation inspected during research showed a macOS 14.0-or-newer requirement for that documentation version; follow the actual installed release's requirement rather than an obsolete generic Apple-Silicon minimum [R19]. An unsupported OS is a blocking condition for that stack, not permission to pretend CPU timings are GPU results. Do not perform an OS upgrade automatically.

### 9.2 Software selection and dependency strategy

**Primary stack:** native arm64 CPython; PyTorch with MPS; NumPy for CPU reference tests; pytest; YAML configuration; a simple local JSONL/CSV logger; Hugging Face dataset/tokenizer tooling plus PyArrow only when the asset gate needs it. No remote GPU, paid API, CUDA dependency, remote tracking service or special equipment is required. Use PyTorch because the relevant published code and custom backward operations can be inspected and adapted in that ecosystem [R28], [R29], [R30].

Select a currently published, native arm64 Python/PyTorch wheel combination that passes the critical-operation suite. Documentation may describe a newer release than the available wheel; **do not infer an installed version from a documentation redirect**. Resolve packages once, archive wheel names/hashes and dependency versions, and freeze the environment before timing. Maintain separate locked environments if a baseline genuinely requires another compatible release; such cross-environment comparisons must still run identical native dense references to detect version confounding.

The following are existing Python packaging commands, to use after selecting a supported native interpreter. They do not assert particular release numbers:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pytest pyyaml psutil
python -m pip freeze > requirements.lock.txt
```

Add dataset/tokenizer packages only after the initial hardware and operator gates. Record package licenses and pin/hash downloaded assets; a list of versions alone is not a complete binary provenance record. Keep local baseline changes as reviewed patches with upstream notices. The observed repository license labels were MIT for INSTANT/DropBP/Fashion-MNIST and Apache-2.0 for VCAS; verify the exact files and bundled dependencies at the execution revision [R23, R28–R30]. Downloadable model weights are not automatically open source; no pretrained weights are needed for the core study.

MLX is a **faithful fallback experiment**, not an assumed faster backend. Its official documentation exposes custom-function VJP hooks [R20]. If PyTorch MPS cannot execute a material critical operation efficiently, first distinguish an operator bug from a backend limitation. Port the same algebra and native dense comparator to MLX, force lazy execution of outputs/parameters/optimizer states in timing, and establish numerical equivalence. Do not compare optimized MLX OAR only against an unoptimized PyTorch baseline. CPU-only execution can establish component correctness but cannot establish an M3 GPU throughput claim.

### 9.3 Critical-operation support suite

Before model downloads or long runs, test, on actual MPS:

- FP32 GEMMs at all relevant \((n,d_i,d_o,r,q)\), including transposed/noncontiguous inputs; explicit reshape/contiguous-copy cost.
- Custom linear autograd, correct exact input/bias derivatives and projected weight derivative; native attention backward, layer norm, GELU and loss masking.
- Gather, scatter, multinomial sampling and reductions for sampling/DropBP baselines; compare their mathematical outputs or distributions with CPU references.
- QR/SVD needed for basis refresh. An explicit CPU FP64 decomposition with measured transfers is a permissible faithful fallback. A silent CPU fallback on every backward GEMM is not.
- Functional optimizer preview, native AdamW variants, Muon availability/semantics, MPS synchronization and memory reporting.
- Reproducible independent CPU-generated Gaussian probes transferred to MPS, model RNG save/restore, checkpointing interactions and exact replay.

Disable automatic MPS CPU fallback during timed critical-path tests; verify the installed release's environment-variable semantics and record the setting. Make unsupported operations raise and categorize them. Start with CPU-generated audit/construction randomness in separate `torch.Generator` or NumPy generator streams to avoid assuming a particular MPS generator interface. Charge generation and transfer costs. Use MPS model RNG APIs only after the local suite verifies them [R19].

Standard finite-difference `gradcheck` is appropriate for the native or full-basis exact operator, **not** for demanding that a deliberately approximated weight derivative equal the true derivative of its forward function. For an approximate operator, compare weight gradients to the specified projection formula and input gradients to the exact branch. A failed full-function gradcheck in this deliberate approximation is not evidence of an implementation bug; an incorrect projection or input gradient is.

### 9.4 Memory model

Let \(P\) be parameter count and all primary tensors be FP32. A dense AdamW run has approximately 4P bytes of parameters, 4P gradients and 8P first/second moments: **16P bytes**, before activations and temporary workspaces. The simple transactional OAR preview additionally retains 8P pending moments and 4P displacement: approximately **28P bytes**. A layerwise recompute-on-commit implementation can reduce that cost but increases computation and requires a separately measured optimization.

Additional contributors are material:

\[
M_{\rm peak}\approx 28P+M_{\rm saved\ activations}+M_{\rm attention}
+M_{\rm logits}+M_{\rm bases/sketches}+M_{\rm temporaries}+M_{\rm runtime}.
\]

For eligible layer \(\ell\), bases cost roughly \(4d_or_{\max}\) bytes; running H costs \(4rd_i\); residual sketch \(4d_oq\); probes \(4d_iq\); temporary \(XZ\) costs \(4nq\). Complement construction may need another dense tensor for the largest processed layer. Process complements layerwise and release them promptly. Store no stack of all incoming adjoints after backward.

Logits alone can cost \(4BSV\) bytes per materialized tensor. Naive attention probabilities cost roughly \(4LBhS^2\) per saved tensor, while a validated fused implementation may have a different memory model. Activation multipliers must be measured, not guessed from parameter count. CPU checkpoint conversion can add about 12P bytes for parameters and moments during serialization. Basis decomposition can add host copies of selected gradients; process them sequentially. Disk checkpoints exclude gradients by default.

Approximate parameter/optimizer-only footprints:

| Parameters | Dense 16P | Transactional 28P | Excludes |
|---:|---:|---:|---|
| 6.82M | 109 MB | 191 MB | Activations, runtime, temporary and checkpoint buffers |
| 17.30M | 277 MB | 484 MB | Same |
| 29.36M | 470 MB | 822 MB | Same |
| 91.23M | 1.46 GB | 2.55 GB | Same |

These decimal-byte estimates do not establish that a complete run fits, nor that the largest model is scientifically necessary. Unified memory does not make duplicate arrays free. Conversely, RSS and Metal-driver allocation can describe overlapping memory, so do not add their readings as separate physical totals.

Set an initial engineering working-set ceiling to the smaller of approximately 70% of physical memory and 80% of the reported recommended Metal working set, with an OS reserve and stricter adjustment if memory pressure rises. These fractions are safety defaults, not claims about a universal hardware limit. Measure the real peak during multiple full steps, a refresh, a rejection/replay and a checkpoint. Never disable allocation safeguards to obtain a benchmark result.

### 9.5 Calibration, runtime and scale rules

For each candidate shape, measure dense step time, candidate time without auditing, audit cost, complement cost, preview overhead, refresh cost and replay probability. Let \(T_D\) be native dense full-step time, \(T_A\) the complete approximate attempt including audit/preview, \(p\) the rejection probability, and \(T_R\) the additional exact replay/commit time after rejection. With amortized refresh overhead \(C/K\), the necessary throughput condition is

\[
T_A+pT_R+C/K<T_D.
\]

Do not double-count the normal work of scheduled dense steps as pure refresh overhead: an exact accounting is a mixture of dense scheduled steps and approximate steps, plus extra basis-building work. The equation above is a useful local break-even simplification; the log-based full-cycle mean is authoritative. In an all-approximate segment, the maximum affordable rejection probability is
\(p<(T_D-T_A-C/K)/T_R\). A nonpositive numerator means no throughput opportunity at that shape even with perfect acceptance. Equivalently, if fraction \(f\) of dense step time is eligible for acceleration, its local speed factor is \(s\), and added overhead is fraction \(h\) of the dense step, the idealized total factor is \(1/(1-f+f/s+h)\). Estimate \(f\) from the actual eligible MLP weight-gradient work, not all backward time; use full-cycle measurements when overlaps make attribution ambiguous.

The probe GEMMs add approximately
\(2nd_iq+2nd_oq+2rd_iq+2d_orq\) operations, while an explicit displacement complement adds about \(4d_ord_i\). These costs and dense AdamW traffic are absent from a “rank divided by width” estimate. Measure shape-dependent launch overhead and effective bandwidth; several small products can lose to one optimized dense product.

Choose the largest useful microbatch that fits safely and improves throughput, not the largest that merely allocates. To increase effective batch without changing its scientific meaning, use the exact accumulation scheme in Section 4.7. If reducing microbatch shrinks GEMMs enough to erase savings, report the resource limitation rather than silently comparing different effective batches. If memory requires changing effective batch, retune all baselines and treat it as a different workload. Checkpointing is shared where appropriate, with hook/recompute correctness tests.

Escalate width or token-batch size only when the smaller experiment cannot express the proposed benefit and measurements predict a useful larger regime. For example, a persistent gradient rank of 16 with a ratio \(r/d\) decreasing as width increases is evidence worth testing at a larger width. A claim that larger models “might help” without measured structure or a cost forecast is not sufficient.

For every substantial stage, estimate runtime from pilot throughput:
\[
T_{\rm planned}\approx N_{\rm processed\ tokens}/v_{\rm measured}
+T_{\rm evaluations}+T_{\rm basis\ work}+T_{\rm serialization},
\]
using a conservative throughput interval and avoiding double-counting costs already included in \(v\). Publish best/central/worst projections and update them after the pilot. No token/s or hours estimate in this document is a measurement of an M3 Max.

For storage, allocate raw assets, tokenized caches, locked wheels/source archives, two atomic checkpoints per active run, aggregate logs and selected diagnostics. One FP32 model-plus-Adam checkpoint is approximately 12P bytes before serialization overhead. Do not retain full gradients at every step. Require available space for a new checkpoint plus the previous valid checkpoint and temporary files before starting an expensive run.

## 10. Repository, interfaces and numerical conventions

### 10.1 Minimal project layout

All names below are **proposed project files/commands to implement**, not existing package interfaces.

```text
oar-poc/
  README.md
  pyproject.toml
  requirements.lock.txt
  manifests/
    environment.json
    assets.json
    upstream_sources.json
    wheels.sha256
  configs/
    discovery.yaml
    operator.yaml
    lm_dev.yaml
    vision_dev.yaml
    protocol_frozen.yaml
  src/oar/
    cli.py
    hardware.py
    data.py
    models.py
    linear.py
    basis.py
    audit.py
    optimizer_preview.py
    controller.py
    methods.py
    train.py
    timing.py
    checkpoints.py
    analysis.py
  tests/
    test_algebra.py
    test_confidence.py
    test_accumulation.py
    test_optimizer_preview.py
    test_exact_adjoints.py
    test_replay_resume.py
    test_data_leakage.py
    test_timing_harness.py
  third_party/                 # pinned sources, notices, minimal patches
  data/                       # private local caches; never commit by default
  runs/<run_id>/
    config_resolved.yaml
    events.jsonl
    layers.jsonl
    resources.jsonl
    evaluations.jsonl
    checkpoints/
  analysis/
    oracle_audit.parquet
    timing_calibration.csv
    confirmatory_results.csv
    report.md
  decisions/
    progress.md
    amendments.jsonl
    gates.jsonl
```

Use deterministic configuration serialization and SHA-256 identities for resolved configuration, code revision and data/tokenizer artifacts. No hidden notebook state or chat instructions are part of execution. A local Git repository is sufficient; pushing to an external service is not required or authorized by this plan.

### 10.2 Shared method interface

```python
class TrainingMethod:
    def begin_step(self, step_context, replayable_batch) -> None: ...
    def backward_microbatch(self, batch, loss_weight: float) -> None: ...
    def propose(self, model, optimizer_state) -> "PendingUpdate": ...
    def decide(self, pending_update) -> "Decision": ...
    def commit_or_replay(self, decision, replayable_batch) -> "StepRecord": ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

The native dense method uses native backward/optimizer execution and does not incur unnecessary preview or audit overhead. Shared interfaces must not force all methods through OAR's slow path. `StepContext` contains logical step, valid-token normalization, parameter version, model-RNG snapshot, method RNG states, source-batch identifiers, schedule position and timing boundary. `PendingUpdate` owns tentative optimizer tensors and displacements, tagged with the exact parameter/state version. Any version mismatch or double commit is an error.

`LayerAudit` contains stable parameter identifier, shape, rank, basis version, probe seed/identifier, residual sketch, confidence allocation, estimated squared residual, bound, orthogonality diagnostic and timing fields. Values have explicit units and dtype. No tensors requiring gradients are stored in audit records. `Decision` is one of `accept_approx`, `scheduled_dense`, `reject_replay`, `numerical_reject`, or `stop_invalid`; it includes reason codes, \(a,B\), safety margin and controller changes for the next step.

### 10.3 AdamW preview conventions

Preserve the **raw** approximate gradient for the inner product in (OAR). In-place clipping must not overwrite it. With global clip factor \(\gamma\), calculate tentative moments using \(h=\gamma\widehat g\):

\[
m'=\beta_1m+(1-\beta_1)h,\quad
v'=\beta_2v+(1-\beta_2)h^2,
\]
\[
d=\eta\left[\frac{m'/(1-\beta_1^t)}{\sqrt{v'/(1-\beta_2^t)}+\epsilon}+\lambda\theta\right].
\]

Use the native implementation's exact clipping convention, including its stabilization term, and validate equivalence rather than assuming it. The primary rule uses no AMSGrad or differentiable optimizer; additional variants require tests. Missing gradients are handled exactly as native AdamW, not converted to zero gradients. Apply weight decay only to the declared groups. Compute \(a\) using raw \(\widehat g\), but \(d\) using the clipped optimizer input. Otherwise the claimed bound is on the wrong gradient.

### 10.4 Whole-step pseudocode

```text
read replayable effective batch, without consuming it twice
save model RNG; freeze rank/basis/parameter versions
if step is scheduled dense:
    native exact backward over correctly weighted microbatches
    preserve exact gradients needed by the scheduled basis refresh
    native optimizer step once
    build future-use bases from the preserved exact gradient
    advance scheduler/data cursor once; save committed state
else:
    allocate independent layer probes and confidence indices before observing them
    for each microbatch:
        forward unchanged
        backward:
            eligible linear layer: exact dX and db; projected dW
            add correctly scaled H and residual sketch R
            release current activations/adjoints normally
    preview AdamW without modifying raw gradients or live states
    form a, residual bounds, and off-subspace displacement norms
    if numerical checks pass and a > 0 and B <= tau*a:
        commit tentative optimizer state and parameter decrement once
    else:
        discard all tentative state; clear gradients and audit accumulators
        restore original model RNG, not the independent audit RNG/counter
        replay identical microbatches with native dense backward
        apply native exact optimizer once
    update next-step controller using frozen policy
    advance scheduler/data cursor once
record valid tokens AND processed/replayed tokens, inclusive time and decision
checkpoint only committed states, including all RNG/controller counters
```

Implement basis refresh before or after optimizer commit only if it uses the same saved **pre-update** gradient and cannot influence that already selected update. Never recompute a post-update gradient and call it the original calibration signal without charging it. To reduce checkpoint inconsistency, serialize the full committed state only after the corresponding future-use basis/controller state is complete.

## 11. Risk-ordered work packages and decision gates

Each package ends with `gates.jsonl` and a human-readable decision. “Stop” means stop the affected claim/implementation branch; it does not justify discarding measurements. Dependencies are functional rather than a mandate to build unnecessary infrastructure.

### WP00 — Discover hardware, provenance and usable operations

**Objective/risk:** prevent an infeasible backend or unusable asset from contaminating all later work. **Entry:** fresh laptop, this plan and permission to create a local project directory. **Inputs/dependencies:** none.

**Actions:** implement `hardware.py` and a `doctor` CLI; collect the Section 9 hardware record; create a native environment; run tiny dense GEMM/backward/optimizer tests; inspect asset notices and file sizes; archive upstream source/license metadata. Do not download large corpora or install CUDA packages. Resolve the actual package versions and binary hashes.

**Files/commands:** proposed `python -m oar.cli doctor --out manifests/environment.json`; `configs/discovery.yaml`; `manifests/assets.json`. **Tests/output:** JSON fields for platform, OS, physical bytes, free-disk bytes, backend availability, package versions, supported operations, fallback settings, redacted hardware and test status. **Resources:** tiny tensors and metadata only; measure runtime rather than preallocating a training budget.

**Gate:** proceed to WP01 if CPU reference and MPS basic execution work and at least one permissible core dataset is available. An unsupported critical operation goes to explicit CPU-calibration or MLX equivalence testing, not silent fallback. No workable GPU path means a correctness-only result. A license/access block changes the workload only through an amendment and narrows claims.

### WP01 — Verify mathematics, conventions and optimizer transactions

**Objective/risk:** falsify incorrect algebra, confidence assumptions or state handling cheaply. **Entry:** WP00 reference environment. **Inputs:** Appendix A plus custom-operator/preview specifications.

**Actions:** implement `linear.py`, scalar bound routine and a minimal functional AdamW preview on CPU FP64; reproduce the recorded checks; test projection/full-basis/zero/rank-deficient cases; test exact incoming adjoints across at least two residual blocks; verify accumulated gradients/probes versus one concatenated effective batch. Add clipping, bias, mixed parameter groups and zero-versus-missing gradients. Test deliberately incorrect adjoints and dependent probes as negative cases.

**Files/commands:** proposed `pytest tests/test_algebra.py tests/test_accumulation.py tests/test_optimizer_preview.py tests/test_exact_adjoints.py`; numerical reference artifacts in `analysis/`. **Outputs:** per-case maximum absolute/relative errors, bound validity experiments, expected failure labels and native-preview state comparisons. **Resources:** CPU-scale matrices, not whole model training; calibrate Monte Carlo repetitions to the failure probability being evaluated. A 50,000-draw test cannot certify a one-in-a-million failure rate.

**Gate:** numerical differences must fall within tolerances derived from dtype, norm and operation count. Default FP64 small-matrix tolerances can start at approximately 1e-10 relative plus an absolute near-zero tolerance, but failing cases require diagnosis rather than blind tolerance inflation. A mathematical or transaction error blocks all subsequent work. A loose yet correct bound advances to WP02/WP04 to assess usefulness; it is not automatically an implementation failure.

### WP02 — Measure kernel headroom before building a trainer

**Objective/risk:** determine whether the target laptop can profit at all after hidden work. **Entry:** WP01 correct algebra and WP00 MPS support. **Inputs:** representative shapes from proposed model tiers and sampling baselines.

**Actions:** benchmark native dense dW, projected dW including reconstruction, probes, complement construction, optimizer preview, explicit CPU decomposition transfers, gather/scatter and complete linear training blocks. Use the timing protocol. Sweep \(n,d_i,d_o,r,q\) across feasible shapes, including tall MLP matrices and multiple accumulation sizes. Profile a complete tiny residual model; do not infer whole-step savings by adding isolated timings with different synchronization patterns.

**Files/commands:** proposed `python -m oar.cli bench --config configs/operator.yaml`; `analysis/timing_calibration.csv`. **Outputs:** shape/dtype/backend, warmup sequence, raw block seconds, confidence interval, memory bytes, copy/launch counts, predicted break-even rejection probability. **Resources:** adapt the sweep to measured memory; no large dataset required.

**Gate:** if no feasible shape has even candidate-only headroom, stop online-audit engineering on that backend and retain a structural study. If candidate-only headroom exists but the audit cannot fit it, proceed only to the cheap oracle-information test, not a full online trainer. If larger widths plausibly enter a useful regime, test those kernels before scaling a model. Faithful MLX or a native-kernel optimization is allowed only after the measured bottleneck identifies the necessary change and the native dense comparison receives equivalent attention.

### WP03 — Build a valid harness and reproduce strong baselines

**Objective/risk:** prevent a weak baseline, leakage or timing artifact from becoming a claimed advance. **Entry:** WP02 identifies at least a structural or computational question worth testing. **Inputs:** approved small dataset, fixed preprocessing, architecture and source revisions.

**Actions:** implement `data.py`, `models.py`, native trainer, complete timing boundaries and atomic checkpoints. Train the dense model enough to establish useful learning and a development curve. Run native/checkpointed variants and the Muon support/tuning gate. Implement D0 A/A controls. Port INSTANT's complete algebra/calibration and attempt VCAS/DropBP support; archive minimal patches and numerical equivalence results. Add B3 and B6. Verify synthetic tasks are learnable by dense BP before using them as adversaries.

**Files/commands:** proposed `python -m oar.cli prepare-data --config configs/lm_dev.yaml`; `python -m oar.cli train --method dense_adamw --config configs/lm_dev.yaml`; `pytest tests/test_data_leakage.py tests/test_timing_harness.py tests/test_replay_resume.py`. **Outputs:** tokenizer/split hashes, dense learning curves, faithful-port checklist, tuning records, all failed runs, native/custom overhead comparison. **Resources:** start at the smallest useful model; forecast full-run duration from a measured pilot and checkpoint footprint.

**Gate:** proceed if the dense model learns, protected data remain sealed, timing detects known differences, resume works and at least the strong native and nearest-art comparison paths are established or explicitly scoped as blocked. A baseline failure is not OAR evidence. If a stronger optimizer removes the entire prospective gain, retain it as the primary comparator; do not revert to a weaker one.

### WP04 — Gradient-information and oracle-value audit

**Objective/risk:** test H1/H2 before investing in an online controller. **Entry:** valid WP03 checkpoints at initialization/early feature learning, mid-development and late-development. **Inputs:** saved parameters/optimizer state and disjoint future training batches; timing map from WP02.

**Actions:** create bases using only batches earlier than the query batch. For sampled layers, capture \(X,D,G\) transiently; calculate exact current SVD oracle, past-basis projections, exact residuals, raw gradient energy/cosine, ordinary and complement-aware update margins. Preview actual optimizer displacements, including a fixed history. On identical copied model states, apply candidate and dense updates, evaluate same-batch finite loss and an independent development batch; discard these counterfactual states rather than feeding oracle information into training. Vary rank, basis age, training stage, effective batch and synthetic rare regime.

Measure the fraction of harmful finite updates detected, false-rejection frequency, margin tightness and predicted compute savings. Estimate whether a policy based only on past rank/fallback schedules performs as well. Include the unfavorably loose bound cases, not just tight examples. Collect only sketches, scalar diagnostics and selected matrices necessary for reproducibility; no blanket all-gradient archive.

**Files/commands:** proposed `python -m oar.cli audit-checkpoints --config configs/operator.yaml`; `analysis/oracle_audit.parquet`, with checkpoint/batch/layer/rank identifiers, all scores, actual loss deltas, basis age and oracle-compute cost. **Resources:** choose checkpoint/layer/batch coverage by uncertainty in the measured event rates and heterogeneity, not a mandatory hundreds-of-gradients dump. Forecast cost from one checkpoint and stream layer by layer.

**Gate:** if no past-only rank preserves useful updates, the selected proposer fails locally; test a better past-only basis only with a recorded reason. If even exact residual knowledge does not beat simple controls, H2 is weakened and an online verifier is unlikely to help. If oracle discrimination exists but statistical probes require more time than saved, report an information–cost gap. Proceed to WP05 only when a measured operating region jointly supports useful proposals, potentially helpful auditing and plausible positive net cost. A rare-event estimate too imprecise to decide is inconclusive; use targeted synthetic counterexamples without misrepresenting their population frequency.

### WP05 — Implement the minimum complete OAR method

**Objective/risk:** validate the actual online mechanism and reject/replay cost. **Entry:** WP04 supporting region, not a promise of success. **Inputs:** frozen provisional projector, rank set, probe count and controller rules.

**Actions:** implement `audit.py`, `controller.py`, `optimizer_preview.py` and `methods.py`; integrate independent probes, effective-batch accumulation, actual displacement complement, guardband rejection and transactional replay. Add all state-version assertions. Preserve raw gradients through clipping. Build B5 ordinary/relative-error auditors and information-matched schedules. Test scheduled refreshes, all-dense mode, all-rejected mode, basis rank changes, interruption during a tentative update and NaN handling.

**Files/commands:** proposed `python -m oar.cli train --method oar --config configs/lm_dev.yaml`; `pytest` transaction/replay suites; a forced-rejection test configuration. **Outputs:** complete step/layer logs, memory peaks during refresh/replay, checksums of resumed versus uninterrupted states, accept/reject reasons and inclusive time. **Resources:** tiny model first, then the measured WP04 region. Charge every CPU transfer and preview tensor; compare actual memory to Section 9's model.

**Gate:** exact/full-basis behavior and replay must match references. Any hidden exact-gradient computation in an approximate step invalidates a compute claim until removed or charged. If OAR is correct but slower, distinguish whether costs arise from an identifiable implementation issue, irreducible repeated work, or an overly conservative policy. Only optimize the measured issue. No extension to approximate input adjoints is permitted in this package.

### WP06 — Pilot, attribution and protocol freeze

**Objective/risk:** select one defensible configuration and a sufficiently sensitive evaluation. **Entry:** WP05 complete/correct implementation and strong baseline reproductions. **Inputs:** development data only, raw timing and structural audits.

**Actions:** tune native baselines and OAR under declared compute budgets; calibrate \(r,q,K,\tau,h\), numerical guardbands and rank re-entry behavior; run all matched controls; measure paired-run variability; determine horizon, a common elapsed-time endpoint checkpoint cadence with acceptable loss discretization, sample size, attainable quality targets and guardrails. Calculate expected total runtime/storage. Test the strongest baseline again after shared optimizations. Select one main architecture and one confirmatory OAR setting, not a post hoc collection of favorable ranks.

**Files/commands:** proposed `python -m oar.cli pilot --config configs/lm_dev.yaml`; `python -m oar.cli freeze-protocol --input configs/lm_dev.yaml --out configs/protocol_frozen.yaml`; freeze utility refuses unresolved essential fields. **Outputs:** selection rationale, all development trials, power/precision calculation, remaining blind spots, immutable protocol hash and confirmation seed list. **Resources:** measured planning, with sensitivity analyses if the variance estimate is unstable.

**Gate:** proceed only if the primary question is identifiable and the planned comparison fits measured resources. A predicted tiny effect requires a precision capable of resolving it; otherwise declare likely inconclusive before launching. A nonpositive estimated frontier can justify stopping with a negative result. An exploratory improvement that vanishes after fair baseline tuning is retained as such, not revived with a weaker comparator.

### WP07 — Execute the fixed confirmatory comparison

**Objective/risk:** test H4 without adaptive selection. **Entry:** WP06 frozen protocol and a clean reproducible environment. **Inputs:** fixed seeds/configurations and protected monitoring population.

**Actions:** execute paired runs in randomized/interleaved order; retain failures; enforce fixed charged-time/checkpoint boundaries; collect prescribed monitoring but do not modify methods in response; run the prespecified paired analysis, censoring/target analysis and guardrail assessment. Seal primary result files before exploratory follow-up. Score official test only for the predeclared selected models, not every trial.

**Files/commands:** proposed `python -m oar.cli confirm --protocol configs/protocol_frozen.yaml`; `python -m oar.cli analyze --protocol configs/protocol_frozen.yaml`. **Outputs:** run-level raw records, primary effect/interval, multiplicity-adjusted tests, absolute times/quality, nonattainment counts and the full list of failures. **Resources:** frozen pilot-derived forecast; checkpoint/resume supported, but methodological changes are not allowed mid-confirmation.

**Gate:** a positive result requires the declared comparisons and guardrails; a smaller nominal p-value without a useful direction is not success. Wide intervals imply inconclusive evidence. Consistent poorer quality or no net speed benefit is evidence against practical value in the tested regime. Continue to WP08 to assess attribution/robustness of a positive claim, or perform only the targeted failure analysis needed to explain a negative one.

### WP08 — Robustness and independent confirmation

**Objective/risk:** determine whether the apparent mechanism survives new conditions. **Entry:** a WP07 result and a prewritten robustness agenda; no retroactive change to primary results. **Inputs:** fresh seeds, secondary workload and failure-mode fixtures.

**Actions:** vary precision, effective batch, rank fraction, basis age, width/depth and dropout; test rare-rule/shift workloads; independently repeat the main result from a clean environment/checkpoint manifest. Vary optimizer/clipping/decay to isolate where complement geometry matters. Test a second width only when scaling evidence justifies it. Run the secondary real-data workload under a separately frozen local comparison. Give each new claim fresh confirmation rather than reusing the original protected outcomes indefinitely.

**Files/commands:** proposed `python -m oar.cli robustness --protocol configs/protocol_frozen.yaml`; separate run labels and amendments. **Outputs:** sensitivity surfaces, subgroup intervals, change-point recovery, independent reproduction status and exceptions to the claim. **Resources:** prioritize tests that distinguish a mechanism from a confound; do not run an arbitrary exhaustive grid or stop at the first favorable slice.

**Gate:** a narrow stable benefit earns a narrow claim. Synthetic-only success is not language-model evidence. Failure confined to launch-bound small models does not negate larger measured structural evidence, but larger-scale promises require local corroboration. A rare-slice regression narrows or defeats an unconditional learning claim. If replication fails, report instability before proposing scale-up.

### WP09 — Synthesize, archive and decide

**Objective/risk:** preserve a reproducible, honest conclusion rather than a cherry-picked demonstration. **Entry:** completed or stopped earlier packages. **Inputs:** all gate records, failed runs, source/asset manifests and analyses.

**Actions:** generate `analysis/report.md`; populate the outcome table in Section 14; distinguish mathematical facts, CPU checks, actual M3 measurements, exploratory findings, confirmed findings and unresolved claims. Include cost breakdowns and unavailable baseline limitations. Run the three audits in Section 15 against the final implementation/report. Export a small reproduction configuration, dependency/assets manifest and analysis code without redistributing restricted assets.

**Outputs:** one-command local reproduction instructions using the implemented CLI, raw-data index, figure/table generation commands, decision record and scope-limited conclusion. **Resources:** analysis/checksum/storage costs only, measured if material. **Gate:** no result is complete until an independent clean-process rerun or a documented inability to reproduce is recorded. The next action is continue, revise, narrow, or stop according to evidence—not whether the project name sounds promising.

## 12. Required ablations, robustness and failure taxonomy

### 12.1 Minimal attribution set

The selected proposer with no audit, relative-error audit, ordinary update norm bound and complement-aware bound must share optimizer semantics, basis-refresh information and training batches. Plot both the quality–time frontier and the quality–rank/refresh frontier. A higher acceptance rate alone is not learning value. Include a no-information schedule matched on development acceptance/rank statistics; do not use the exact confirmatory OAR schedule retroactively as a deployable competitor. Replaying that schedule afterward is a diagnostic only because it contains future information.

Separate (a) a better proposer, (b) a better decision rule, (c) more dense updates, (d) optimizer/schedule tuning and (e) kernel engineering. Share each generic improvement with relevant baselines and recompute the headline comparison. An oracle-error control separates inaccurate probing from an inadequate acceptance criterion. An exact-current-subspace oracle separates stale-basis failure from inherently high-rank useful information.

### 12.2 Boundary conditions that must be probed

Rank zero, full basis, zero incoming adjoint, rank deficiency, bias/no-bias, tied parameters, extreme scales, loss masking, variable microbatch token counts, noncontiguous tensors and gradient accumulation all receive unit tests. Flat or rapidly rotating spectra, low-energy important coordinates, stale momentum, extreme second moments, weight decay and clipping receive adversarial tests. Primary models avoid BatchNorm because replay would require restoring running statistics; any later model with stateful forward buffers must snapshot and restore those buffers too.

Checkpointing may recompute forwards and can duplicate hook side effects. Audit accumulators must be updated exactly once per intended backward contribution, with recomputation tests. A custom backward that returns the right weight shape but accidentally modifies input adjoints violates the main mechanism. Higher-order derivatives are out of scope initially; fail explicitly when requested rather than returning misleading values.

Sampling baselines must handle zero scores and nonuniform probabilities correctly. DropBP must actually remove backward computation, not multiply already-computed gradients by zero. Approximate methods must not receive a larger effective batch, more calibration tokens or a cheaper tokenizer silently. A rare-condition test without a task marker can become intrinsically ambiguous; the specified synthetic generator includes one to avoid that confound.

### 12.3 Failure categories and response

| Category | Typical observation | Correct response |
|---|---|---|
| Implementation/measurement defect | Wrong input adjoint, duplicate update, uncharged fallback, asynchronous timing | Block dependent results, repair, rerun affected experiments; retain invalid logs. |
| Inadequate sensitivity | Wide intervals, too few rare cases, short horizon dominated by warmup | Recalculate precision/cost; add justified independent data or state inconclusive. |
| Resource/backend limitation | Required shape cannot fit; critical operation only slow CPU fallback | Narrow hardware claim or test a faithful alternative; not a universal mechanism rejection. |
| Structural hypothesis weakened | Future-batch oracle needs high rank or refresh every step | Reject current reuse assumption in that regime; diagnose basis/noise before a bounded pivot. |
| Audit hypothesis weakened | Exact residual audit no better than simpler schedules | Do not build a larger auditor; report absent information value. |
| Works but offers no advantage | Correct low-rank updates, savings consumed by overhead or baseline tuning | Negative practical result; preserve component findings without claiming acceleration. |
| Uncompensated trade-off | Better mean NLL/time but worse rare performance or stability | Report a conditional frontier; no unconditional improvement claim. |
| Inconsistent evidence | One favorable seed/workload, failed independent repeat | Treat as exploratory/unstable; no scale-up on that evidence alone. |

## 13. Reproducibility, logging and recovery

### 13.1 Required event schemas

Use append-only JSONL with schema versions. Core run records include:

```json
{
  "schema_version": 1,
  "run_id": "content-addressed-id",
  "phase": "exploratory_or_confirmatory",
  "method": "oar",
  "config_sha256": "...",
  "code_revision": "...",
  "logical_step": 0,
  "valid_tokens_total": 0,
  "processed_tokens_total": 0,
  "active_wall_s": 0.0,
  "decision": "accept_approx",
  "a_raw_gradient_dot_displacement": 0.0,
  "B_residual_margin": 0.0,
  "model_rng_state_id": "...",
  "audit_counter": 0,
  "checkpoint_id": null,
  "status": "committed"
}
```

The example is a schema illustration, not an actual measurement. Per-layer records include parameter ID, shape, rank, basis age/hash, probe count/seed, delta allocation, residual estimate/bound, numerical guardband, complement norm, ordinary norm, actual-error fields when paid audits exist and component timings from separate profiling runs. Memory records specify tensor/driver/RSS bytes and system-pressure context. Evaluation records include split/article-population hash, scored-token count, summed NLL, normalized NLL, rare-slice counts, checkpoint ID, inclusive training time and evaluation time.

Raw training losses use token-weighted aggregation; do not average batch means with unequal token counts. Use explicit missing/null values for measurements not taken, not zero. Record failed runs with exception, resource state, last valid checkpoint and cost consumed. Retain all attempted configurations and predeclared invalidation reasons.

### 13.2 Checkpoint content and atomicity

Save model parameters and forward buffers, optimizer state, scheduler/token counters, current bases/ranks/refresh schedule, controller streaks, confidence-allocation counter, all RNG states, data-shuffle epoch/cursor and replay descriptor, tokenizer/config/source hashes, committed step ID and accumulated active time. Include schema version and a checksum. Write a temporary file, flush and atomically rename; retain at least one prior valid checkpoint. Do not overwrite the only good state in place.

Resume only after checking hashes and shapes. A changed batch, tokenizer, data order, precision, optimizer, confidence budget or controller is a new run or explicit amendment, not a transparent resume. If interruption occurred in a tentative update, discard it and restart from the last committed checkpoint; do not mix new moments with old parameters. Preserve audit counter monotonicity for committed/randomness-consumed events according to the checkpoint transaction design; the safe default is to journal allocated confidence indices before execution and never reuse them after a crash. If exact deterministic replay is unavailable on the installed backend, quantify drift and classify reproducibility accordingly.

The agent may make routine non-destructive engineering decisions—filenames, local caching and refactoring—without approval, provided semantics and provenance are preserved. Changes to hypothesis, objective, evaluation population, baselines, primary metric, information access, numerical guarantee or comparison scope require an entry in `decisions/amendments.jsonl` and a new protocol hash before dependent experiments.

### 13.3 Progress and reporting

At every gate write what ran, elapsed/forecast resources, outputs, blockers, interpretation and next action into `decisions/progress.md`. A future process must resume from those artifacts without chat memory. Publish both mean effects and individual runs, including unfavorable seeds, runtime/memory regressions and blocked baselines. Classify novelty as replication, engineering, new application, empirical finding or plausible methodological difference. Do not promote the strongest label without evidence.

## 14. Interpretation and evidence-triggered alternatives

### 14.1 Outcome decision table

| Evidence | Permitted conclusion | Next action |
|---|---|---|
| Correct mathematics only | The specified identities and conservative real-arithmetic bound are implemented correctly in tested cases. | Complete local compatibility/headroom gates; no training claim. |
| Positive component timing, no learning evidence | The component is faster at reported shapes/backend. | Run matched learning experiments; do not extrapolate full-step gains. |
| Better fixed-time quality only against weak/untuned baseline | Exploratory indication, not competitive evidence. | Strengthen baseline and repeat development selection. |
| Better fixed-time quality versus strongest local baseline and matched controls, guardrails intact | Scope-limited learning-efficiency improvement on tested workloads/hardware. | Independent confirmation and informative scale test. |
| Faster step but slower/equal time-to-quality | Computational saving with no demonstrated practical learning advantage. | Diagnose update quality or stop practical branch. |
| Better reconstruction but no benefit versus fixed projection | Audit information did not improve this frontier. | Prefer the simpler method; retain negative audit result. |
| No useful past-only oracle subspace | Current reuse hypothesis unsupported in that regime. | Test a specifically motivated basis alternative or abandon this proposer. |
| Positive smaller/wider scaling trend with local bottleneck explanation | A scale-dependent prediction worth testing, not a frontier-model result. | Escalate only within measured capacity; state external scale remains unverified. |
| Confidence intervals include meaningful benefit and harm | Inconclusive, not equivalence and not failure proof. | Improve justified precision or narrow the conclusion. |

A genuine small improvement is worth reporting when distinguishable from noise and accompanied by absolute costs. A statistically detectable but operationally negligible effect is not a breakthrough. No fixed percentage is required for scientific interest; usefulness depends on complexity, deployment cost and regression risk.

### 14.2 Pivot B — audited proxy/control-variate gradients

Trigger this branch only if the gradient-information audit finds a compact past-only representation with small **per-sequence residual variance**, and a measured cost calculation beats both OAR and simply using a smaller exact batch. The established estimator is

\[
\widehat g=\frac1N\sum_{i=1}^N p_i+\frac1k\sum_{i\in S}(g_i-p_i),
\]
where \(S\) is a uniformly sampled size-\(k\) subset without replacement, selected after proxies are fixed. For residuals \(r_i=g_i-p_i\),
\[
\mathbb E\|\widehat g-g\|^2=
\frac{1-k/N}{k}\frac1{N-1}\sum_i\|r_i-\bar r\|^2.
\]
This is standard finite-population control-variate logic, not a proposed new theorem. Sequences, not dependent tokens, are the sampling units. If sampling becomes nonuniform, derive the appropriate inclusion-probability estimator before implementation; do not reuse this formula.

First use oracle coefficients in a **past-only** dictionary on future held-out batches. Compare residual variance and actual short-rollout learning with a smaller exact batch at equal cost. Only if that bound is favorable, train a cheap coefficient predictor from forward features available at runtime. A dense dictionary of K directions costs O(PK) storage and expansion; use explicitly factored per-layer bases and measure costs. Training the predictor and exact calibration is charged. Unbiased gradients do not imply unbiased adaptive-optimizer displacements. Physically skip unsampled backward work, including recomputation cost for selected examples. Failed variance/cost gates stop this branch before constructing a predictor.

### 14.3 Pivot C — reusable gradient instruction dictionary

Trigger only when future-batch **oracle** dictionary updates preserve learning across representation changes, not merely high gradient energy. Fit a dictionary from earlier exact gradients; freeze it for a held-out temporal block; compare optimal coefficients, learned coefficients, stale gradients, fixed random subspaces and periodically refreshed subspaces. Evaluate actual update progress and new-pattern acquisition. DOME and older subspace observations warn against equating dominant variance with useful learning [R11], [R27].

A positive oracle test is necessary, not sufficient. Selecting coefficients must be cheaper than obtaining them through exact backward differentiation; an oracle selecting them using the current full gradient is not a deployable saving. Track dictionary refresh, factor expansion, state memory and rank growth. If a predictor fails but oracle succeeds, the remaining research problem is coefficient information, not proof that gradient grammar works. A successful dictionary may reduce the frequency of full backward passes, but whole-forward cost still limits step throughput.

### 14.4 Pivot D — architecture/credit-channel co-design

This branch is justified when the scientific objective explicitly changes to learning mechanisms/architectures or when measured credit-signal structure supports a testable co-design, not simply because OAR failed. State the new object of comparison. Construct a residual network with a learned low-dimensional feedback interface; train local update modules from occasional exact adjoints and compare with full BP on the **same forward architecture**, fixed random feedback, synthetic gradients and a bottlenecked exact-gradient control. Match forward capacity, optimizer/calibration expenditure and inference procedure. Test depth dependence, feature discovery, signal conditioning and stability before broad task benchmarks.

NoProp is relevant architecture-level precedent but trains block-local objectives and changes forward organization; it is not an equal-architecture drop-in baseline [R15]. Structured fast forward operators are another possible co-design direction, but require their own capacity and kernel study before a large-speedup claim. The local PoC can establish a feedback-dimensionality/learning trade-off or a component scaling law; it cannot establish general replacement of backpropagation or a 100× frontier-training result.

## 15. Audit record and remaining risks

These are self-audits of the plan using inspected evidence and executed small checks, not independent peer review or an executed M3 implementation audit.

### 15.1 Scientific and prior-art audit

**Defect found:** the brainstorm's broad mechanisms overlap extensively with known work. **Revision:** narrowed the candidate contribution to optimizer-displacement/complement-aware allocation, made simple adaptive compression a baseline, added recent low-rank-certification prior art and explicitly withheld novelty certification. **Residual risk:** an equivalent criterion may exist under another numerical-optimization terminology, in unpublished work or patents.

**Defect found:** gradient cosine and raw descent checks do not characterize an adaptive optimizer's actual step. **Revision:** actual AdamW preview, the demonstrated counterexample and raw-gradient/clipping conventions are required. **Residual risk:** first-order minibatch usefulness may still have little relation to generalization or finite updates.

**Defect found:** an ordinary residual-norm bound can be unnecessarily conservative. **Revision:** derived and numerically checked the orthogonal-complement refinement, while adding its extra GEMMs to cost accounting. **Residual risk:** it remains badly loose on some constructed displacements and may add no value on real states.

**Defect found:** claims about locally verified approximate adjoints can certify the wrong objective. **Revision:** exact input gradients are invariant; the wrong-adjoint example is a negative test. **Residual risk:** this deliberately limits speedup headroom and does not resolve full backward compression.

**Defect found:** arithmetic headroom was too easily confused with total training. **Revision:** corrected the 100× backward calculation, stated the 1.5× ideal weight-gradient-only ceiling and required actual time-to-quality evidence. **Residual risk:** scale-dependent learning efficiency remains unknown.

### 15.2 Experimental-validity audit

**Defect found:** oracle/current-gradient information and matched schedule replays can leak future information into a purported cheap method. **Revision:** past-only bases, explicit paid oracles, frozen development schedules and separate diagnostic replay labels. **Residual risk:** careful state/provenance checks are still needed in code.

**Defect found:** probe-dependent rank selection and repeated testing invalidate naive confidence claims. **Revision:** next-step-only adaptation, independent streams and persisted run-wide confidence allocations. Recent independent-family reuse is a separately labeled extension, not presumed valid. **Residual risk:** conservative allocations may be practically unusable.

**Defect found:** real-arithmetic concentration is not a finite-precision certificate. **Revision:** explicit numerical uncertainty, orthogonality tests, empirical-guardband labeling and rejection of near-uncertain margins. **Residual risk:** formal floating-point certification is not supplied by this plan.

**Defect found:** a slower custom dense baseline, unequal tuning or ignoring checkpointed BP can manufacture a gain. **Revision:** native baseline remains headline comparator; contemporary optimizer, checkpointed and nearest-art gates are required. **Residual risk:** some strongest published kernels cannot be faithfully reproduced on the laptop; claims must remain limited.

**Defect found:** dataset family overlap, license metadata inconsistency and repeated validation selection could overstate generality. **Revision:** archive notices, use train-only development, protect official evaluation, do not call WikiText-2/103 independent datasets, and require a second workload for broader claims. **Residual risk:** corpus-level generality and the original license provenance need execution-time resolution.

**Defect found:** fixed-time endpoints, target nonattainment and repeated timings can create subtle selection/statistical errors. **Revision:** frozen checkpoint/budget policy, explicit censoring, run-level pairing and fixed sample sizes based on a pilot. **Residual risk:** a tiny effect may remain too expensive to resolve locally.

### 15.3 Implementation and feasibility audit

**Defect found:** published code contains device-specific assumptions, and M3 unified memory is unspecified. **Revision:** actual hardware discovery, per-operation tests, explicit transfer accounting, safety memory model and faithful fallback rules precede long runs. **Residual risk:** actual performance/compatibility is unmeasured.

**Defect found:** an apparently straightforward nearest-art port would omit INSTANT's energy rescaling, extra calibration passes and source-helper indexing behavior. **Revision:** inspected the callback/SVD helper, specified the literal behavior and a separately labeled paper-intent variant, and charged extra data exposure. **Residual risk:** the complete port still requires source-equivalence tests and zero-energy handling; no author benchmark was executed.

**Defect found:** late rank changes could require keeping all incoming adjoints; rejection could double-update moments or consume data twice. **Revision:** aggregate small sketches during backward, adapt next step, preview transactionally and replay from original state. **Residual risk:** checkpointing hooks and RNG restoration can still be implementation traps; they are blocking tests.

**Defect found:** gradient accumulation and clipping can change the quantity audited. **Revision:** common probes for correctly weighted aggregate residuals, one layer test per effective batch, raw gradients preserved through optimizer clipping. **Residual risk:** variable token masks and tied parameters require explicit fixtures.

**Defect found:** interruption can reset confidence counters or mix parameter/optimizer versions. **Revision:** committed checkpoints, allocation journal, atomic writes and checksum/version checks. **Residual risk:** bitwise MPS reproducibility may not be available; disclose numerical reproducibility rather than promising identical bits.

A final challenge pass must be repeated against the implemented repository before confirmation. It must test: no available MPS kernel; low memory; missing corpus; all approximate attempts rejected; nonzero dropout; zero gradients with weight decay; duplicate/shared parameters; rare patterns outside the basis; ordinary SGD where auditing is redundant; Muon outperforming both AdamW variants; positive synthetic but negative real-task results; and a resumed run whose RNG/counter state differs. A passing plan audit does not mean those implementation tests have already passed.

## 16. References and search-coverage record

References identify the material actually used; an inspection label distinguishes full-text/code inspection from metadata-only context. Access date for live documentation and repositories: **14 October 2026**. Stable paper identifiers are preferred to search snippets. Repository branches remain mutable; execution must pin a commit and archive any patch.

### Bibliography and evidence ledger

**R1.** Tuan-Kiet Doan, Trung-Hieu Tran, Enzo Tartaglione, Nikola Simidjievski and Van-Tam Nguyen. *INSTANT: Compressing Gradients and Activations for Resource-Efficient Training.* ICLR 2026. [Official paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/d368aba36f74776cc7a1079332a31973-Paper-Conference.pdf). Inspected full-text method/calibration descriptions and rendered timing appendix pages 25–26. Supports the nearest-method comparison and FLOPs-versus-runtime caveat; not evidence of M3 speed. Repository inspection is separately recorded in R28.

**R2.** Ziteng Wang, Jianfei Chen and Jun Zhu. *Efficient Backpropagation with Variance-Controlled Adaptive Sampling.* ICLR 2024; arXiv:2402.17227. [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/56e8943ecd44b67bbea2702b9117b858-Paper-Conference.pdf); [stable record](https://arxiv.org/abs/2402.17227). Inspected method text, rendered algorithm page and reported experimental scope. Establishes adaptive unbiased data/token sampling as close prior art. Published FLOP reductions are not imported as laptop time savings.

**R3.** Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu and Radu Marculescu. *Efficient Low-rank Backpropagation for Vision Transformer Adaptation.* NeurIPS 2023; arXiv:2309.15275. [Record](https://arxiv.org/abs/2309.15275). Inspected author abstract/metadata for the Walsh–Hadamard low-rank mechanism and adaptation scope; a faithful M3 implementation was not inspected or executed. Contextual closest vision method, not a locally reproduced result.

**R4.** Daniel Bershatsky, Aleksandr Mikhalev, Alexandr Katrutsa, Julia Gusak, Daniil Merkulov and Ivan Oseledets. *Memory-Efficient Backpropagation through Large Linear Layers.* arXiv:2201.13195. [Record](https://arxiv.org/abs/2201.13195); [inspected full-text version](https://arxiv.org/html/2201.13195v3). Supports randomized approximate weight gradients, activation storage and variance considerations. Its memory objective is distinct from OAR's primary fixed-time objective.

**R5.** Killian Bakong, Laurent Massoulié, Edouard Oyallon and Kevin Scaman. *Unbiased Approximate Vector-Jacobian Products for Efficient Backpropagation.* February 2026; arXiv:2602.14701. [Record](https://arxiv.org/abs/2602.14701); [full text](https://arxiv.org/html/2602.14701v1). Inspected estimator/optimality discussion and experiment/code-availability description. Directly limits novelty claims about unbiased sparse/low-rank VJPs; no author implementation was executed.

**R6.** Harbir Antil and Deepanshu Verma. *Randomized Matrix Sketching for Neural Network Training and Gradient Monitoring.* October 2025; arXiv:2510.00442. [Record](https://arxiv.org/abs/2510.00442); [full text](https://arxiv.org/html/2510.00442v1). Inspected adaptive sketching, reconstruction and monitoring portions. Supports mechanism overlap, not a general wall-time advantage for sketched training.

**R7.** Sunghyeon Woo, Baeseong Park, Byeongwook Kim, Minjung Jo, Se Jung Kwon, Dongsuk Jeon and Dongsoo Lee. *DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation.* NeurIPS 2024; arXiv:2402.17812. [Official proceedings record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2402.17812). Inspected method/experimental summary and the operator in R30. Fine-tuning evidence does not establish scratch-training behavior.

**R8.** Jiawei Zhao and colleagues. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* 2024; arXiv:2403.03507. [Record](https://arxiv.org/abs/2403.03507). Inspected paper record and mechanism description. Establishes a strong low-rank optimizer-state alternative; it should not be confused with avoiding the original dense weight-gradient computation.

**R9.** Thijs Vogels, Sai Praneeth Karimireddy and Martin Jaggi. *PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization.* NeurIPS 2019; arXiv:1905.13727. [Record](https://arxiv.org/abs/1905.13727). Inspected primary record for low-rank distributed gradient compression and reuse context. Communication reduction is not the M3 compute bottleneck being tested.

**R10.** Ilia Markov, Kaveh Alimohammadi, Elias Frantar and Dan Alistarh. *L-GreCo: Layerwise-Adaptive Gradient Compression for Efficient Data-Parallel Deep Learning.* MLSys 2024; earlier arXiv:2210.17357. [Official paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/9069a8976ff06f6443e7f4172990a580-Paper-Conference.pdf); [preprint record](https://arxiv.org/abs/2210.17357). Inspected the official first-page author/title metadata, problem/sensitivity formulation and rendered pages 1 and 3. Relevant error-budget allocation prior art; experiments concern distributed compression.

**R11.** Sheikh Shams Azam, Seyyedali Hosseinalipour, Qiang Qiu and Christopher Brinton. *Recycling Model Updates in Federated Learning: Are Gradient Subspaces Low-Rank?* ICLR 2022; arXiv:2202.00280. [Record](https://arxiv.org/abs/2202.00280). Inspected primary abstract/metadata. Supports temporal reuse as an existing idea, not its sufficiency for language-model pretraining.

**R12.** Max Jaderberg, Wojciech Marian Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, David Silver and Koray Kavukcuoglu. *Decoupled Neural Interfaces using Synthetic Gradients.* ICML 2017, PMLR 70:1627–1635. [Official record and paper](https://proceedings.mlr.press/v70/jaderberg17a.html). Inspected official mechanism summary. Establishes learned gradient prediction and local decoupling as old concepts.

**R13.** Kang Liu and Bohao Qu. *Information Limits of Low-Rank Approximation Certification.* 2 October 2026; arXiv:2610.03321. [Record/full-text rendering](https://arxiv.org/abs/2610.03321); [PDF](https://arxiv.org/pdf/2610.03321). Inspected problem definitions and single-candidate/independent-family reuse statements in Sections 1–4; PDF rendering attempt timed out. This very recent preprint is important novelty context, not independently validated theory or a neural-training benchmark. Its specific minimax results are not assumed in OAR's elementary bound.

**R14.** Raghu Bollapragada, Richard H. Byrd and Jorge Nocedal. *Adaptive Sampling Strategies for Stochastic Optimization.* SIAM Journal on Optimization 28(4):3312–3343, 2018; arXiv:1710.11258. [DOI](https://doi.org/10.1137/17M1154679); [author preprint](https://optimization-online.org/wp-content/uploads/2017/10/6305.pdf). Inspected inner-product-test section and rendered page 4. Establishes antecedents for controlling gradient information by descent value rather than only norm accuracy.

**R15.** Qinyu Li, Yee Whye Teh and Razvan Pascanu. *NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation.* CoLLAs 2025; arXiv:2503.24322v2. [Paper](https://arxiv.org/pdf/2503.24322). Inspected local-denoising objective, architecture and scope. Uses local differentiation inside blocks; it is not evidence that all differentiation or ordinary transformer backward computation has been eliminated.

**R16.** Kunjal Panchal, Sunav Choudhary, Yuriy Brun and Hui Guan. *Memory Savings at What Cost? A Study of Alternatives to Backpropagation.* ICML 2026; arXiv:2506.21833v2, revised 10 July 2026. Earlier version titled *The Cost of Avoiding Backpropagation*. [Record](https://arxiv.org/abs/2506.21833); [full text](https://arxiv.org/html/2506.21833v2). Inspected comparison and scope. Supports strong checkpointed-BP baselines and caution about forward-mode/zeroth-order claims; not a universal impossibility result.

**R17.** Nathan Halko, Per-Gunnar Martinsson and Joel A. Tropp. *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.* SIAM Review, 2011; arXiv:0909.4061. [Record](https://arxiv.org/abs/0909.4061). Foundational randomized range-finding/error-estimation reference; primary record inspected. The specific contractions and bound used here are provided and checked in this document rather than outsourced to the survey.

**R18.** Haim Avron and Sivan Toledo. *Randomized Algorithms for Estimating the Trace of an Implicit Symmetric Positive Semi-definite Matrix.* Journal of the ACM, 2011. [DOI](https://doi.org/10.1145/1944345.1944349). Bibliographic/primary-source discovery only; publisher full text was not used as an uninspected source of a specific theorem. Included as foundational trace-estimation context; OAR's confidence inequality is derived in Section 4.4.

**R19.** PyTorch contributors. *MPS backend notes, MPS APIs and MPS environment variables.* Documentation inspected on 14 October 2026; resolved documentation paths labeled version 2.14. [Backend notes](https://docs.pytorch.org/docs/2.14/notes/mps.html); [MPS API](https://docs.pytorch.org/docs/2.14/mps.html); [environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html). Inspected availability, synchronization, memory and RNG/profiling interfaces. Documentation is not an executed compatibility test or proof that the same release wheel is installed.

**R20.** Apple MLX contributors. *Custom function API.* [Official documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html). Inspected custom VJP/JVP registration documentation. Establishes a possible faithful fallback path; no MLX/M3 implementation was executed.

**R21.** PyTorch contributors. *AdamW and Muon optimizer documentation.* [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html); [Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html). Inspected algorithm/API and parameter-scope guidance. Supports optimizer selection/preview semantics; actual backend behavior must be verified. Do not treat all 2D parameters, including embeddings, as automatically eligible for Muon.

**R22.** Salesforce. *WikiText dataset repository/card.* [Dataset](https://huggingface.co/datasets/Salesforce/wikitext); [inspected card](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md). Inspected raw configurations, sizes, split descriptions and inconsistent license labels. Original creator blog access failed during research. Archive the execution revision; do not infer redistribution permission from a dataset-loader success.

**R23.** Zalando Research. *Fashion-MNIST official dataset repository.* [Repository](https://github.com/zalandoresearch/fashion-mnist). Inspected dataset description and repository license label. File checksums/notices must be verified at the pinned revision before execution. No dataset benchmark was run during this assignment.

**R24.** Patent-publication landscape examples, not legal conclusions: *Systems and Methods for Exchange of Data in Distributed Training of Machine Learning Algorithms*, [US20190156214A1](https://patents.google.com/patent/US20190156214A1/en), 2019; and *A Multi-level Fine-grained Gradient Compression Method for Distributed Learning of Neural Networks*, [CN121413675A](https://patents.google.com/patent/CN121413675A/en), 2026. Inspected publication descriptions for compression, residual feedback and distributed-training overlap. No exhaustive family/claim/status analysis or freedom-to-operate opinion was performed.

**R25.** Xi Chen, Kaituo Feng, Changsheng Li, Xunhao Lai, Xiangyu Yue, Ye Yuan and Guoren Wang. *Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?* NeurIPS 2025; arXiv:2410.01623v3. [Record](https://arxiv.org/abs/2410.01623). Inspected primary abstract/metadata. Relevant residual-gradient and optimizer-scaling prior art, not an implementation reproduced here.

**R26.** Thien Hang Nguyen and Huy Nguyen. *Lean and Mean Adaptive Optimization via Subset-Norm and Subspace-Momentum with Convergence Guarantees.* ICML 2025, PMLR 267:46116–46161. [Official record](https://proceedings.mlr.press/v267/nguyen25f.html). Inspected primary mechanism summary and scope. Shows optimizer-state/subspace/complement design is not novel territory; no reported large-model number is used as a local comparison.

**R27.** Julien Nicolas, Mohamed Maouche, Sonia Ben Mokhtar and Mark Coates. *DOME: Improving Signal-to-Noise in Stochastic Gradient Descent via Sharp-Direction Subspace Filtering.* arXiv:2507.03545v2, February 2026. [Full text](https://arxiv.org/html/2507.03545v2). Inspected subspace-filtering mechanism and stated assumptions. Relevant caution against equating dominant gradient variance with useful updates; its broader theoretical interpretations were not independently verified.

**R28.** INSTANT authors. *Official implementation.* [Repository](https://github.com/hieu-trannn/INSTANT); [inspected operator](https://github.com/hieu-trannn/INSTANT/blob/main/language/custom_op.py). README/license label, language directory, full operator, calibration callback and SVD helper inspected; file-blob SHAs recorded in Section 3.2. No complete repository clone or author training run was executed. Portability conclusions are limited to inspected code.

**R29.** Tsinghua University Machine Learning Group / VCAS authors. *Official VCAS implementation.* [Repository](https://github.com/thu-ml/VCAS); [inspected linear layer](https://github.com/thu-ml/VCAS/blob/main/vcas/layers/linear.py). README, layer directory and full linear operator inspected; blob SHA recorded above. Sampling utility/controller behavior still requires a faithful execution-time source audit before a complete named-baseline claim.

**R30.** DropBP authors. *Official DropBP implementation.* [Repository](https://github.com/WooSunghyeon/dropbp); [inspected layer](https://github.com/WooSunghyeon/dropbp/blob/main/dropbp/layer.py). README and full detach operator inspected; blob SHA recorded above. A detached residual branch has different graph behavior from a dense gradient mask.

**R31.** Guy Gur-Ari, Daniel A. Roberts and Ethan Dyer. *Gradient Descent Happens in a Tiny Subspace.* 2018; arXiv:1812.04754. [Record](https://arxiv.org/abs/1812.04754). Primary record inspected as historical context for gradient-subspace observations. Classification evidence does not establish a small sufficient update dictionary for language-model pretraining.

### Search scope, stopping rationale and blind spots

Searches covered approximate backpropagation, randomized matrix multiplication, activation sketching, low-rank VJPs, adaptive rank/error allocation, sample-and-rescale estimators, variance-controlled sampling, control variates, synthetic gradients, temporal gradient reuse, low-rank optimizer states, optimizer residuals/orthogonal complements, inexact-descent tests, trace estimation, low-rank certification, local credit assignment, negative comparisons and Apple-Silicon operator support. Backward tracing used references in INSTANT/VCAS and numerical-linear-algebra papers; forward/current tracing found 2025–2026 VJP, sketching, certification and optimizer work. Exact title and mechanism searches were supplemented with public code and patent-publication searches.

The final targeted refinement search included combinations of “gradient compression,” “orthogonal complement,” “Adam,” “descent,” “residual,” “verification” and “preconditioner.” It found additional optimizer/subspace precedents, not a verified duplicate of the exact OAR execution rule. This is **unresolved differentiation**, not proof of no prior art. The recent certification work materially narrowed the novelty assessment. Additional related hits after that refinement primarily concerned memory-state optimization, distributed communication or different constraints rather than changing the selected local test. This is a practical coverage stopping point, not a paper quota or exhaustive census.

Patent searches included gradient compression, adaptive residual checking, backward verification and distributed-training families. The two publication examples in R24 illustrate overlapping terrain; many “Freivalds” hits were author-name matches rather than the desired verification mechanism. No legal conclusion follows. A professional claims/family search would be a separate requirement before an IP-dependent commercialization decision.

Access limits: some OpenReview pages were blocked; official proceedings supplied the important INSTANT/VCAS papers. A direct container request to GitHub failed network resolution, but subsequent connector reads successfully inspected the named files. The original WikiText blog was unavailable. One very recent matrix-free low-rank reference, arXiv:2607.06758, could not be fetched and was not used to substantiate a claim. Some sources were inspected at abstract level and are explicitly labeled as such; their detailed experimental claims were not imported. The mathematical checks did not execute on Apple hardware. Very recent preprints, non-English literature, unindexed implementations and patents remain blind spots.

## Appendix A. Reproducible CPU reference checks

The first script below is the mathematical validation program executed during this assignment, reformatted only for inclusion. Save it as `validate_math.py` in the project, run with the locked NumPy environment and redirect stdout to a JSON file. The command is an ordinary Python invocation once that file is created:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_math.py > math_results.json
```

Exact last-bit results can differ across BLAS builds; compare the identities and tolerances, not an unqualified byte-for-byte JSON match. This is not an MPS performance benchmark.

```python
"""Small mathematical checks; not a learning benchmark or an M3 test."""
import itertools
import json
import math
import platform
import sys
import numpy as np

rng = np.random.default_rng(20261014)
out = {"environment": {"platform": platform.platform(), "python":sys.version.split()[0], "numpy":np.__version__, "device":"CPU"}}

def lower_fraction(k, delta):
    """Solve c - 1 - log(c) = 2 log(1/delta)/k for c in (0,1)."""
    if k <= 0 or not 0 < delta < 1:
        raise ValueError("invalid confidence arguments")
    target = 2.0 * math.log(1.0 / delta) / k
    lo, hi = -max(2.0, target + 2.0), 0.0
    for _ in range(100):
        y = (lo + hi) / 2
        if math.exp(y) - 1 - y > target:
            lo = y
        else:
            hi = y
    return math.exp((lo + hi)/2)

n, di, do, r, k = 97, 31, 43, 8, 16
X, D = rng.standard_normal((n,di)), rng.standard_normal((n,do))
G = D.T @ X
U = np.linalg.qr(rng.standard_normal((do,r)))[0]
H = (D @ U).T @ X
Gh = U @ H
Z = rng.standard_normal((di,k))
R = D.T @ (X @ Z) - U @ (H @ Z)
out["matrix_free_residual_relative_error"] = float(np.linalg.norm(R-(G-Gh)@Z)/np.linalg.norm((G-Gh)@Z))
out["projection_descent_identity_relative_error"] = float(abs(np.sum(G*Gh)-np.sum(Gh*Gh))/np.sum(Gh*Gh))
# A full-rank left basis recovers G, including rectangular shapes.
Ufull = np.linalg.qr(rng.standard_normal((do,do)))[0]
out["full_basis_gradient_relative_error"] = float(np.linalg.norm(Ufull@((D@Ufull).T@X)-G)/np.linalg.norm(G))
# Exact equality for a generic low-rank direction, not only an orthogonal projection.
V = rng.standard_normal((di,r))
out["directional_contraction_relative_error"] = float(abs(np.sum(G*(U@V.T))-np.sum((D@U)*(X@V)))/max(1,abs(np.sum(G*(U@V.T)))))
# An unbiased audit norm estimate and its conservative one-sided bound.
reps, q, alpha = 50000, 16, 0.05
c = lower_fraction(q, alpha)
out["probe_mc"] = []
for name, eig in [("rank1",np.array([1.])),("flat_rank8",np.ones(8)/8), ("decaying_rank8",2.0**(-np.arange(8)))]:
    eig=eig/eig.sum()
    z=rng.standard_normal((reps,q,len(eig)))
    s=(z*z*eig).sum(axis=(1,2))/q
    out["probe_mc"].append({"spectrum":name,"replicates":reps,"q":q,"delta":alpha,"c":c,"mean_estimated_norm_squared":float(s.mean()),"true_norm_squared":1.0,"upper_bound_failure_rate":float(np.mean(s/c<1))})
out["confidence_factors"]=[{"q":q,"delta":a,"c":lower_fraction(q,a),"norm_multiplier":1/math.sqrt(lower_fraction(q,a))} for q in [8,16,32] for a in [0.01,1e-6]]
# Proxy correction: exhaustive finite-population unbiasedness and covariance trace.
N, sample_k, p = 8, 2, 5
exact=rng.standard_normal((N,p)); proxy=exact+0.4*rng.standard_normal((N,p))
ests=np.array([proxy.mean(0)+(exact[list(S)]-proxy[list(S)]).mean(0) for S in itertools.combinations(range(N),sample_k)])
residual=exact-proxy; Rcenter=residual-residual.mean(0)
formula=(1-sample_k/N)/sample_k*np.sum(Rcenter*Rcenter)/(N-1)
out["proxy_correction"]={"N":N,"k":sample_k,"subsets":len(ests),"mean_gradient_error":float(np.linalg.norm(ests.mean(0)-exact.mean(0))),"variance_measured":float(np.mean(np.sum((ests-exact.mean(0))**2,axis=1))),"variance_formula":float(formula)}
# Raw-gradient agreement can conceal a harmful actual Adam direction.
g=np.array([1.,0.02]); gh=np.array([1.,-0.02]); m=np.zeros(2); v=np.array([1e4,1e-8])
b1,b2,t,eps=0.9,0.999,100,1e-8
mp=b1*m+(1-b1)*gh; vp=b2*v+(1-b2)*gh**2
d=(mp/(1-b1**t))/(np.sqrt(vp/(1-b2**t))+eps)
out["adam_counterexample"]={"true_gradient":g.tolist(),"approx_gradient":gh.tolist(),"cosine":float(g@gh/(np.linalg.norm(g)*np.linalg.norm(gh))),"raw_alignment":float(g@gh),"subtract_direction":d.tolist(),"true_directional_alignment":float(g@d),"approx_directional_alignment":float(gh@d),"exact_residual_norm_bound_margin":float(gh@d-np.linalg.norm(g-gh)*np.linalg.norm(d))}
# A verifier using corrupted incoming adjoints can return zero error for -G.
Gbad=(-D).T@X
out["wrong_adjoint_counterexample"]={"local_probe_error":float(np.linalg.norm((-D).T@(X@Z)-Gbad@Z)),"alignment_with_true_gradient":float(np.sum(G*Gbad))}
out["amdahl"]={"backward_100x_step_speedup":3/(1+2/100),"weight_gradient_eliminated_speedup":1.5,"all_backward_eliminated_speedup":3.0}
print(json.dumps(out,indent=2))
```

The complementary script checks the main refinement and effective-batch accumulation. The executed version wrote an additional local JSON artifact; this portable version prints the same result object rather than assuming a sandbox path. Save it as `validate_complement.py` and redirect stdout as above.

```python
import json, numpy as np
rng=np.random.default_rng(20261015)
X=rng.normal(size=(97,31)); D=rng.normal(size=(97,43))
U=np.linalg.qr(rng.normal(size=(43,8)))[0]
G=D.T@X; H=(D@U).T@X; Gh=U@H; E=G-Gh
norm=np.linalg.norm
out={}
for name,d in [('random',rng.normal(size=G.shape)),('projected_sgd',0.01*Gh),('mostly_in_subspace',Gh+0.01*rng.normal(size=G.shape))]:
    dp=d-U@(U.T@d)
    actual=float(np.vdot(E,d)); reduced=float(np.vdot(E,dp))
    old=norm(E)*norm(d); new=norm(E)*norm(dp)
    out[name]={'identity_absolute_error':abs(actual-reduced),'actual_error_alignment':actual,
       'old_cauchy_bound':float(old),'complement_cauchy_bound':float(new),
       'bound_ratio':float(new/old),'true_alignment':float(np.vdot(G,d)),
       'lower_bound_exact_residual':float(np.vdot(Gh,d)-new)}
# Identical probes across accumulated microbatches estimate the aggregate, not sum of norms.
Z=rng.normal(size=(31,16)); R=np.zeros((43,16)); Ha=np.zeros((8,31))
for idx in np.array_split(np.arange(97),7):
    h=(D[idx]@U).T@X[idx]; Ha+=h
    R+=D[idx].T@(X[idx]@Z)-U@(h@Z)
out['accumulation']={'relative_residual_error':float(norm(R-E@Z)/norm(E@Z)),
                     'relative_H_error':float(norm(Ha-H)/norm(H))}
print(json.dumps(out,indent=2))
```

## Appendix B. Completion checklist for the coding agent

The handoff is executable only through the gates, not by assuming positive results. Before claiming completion, verify that the artifact set contains the actual hardware/backend record; pinned code/dependencies/data; operator and transaction tests; measured full-cost timing; tuned native and nearest-art baselines or documented exclusions; future-held-out oracle tests; a frozen confirmatory protocol; all successful and failed runs; uncertainty and guardrail analyses; and a clean-process reproduction attempt.

The report's opening paragraph must state exactly what improved, compared with what, on which workload, at what quality/time/memory trade-off, with what uncertainty, and whether the evidence is exploratory or confirmed. It must separately state what remains unverified: methodology novelty, broader hardware/model scaling, finite-step/global learning guarantees, rare real-world capabilities and any 10×–100× ambition.

**Default scientific decision:** prefer the simpler method when OAR does not demonstrate added value. Preserve a modest reproducible advance if it is real; preserve a negative or inconclusive result if that is what the evidence supports.


<!-- Stable Markdown reference destinations; see Section 16 for the evidence ledger. -->
[R1]: https://proceedings.iclr.cc/paper_files/paper/2026/file/d368aba36f74776cc7a1079332a31973-Paper-Conference.pdf
[R2]: https://proceedings.iclr.cc/paper_files/paper/2024/file/56e8943ecd44b67bbea2702b9117b858-Paper-Conference.pdf
[R3]: https://arxiv.org/abs/2309.15275
[R4]: https://arxiv.org/abs/2201.13195
[R5]: https://arxiv.org/abs/2602.14701
[R6]: https://arxiv.org/abs/2510.00442
[R7]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/240225294cdd2c9b692c2519d3278a08-Abstract-Conference.html
[R8]: https://arxiv.org/abs/2403.03507
[R9]: https://arxiv.org/abs/1905.13727
[R10]: https://proceedings.mlsys.org/paper_files/paper/2024/file/9069a8976ff06f6443e7f4172990a580-Paper-Conference.pdf
[R11]: https://arxiv.org/abs/2202.00280
[R12]: https://proceedings.mlr.press/v70/jaderberg17a.html
[R13]: https://arxiv.org/abs/2610.03321
[R14]: https://doi.org/10.1137/17M1154679
[R15]: https://arxiv.org/pdf/2503.24322
[R16]: https://arxiv.org/abs/2506.21833
[R17]: https://arxiv.org/abs/0909.4061
[R18]: https://doi.org/10.1145/1944345.1944349
[R19]: https://docs.pytorch.org/docs/2.14/notes/mps.html
[R20]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html
[R21]: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
[R22]: https://huggingface.co/datasets/Salesforce/wikitext
[R23]: https://github.com/zalandoresearch/fashion-mnist
[R24]: https://patents.google.com/patent/US20190156214A1/en
[R25]: https://arxiv.org/abs/2410.01623
[R26]: https://proceedings.mlr.press/v267/nguyen25f.html
[R27]: https://arxiv.org/html/2507.03545v2
[R28]: https://github.com/hieu-trannn/INSTANT
[R29]: https://github.com/thu-ml/VCAS
[R30]: https://github.com/WooSunghyeon/dropbp
[R31]: https://arxiv.org/abs/1812.04754
