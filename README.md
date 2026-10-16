# Seller AI Copilot — reproducibility supplement v2.0.0

This supplement (release v2.0.0) accompanies **Engineering Financial Trust in Seller AI Copilots** (manuscript v10, IEEE Software format, September 2026; the v9 manuscript is the long-form version of the same experiments)
(`Seller_AI_Copilot_IEEE_Software_v10.docx`). It contains synthetic experiments,
not production observations. It makes no external API calls and uses no private data.

## Run

Reference environment: Python 3.12.14, NumPy 2.3.5. Matplotlib is not required.

Verify the manifest **before** running: the runner rewrites `results.revision.json`,
so `manifest.sha256` matches only the package as distributed.

```sh
sha256sum -c manifest.sha256
python -m pip install -r requirements.txt
python run_revision.py
python verify_revision.py
```

The first command installs the numerical dependency in the environment chosen by
the reader. The runner writes `results.revision.json` and `primary_fraud_scores.npz`.
`results.reference.json` is the retained reference output. JSON output was compared
byte for byte across repeat runs in the reference environment. Other environments
are not claimed to produce bitwise-identical random draws or floating-point results.
When comparing a fresh run with `results.reference.json`, expect the `environment`
block to differ; it records the interpreter and NumPy versions actually used. All
other fields reproduced identically in testing, including on Python 3.13 with NumPy
pinned to 2.3.5.

## Files and provenance

- `evaluation_original.py`: unchanged copy of the local v7/v1.0.0 baseline source.
  Its historical `main()` contains the disputed arithmetic cascade row; do not use
  that row as evidence. The revision runner imports baseline functions only.
- `evaluation_controls.py`: extracted baseline generator and control functions with
  explicit noise scaling, an adjustable absolute floor, a promotional-credit
  consistency check before prior-rate suppression, and grouped-tie ROC/AUC.
- `run_revision.py`: all v9 experiments, complete per-seed output and primary scores.
- `verify_revision.py`: independent invariants, pairwise AUC checks and known-case
  replay checks. It also checks the subgroup and threshold findings.
- `results.revision.json`: current generated output; full metrics before formatting.
- `results.reference.json`: preserved output for checking reproduction.
- `primary_fraud_scores.npz`: cohort and universal scores, binary labels and groups.
- `revision_notes.md`: review-point map and methodological changes.
- `reference_verification.md`: source-resolution checks, including the three
  disputed 2026 preprints.
- `manifest.sha256`: file hashes for the package contents.

The original source came from `copilot-repro-package/evaluation_simulation.py` in
the author's project. Original data-generation order is preserved for primary E1,
E2 and E3: reconciliation, seller scores, then traces, using seeds 42 through 51.
The original source is retained to make the changed assumptions inspectable. The
v9 supplement has not been uploaded to the earlier GitHub/Zenodo release. A prior
release or DOI must not be presented as containing these revised experiments.

## Manuscript-to-output map

| Item (v9 long-form numbering) | Where it appears in v10 (IEEE Software) | JSON key |
|---|---|---|
| Table 3, primary class precision/recall | **Table 1** | `primary.legacy_e1`, `primary.corrected_e1` |
| Credit-miss causes and expected dollars | §"A suppression rule…", para 2 | `primary.promo_audit` |
| Ten-seed detection and macro-F1 | §"A suppression rule…", para 4 | `legacy_reconciliation`, `corrected_reconciliation` |
| Removal of prior-rate suppression | §"A suppression rule…", para 4 | `primary.no_prior_suppression` |
| Table 4, reconciliation noise | §"A suppression rule…", final para (prose) | `reconciliation_noise_sweep` |
| Table 5, absolute-floor sweep | §"A suppression rule…", final para (prose) | `reconciliation_floor_sweep` |
| Table 6, primary score thresholds and AUC | **Table 2** | `primary.fraud` |
| Table 7, subgroup denominators and errors | **Table 3** | `primary.fraud.*.groups` |
| Ten-seed ranking summary | Table 2, last two rows | `fraud_summary` |
| Original local-boundary diagnostic | §"Direct reads…", para 2 | `primary.legacy_boundary_check` |
| Table 8, independent boundary trials | §"Direct reads…", para 2 (prose) | `boundary_noise_sweep` |
| Table 9, executed policy replays | §"Direct reads…", para 2 (prose) | `execution_replay` |

v9 Tables 1–2 and Figures 1–4 (v10 Figure 1) are proposed architecture and policy illustrations;
they are not empirically calibrated results. All aggregate SD values are sample
SD over ten seeds (`ddof=1`). Means are computed on unrounded per-seed metrics.
The five injected types define the macro-F1 denominator. Primary per-class
precision/recall are formatted to three decimals by the preserved scorer.

## Experiment definitions

**Reconciliation.** The baseline rule suppresses a fee/ship signature compatible
with a prior rate. It is not an authoritative rule-version lookup. The corrected
suppression additionally requires `abs(actual_promo - expected_promo) <= 0.01`.
Noise scales 1, 2, 4 and 8 multiply the original legitimate-adjustment and both
observation-noise draws; labels, anomaly effects, relative threshold and typing
tolerances are fixed. Absolute floors 0.10, 0.05 and 0.01 are compared at scale 1.
The 0.5% relative floor is always retained. Prior-rate confounders are not
scaled; the residual bound in Table 4 excludes their deliberate rate shift.

**Fraud scores.** Identical synthetic sellers are scored using either the true
cohort mean or the mean of the six true cohort means. Both are oracle inputs.
Score thresholds 0.60 and 0.65 are diagnostic, not validation-selected policies.
ROC endpoints are constructed after an entire equal-score group is added.
Trapezoidal integration then gives tied positive-negative pairs half credit.
FPR at recall >= 0.85 uses the first attainable distinct-score ROC point,
without interpolation. This change removes within-tie sorting dependence;
it does not certify every aspect of cross-version RNG reproduction.

**Boundary trials.** 100,000 independent trials per seed use a separate seed
offset of 2000. Injection prevalence is 0.07; magnitudes are log-uniform 2–60%
with a random sign. Uniform signed noise of amplitude 1%, 3%, 5%, 7% or 10%
is added to every trial, including injected trials. Detection threshold is 5%.
All-error recall labels every injected case positive; material recall is
conditioned on injection magnitude >5% before observation noise is added.
For clean samples, theoretical FPR is max(0, 1 - 0.05/amplitude).

**Executed replay.** 2,500 new traces per seed use seed offset 1000. Each has
4–8 steps, a base value $500–$20,000 and transition factors uniform within
1 +/- 0.002. Reference values are computed independently of all observations.
Noise is multiplicative and accumulates along the unchecked trajectory. The
same exogenous factors, noise and injected errors are replayed for all policies.
The evaluator counts >5% erroneous receiving inputs after the designated
injection point, divided by all injected traces, including traces halted before
injection. This is an intention-to-treat policy denominator, not a count of
errors only among surviving traces. No claim is made that every counted error
was caused solely by the injection; benign noise also contributes.

Post-hoc execution does not intervene. Inline checks compare each outgoing value
to its independent reference and halt before the receiver executes. Direct-read
execution substitutes the corresponding reference at every receiving step. The
two zero-error outcomes under perfect reference access are invariants of those
policies, confirmed by counters rather than assigned constants. Non-injected
trace halts quantify interruption from accumulated benign noise. References
remain perfectly available and correct; access cost, failures and latency are
not modeled. Final narrated outputs are outside the receiving-input metric.

## Interpretation limits

These models do not run an LLM, learn a fraud model, estimate a cohort baseline,
send a payment, test a legal policy, demonstrate demographic fairness or establish
production effectiveness. The original zero-noise-overlap tests are labeled as
construction diagnostics. The new sweeps expose failure boundaries and should
travel with the paper when it is submitted. Publication of a revised repository
or archive remains a separate action.
