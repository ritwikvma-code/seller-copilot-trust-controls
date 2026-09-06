# Trustworthy AI Copilots for E-Commerce Sellers — Reproducibility Package

Reproducible simulation accompanying the paper *"Trustworthy AI Copilots for
E-Commerce Sellers: A Practitioner's Playbook for Engineering Financial Trust
on Managed Payment Platforms."*

This package regenerates **every quantitative value in Section 9** of the
paper (Tables 3 and 4) and the three result figures.

## Data statement

Fully synthetic. No proprietary, confidential, customer, or internal business
data of any company is used, referenced, or required. Every population in this
package is generated at runtime from a seeded random number generator; there
are no input data files.

## What is validated — and what is not

The simulation exercises the **deterministic control machinery** of Sections 5
and 6 (reconciliation, cohort-aware fraud baselines, inline boundary
verification). **No language model executes anywhere in this code.** The
generative layer of the reference architecture — intent classification,
orchestration, narration — is designed in the paper but is neither executed
nor measured here. Results are therefore evidence that the control machinery
behaves as specified, not evidence of production system performance.

## The three experiments

**E1 — Settlement reconciliation.** Generates 40,000 synthetic transactions
across six categories with distinct fee rates, then injects five anomaly types
(`duplicate_fee`, `missing_promo_credit`, `incorrect_category_rate`,
`erroneous_refund_deduction`, `shipping_miscalculation`) into ~3% of them.
Two benign confounders are injected alongside, and this is the point of the
experiment: 1.5% of transactions settle under the *prior* fee-schedule version
(+0.4pp), and 2% carry small legitimate adjustments; all transactions carry
±$0.02 observation noise. A naive detector reports every schedule-lag
transaction as an anomaly. The version-aware detector recomputes expected fees
from the current schedule, suppresses discrepancies fully explained by the
immediately prior version, and classifies the remainder by delta signature.

**E2 — Return-fraud baselines.** Generates 600 synthetic sellers across six
return-rate cohorts (fashion 28% down to industrial 3%), each with 8–15 SKUs.
For ~10% of sellers, a fraud ring of 2–4 buyer accounts is injected against one
target SKU, filing 5–15 fast claims (median ~4.5h vs. a ~55h legitimate
median). A composite score combines return-rate z-score (0.40), buyer
concentration (0.35), and claim-filing speed (0.25), gated by a minimum of 6
returns for SKU-level signal. The experiment compares scoring each seller
against **their own cohort's** baseline versus a **single universal** baseline.

**E3 — Cascade detection.** Generates 2,500 multi-agent traces of 4–8 steps,
propagating a financial value across agent boundaries. In 7% of traces a
hallucinated error of 2%–60% relative magnitude (log-uniform) is injected at a
non-terminal step. Clean transitions carry ±1% measurement noise. Inline
verification re-checks the value at each cross-agent boundary against the
system of record and halts when relative error exceeds the 5% materiality
threshold; the ablation compares this to post-hoc auditing after trace
completion.

**E4 — Ablations.** Re-runs E1 with version-aware suppression removed, and
reports the cohort-vs-universal and inline-vs-post-hoc comparisons on the
primary seed. Backs Table 4.

**E5 — Multi-seed robustness.** Re-runs E1–E3 across seeds 42–51 and reports
mean and sample standard deviation. Backs Table 3.

## Reproducing

```bash
pip install -r requirements.txt
python evaluation_simulation.py
```

Single run, no arguments. Completes in about 6 seconds on a 2020-era
laptop (Python 3.10, numpy 2.2.6). Writes `results.json` plus
`fig1_reconciliation.png`, `fig2_fraud_baselines.png`, and
`fig3_cascade.png`. The primary seed is 42; E5 sweeps 42–51.

To confirm the committed values are unchanged:

```bash
python -c "import json;a=json.load(open('results.json'));b=json.load(open('results.reference.json'));print('MATCH' if a==b else 'DIFFERS')"
```

`results.reference.json` is the committed output this package was verified
against. `arch_figures.py` separately renders the four architecture diagrams
(Figures 1–4); it is illustration code and computes no results.

## Mapping results.json to the paper

| Paper | Key in `results.json` |
|---|---|
| Table 3, precision / recall / macro-F1 | `E5_multiseed.det_precision`, `.det_recall`, `.macro_f1` |
| Table 3, fraud precision (cohort / universal) | `E5_multiseed.fraud_prec_cohort`, `.fraud_prec_universal` |
| Table 3, material cascade interception | `E5_multiseed.material_cascade_detection` |
| Table 3, false flags | `E5_multiseed.recon_false_flags`, `.cascade_false_flags` |
| Table 4, all rows | `E4_ablations` |
| Section 9 narrative, 2.4 downstream steps | `E3_cascade.mean_downstream_steps_halted` |

## Reading the cascade result correctly

`E5_multiseed.material_cascade_detection` is 1.000: every injected error
**above** the 5% materiality threshold is intercepted. The unconditional rate,
`E3_cascade.overall_detection_rate`, is **0.733**, because errors in the 2–5%
band are never flagged — 0 of 51 on the primary seed. This is by design, not
by failure: Section 6 defines the materiality threshold as a *declared risk
budget*, and the system formally accepts undetected errors below that line.
Both numbers are reported here so the distinction is auditable.

## Limitations

1. **Anomalies are injected from the same taxonomy the detector recognizes.**
   E1 recall is therefore an upper bound; it measures whether the detector
   catches what it was built to catch, not whether the taxonomy is complete.
2. **Cascade interception at 1.000 is partly by construction of the noise
   model** — clean transitions carry ±1% noise against a 5% threshold, so
   there is no overlap region. The experiment demonstrates that the mechanism
   works and quantifies what post-hoc auditing costs; it does not estimate a
   real-world interception rate.
3. **Cohort baselines remove the concentration of false accusations, not
   false accusations.** Cohort precision is ~0.78, so roughly one accusation
   in five is still wrong. This is why the paper routes fraud accusations
   through the escalation band rather than allowing auto-execution.
4. **Detector parameters are not tuned against held-out data.** Thresholds
   and signal weights are specified a priori from the design in Sections 5–6.
5. **No language model is evaluated.** See "What is validated" above.

## Files

- `evaluation_simulation.py` — the entire simulation (generators, detectors,
  scorers, ablations, multi-seed harness, figure rendering)
- `results.reference.json` — committed reference output (seed 42; E5 seeds 42–51)
- `arch_figures.py` — renders the four architecture diagrams used as Figures 1–4
- `fig1_reconciliation.png`, `fig2_fraud_baselines.png`, `fig3_cascade.png` —
  committed figures, regenerated by the run above
- `requirements.txt` — pinned dependencies
- `LICENSE` — MIT

## Verification record

This package was verified on 6 September 2026 by running
`evaluation_simulation.py` in a clean directory containing no prior outputs,
under Python 3.10.12 with numpy 2.2.6. The freshly generated `results.json`
was compared key-by-key against `results.reference.json` and matched
**exactly**, with no numerical drift in any field. Every value in the paper's
Tables 3 and 4 was then checked against that output and reconciled.

## License

MIT — see [LICENSE](LICENSE).
