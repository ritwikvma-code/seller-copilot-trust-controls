# Revision map

The manuscript was revised from the supplied v7 Word file. The original Word
file and original source package were left intact. The revised paper is a
practitioner contribution with synthetic mechanism tests, not a claim of
production effectiveness. Editorial acceptance is not asserted.

| Review concern | Revision |
|---|---|
| “Roughly double” and threshold confounding | Abstract and §9.3 remove the superiority claim; Table 6 shows precision, recall, FPR and tie-aware AUC at both 0.60 and 0.65. Matched-recall FPR is explicitly defined. |
| Category fairness | Table 7 shows denominators, false flags, FPR and victim recall by category. The 58-to-zero fashion/shoes finding is retained with its 18/24-to-13/24 recall tradeoff. Residual false flags in other categories are explicit. |
| Perfect precision by construction | §9.2 states the $0.07 pre-rounding bound and suppressor assumptions. Table 4 tests 1×, 2×, 4× and 8× noise with ten seeds. Table 8 adds noise to both clean and injected boundaries. |
| Missing promotional credits | Direct code inspection identifies four threshold misses and **31 prior-rate suppression misses**, correcting the reviewer’s attribution of all 35 to the floor. A credit-component predicate recovers 31, raising primary credit recall from 0.602 to 0.955. Table 3 gives all anomaly classes; Table 5 varies the absolute floor. |
| Arithmetic cascade “ablation” | The old 0.00/2.43 comparison is no longer presented as a measured experiment. §9.4 identifies what it computes. Table 9 uses actually executed traces and an independent reference trajectory. |
| Perfect access makes inline verification redundant | §6 and Figure 4 prefer direct retrieval when available. Table 9 includes it explicitly; the discussion identifies the shared oracle assumption and interruption cost. No novel performance is claimed from zero-error invariants. |
| Hidden sub-threshold cascade misses | §9.4 reports both 140/140 material detection and 140/191 = 0.733 unconditional detection, including 0/51 below the 5% threshold. |
| Tie-sensitive AUC and exact reproduction | `evaluation_controls.py` groups tied scores. Tests compare it to pairwise AUC and permute ties. New results are recomputed before rounding, with environment-pinned reproduction claims only. |
| Thresholds and calibration | Table 1 is illustrative. §5 derives the binary cost threshold and discusses separate review costs and selected approval labels. Figures 2–3 place checks before commitment and remove the unsupported calibrated-posterior label. |
| Experience, affiliation and disclosure | Corporate affiliation is removed from the author line. At the author’s express direction, the paper identifies experience launching DoorDash’s MCA program in 2022, personal-capacity authorship and use of no proprietary data. It does not claim an employer’s funding or publication approval. |
| References [8], [9], [12] | All resolve on arXiv with matching titles/authors. §6 distinguishes the attenuation result in [8] from amplification studied in [9]. Reference [4] uses a single bibliographic form. |
| Positioning and existing tooling | Introduction and related work acknowledge existing Stripe/eBay reconciliation, add financial QA and group-conditional evaluation sources, and remove first-of-kind/production-effect assertions. |

## Items outside the evidence

The author should submit the supplement with the revised Word file. A public
release containing the v9 code has not been created. The earlier archive does
not substantiate the newly added measurements. Journal-specific formatting,
employer publication approval (if applicable) and a complete journal disclosure
form are separate from the authorized manuscript edit; none is represented as
already obtained.
