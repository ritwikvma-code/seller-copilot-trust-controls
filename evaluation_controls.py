"""
Synthetic Evaluation Suite for the Seller AI Copilot Paper
===========================================================
Three experiments on fully synthetic data (no platform data used):

  E1. Settlement reconciliation: precision/recall of anomaly detection
      and classification across 5 injected anomaly types.
  E2. Return fraud detection: within-cohort baselines vs. a universal
      baseline (TPR / FPR comparison).
  E3. Financial error cascade detection (Algorithm 1): detection rate by
      injected error magnitude, false-flag rate, downstream steps saved.

Reproducible with fixed seed. Outputs: results.json + figures (PNG).
"""

import json
import math
import os
import random
from collections import defaultdict

import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)


def set_seed(s):
    """Reset global RNGs (used for the multi-seed robustness runs)."""
    global rng
    rng = np.random.default_rng(s)
    random.seed(s)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# E1. SETTLEMENT RECONCILIATION
# ----------------------------------------------------------------------

CATEGORIES = {
    # category: (fee_rate, price_lognorm_mu, price_lognorm_sigma)
    "electronics": (0.0655, 4.2, 0.9),
    "fashion": (0.1350, 3.2, 0.7),
    "home_garden": (0.1325, 3.5, 0.8),
    "collectibles": (0.1235, 3.8, 1.1),
    "industrial": (0.1230, 4.6, 0.9),
    "media": (0.1450, 2.6, 0.6),
}
FIXED_FEE = 0.30
PROMO_CREDIT_RATE = 0.10          # 10% discount on final value fee for enrolled sellers
SHIP_BASE, SHIP_PER_LB = 4.50, 0.85
ANOMALY_TYPES = [
    "duplicate_fee", "missing_promo_credit", "incorrect_category_rate",
    "erroneous_refund_deduction", "shipping_miscalculation",
]

N_TXN = 40_000
N_SELLERS = 250
ANOMALY_RATE = 0.03               # ~3% of transactions carry an injected anomaly
STALE_SCHEDULE_RATE = 0.015       # benign confounder: txns settled under a prior
STALE_RATE_BUMP = 0.004           # fee-schedule version (+0.4pp) — not true anomalies
LEGIT_ADJ_RATE = 0.02             # small legitimate adjustments (currency rounding)


def expected_components(cat, price, weight, enrolled, rate_override=None):
    rate = rate_override if rate_override is not None else CATEGORIES[cat][0]
    fvf = rate * price + FIXED_FEE
    promo = -PROMO_CREDIT_RATE * fvf if enrolled else 0.0
    ship = SHIP_BASE + SHIP_PER_LB * weight
    return fvf, promo, ship


def gen_reconciliation_corpus(noise_scale=1.0):
    cats = list(CATEGORIES)
    sellers = [f"SLR-{i:05d}" for i in range(N_SELLERS)]
    enrolled_sellers = set(rng.choice(sellers, size=int(0.4 * N_SELLERS), replace=False))
    txns = []
    for i in range(N_TXN):
        cat = cats[rng.integers(len(cats))]
        _, mu, sg = CATEGORIES[cat]
        price = round(float(rng.lognormal(mu, sg)), 2)
        price = max(1.0, min(price, 2500.0))
        weight = round(float(rng.uniform(0.2, 18.0)), 1)
        seller = sellers[rng.integers(N_SELLERS)]
        enrolled = seller in enrolled_sellers
        fvf, promo, ship = expected_components(cat, price, weight, enrolled)

        actual_fvf, actual_promo, actual_ship, actual_refund = fvf, promo, ship, 0.0
        label = None

        u = rng.random()
        if u < ANOMALY_RATE:
            label = ANOMALY_TYPES[rng.integers(len(ANOMALY_TYPES))]
            if label == "duplicate_fee":
                actual_fvf = fvf * 2
            elif label == "missing_promo_credit":
                if enrolled:
                    actual_promo = 0.0
                else:                       # cannot miss a credit that doesn't exist
                    label = "duplicate_fee"
                    actual_fvf = fvf * 2
            elif label == "incorrect_category_rate":
                alt = cats[rng.integers(len(cats))]
                while alt == cat:
                    alt = cats[rng.integers(len(cats))]
                actual_fvf = CATEGORIES[alt][0] * price + FIXED_FEE
            elif label == "erroneous_refund_deduction":
                actual_refund = round(price * float(rng.uniform(0.3, 1.0)), 2)
            elif label == "shipping_miscalculation":
                actual_ship = SHIP_BASE + SHIP_PER_LB * (weight + float(rng.uniform(1.5, 9.0)))
        elif u < ANOMALY_RATE + STALE_SCHEDULE_RATE:
            # benign: settled under prior fee-schedule version (rate +0.4pp).
            actual_fvf = (CATEGORIES[cat][0] + STALE_RATE_BUMP) * price + FIXED_FEE
        elif u < ANOMALY_RATE + STALE_SCHEDULE_RATE + LEGIT_ADJ_RATE:
            actual_fvf = fvf + noise_scale * float(rng.uniform(-0.03, 0.03))

        # observation noise on settled amounts (rounding, FX residuals)
        actual_fvf += noise_scale * float(rng.uniform(-0.02, 0.02))
        actual_ship += noise_scale * float(rng.uniform(-0.02, 0.02))
        txns.append(dict(
            id=i, cat=cat, price=price, weight=weight, seller=seller,
            enrolled=enrolled, label=label,
            actual=dict(fvf=round(actual_fvf, 2), promo=round(actual_promo, 2),
                        ship=round(actual_ship, 2), refund=round(actual_refund, 2)),
        ))
    return txns


def reconcile(txns, version_aware=True, floor=0.10, credit_guard=True):
    """Detector: recompute expected fees under the CURRENT schedule, flag and
    classify deltas by signature match. Discrepancies fully explained by the
    immediately prior fee-schedule version are suppressed as benign
    (version-aware reconciliation, cf. fee_schedule_version parameter).
    Unmatched deltas are flagged as 'unclassified_discrepancy'.
    version_aware=False ablates the prior-version suppression step."""
    preds = {}
    cats = list(CATEGORIES)
    for t in txns:
        fvf, promo, ship = expected_components(t["cat"], t["price"], t["weight"], t["enrolled"])
        a = t["actual"]
        delta_total = (a["fvf"] + a["promo"] + a["ship"] + a["refund"]) - (fvf + promo + ship)
        threshold = max(floor, 0.005 * (fvf + ship))
        if abs(delta_total) <= threshold:
            continue

        # version-aware suppression: is the fee exactly explained by the
        # prior schedule version (rate + STALE_RATE_BUMP)?
        prior_fvf = (CATEGORIES[t["cat"]][0] + STALE_RATE_BUMP) * t["price"] + FIXED_FEE
        tol = max(0.08, 0.003 * fvf)
        if version_aware and abs(a["fvf"] - prior_fvf) < tol and a["refund"] == 0 \
                and abs(a["ship"] - ship) < tol \
                and (not credit_guard or abs(a["promo"] - promo) <= 0.01):
            continue  # benign: settled under prior schedule version

        d_fvf = a["fvf"] - fvf
        d_promo = a["promo"] - promo
        d_ship = a["ship"] - ship

        pred, conf = None, 0.0
        if a["refund"] > 0.5:
            pred, conf = "erroneous_refund_deduction", 0.95
        elif abs(d_fvf - fvf) < tol:
            pred, conf = "duplicate_fee", 0.95
        elif t["enrolled"] and abs(d_promo - (PROMO_CREDIT_RATE * fvf)) < tol and abs(d_fvf) < tol:
            pred, conf = "missing_promo_credit", 0.90
        elif abs(d_ship) > 0.5 and abs(d_fvf) < tol:
            pred, conf = "shipping_miscalculation", 0.90
        elif abs(d_fvf) > tol:
            # does the actual fee match another category's schedule?
            best, best_err = None, float("inf")
            for alt in cats:
                if alt == t["cat"]:
                    continue
                err = abs(a["fvf"] - (CATEGORIES[alt][0] * t["price"] + FIXED_FEE))
                if err < best_err:
                    best, best_err = alt, err
            if best_err < tol:
                pred, conf = "incorrect_category_rate", 0.85
            else:
                pred, conf = "unclassified_discrepancy", 0.55
        else:
            pred, conf = "unclassified_discrepancy", 0.50
        preds[t["id"]] = (pred, conf, round(delta_total, 2))
    return preds


def score_reconciliation(txns, preds):
    per_type = {a: dict(tp=0, fn=0, fp_class=0)
                for a in ANOMALY_TYPES + ["unclassified_discrepancy"]}
    flagged_clean = 0
    n_anom = 0
    detected_any = 0
    for t in txns:
        p = preds.get(t["id"])
        if t["label"]:
            n_anom += 1
            if p is None:
                per_type[t["label"]]["fn"] += 1
            else:
                detected_any += 1
                if p[0] == t["label"]:
                    per_type[t["label"]]["tp"] += 1
                else:
                    per_type[t["label"]]["fn"] += 1
                    per_type[p[0]]["fp_class"] += 1
        elif p is not None:
            flagged_clean += 1
            per_type[p[0]]["fp_class"] += 1

    results = {}
    for a, c in per_type.items():
        prec = c["tp"] / (c["tp"] + c["fp_class"]) if c["tp"] + c["fp_class"] else 0
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        results[a] = dict(precision=round(prec, 3), recall=round(rec, 3),
                          f1=round(f1, 3), support=c["tp"] + c["fn"])
    overall_det = detected_any / n_anom
    overall_prec = detected_any / (detected_any + flagged_clean)
    results["_overall"] = dict(
        detection_recall=round(overall_det, 3),
        detection_precision=round(overall_prec, 3),
        n_transactions=N_TXN, n_anomalies=n_anom,
        false_flags_on_clean=flagged_clean,
        false_flag_rate_clean=round(flagged_clean / (N_TXN - n_anom), 5),
    )
    return results


# ----------------------------------------------------------------------
# E2. RETURN FRAUD DETECTION — cohort vs. universal baselines
# ----------------------------------------------------------------------

COHORTS = {
    "fashion": 0.28, "shoes": 0.24, "electronics": 0.10,
    "home_garden": 0.08, "collectibles": 0.05, "industrial": 0.03,
}
N_FRAUD_SELLERS = 600
FRAUD_VICTIM_RATE = 0.10
W_Z, W_CONC, W_TIME = 0.40, 0.35, 0.25
FRAUD_THRESHOLD = 0.60


def gen_fraud_corpus():
    """SKU-level modeling, per the paper: fraud rings target a specific SKU
    of a victim seller with concentrated, fast-filed return claims."""
    sellers = []
    cohort_names = list(COHORTS)
    for s in range(N_FRAUD_SELLERS):
        cohort = cohort_names[rng.integers(len(cohort_names))]
        base = COHORTS[cohort]
        srate = max(0.005, base + float(rng.normal(0, base * 0.20)))
        n_skus = int(rng.integers(8, 16))
        is_victim = rng.random() < FRAUD_VICTIM_RATE
        target_sku = int(rng.integers(n_skus)) if is_victim else -1
        skus = []
        for k in range(n_skus):
            n_orders = max(5, int(rng.poisson(30)))
            n_returns = int(rng.binomial(n_orders, min(srate, 0.95)))
            n_buyers = max(1, int(n_returns * float(rng.uniform(0.7, 1.0)))) if n_returns else 0
            buyer_ids = [f"B{rng.integers(1_000_000)}" for _ in range(n_buyers)]
            ret_buyers = [random.choice(buyer_ids) for _ in range(n_returns)] if n_buyers else []
            hours = list(rng.lognormal(4.0, 0.6, size=n_returns))       # median ~55h
            if k == target_sku:
                ring = [f"RING{s}_{r}" for r in range(int(rng.integers(2, 5)))]
                n_fraud = int(rng.integers(5, 15))
                ret_buyers += [random.choice(ring) for _ in range(n_fraud)]
                hours += list(rng.lognormal(1.5, 0.5, size=n_fraud))    # median ~4.5h
                n_returns += n_fraud
            skus.append(dict(orders=n_orders, returns=n_returns,
                             buyers=ret_buyers, hours=hours))
        sellers.append(dict(cohort=cohort, skus=skus, victim=is_victim))
    return sellers


def _composite(z, buyers, hours):
    z_norm = 1 / (1 + math.exp(-(z - 2.0)))
    if buyers:
        counts = defaultdict(int)
        for b in buyers:
            counts[b] += 1
        top3 = sum(sorted(counts.values(), reverse=True)[:3]) / len(buyers)
    else:
        top3 = 0.0
    conc_norm = max(0.0, (top3 - 0.15) / 0.85)
    med_h = float(np.median(hours)) if hours else 100.0
    time_norm = max(0.0, min(1.0, (24.0 - med_h) / 24.0))
    return W_Z * z_norm + W_CONC * conc_norm + W_TIME * time_norm


MIN_RETURNS_FOR_SIGNAL = 6      # minimum support gate for SKU-level signals


def fraud_scores(sellers, use_cohort_baseline=True):
    universal_mean = float(np.mean(list(COHORTS.values())))
    scores, labels = [], []
    for s in sellers:
        base = COHORTS[s["cohort"]] if use_cohort_baseline else universal_mean
        best = 0.0
        for sku in s["skus"]:
            if sku["returns"] < MIN_RETURNS_FOR_SIGNAL:
                continue
            rate = sku["returns"] / sku["orders"]
            # binomial standard error under the baseline hypothesis
            se = math.sqrt(base * (1 - base) / sku["orders"])
            z = (rate - base) / max(se, 1e-6)
            best = max(best, _composite(z, sku["buyers"], sku["hours"]))
        scores.append(best)
        labels.append(s["victim"])
    return np.array(scores), np.array(labels)


def _roc_auc(scores, labels):
    """Tie-aware ROC: move all examples with the same score together.

    Trapezoids between distinct-score endpoints give tied positive-negative
    pairs half credit, independently of their order within each tie block.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tpr = np.r_[0.0, np.cumsum(y)[ends] / y.sum()]
    fpr = np.r_[0.0, np.cumsum(~y)[ends] / (~y).sum()]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return auc, tpr, fpr


def score_fraud(sellers):
    out = {}
    for mode, use_cohort in [("cohort_baseline", True), ("universal_baseline", False)]:
        scores, labels = fraud_scores(sellers, use_cohort)
        auc, tpr_c, fpr_c = _roc_auc(scores, labels)
        pred = scores >= FRAUD_THRESHOLD
        tp = int(np.sum(pred & labels)); fp = int(np.sum(pred & ~labels))
        fn = int(np.sum(~pred & labels)); tn = int(np.sum(~pred & ~labels))
        hi = np.array([s["cohort"] in ("fashion", "shoes") for s in sellers])
        fp_hi = int(np.sum(pred & ~labels & hi))
        # FPR at matched TPR >= 0.85 (threshold sweep)
        idx = np.searchsorted(tpr_c, 0.85)
        fpr_at_85 = float(fpr_c[min(idx, len(fpr_c) - 1)])
        out[mode] = dict(
            roc_auc=round(auc, 3),
            tpr_at_threshold=round(tp / (tp + fn), 3),
            fpr_at_threshold=round(fp / (fp + tn), 4),
            precision_at_threshold=round(tp / (tp + fp), 3) if tp + fp else None,
            fpr_at_matched_tpr_085=round(fpr_at_85, 4),
            false_positives=fp, false_positives_high_return_cohorts=fp_hi,
            n_sellers=len(sellers), n_victims=int(np.sum(labels)),
            threshold=FRAUD_THRESHOLD,
        )
    return out


# ----------------------------------------------------------------------
# E3. FINANCIAL ERROR CASCADE DETECTION (Algorithm 1)
# ----------------------------------------------------------------------

N_TRACES = 2_500
HALLUCINATION_RATE = 0.07
ERROR_THRESHOLD = 0.05            # 5% relative error for financial values
MEASUREMENT_NOISE = 0.01          # legit ±1% discrepancy on clean transitions
AGENTS = ["payments", "orders", "listings", "post_txn", "orchestrator"]


def gen_traces():
    traces = []
    for t in range(N_TRACES):
        n_steps = int(rng.integers(4, 9))
        inject = rng.random() < HALLUCINATION_RATE
        # inject only at non-terminal steps: cascade detection is scoped to
        # errors that cross an agent boundary (terminal-output errors are the
        # province of output evaluation, not cascade detection)
        inject_at = int(rng.integers(1, n_steps - 1)) if inject else -1
        # relative error magnitude for the hallucination: 2%..60%, log-uniform
        err_mag = float(np.exp(rng.uniform(np.log(0.02), np.log(0.60)))) if inject else 0.0
        steps = []
        value = float(rng.uniform(500, 20_000))          # ground-truth financial value
        for k in range(n_steps):
            agent = AGENTS[k % len(AGENTS)]
            gt = value
            observed = gt * (1 + float(rng.uniform(-MEASUREMENT_NOISE, MEASUREMENT_NOISE)))
            if k == inject_at:
                sign = 1 if rng.random() < 0.5 else -1
                observed = gt * (1 + sign * err_mag)
            steps.append(dict(agent=agent, gt=gt, observed=observed, k=k))
            value = observed * (1 + float(rng.uniform(-0.002, 0.002)))  # value propagates
        traces.append(dict(steps=steps, inject=inject, inject_at=inject_at, err_mag=err_mag))
    return traces


def detect_cascades(traces):
    buckets = {"2-5%": [0, 0], "5-10%": [0, 0], "10-25%": [0, 0], ">25%": [0, 0]}
    false_flags = 0
    clean_transitions = 0
    steps_saved = []
    detected, total_inj = 0, 0
    detect_latency = []
    for tr in traces:
        flagged_step = None
        for j in range(len(tr["steps"]) - 1):
            a, b = tr["steps"][j], tr["steps"][j + 1]
            if a["agent"] == b["agent"]:
                continue
            rel = abs(a["observed"] - a["gt"]) / a["gt"]
            if rel > ERROR_THRESHOLD and flagged_step is None:
                flagged_step = j
            if not tr["inject"] or j < tr["inject_at"]:
                clean_transitions += 1
                if rel > ERROR_THRESHOLD:
                    false_flags += 1
        if tr["inject"]:
            total_inj += 1
            m = tr["err_mag"]
            key = "2-5%" if m < 0.05 else "5-10%" if m < 0.10 else "10-25%" if m < 0.25 else ">25%"
            buckets[key][1] += 1
            hit = flagged_step is not None and flagged_step >= tr["inject_at"]
            if hit:
                buckets[key][0] += 1
                detected += 1
                detect_latency.append(flagged_step - tr["inject_at"])
                steps_saved.append(len(tr["steps"]) - 1 - flagged_step)
    return dict(
        overall_detection_rate=round(detected / total_inj, 3),
        detection_by_magnitude={k: dict(detected=v[0], total=v[1],
                                        rate=round(v[0] / v[1], 3) if v[1] else None)
                                for k, v in buckets.items()},
        false_flag_rate_clean_transitions=round(false_flags / clean_transitions, 5),
        mean_detection_latency_steps=round(float(np.mean(detect_latency)), 2),
        mean_downstream_steps_halted=round(float(np.mean(steps_saved)), 2),
        n_traces=N_TRACES, n_injected=total_inj,
        error_threshold=ERROR_THRESHOLD,
    )


# ----------------------------------------------------------------------
