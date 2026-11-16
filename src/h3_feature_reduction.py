# Title: h3_feature_reduction.py
# Description: H3 (time permitting). Compares the full ten-group model with the prespecified reduced set
# (education, household income, age, disability type) under exactly the same protocol as main.py:
# same split, same household-grouped folds, same grids, same scoring. The tolerance for "a few percentage
# points" of accuracy is fixed here (ACCURACY_TOLERANCE_POINTS) before any result is seen, and the
# exclusion-class recall and F1 differences are reported alongside so similar accuracy cannot hide lost
# detection of excluded people.
#
# Run:  python h3_feature_reduction.py                 (pooled experiment A protocol)
#       python h3_feature_reduction.py --experiment B  (temporal protocol)
#       python h3_feature_reduction.py --tolerance 3 --recall-tolerance 5

import argparse
import os

import pandas as pd

import main as m

ACCURACY_TOLERANCE_POINTS = 3.0   # reduced model counts as "within a few points" if accuracy drop <= this
RECALL_TOLERANCE_POINTS = 5.0     # and exclusion-class recall drop <= this


def compare(full_results, reduced_results, acc_tol, rec_tol):
    full = {r["Model"]: r for r in full_results}
    red = {r["Model"]: r for r in reduced_results}
    rows = []
    for name in m.MODEL_ORDER:
        if name not in full or name not in red or name.startswith("Dummy"):
            continue
        acc_drop = 100 * (full[name]["Accuracy"] - red[name]["Accuracy"])
        rec_drop = 100 * (full[name]["Recall (excluded)"] - red[name]["Recall (excluded)"])
        f1_drop = 100 * (full[name]["F1 (excluded)"] - red[name]["F1 (excluded)"])
        auc_drop = 100 * (full[name]["ROC AUC"] - red[name]["ROC AUC"])
        rows.append({
            "Model": name,
            "Full accuracy (%)": round(100 * full[name]["Accuracy"], 2),
            "Reduced accuracy (%)": round(100 * red[name]["Accuracy"], 2),
            "Accuracy drop (points)": round(acc_drop, 2),
            "Full recall excluded (%)": round(100 * full[name]["Recall (excluded)"], 2),
            "Reduced recall excluded (%)": round(100 * red[name]["Recall (excluded)"], 2),
            "Recall drop (points)": round(rec_drop, 2),
            "F1 excluded drop (points)": round(f1_drop, 2),
            "ROC AUC drop (points)": round(auc_drop, 2),
            "Within accuracy tolerance": bool(acc_drop <= acc_tol),
            "Within recall tolerance": bool(rec_drop <= rec_tol),
            "H3 holds for this model": bool(acc_drop <= acc_tol and rec_drop <= rec_tol),
        })
    return pd.DataFrame(rows)


def run(experiment="A", cv=5, quick=False, outdir=m.OUTPUT_DIR, acc_tol=ACCURACY_TOLERANCE_POINTS,
        rec_tol=RECALL_TOLERANCE_POINTS):
    df = m.load_predictor_table()
    if df is None:
        return None
    if experiment == "A":
        train, test = m.pooled_split(df)
        label = "pooled"
    else:
        train, test = m.temporal_split(df)
        label = "temporal"

    print(f"H3 prespecified tolerances: accuracy drop <= {acc_tol} points AND exclusion recall drop <= {rec_tol} points")
    print(f"Reduced inputs: {m.REDUCED_FEATURES}")

    base = os.path.join(outdir, f"H3_{label}")
    full_results, _ = m.run_experiment(f"H3_{label}_full", train, test, os.path.join(base, "full"),
                                       f"{label} protocol, all ten predictor groups", features=m.FEATURES,
                                       cv=cv, quick=quick, interpret=False)
    red_results, _ = m.run_experiment(f"H3_{label}_reduced", train, test, os.path.join(base, "reduced"),
                                      f"{label} protocol, reduced set: age, income, education, disability type",
                                      features=m.REDUCED_FEATURES, cv=cv, quick=quick, interpret=False)

    table = compare(full_results, red_results, acc_tol, rec_tol)
    table.to_csv(os.path.join(base, "h3_comparison.csv"), index=False)
    print("\n===== H3: full vs reduced (held-out) =====")
    print(table.to_string(index=False))
    print(f"\nSaved {os.path.join(base, 'h3_comparison.csv')}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B"], default="A")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outdir", default=m.OUTPUT_DIR)
    parser.add_argument("--tolerance", type=float, default=ACCURACY_TOLERANCE_POINTS)
    parser.add_argument("--recall-tolerance", type=float, default=RECALL_TOLERANCE_POINTS)
    args = parser.parse_args()
    run(args.experiment, args.cv, args.quick, args.outdir, args.tolerance, args.recall_tolerance)
