"""
Self-checks for the v3 adaptive ridge calibration.

Run with:  python tests/test_adaptive_ridge.py

Test 1 always runs and needs no data: it verifies that the inner selector
recovers the correct feature design on synthetic data with known structure.

Test 2 runs only if the ADNI-derived inputs are present in the working
directory (they cannot be redistributed under the ADNI Data Use Agreement).
It reproduces the published agreement statistics exactly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecvrs.calibration.adaptive_ridge import (  # noqa: E402
    select_model,
    train_adaptive_ridge_mapping,
    predict_adaptive_ridge_mapping,
)

# Published values (ADNI MCI, N=339, five-fold outer CV, seed 42).
PUBLISHED = {
    "icc_total": 0.7467,
    "kappa": {
        "Hippo_Lt": 0.6000,
        "Hippo_Rt": 0.6219,
        "Front": 0.4698,
        "Parietal": 0.6312,
        "Temporal": 0.6524,
    },
    "icc_mean_8_seeds": 0.7452,
}
SEEDS = [0, 1, 2, 3, 4, 42, 100, 2024]
LOBES = [("Hippo_Lt", 5), ("Hippo_Rt", 5), ("Front", 4),
         ("Parietal", 4), ("Temporal", 4)]


def test_design_selection_recovers_known_structure():
    """A curved generating function should pull the selector to 'quadratic'."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 4))

    y_lin = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(scale=0.1, size=400)
    variant, _ = select_model(X, y_lin, seed=42)
    assert variant == "linear", f"expected linear, got {variant}"

    y_quad = 3.0 * X[:, 0] ** 2 + rng.normal(scale=0.1, size=400)
    variant, _ = select_model(X, y_quad, seed=42)
    assert variant == "quadratic", f"expected quadratic, got {variant}"

    # Heavy noise and near-collinear features should attract a non-zero penalty.
    Xc = np.column_stack([X[:, 0], X[:, 0] + rng.normal(scale=1e-3, size=400)])
    y_noisy = X[:, 0] + rng.normal(scale=5.0, size=400)
    _, lam = select_model(Xc, y_noisy, seed=42)
    assert lam > 0, f"expected shrinkage under collinearity, got lambda={lam}"

    print("PASS  design and penalty selection recover known structure")


def _load_cohort():
    """Return (feature matrix, dataframe) or None if ADNI inputs are absent."""
    scores, ratings = "e-CVRS_automated_scores.csv", "ADNI_MRI_rating.xlsx"
    if not (os.path.exists(scores) and os.path.exists(ratings)):
        return None
    import pandas as pd

    df = pd.merge(pd.read_csv(scores), pd.read_excel(ratings), on="RID")
    ratio = ["Auto_Lt_Ratio", "Auto_Rt_Ratio", "Auto_Front_Ratio",
             "Auto_Parietal_Ratio", "Auto_Temporal_Ratio"]
    vol = ["Vol_Hippo_Lt", "Vol_Hippo_Rt", "Vol_Front", "Vol_Parietal", "Vol_Temporal"]
    tbv = df["Vol_TBV"].values.astype(float)
    X = np.column_stack(
        [df[c].values.astype(float) for c in ratio]
        + [df[v].values.astype(float) / tbv for v in vol]
        + [(df["Vol_Ventricle"] / df["Vol_TBV"]).values.astype(float)]
    )
    df["Manual_Atrophy_Sum"] = df[[r for r, _ in LOBES]].sum(axis=1)
    return X, df


def _out_of_fold_grades(X, df, seed):
    """Replicate the CLI's outer split and return out-of-fold grades."""
    import pandas as pd

    n = len(df)
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    folds = np.array_split(idx, 5)

    preds = {}
    for region, ncls in LOBES:
        y = df[region].values.astype(float)
        p = np.zeros(n, dtype=int)
        for k in range(5):
            te = folds[k]
            tr = np.hstack([folds[j] for j in range(5) if j != k])
            model = train_adaptive_ridge_mapping(X[tr], y[tr], ncls, seed=seed)
            p[te] = predict_adaptive_ridge_mapping(X[te], model)
        preds[region] = p
    e = pd.DataFrame(preds)
    e["e_sum"] = e.sum(axis=1)
    return e


def test_reproduces_published_results():
    from ecvrs.stats.agreement import cohen_kappa_quadratic, intraclass_correlation

    loaded = _load_cohort()
    if loaded is None:
        print("SKIP  ADNI-derived inputs not found; see README for data access")
        return
    X, df = loaded
    assert len(df) == 339, f"expected the 339-subject paired cohort, got {len(df)}"

    e42 = _out_of_fold_grades(X, df, 42)
    icc = intraclass_correlation(df["Manual_Atrophy_Sum"].values, e42["e_sum"].values)
    assert abs(icc - PUBLISHED["icc_total"]) < 5e-4, \
        f"total ICC {icc:.4f} != published {PUBLISHED['icc_total']}"
    print(f"PASS  seed 42 total ICC {icc:.4f}")

    for region, ncls in LOBES:
        k = cohen_kappa_quadratic(df[region].values, e42[region].values, ncls)
        assert abs(k - PUBLISHED["kappa"][region]) < 5e-4, \
            f"{region} kappa {k:.4f} != published {PUBLISHED['kappa'][region]}"
        print(f"PASS  seed 42 kappa {region:<10} {k:.4f}")

    iccs = [
        intraclass_correlation(
            df["Manual_Atrophy_Sum"].values, _out_of_fold_grades(X, df, s)["e_sum"].values
        )
        for s in SEEDS
    ]
    mean_icc = float(np.mean(iccs))
    assert abs(mean_icc - PUBLISHED["icc_mean_8_seeds"]) < 5e-4, \
        f"8-seed mean ICC {mean_icc:.4f} != published {PUBLISHED['icc_mean_8_seeds']}"
    print(f"PASS  8-seed mean ICC {mean_icc:.4f} +/- {np.std(iccs, ddof=1):.4f}")


if __name__ == "__main__":
    test_design_selection_recovers_known_structure()
    test_reproduces_published_results()
    print("\nAll checks completed.")
