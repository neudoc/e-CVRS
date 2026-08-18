"""Self-checks for the published e-CVRS calibration.

Run with:  python tests/test_calibration.py

Test 1 always runs and needs no data: it verifies that the ridge penalty
selector behaves correctly on synthetic data with known structure, and that a
trained model is a plain, inspectable dictionary.

Test 2 runs only if the ADNI-derived inputs are present in the working
directory (they cannot be redistributed under the ADNI Data Use Agreement).
It reproduces the published agreement statistics exactly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecvrs.calibration.quadratic_ridge import (  # noqa: E402
    select_lambda,
    quadratic_design,
    train_quadratic_ridge_mapping,
    predict_quadratic_ridge_mapping,
)

# Published values (ADNI MCI, N=339, five-fold outer CV, seed 42,
# hippocampus-anchored windows, full_quadratic_ridge calibration).
PUBLISHED = {
    "icc_total": 0.7482,
    "kappa": {
        "Hippo_Lt": 0.5987,
        "Hippo_Rt": 0.6225,
        "Front": 0.4981,
        "Parietal": 0.5944,
        "Temporal": 0.6655,
    },
    "icc_mean_8_seeds": 0.7341,
}

LOBES = [("Hippo_Lt", 5), ("Hippo_Rt", 5), ("Front", 4), ("Parietal", 4), ("Temporal", 4)]
SEEDS = [0, 1, 2, 3, 4, 42, 100, 2024]


def test_penalty_selection_and_model_transparency():
    rng = np.random.RandomState(0)

    # Well-conditioned, near-noiseless signal: little shrinkage is needed.
    X = rng.normal(size=(400, 4))
    y = X @ np.array([1.0, -0.5, 0.25, 0.0]) + 0.01 * rng.normal(size=400)
    lam_clean = select_lambda(X, y, seed=0)

    # Same signal with heavily collinear duplicate columns: ridge must shrink more.
    Xc = np.hstack([X, X + 1e-3 * rng.normal(size=X.shape)])
    lam_collinear = select_lambda(Xc, y, seed=0)
    assert lam_collinear >= lam_clean, (
        f"expected at least as much shrinkage under collinearity, "
        f"got {lam_collinear} vs {lam_clean}"
    )
    print(f"PASS  penalty selection: lambda {lam_clean} (clean) -> "
          f"{lam_collinear} (collinear)")

    # The quadratic basis is explicit: [x, x^2], nothing latent.
    D = quadratic_design(X)
    assert D.shape == (400, 8)
    assert np.allclose(D[:, :4], X)
    assert np.allclose(D[:, 4:], X ** 2)
    print("PASS  quadratic design is the explicit basis [x, x^2]")

    # A trained model is a plain dict of named arrays that can be audited.
    grades = np.clip(np.round(2 + y), 0, 4)
    model = train_quadratic_ridge_mapping(X, grades, num_classes=5, seed=0)
    for key in ("variant", "lam", "mu", "sd", "intercept", "beta", "thresholds"):
        assert key in model, f"model is missing the inspectable field {key!r}"
    assert model["beta"].shape == (8,), model["beta"].shape
    assert len(model["thresholds"]) == 4, model["thresholds"]

    pred = predict_quadratic_ridge_mapping(X, model)
    assert pred.min() >= 0 and pred.max() <= 4
    print("PASS  trained model exposes coefficients and explicit grade cutoffs")


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
            model = train_quadratic_ridge_mapping(X[tr], y[tr], ncls, seed=seed)
            p[te] = predict_quadratic_ridge_mapping(X[te], model)
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
    test_penalty_selection_and_model_transparency()
    test_reproduces_published_results()
    print("\nAll checks completed.")
