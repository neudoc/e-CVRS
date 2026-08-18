"""
Adaptive nested ridge calibration for e-CVRS.

This is the RUNNER-UP calibration in the published benchmark. The method
actually reported in the manuscript is the quadratic-basis ridge in
``ecvrs/calibration/quadratic_ridge.py``; this module is retained because it
is the closest competitor and is referenced in the candidate comparison.

Rationale
---------
A single ordinary least-squares line through the 11-feature panel assumes each
feature acts on every regional grade through a straight-line effect of equal
stability. This module relaxes both assumptions while remaining fully
transparent:

1. **Design selection.** For each region, a nested inner cross-validation
   decides between a linear design ``x`` and a quadratic basis ``[x, x^2]``.
2. **Shrinkage selection.** The same inner loop selects an L2 (ridge) penalty
   ``lambda`` from a fixed grid, damping unstable coefficients.

Both choices are made *inside the outer training fold only*, so the procedure
is leakage-controlled. The fitted model is an explicit weighted sum of named,
inspectable measurements followed by ordinal thresholds -- no latent image
representation is learned.

Performance with the published hippocampus-anchored windows (ADNI MCI, N=339,
leakage-controlled 5-fold CV, mean over 8 fold-assignment seeds): total-score
ICC(2,1) = 0.729, FreeSurfer rho = 0.403, mean cognitive incremental F = 15.61,
mean absolute CSF rho = 0.164, composite rank 2.29 -- second to
``quadratic_ridge`` (0.734 / 0.405 / 15.65 / 0.167, composite rank 1.67).
"""
import numpy as np

from ecvrs.calibration.ordinal import (
    train_kappa_optimized_mapping,
    predict_kappa_optimized_mapping,
)

#: Ridge penalties searched by the inner cross-validation. lambda = 0 is plain OLS.
DEFAULT_LAMBDAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)

#: Feature designs searched by the inner cross-validation.
DEFAULT_VARIANTS = ("linear", "quadratic")

#: Number of inner folds used for model selection inside each outer training fold.
DEFAULT_INNER_FOLDS = 4


def _zstats(X):
    """Training-fold mean/SD, with zero-variance columns protected."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def _design(X, variant):
    """Build the feature design matrix for a given variant."""
    if variant == "linear":
        return X
    if variant == "quadratic":
        return np.column_stack([X, X ** 2])
    raise ValueError(f"unknown variant: {variant!r}")


def _ridge_fit(Xz, y, lam):
    """
    Closed-form ridge regression on standardized features.

    The intercept is the training mean of y and is not penalized; returning it
    separately keeps the penalty from shrinking the grade level itself.
    """
    p = Xz.shape[1]
    A = Xz.T @ Xz + lam * np.eye(p)
    b = np.linalg.solve(A, Xz.T @ (y - y.mean()))
    return y.mean(), b


def select_model(train_features, train_manual, seed=42,
                 variants=DEFAULT_VARIANTS, lambdas=DEFAULT_LAMBDAS,
                 n_inner=DEFAULT_INNER_FOLDS):
    """
    Choose (variant, lambda) by inner cross-validation on the training fold.

    Parameters
    ----------
    train_features : (n, p) array
        Outer training-fold feature panel.
    train_manual : (n,) array
        Manual reference grades for the training fold.
    seed : int
        Seed for the inner fold shuffle. Uses an isolated RandomState so the
        caller's global numpy seed (which drives the outer split) is untouched.

    Returns
    -------
    (variant, lam) : tuple
        The design and penalty minimizing inner-validation mean squared error.
    """
    X = np.asarray(train_features, dtype=float)
    y = np.asarray(train_manual, dtype=float)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_inner)

    best_variant, best_lam, best_err = None, None, np.inf
    for variant in variants:
        D = _design(X, variant)
        for lam in lambdas:
            errs = []
            for k in range(n_inner):
                te = folds[k]
                tr = np.hstack([folds[j] for j in range(n_inner) if j != k])
                mu, sd = _zstats(D[tr])
                Ztr = (D[tr] - mu) / sd
                Zte = (D[te] - mu) / sd
                b0, b = _ridge_fit(Ztr, y[tr], lam)
                errs.append(np.mean((y[te] - (b0 + Zte @ b)) ** 2))
            err = float(np.mean(errs))
            if err < best_err:
                best_variant, best_lam, best_err = variant, lam, err

    return best_variant, best_lam


def train_adaptive_ridge_mapping(train_features, train_manual, num_classes, seed=42,
                                 variants=DEFAULT_VARIANTS, lambdas=DEFAULT_LAMBDAS,
                                 n_inner=DEFAULT_INNER_FOLDS):
    """
    Fit the adaptive calibration on one outer training fold.

    Steps: (1) inner CV selects design and penalty; (2) the chosen model is refit
    on the full training fold; (3) grade cutoffs maximizing quadratic-weighted
    kappa are learned from the same training fold.

    Parameters
    ----------
    train_features : (n, p) array
        Feature panel; the published model uses 11 columns (5 regional CSF
        ratios, 5 proxy volumes normalized by total brain volume, ventricular
        fraction).
    train_manual : (n,) int array
        Manual reference grades.
    num_classes : int
        Ordinal classes for this sub-scale (5 for hippocampus, 4 otherwise).
    seed : int
        Seed for the inner fold shuffle.

    Returns
    -------
    model : dict
        Serialisable parameters: selected variant and penalty, feature
        means/SDs, intercept, coefficients, and score thresholds.
    """
    X = np.asarray(train_features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(train_manual, dtype=float)

    variant, lam = select_model(X, y, seed=seed, variants=variants,
                                lambdas=lambdas, n_inner=n_inner)

    D = _design(X, variant)
    mu, sd = _zstats(D)
    Z = (D - mu) / sd
    b0, b = _ridge_fit(Z, y, lam)

    train_score = b0 + Z @ b
    thresholds = train_kappa_optimized_mapping(train_score, train_manual, num_classes)

    return {
        "variant": variant,
        "lam": float(lam),
        "mu": mu,
        "sd": sd,
        "intercept": float(b0),
        "beta": b,
        "thresholds": np.asarray(thresholds, dtype=float),
        "num_classes": int(num_classes),
    }


def predict_adaptive_ridge_mapping(test_features, model):
    """Apply a fitted adaptive model to new features -> ordinal grades."""
    X = np.asarray(test_features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    D = _design(X, model["variant"])
    Z = (D - model["mu"]) / model["sd"]
    score = model["intercept"] + Z @ model["beta"]
    return predict_kappa_optimized_mapping(score, model["thresholds"], model["num_classes"])


def score_adaptive_ridge_mapping(test_features, model):
    """Return the continuous atrophy score before ordinal thresholding."""
    X = np.asarray(test_features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    D = _design(X, model["variant"])
    Z = (D - model["mu"]) / model["sd"]
    return model["intercept"] + Z @ model["beta"]
