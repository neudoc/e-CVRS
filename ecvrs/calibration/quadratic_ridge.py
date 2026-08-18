"""Quadratic-basis ridge calibration -- the published e-CVRS mapping.

Each regional grade is predicted from the 11-feature panel expanded into an
explicit quadratic basis ``[x, x^2]`` and fitted by ridge regression. The ridge
penalty is chosen by four-fold inner cross-validation run strictly inside the
outer training fold; the fitted continuous score is then mapped to ordinal
grades by cutoffs that maximize quadratic-weighted kappa on the same training
fold.

Nothing latent is learned. The trained model is a plain dictionary of named
arrays -- standardization constants, coefficients, and thresholds -- so a
grade can be recomputed, inspected, and audited by hand.

This candidate was selected over four alternatives (linear least squares,
linear ridge, an adaptive linear/quadratic design selector, and rank-transformed
least squares) by a prespecified equal-weight composite of three external
validity domains computed strictly out-of-fold: FreeSurfer hippocampal
volumetry, six cognitive outcomes, and CSF biomarkers.
"""

import numpy as np

from ecvrs.calibration.ordinal import (
    train_kappa_optimized_mapping,
    predict_kappa_optimized_mapping,
)

LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
N_INNER_FOLDS = 4


def quadratic_design(X):
    """Expand the feature panel into the explicit quadratic basis [x, x^2]."""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, X ** 2])


def _zstats(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def _ridge_fit(Xz, y, lam):
    """Ridge solution on standardized features, with the intercept held out."""
    p = Xz.shape[1]
    beta = np.linalg.solve(Xz.T @ Xz + lam * np.eye(p), Xz.T @ (y - y.mean()))
    return float(y.mean()), beta


def select_lambda(X_train, y_train, seed=42, lambdas=LAMBDAS):
    """Pick the ridge penalty by inner cross-validation on the training fold."""
    D = quadratic_design(X_train)
    y_train = np.asarray(y_train, dtype=float)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(y_train))
    rng.shuffle(idx)
    folds = np.array_split(idx, N_INNER_FOLDS)

    best_lam, best_err = lambdas[0], np.inf
    for lam in lambdas:
        errs = []
        for k in range(N_INNER_FOLDS):
            te = folds[k]
            tr = np.hstack([folds[j] for j in range(N_INNER_FOLDS) if j != k])
            mu, sd = _zstats(D[tr])
            Ztr = (D[tr] - mu) / sd
            Zte = (D[te] - mu) / sd
            try:
                b0, b = _ridge_fit(Ztr, y_train[tr], lam)
            except np.linalg.LinAlgError:
                errs.append(np.inf)
                continue
            errs.append(np.mean((y_train[te] - (b0 + Zte @ b)) ** 2))
        err = float(np.mean(errs))
        if err < best_err:
            best_lam, best_err = lam, err
    return best_lam


def train_quadratic_ridge_mapping(X_train, y_train, num_classes, seed=42):
    """Fit the published calibration on a training fold only.

    Returns a dict with the standardization constants, the ridge penalty chosen
    by inner CV, the explicit coefficients, and the ordinal grade cutoffs.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    lam = select_lambda(X_train, y_train, seed=seed)

    D = quadratic_design(X_train)
    mu, sd = _zstats(D)
    Z = (D - mu) / sd
    try:
        b0, beta = _ridge_fit(Z, y_train, lam)
    except np.linalg.LinAlgError:
        lam = 0.01
        b0, beta = _ridge_fit(Z, y_train, lam)

    score_train = b0 + Z @ beta
    thresholds = train_kappa_optimized_mapping(score_train, y_train, num_classes)

    return {
        'variant': 'quadratic',
        'lam': lam,
        'mu': mu,
        'sd': sd,
        'intercept': b0,
        'beta': beta,
        'thresholds': thresholds,
        'num_classes': num_classes,
    }


def score_quadratic_ridge(X, model):
    """Continuous (pre-threshold) score, exposed for inspection and plotting."""
    D = quadratic_design(X)
    Z = (D - model['mu']) / model['sd']
    return model['intercept'] + Z @ model['beta']


def predict_quadratic_ridge_mapping(X, model):
    """Ordinal grades for held-out data using a model fitted on training data."""
    return predict_kappa_optimized_mapping(
        score_quadratic_ridge(X, model), model['thresholds'], model['num_classes'])
