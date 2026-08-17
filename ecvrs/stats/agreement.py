import numpy as np

def cohen_kappa_quadratic(y_true, y_pred, num_classes):
    """Computes Cohen's quadratic weighted Kappa."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    O = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            O[t, p] += 1
        
    w = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
            
    hist_true = np.sum(O, axis=1)
    hist_pred = np.sum(O, axis=0)
    E = np.outer(hist_true, hist_pred) / len(y_true)
    
    num = np.sum(w * O)
    den = np.sum(w * E)
    
    if den == 0:
        return 1.0 if num == 0 else 0.0
        
    return 1.0 - (num / den)

def intraclass_correlation(y_true, y_pred):
    """Computes ICC(2,1) - two-way random effects, single rater, absolute agreement."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    targets = np.concatenate([y_true, y_pred])
    n = len(y_true)
    k = 2
    
    grand_mean = np.mean(targets)
    sst = np.sum((targets - grand_mean) ** 2)
    
    subject_means = np.array([np.mean([y_true[i], y_pred[i]]) for i in range(n)])
    ssr = k * np.sum((subject_means - grand_mean) ** 2)
    msr = ssr / (n - 1)
    
    rater_means = np.array([np.mean(y_true), np.mean(y_pred)])
    ssc = n * np.sum((rater_means - grand_mean) ** 2)
    msc = ssc / (k - 1)
    
    sse = sst - ssr - ssc
    mse = sse / ((n - 1) * (k - 1))
    
    denominator = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if denominator == 0:
        return 0.0
        
    icc = (msr - mse) / denominator
    return icc

def bootstrap_confidence_intervals(y_true, y_pred, metric_func, num_classes=None, n_bootstrap=2000, seed=42):
    """
    Computes 95% bootstrap confidence intervals for a given agreement metric.
    """
    np.random.seed(seed)
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    estimates = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_true = y_true[boot_idx]
        boot_pred = y_pred[boot_idx]
        
        if num_classes is not None:
            val = metric_func(boot_true, boot_pred, num_classes)
        else:
            val = metric_func(boot_true, boot_pred)
            
        estimates.append(val)
        
    estimates = np.sort(estimates)
    ci_lower = np.percentile(estimates, 2.5)
    ci_upper = np.percentile(estimates, 97.5)
    
    return ci_lower, ci_upper
