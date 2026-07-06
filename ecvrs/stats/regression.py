import numpy as np
import scipy.stats as stats

def fit_ols(X, y):
    """
    Fits an Ordinary Least Squares (OLS) linear regression model.
    X: design matrix (shape n x p), should NOT include intercept.
    y: target vector (shape n)
    """
    y = np.array(y, dtype=float)
    X = np.array(X, dtype=float)
    n, p_features = X.shape
    
    # Add intercept column
    X_design = np.hstack([np.ones((n, 1)), X])
    p = p_features + 1 # total parameters including intercept
    
    # Solve beta = (X^T X)^-1 X^T y
    try:
        beta = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
        
    y_pred = X_design @ beta
    residuals = y - y_pred
    rss = np.sum(residuals ** 2)
    
    y_mean = np.mean(y)
    tss = np.sum((y - y_mean) ** 2)
    
    r_squared = 1.0 - (rss / tss) if tss != 0 else 0.0
    
    # Adjusted R-squared
    if n > p:
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - p)
    else:
        adj_r_squared = r_squared
        
    # Variance of beta
    sigma_sq = rss / (n - p) if n > p else 0.0
    try:
        cov_beta = sigma_sq * np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(cov_beta))
    except np.linalg.LinAlgError:
        cov_beta = sigma_sq * np.linalg.pinv(X_design.T @ X_design)
        se = np.sqrt(np.maximum(0.0, np.diag(cov_beta)))
        
    t_stats = np.zeros(p)
    p_values = np.ones(p)
    for j in range(p):
        if se[j] > 0:
            t_stats[j] = beta[j] / se[j]
            # 2-tailed p-value using Student's t distribution
            p_values[j] = 2.0 * (1.0 - stats.t.cdf(abs(t_stats[j]), df=n - p))
            
    return {
        'coef': beta,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'rss': rss,
        'n': n,
        'p': p,
        'y_pred': y_pred
    }

def compare_models(model_restricted, model_unrestricted):
    """
    Performs an F-test to compare nested regression models.
    model_restricted: dict from fit_ols (fewer parameters)
    model_unrestricted: dict from fit_ols (more parameters)
    """
    rss_r = model_restricted['rss']
    rss_u = model_unrestricted['rss']
    n = model_unrestricted['n']
    p_r = model_restricted['p']
    p_u = model_unrestricted['p']
    
    if p_u <= p_r:
        raise ValueError("Unrestricted model must have more parameters than restricted model.")
        
    df1 = p_u - p_r
    df2 = n - p_u
    
    numerator = (rss_r - rss_u) / df1
    denominator = rss_u / df2
    
    if denominator == 0:
        f_stat = 0.0
        p_val = 1.0
    else:
        f_stat = numerator / denominator
        # Cumulative F-distribution
        p_val = 1.0 - stats.f.cdf(f_stat, df1, df2)
        
    return f_stat, p_val

def fdr_correction(p_values):
    """
    Applies Benjamini-Hochberg FDR correction.
    Returns adjusted p-values.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]
    
    adjusted_p = np.zeros(n)
    for i in range(n):
        rank = i + 1
        # FDR factor is n / rank
        adjusted_p[sort_idx[i]] = sorted_p[i] * (n / rank)
        
    # Ensure monotonicity: adjusted p-value must be non-decreasing
    for i in range(n - 2, -1, -1):
        idx_curr = sort_idx[i]
        idx_next = sort_idx[i + 1]
        if adjusted_p[idx_curr] > adjusted_p[idx_next]:
            adjusted_p[idx_curr] = adjusted_p[idx_next]
            
    # Clip to max of 1.0
    return np.clip(adjusted_p, 0.0, 1.0)
