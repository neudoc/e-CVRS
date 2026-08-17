import numpy as np
from scipy.optimize import minimize
from ecvrs.stats.agreement import cohen_kappa_quadratic
from ecvrs.calibration.quantile import train_quantile_mapping, predict_quantile_mapping

def train_kappa_optimized_mapping(train_ratios, train_manual, num_classes):
    """
    Directly optimizes ratio-to-score thresholds on the training set to maximize quadratic weighted Kappa.
    """
    # Use quantile thresholds as starting values
    x0 = train_quantile_mapping(train_ratios, train_manual, num_classes)
    
    def loss(thresholds):
        # Ensure thresholds remain sorted during search
        sorted_thresh = np.sort(thresholds)
        preds = predict_quantile_mapping(train_ratios, sorted_thresh, num_classes)
        return -cohen_kappa_quadratic(train_manual, preds, num_classes)
        
    # Nelder-Mead works well for discontinuous/step-like objective functions
    res = minimize(loss, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4})
    
    # Return sorted optimized thresholds
    return np.sort(res.x)

def predict_kappa_optimized_mapping(test_ratios, thresholds, num_classes):
    """Predicts scores using optimized thresholds."""
    return predict_quantile_mapping(test_ratios, thresholds, num_classes)
