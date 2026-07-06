import numpy as np

def train_quantile_mapping(train_ratios, train_manual, num_classes):
    """
    Calibrates ratio-to-score thresholds based on target quantile proportions of training manual scores.
    """
    n = len(train_ratios)
    if n == 0:
        return [0.0] * (num_classes - 1)
        
    counts = [np.sum(train_manual == c) for c in range(num_classes)]
    cum_props = np.cumsum(counts) / n
    
    thresholds = []
    for p in cum_props[:-1]:
        if p == 0:
            thresholds.append(-1.0)
        else:
            thresholds.append(np.percentile(train_ratios, p * 100))
            
    return thresholds

def predict_quantile_mapping(test_ratios, thresholds, num_classes):
    """Maps continuous ratios to discrete classes using calibrated thresholds."""
    preds = []
    for r in test_ratios:
        assigned_class = num_classes - 1
        for class_idx, thresh in enumerate(thresholds):
            if r < thresh:
                assigned_class = class_idx
                break
        preds.append(assigned_class)
    return np.array(preds)
