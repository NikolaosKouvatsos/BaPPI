import numpy as np
from scipy.stats import beta

def compute_posterior_metrics(n_a, k_a, n_b, k_b, samples=100000):
    """Calculates Bayesian metrics for an A/B test."""
    # Generate Samples
    s_a = beta.rvs(k_a + 1, (n_a - k_a) + 1, size=samples)
    s_b = beta.rvs(k_b + 1, (n_b - k_b) + 1, size=samples)
    
    # Prob B > A
    prob_b_better = (s_b > s_a).mean()
    
    # Expected Loss
    loss_b = np.maximum(s_a - s_b, 0).mean()
    
    return {
        "prob_b_better": prob_b_better,
        "expected_loss": loss_b,
        "hdi_a": np.percentile(s_a, [2.5, 97.5]),
        "hdi_b": np.percentile(s_b, [2.5, 97.5])
    }