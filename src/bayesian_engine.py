import numpy as np
from scipy.stats import beta

def run_bayesian_analysis(n_a, k_a, n_b, k_b, samples=100000):
    """
    Computes the probability that B is better than A and the expected loss.
    Uses Beta-Binomial conjugacy for the posterior.
    """
    # Posterior parameters (using a flat Beta(1,1) prior)
    # Posterior A ~ Beta(k_a + 1, n_a - k_a + 1)
    s_a = beta.rvs(k_a + 1, (n_a - k_a) + 1, size=samples)
    s_b = beta.rvs(k_b + 1, (n_b - k_b) + 1, size=samples)
    
    # Probability B > A
    prob_b_better = (s_b > s_a).mean()
    
    # Expected Loss (The "Regret" of choosing B if A is actually better)
    # Logic: max(A - B, 0)
    loss_b = np.maximum(s_a - s_b, 0).mean()
    
    # 95% High Density Interval (Credible Interval)
    hdi_a = np.percentile(s_a, [2.5, 97.5])
    hdi_b = np.percentile(s_b, [2.5, 97.5])
    
    return {
        "prob_better": prob_b_better,
        "expected_loss": loss_b,
        "hdi_a": hdi_a,
        "hdi_b": hdi_b,
        "samples_a": s_a,
        "samples_b": s_b
    }