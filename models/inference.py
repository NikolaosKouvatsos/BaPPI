from src.bayesian_engine import run_bayesian_analysis

def get_experiment_results(n_a, k_a, n_b, k_b):
    """Wraps the engine with validation and business logic."""
    # Data Validation
    if any(val < 0 for val in [n_a, k_a, n_b, k_b]):
        raise ValueError("Sample sizes and conversions must be non-negative.")
    if k_a > n_a or k_b > n_b:
        raise ValueError("Conversions cannot exceed total sample size.")
        
    # Execution
    results = run_bayesian_analysis(n_a, k_a, n_b, k_b)
    
    # Logic for Decision Status
    # Standard industry threshold for Expected Loss is often 0.001 (0.1%)
    results['status'] = "SWITCH" if results['expected_loss'] < 0.001 else "CONTINUE TEST"
    
    return results