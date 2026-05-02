import pandas as pd
import numpy as np

def generate_experiment_data(n_users=2000):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Half the users see Version A, half see Version B
    groups = ['A', 'B']
    user_groups = np.random.choice(groups, size=n_users)
    
    # Conversion rates (Version B is slightly better)
    # A = 10% conversion, B = 12% conversion
    rates = {'A': 0.10, 'B': 0.12}
    
    results = []
    for group in user_groups:
        # Simulate a click (1) or no click (0)
        conversion = np.random.binomial(1, rates[group])
        results.append(conversion)
    
    df = pd.DataFrame({
        'user_id': range(n_users),
        'group': user_groups,
        'converted': results
    })
    
    return df

if __name__ == "__main__":
    df = generate_experiment_data()
    df.to_csv('data/experiment_results.csv', index=False)
    print("File 'data/experiment_results.csv' created successfully.")