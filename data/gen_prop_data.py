import numpy as np
import pandas as pd

def generate_property_data(n_samples=1000):
    """
    Generates synthetic real estate data based on multivariate distributions.
    """
    
    # 1. House or Flat? (75% Flat / 25% House)
    prop_types = np.random.choice(['Flat', 'House'], size=n_samples, p=[0.75, 0.25])
    
    # 2. Number of Rooms (Poisson distribution, mean=2.5)
    # We add 1 because a 0-room property is impossible in this context
    rooms = np.random.poisson(lam=2.5, size=n_samples) + 1
    
    # 3. Distance from City Centre (Exponential distribution, scale=5km)
    # Most properties are near the centre; fewer are far away.
    distance = np.random.exponential(scale=5.0, size=n_samples)
    
    # 4. Outdoor Space (Garden: 30%, Balcony: 10%, Terrace: 20%, None: 40%)
    outdoor_options = ['Garden', 'Balcony', 'Terrace', 'None']
    outdoor_space = np.random.choice(outdoor_options, size=n_samples, p=[0.3, 0.1, 0.2, 0.4])
    
    # 5. Underground within 500m (Bernoulli distribution, p=0.3)
    # 0 (No) is more likely (70%) than 1 (Yes) (30%)
    underground = np.random.binomial(n=1, p=0.3, size=n_samples)
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'property_type': prop_types,
        'n_rooms': rooms,
        'dist_centre_km': distance,
        'outdoor_space': outdoor_space,
        'near_underground': underground
    })
    
    return df

if __name__ == "__main__":
    # Quick test run
    data = generate_property_data(10)
    print("--- Sample Generated Data ---")
    print(data.head())