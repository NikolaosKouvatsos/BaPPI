import numpy as np
import pandas as pd

def generate_property_data(n_samples=1000, seed=None):
    """
    Generates structural features for synthetic London properties.
    
    Units & Distributions:
    - property_type: Categorical (Flat or House).
    - n_rooms: Discrete (Poisson mean 2.5, shifted by +1). Represents total rooms.
    - dist_centre_km: Continuous (Exponential scale 5.0). Distance in Kilometers.
    - outdoor_space: Categorical (Garden, Balcony, Terrace, Nothing).
    - near_underground: Binary (Bernoulli p=0.3). 1 = <500m to station, 0 = far.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 75% Flats, 25% Houses (Typical for London rental market)
    prop_types = np.random.choice(['Flat', 'House'], size=n_samples, p=[0.75, 0.25])
    
    # Poisson + 1 ensures we don't have 0-room properties. Mean ~3.5 rooms.
    rooms = np.random.poisson(lam=2.5, size=n_samples) + 1
    
    # Exponential distribution: most properties are concentrated near the center
    distance = np.random.exponential(scale=5.0, size=n_samples)
    
    outdoor_options = ['Garden', 'Balcony', 'Terrace', 'Nothing']
    outdoor_space = np.random.choice(outdoor_options, size=n_samples, p=[0.3, 0.1, 0.2, 0.4])
    
    # Bernoulli: 0 (No) is 70% likely, 1 (Yes) is 30% likely
    underground = np.random.binomial(n=1, p=0.3, size=n_samples)
    
    df = pd.DataFrame({
        'property_type': prop_types,
        'n_rooms': rooms,
        'dist_centre_km': distance,
        'outdoor_space': outdoor_space,
        'near_underground': underground
    })
    
    return df

def generate_price(df, model_case='A', sigma=0.1, seed=None):
    """
    Calculates monthly rental price (GBP) using a Log-Linear model.
    
    The formula follows: ln(Price) = Intercept + sum(beta * feature) + Bias + noise
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data from generate_property_data.
    model_case : str
        'A' for Owner (Direct) - No markup.
        'B' for Agent - Includes a ~16% (0.15 log-point) premium.
    sigma : float
        Standard deviation of the Gaussian noise in log-space. 
        Reflects market volatility (0.1 = ~10% fluctuation).
        
    Returns:
    --------
    pd.Series
        Monthly rental prices in GBP (£).
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Base Parameters (Log-Space)
    # Intercept of 7.5 corresponds to a base price of exp(7.5) ≈ £1,800
    intercept = 7.5  
    beta_room = 0.15  # Each additional room adds ~15% to price
    beta_dist = -0.05 # Each km from city centre reduces price by ~5%
    beta_underground = 0.10 # Being near a station adds ~10%
    
    # 2. Categorical Offsets (Log-Space)
    # Houses carry a ~22% premium over Flats (exp(0.2))
    type_map = {'House': 0.2, 'Flat': 0.0}
    # Amenities add progressive percentage bonuses
    outdoor_map = {'Garden': 0.15, 'Terrace': 0.1, 'Balcony': 0.05, 'Nothing': 0.0}
    
    # 3. Model Logic (The Hidden Signal)
    # Case B adds a 0.15 bias, which shifts the price distribution rightward.
    agent_premium = 0.15 if model_case == 'B' else 0.0
    
    # 4. Core Linear Combination
    # We add all log-contributions together.
    log_price = (
        intercept + 
        (df['n_rooms'] * beta_room) +
        (df['dist_centre_km'] * beta_dist) +
        (df['near_underground'] * beta_underground) +
        df['property_type'].map(type_map) +
        df['outdoor_space'].map(outdoor_map) +
        agent_premium
    )
    
    # 5. Stochasticity
    # Adding epsilon ~ N(0, sigma^2) makes the inference problem non-trivial.
    noise = np.random.normal(0, sigma, size=len(df))
    log_price_final = log_price + noise
    
    # 6. Transform back to GBP
    # exp(log_price) converts additive percentage changes into multiplicative ones.
    return np.exp(log_price_final)

if __name__ == "__main__":
    # --- GLOBAL SEED SETTING ---
    MASTER_SEED = 42

    # Generate 1000 properties
    n = 1000
    data = generate_property_data(n, seed=MASTER_SEED)
    
    # Assign half to Owner (A) and half to Agent (B) to create a balanced dataset
    # This represents our prior P(A) = P(B) = 0.5
    half = n // 2
    prices_a = generate_price(data.iloc[:half], model_case='A', seed=MASTER_SEED)
    prices_b = generate_price(data.iloc[half:], model_case='B', seed=MASTER_SEED + 1)
    
    data['monthly_rent_gbp'] = pd.concat([prices_a, prices_b]).values
    data['listing_type'] = ['Owner']*half + ['Agent']*half
    
    # Output results
    print(f"--- Dataset Generated with Seed {MASTER_SEED} ---")
    print("--- London Rental Dataset (First 10 Rows) ---")
    print(data.head(10))
    print("\nMean Rent (Owner):", round(data[data['listing_type']=='Owner']['monthly_rent_gbp'].mean(), 2))
    print("Mean Rent (Agent):", round(data[data['listing_type']=='Agent']['monthly_rent_gbp'].mean(), 2))

    # Optional: Save for the next step
    data.to_csv("data/london_rentals.csv", index=False)
    print("\nFile saved as london_rentals.csv")