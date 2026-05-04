import numpy as np
import pandas as pd
import os

def generate_property_data(n_samples=1000, seed=None):
    """
    Generates the structural 'Physical' features of synthetic London properties.
    
    This function creates the independent variables (X) that define a property's
    intrinsic profile before any pricing logic is applied.
    
    Parameters:
    -----------
    n_samples : int
        Number of properties to generate.
    seed : int, optional
        Reproducibility seed for the structural features.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Typical London split: mostly flats (75%), fewer houses (25%)
    prop_types = np.random.choice(['Flat', 'House'], size=n_samples, p=[0.75, 0.25])
    
    # Rooms follow a Poisson distribution (mean 2.5) + 1 to avoid 'zero-room' homes.
    # This results in a realistic range of 1 to 6+ rooms.
    rooms = np.random.poisson(lam=2.5, size=n_samples) + 1
    
    # Distance from city center follows an Exponential distribution.
    # Most properties are clustered near the center; few are very far out.
    distance = np.random.exponential(scale=5.0, size=n_samples)
    
    # Categorical amenities with varied probability of occurrence.
    outdoor_options = ['Garden', 'Balcony', 'Terrace', 'Nothing']
    outdoor_space = np.random.choice(outdoor_options, size=n_samples, p=[0.3, 0.1, 0.2, 0.4])
    
    # Proximity to London Underground (Binary Bernoulli trial with 30% 'Yes').
    underground = np.random.binomial(n=1, p=0.3, size=n_samples)
    
    return pd.DataFrame({
        'property_type': prop_types,
        'n_rooms': rooms,
        'dist_centre_km': distance,
        'outdoor_space': outdoor_space,
        'near_underground': underground
    })

def generate_price_fix_params(df, model_case='A', sigma=0.1, seed=None):
    """
    FIXED PARAMETER PRICE GENERATOR (Log-Linear Model)
    -------------------------------------------------
    In this model, every house follows identical market rules. The impact 
    of a 'Room' or 'Garden' is a universal constant for all properties.
    
    Logic: ln(Price) = Intercept + (Beta * Feature) + Agent_Premium + Noise
    
    Parameters:
    -----------
    model_case : str
        'A' for Owner (Direct) - No markup.
        'B' for Agent - Exactly 0.15 log-point (~16%) premium.
    sigma : float
        Standard deviation of the residual noise (market chaos).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Universal Coefficients (Point Estimates)
    intercept, beta_room, beta_dist, beta_under = 7.5, 0.15, -0.05, 0.10
    type_map = {'House': 0.2, 'Flat': 0.0}
    outdoor_map = {'Garden': 0.15, 'Terrace': 0.1, 'Balcony': 0.05, 'Nothing': 0.0}
    
    # 2. Case Selection: Shift the entire distribution by a fixed amount if Agent
    agent_premium = 0.15 if model_case == 'B' else 0.0
    
    # 3. Calculation: Additive log-contributions become multiplicative in GBP
    log_price = (
        intercept + 
        (df['n_rooms'] * beta_room) + 
        (df['dist_centre_km'] * beta_dist) + 
        (df['near_underground'] * beta_under) + 
        df['property_type'].map(type_map) + 
        df['outdoor_space'].map(outdoor_map) + 
        agent_premium
    )
    
    # Add homogeneous Gaussian noise
    noise = np.random.normal(0, sigma, size=len(df))
    return np.exp(log_price + noise)

def generate_price_rand_params(df, model_case='A', market_volatility=0.03, seed=None):
    """
    STOCHASTIC (RANDOM) PARAMETER PRICE GENERATOR - Non-Gaussian Version
    --------------------------------------------------------------------
    Features realistic, non-idealized priors to simulate market complexity:
    - Laplace: For 'fat-tailed' room values (more outliers).
    - Gamma: For Agent Premiums (strictly positive, right-skewed).
    - Normal: For Intercept and Distance (standard market noise).
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(df)
    
    # 1. Intercept (Normal)
    # The base cost remains Gaussian as it represents the central market equilibrium.
    intercepts = np.random.normal(7.5, market_volatility, size=n)
    
    # 2. Beta Room (Laplace / Double Exponential)
    # REALISM: The value of an extra room often has 'fat tails'. 
    # Most are standard, but some luxury or tiny rooms vary wildly.
    betas_room = np.random.laplace(0.15, market_volatility, size=n)
    
    # 3. Beta Distance & Underground (Normal)
    betas_dist = np.random.normal(-0.05, market_volatility, size=n)
    betas_under = np.random.normal(0.10, market_volatility, size=n)
    
    # 4. Agent Premium (Gamma Distribution)
    # REALISM: An Agent Premium cannot be negative. A Gamma distribution 
    # is strictly positive and right-skewed, meaning most agents charge a 
    # moderate fee, but a few 'premium' agents charge significantly more.
    if model_case == 'B':
        # Shape (k) and Scale (theta). Mean = k*theta. 
        # We target a mean of 0.15.
        shape = 5.0
        scale = 0.15 / shape
        agent_premium = np.random.gamma(shape, scale, size=n)
    else:
        agent_premium = 0.0
    
    # 5. Categorical Effects (Normal for simplicity)
    house_bonus_dist = np.random.normal(0.20, market_volatility, size=n)
    garden_bonus_dist = np.random.normal(0.15, market_volatility, size=n)
    terrace_bonus_dist = np.random.normal(0.10, market_volatility, size=n)
    balcony_bonus_dist = np.random.normal(0.05, market_volatility, size=n)
    
    # 6. Mapping: Connecting structural choices to their specific random betas
    type_impact = np.where(df['property_type'] == 'House', house_bonus_dist, 0.0)
    
    outdoor_impact = np.zeros(n)
    outdoor_impact += np.where(df['outdoor_space'] == 'Garden', garden_bonus_dist, 0.0)
    outdoor_impact += np.where(df['outdoor_space'] == 'Terrace', terrace_bonus_dist, 0.0)
    outdoor_impact += np.where(df['outdoor_space'] == 'Balcony', balcony_bonus_dist, 0.0)
    
    # 7. Calculation
    log_price = (
        intercepts + 
        (df['n_rooms'] * betas_room) + 
        (df['dist_centre_km'] * betas_dist) + 
        (df['near_underground'] * betas_under) + 
        type_impact + 
        outdoor_impact + 
        agent_premium
    )
    
    # 8. Unexplained Residual (Student's T-Distribution)
    # REALISM: 'Market Noise' in the real world has more 'black swan' events 
    # than a Normal distribution. A T-distribution with low degrees of freedom (df=5)
    # creates more extreme price outliers.
    noise = np.random.standard_t(df=5, size=n) * 0.1
    
    return np.exp(log_price + noise)

if __name__ == "__main__":
    # --- SETUP & CONFIGURATION ---
    MASTER_SEED = 42
    
    # MODE SWITCH: 
    # 'fixed' -> Standard Log-Linear model (london_rentals_fix_params.csv)
    # 'random' -> Stochastic Market model (london_rentals_rand_params.csv)
    MODE = 'random' 
    
    # Generate common structural features for 1000 properties
    data = generate_property_data(1000, seed=MASTER_SEED)
    half = len(data) // 2
        
    # --- DATA GENERATION EXECUTION ---
    if MODE == 'fixed':
        # Generate 500 Owners and 500 Agents using identical beta rules
        prices_a = generate_price_fix_params(data.iloc[:half], model_case='A', seed=MASTER_SEED)
        prices_b = generate_price_fix_params(data.iloc[half:], model_case='B', seed=MASTER_SEED + 1)
        filename = "data/london_rentals_fix_params.csv"
    else:
        # Generate 500 Owners and 500 Agents using stochastic property-level betas
        prices_a = generate_price_rand_params(data.iloc[:half], model_case='A', seed=MASTER_SEED)
        prices_b = generate_price_rand_params(data.iloc[half:], model_case='B', seed=MASTER_SEED + 1)
        filename = "data/london_rentals_rand_params.csv"

    # Merge structural features with generated prices and labels
    data['monthly_rent_gbp'] = pd.concat([prices_a, prices_b]).values
    data['listing_type'] = ['Owner']*half + ['Agent']*half
    
    # --- EXPORT ---
    data.to_csv(filename, index=False)
    print(f"--- Dataset Generated with Seed {MASTER_SEED} ---")
    print(f"--- Mode: {MODE.upper()} | Observations: {len(data)} ---")
    print("--- London Rental Dataset (First 10 Rows) ---")
    print(data.head(10))
    print("\nMean Rent (Owner):", round(data[data['listing_type']=='Owner']['monthly_rent_gbp'].mean(), 2))
    print("Mean Rent (Agent):", round(data[data['listing_type']=='Agent']['monthly_rent_gbp'].mean(), 2))
    print(f"--- File saved as {filename} ---")
