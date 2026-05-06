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

# --- PACKAGING THE FIXED PARAMETERS ---
    # We broadcast the single values into arrays so they match the dataframe length
    n = len(df)
    fixed_params = {
        "intercept": np.full(n, intercept),
        "beta_house": np.full(n, type_map["House"]),
        "beta_room": np.full(n, beta_room),
        "beta_dist": np.full(n, beta_dist),
        "beta_garden": np.full(n, outdoor_map["Garden"]),
        "beta_terrace": np.full(n, outdoor_map["Terrace"]),
        "beta_balcony": np.full(n, outdoor_map["Balcony"]),
        "beta_under": np.full(n, beta_under),
        "premium": np.full(n, agent_premium)
    }

    prices = np.exp(log_price + noise)

    return prices, fixed_params

def generate_price_rand_params(df, model_case='A', market_volatility=0.03, seed=None):
    """
    STOCHASTIC (RANDOM) PARAMETER PRICE GENERATOR - Non-Gaussian Version
    --------------------------------------------------------------------
    Returns:
    --------
    prices : np.ndarray
        The generated prices in original currency units (exp of log-price).
    sampled_params : dict
        The actual property-level parameter values (arrays of size n) 
        randomly generated for this specific dataset.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(df)
    
    # Target central values (used to center the random sampling)
    targets = {
        "intercept": 7.5, "beta_room": 0.15, "beta_dist": -0.05, 
        "beta_under": 0.10, "beta_house": 0.20, "beta_garden": 0.15, 
        "beta_terrace": 0.10, "beta_balcony": 0.05,
        "premium": 0.15 if model_case == 'B' else 0.0
    }
    
    # --- STOCHASTIC GENERATION (Individual values for every row) ---
    intercepts = np.random.normal(targets["intercept"], market_volatility, size=n)
    betas_dist = np.random.normal(targets["beta_dist"], market_volatility, size=n)
    betas_under = np.random.normal(targets["beta_under"], market_volatility, size=n)
    betas_room = np.random.laplace(targets["beta_room"], market_volatility, size=n)
    
    if model_case == 'B':
        shape = 5.0
        scale = targets["premium"] / shape
        agent_premium = np.random.gamma(shape, scale, size=n)
    else:
        agent_premium = np.zeros(n)
    
    house_bonus = np.random.normal(targets["beta_house"], market_volatility, size=n)
    garden_bonus = np.random.normal(targets["beta_garden"], market_volatility, size=n)
    terrace_bonus = np.random.normal(targets["beta_terrace"], market_volatility, size=n)
    balcony_bonus = np.random.normal(targets["beta_balcony"], market_volatility, size=n)
    
    # --- CALCULATION ---
    # Map bonuses based on property structural features
    type_impact = np.where(df['property_type'] == 'House', house_bonus, 0.0)
    
    # Determine the specific outdoor bonus applied to each row
    outdoor_impact = np.zeros(n)
    outdoor_impact += np.where(df['outdoor_space'] == 'Garden', garden_bonus, 0.0)
    outdoor_impact += np.where(df['outdoor_space'] == 'Terrace', terrace_bonus, 0.0)
    outdoor_impact += np.where(df['outdoor_space'] == 'Balcony', balcony_bonus, 0.0)
    
    log_price = (
        intercepts + 
        (df['n_rooms'] * betas_room) + 
        (df['dist_centre_km'] * betas_dist) + 
        (df['near_underground'] * betas_under) + 
        type_impact + 
        outdoor_impact + 
        agent_premium
    )
    
    noise = np.random.standard_t(df=5, size=n) * 0.1
    prices = np.exp(log_price + noise)

    # --- PACKAGING THE SAMPLED PARAMETERS ---
    # We collect the specific values used for each observation to save to our CSV
    sampled_params = {
        "intercept": intercepts,
        "beta_house": house_bonus,
        "beta_room": betas_room,
        "beta_dist": betas_dist,
        "beta_garden": garden_bonus,
        "beta_terrace": terrace_bonus,
        "beta_balcony": balcony_bonus,
        "beta_under": betas_under,
        "premium": agent_premium
    }
    
    return prices, sampled_params

if __name__ == "__main__":
    # --- SETUP & CONFIGURATION ---
    MASTER_SEED = 42
    
    # MODE SWITCH: 
    # 'fixed' -> Standard Log-Linear model (london_rentals_fix_params.csv)
    # 'random' -> Stochastic Market model (london_rentals_rand_params.csv)
    MODE = 'fixed' 
    
    # Generate common structural features for 1000 properties
    data = generate_property_data(1000, seed=MASTER_SEED)
    half = len(data) // 2
        
# --- DATA GENERATION EXECUTION ---
    if MODE == 'fixed':
        # Unpack both the prices and the parameter dictionaries
        prices_a, params_a = generate_price_fix_params(data.iloc[:half], model_case='A', seed=MASTER_SEED)
        prices_b, params_b = generate_price_fix_params(data.iloc[half:], model_case='B', seed=MASTER_SEED + 1)
        filename = "data/london_rentals_fix_params.csv"
        
        # Merge the parameter dictionaries into the main dataframe
        for key in params_a.keys():
            data[key] = np.concatenate([params_a[key], params_b[key]])
            
    else:
        # Generate 500 Owners and 500 Agents using stochastic property-level betas
        # Unpack the price array and the dictionary of sampled random parameters
        prices_a, params_a = generate_price_rand_params(data.iloc[:half], model_case='A', seed=MASTER_SEED)
        prices_b, params_b = generate_price_rand_params(data.iloc[half:], model_case='B', seed=MASTER_SEED + 1)
        filename = "data/london_rentals_rand_params.csv"

        # Merge the dictionary of sampled random parameters into the main dataframe
        for key in params_a.keys():
            data[key] = np.concatenate([params_a[key], params_b[key]])

    # Merge structural features with generated prices and labels
    # Use pd.Series to ensure smooth concatenation before taking .values
    data['monthly_rent_gbp'] = np.concatenate([prices_a, prices_b])
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
