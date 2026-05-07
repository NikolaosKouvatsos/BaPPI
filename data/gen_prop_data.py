import numpy as np
import pandas as pd
import json

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
    In this model, every house follows identical market rules. 
    Returns:
        prices: np.ndarray of property prices.
        fixed_params: dict containing the actual constant betas applied per row.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(df)
    
    # 1. Universal Coefficients (Point Estimates)
    intercept, beta_room, beta_dist, beta_under = 7.5, 0.15, -0.05, 0.10
    type_map = {'House': 0.2, 'Flat': 0.0}
    outdoor_map = {'Garden': 0.15, 'Terrace': 0.1, 'Balcony': 0.05, 'Nothing': 0.0}
    
    # 2. Case Selection: Constant premium for all observations in Case B
    agent_premium = 0.15 if model_case == 'B' else 0.0
    
    # 3. Structural Beta Mapping (Fixed but conditional on the feature)
    beta_prop = df['property_type'].map(type_map).values
    beta_outdoor = df['outdoor_space'].map(outdoor_map).values
    
    # NEW: Calculate the applied underground beta (0.1 if 1, 0.0 if 0)
    applied_beta_under = (df['near_underground'] * beta_under).values
    
    # 4. Calculation
    log_price = (
        intercept + 
        (df['n_rooms'] * beta_room) + 
        (df['dist_centre_km'] * beta_dist) + 
        applied_beta_under + # Uses the conditional array
        beta_prop + 
        beta_outdoor + 
        agent_premium
    )
    
    # Add homogeneous Gaussian noise
    noise = np.random.normal(0, sigma, size=n)
    prices = np.exp(log_price + noise)

    # --- PACKAGING THE FIXED PARAMETERS ---
    fixed_params = {
        "intercept": np.full(n, intercept),
        "beta_prop": beta_prop,
        "beta_room": np.full(n, beta_room),
        "beta_dist": np.full(n, beta_dist),
        "beta_outdoor": beta_outdoor,
        "beta_under": applied_beta_under,
        "premium": np.full(n, agent_premium)
    }

    return prices, fixed_params

def generate_price_rand_params(df, model_case='A', market_volatility=0.03, seed=None):
    """
    STOCHASTIC PARAMETER GENERATOR - Optimized Version
    --------------------------------------------------
    Generates property-specific betas based on actual features present.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(df)
    
    # 1. Define the "Global Targets" for the distributions
    targets = {
        "intercept": 7.5, "beta_room": 0.15, "beta_dist": -0.05, 
        "beta_under": 0.10, "beta_house": 0.20,
        "beta_garden": 0.15, "beta_terrace": 0.10, "beta_balcony": 0.05,
        "premium": 0.15 if model_case == 'B' else 0.0
    }
    
    # 2. Generate the "Pool" of potential random betas
    intercepts = np.random.normal(targets["intercept"], market_volatility, size=n)
    betas_room = np.random.laplace(targets["beta_room"], market_volatility, size=n)
    betas_dist = np.random.normal(targets["beta_dist"], market_volatility, size=n)
    
    # Pool for underground: We generate values for everyone, then mask them
    under_pool = np.random.normal(targets["beta_under"], market_volatility, size=n)
    
    # Potential bonuses pool
    house_pool = np.random.normal(targets["beta_house"], market_volatility, size=n)
    garden_pool = np.random.normal(targets["beta_garden"], market_volatility, size=n)
    terrace_pool = np.random.normal(targets["beta_terrace"], market_volatility, size=n)
    balcony_pool = np.random.normal(targets["beta_balcony"], market_volatility, size=n)
    
    if model_case == 'B':
        agent_premium = np.random.gamma(5.0, targets["premium"]/5.0, size=n)
    else:
        agent_premium = np.zeros(n)

    # 3. Apply Logic: Mask betas so the "Truth" is 0 if the feature is absent
    # Underground Beta: Only apply if near_underground == 1
    beta_under_final = np.where(df['near_underground'] == 1, under_pool, 0.0)
    
    # Property Type Beta: 0 if Flat, random house_pool if House
    beta_prop = np.where(df['property_type'] == 'House', house_pool, 0.0)
    
    # Outdoor Beta: Logic switch for the specific outdoor feature
    beta_outdoor = np.zeros(n)
    beta_outdoor = np.where(df['outdoor_space'] == 'Garden', garden_pool, beta_outdoor)
    beta_outdoor = np.where(df['outdoor_space'] == 'Terrace', terrace_pool, beta_outdoor)
    beta_outdoor = np.where(df['outdoor_space'] == 'Balcony', balcony_pool, beta_outdoor)
    
    # 4. Final Calculation
    # Note: Since beta_under_final is already 0 where near_underground is 0, 
    # we just add it directly.
    log_price = (
        intercepts + 
        (df['n_rooms'] * betas_room) + 
        (df['dist_centre_km'] * betas_dist) + 
        beta_under_final + 
        beta_prop + 
        beta_outdoor + 
        agent_premium
    )
    
    noise = np.random.standard_t(df=5, size=n) * 0.1
    prices = np.exp(log_price + noise)

    # 5. Package only the relevant applied parameters
    sampled_params = {
        "intercept": intercepts,
        "beta_prop": beta_prop,
        "beta_room": betas_room,
        "beta_dist": betas_dist,
        "beta_outdoor": beta_outdoor,
        "beta_under": beta_under_final,
        "premium": agent_premium
    }
    
    return prices, sampled_params

def generate_hierarchical_data_advanced(df, model_case='A', market_volatility=0.03, seed=None):
    """
    ADVANCED HIERARCHICAL GENERATOR
    -------------------------------
    - Every feature has a unique Market Mean (mu) and Market Sigma (sigma).
    - Market Sigmas are derived from the base volatility to create diverse spreads.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(df)
    
    # 1. Define Market-Level Means (Hyper-means)
    market_means = {
        "intercept": 7.50, "beta_room": 0.15, "beta_dist": -0.05, 
        "beta_under": 0.10, "beta_house": 0.20,
        "beta_garden": 0.15, "beta_terrace": 0.10, "beta_balcony": 0.05,
        "premium": 0.15 if model_case == 'B' else 0.0
    }
    
    # 2. Define Market-Level Sigmas (Hyper-sigmas)
    # We apply different scaling factors to the base volatility for each feature
    # to simulate different 'standardization' levels in the market.
    vol_scales = {
        "intercept": 0.8,    # Very stable
        "beta_room": 1.5,    # High variation
        "beta_dist": 1.0,    # Baseline
        "beta_under": 0.5,   # Highly standardized bonus
        "beta_house": 1.2,   # Varies by house quality
        "beta_garden": 1.3, 
        "beta_terrace": 1.1,
        "beta_balcony": 0.9,
        "premium": 1.4 if model_case == 'B' else 0.0 # Premium spread
    }
    
    market_sigmas = {k: market_volatility * vol_scales[k] for k in market_means.keys()}

    # 3. LEVEL 1: Generate Individual Property Betas
    # We draw from specific distributions (Normal/Laplace) based on the feature type
    intercepts = np.random.normal(market_means["intercept"], market_sigmas["intercept"], size=n)
    betas_room  = np.random.laplace(market_means["beta_room"], market_sigmas["beta_room"], size=n)
    betas_dist  = np.random.normal(market_means["beta_dist"], market_sigmas["beta_dist"], size=n)
    
    # Generate potential pools for conditional features
    under_pool   = np.random.normal(market_means["beta_under"], market_sigmas["beta_under"], size=n)
    house_pool   = np.random.normal(market_means["beta_house"], market_sigmas["beta_house"], size=n)
    garden_pool  = np.random.normal(market_means["beta_garden"], market_sigmas["beta_garden"], size=n)
    terrace_pool = np.random.normal(market_means["beta_terrace"], market_sigmas["beta_terrace"], size=n)
    balcony_pool = np.random.normal(market_means["beta_balcony"], market_sigmas["beta_balcony"], size=n)
    
    if model_case == 'B':
        shape_k = (market_means["premium"] / market_sigmas["premium"])**2
        scale_theta = (market_sigmas["premium"]**2) / market_means["premium"]
        agent_premium = np.random.gamma(shape_k, scale_theta, size=n)
    else:
        agent_premium = np.zeros(n)

    # 4. Logical Masking (Feature-to-Beta Mapping)
    beta_under_final = np.where(df['near_underground'] == 1, under_pool, 0.0)
    beta_prop = np.where(df['property_type'] == 'House', house_pool, 0.0)
    
    beta_outdoor = np.zeros(n)
    beta_outdoor = np.where(df['outdoor_space'] == 'Garden', garden_pool, beta_outdoor)
    beta_outdoor = np.where(df['outdoor_space'] == 'Terrace', terrace_pool, beta_outdoor)
    beta_outdoor = np.where(df['outdoor_space'] == 'Balcony', balcony_pool, beta_outdoor)
    
    # 5. Calculate Final Log-Prices and Prices
    log_price_mu = (
        intercepts + 
        (df['n_rooms'] * betas_room) + 
        (df['dist_centre_km'] * betas_dist) + 
        beta_under_final + 
        beta_prop + 
        beta_outdoor + 
        agent_premium
    )
    
    # Observation noise (residual)
    noise = np.random.standard_t(df=5, size=n) * 0.05 
    prices = np.exp(log_price_mu + noise)

    # 6. Structured Output for Recovery Analysis
    sampled_params = {
        "individual": {
            "intercept": intercepts,
            "beta_room": betas_room,
            "beta_dist": betas_dist,
            "beta_under": beta_under_final,
            "beta_prop": beta_prop,
            "beta_outdoor": beta_outdoor,
            "premium": agent_premium
        },
        "market": {
            "means": market_means,
            "sigmas": market_sigmas
        }
    }
    
    return prices, sampled_params

if __name__ == "__main__":
    # --- SETUP & CONFIGURATION ---
    MASTER_SEED = 42
    
    # MODE SWITCH: 'fixed', 'random', or 'hierarchical'
    MODE = 'hierarchical' 
    
    data = generate_property_data(1000, seed=MASTER_SEED)
    half = len(data) // 2

    # --- DATA GENERATION EXECUTION ---
    if MODE == 'hierarchical':
        # Advanced Multi-level generation
        prices_a, params_a = generate_hierarchical_data_advanced(
            data.iloc[:half], model_case='A', market_volatility=0.03, seed=MASTER_SEED
        )
        prices_b, params_b = generate_hierarchical_data_advanced(
            data.iloc[half:], model_case='B', market_volatility=0.03, seed=MASTER_SEED + 1
        )
        
        # 1. Map Individual Betas to columns
        ind_a, ind_b = params_a['individual'], params_b['individual']
        for key in ind_a.keys():
            data[key] = np.concatenate([ind_a[key], ind_b[key]])
            
        data['monthly_rent_gbp'] = np.concatenate([prices_a, prices_b])
        data['listing_type'] = ['Owner']*half + ['Agent']*half

        # 2. Package EVERYTHING (Individual + Market) into JSON
        filename = "data/datasets/london_rentals_hierarchical.json"
        full_output = {
            "metadata": {
                "mode": MODE,
                "owner_market_truth": params_a['market'],
                "agent_market_truth": params_b['market']
            },
            "properties": data.to_dict(orient='records')
        }
        with open(filename, 'w') as f:
            json.dump(full_output, f, indent=4)
        
    else:
        # Fixed or Random modes (Standard 2D Logic)
        if MODE == 'fixed':
            prices_a, params_a = generate_price_fix_params(data.iloc[:half], model_case='A', seed=MASTER_SEED)
            prices_b, params_b = generate_price_fix_params(data.iloc[half:], model_case='B', seed=MASTER_SEED + 1)
            filename = "data/datasets/london_rentals_fixed.csv"
        else: # random
            prices_a, params_a = generate_price_rand_params(data.iloc[:half], model_case='A', seed=MASTER_SEED)
            prices_b, params_b = generate_price_rand_params(data.iloc[half:], model_case='B', seed=MASTER_SEED + 1)
            filename = "data/datasets/london_rentals_random.csv"

        # Map flat parameters to columns
        for key in params_a.keys():
            data[key] = np.concatenate([params_a[key], params_b[key]])
            
        data['monthly_rent_gbp'] = np.concatenate([prices_a, prices_b])
        data['listing_type'] = ['Owner']*half + ['Agent']*half
        
        # Save as standard flat CSV
        data.to_csv(filename, index=False)

# --- CONSOLE SUMMARY ---
    print("\n" + "="*50)
    print(f"DATA GENERATION SUMMARY")
    print("="*50)
    print(f"Seed:         {MASTER_SEED}")
    print(f"Mode:         {MODE.upper()}")
    print(f"Observations: {len(data)}")
    print(f"File Saved:   {filename}")
    print("-"*50)

    # 1. Statistical Summary (Works for all modes because 'data' DF exists in all)
    mean_owner = data[data['listing_type']=='Owner']['monthly_rent_gbp'].mean()
    mean_agent = data[data['listing_type']=='Agent']['monthly_rent_gbp'].mean()
    print(f"Mean Rent (Owner): £{mean_owner:,.2f}")
    print(f"Mean Rent (Agent): £{mean_agent:,.2f}")
    print(f"Agent Markup:      {((mean_agent/mean_owner)-1)*100:+.2f}%")

    # 2. Data Preview
    print("-"*50)
    print(f"PREVIEW (First 10 Properties):")
    print(data.head(10))
    print("="*50 + "\n")
