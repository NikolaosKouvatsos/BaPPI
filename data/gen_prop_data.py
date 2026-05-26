import numpy as np
import pandas as pd
import json
import configparser
import os
import shutil

def load_config(filename='app/config.ini'):
    """Parses the config.ini file into a usable dictionary."""
    # Use inline_comment_prefixes to ignore everything after '#'
    parser = configparser.ConfigParser(inline_comment_prefixes=('#',))
    parser.read(filename)
    
    # Helper to remove quotes and whitespace
    def clean_val(val):
        return val.strip().strip('"').strip("'")

    # Helper to convert string lists like "[0.3, 0.1, 0.2, 0.4]" to actual lists
    def parse_list(string):
        clean_s = clean_val(string).strip('[]')
        return [float(x.strip()) for x in clean_s.split(',')]

    c = {}
    
    # 1. GENERAL Section
    gen_section = parser['GENERAL']
    c['mode'] = clean_val(gen_section['mode'])
    raw_seed = gen_section.get('seed', '').strip().lower()
    if raw_seed in ['none', 'null', '']:
        c['seed'] = None
    else:
        c['seed'] = int(gen_section['seed'])

    # 2. DATA_GENERATION Section
    # Note: Use exact string 'DATA GENERATION' as per your .ini file
    dg = parser['DATA_GENERATION']
    c['num_properties'] = int(dg['num_properties'])
    c['flat_house_ratio'] = float(dg['flat_house_ratio'])
    c['room_exp_num'] = float(dg['room_exp_num'])
    c['distance_scale'] = float(dg['distance_scale'])
    c['outdoor_space_weights'] = parse_list(dg['outdoor_space_weights'])
    c['underground_prob'] = float(dg['underground_prob'])
    
    # 3. PRICE ARGUMENTS Section
    price = parser['PRICE_ARGUMENTS']
    c['noise_scale'] = float(price['noise_scale'])
    c['base'] = float(price['base'])
    c['room_coeff'] = float(price['room_coeff'])
    c['distance_coeff'] = float(price['distance_coeff'])
    c['underground_fee'] = float(price['underground_fee'])
    c['house_fee'] = float(price['house_fee'])
    c['garden_fee'] = float(price['garden_fee'])
    c['terrace_fee'] = float(price['terrace_fee'])
    c['balcony_fee'] = float(price['balcony_fee'])
    c['agent_premium'] = float(price['agent_premium'])
    c['market_vol'] = float(price['market_vol'])
    # Vol Scales
    c['vol_scales'] = {
        "intercept": float(price['base_vol_scale']),
        "beta_room": float(price['room_coeff_vol_scale']),
        "beta_dist": float(price['distance_coeff_vol_scale']),
        "beta_under": float(price['underground_fee_vol_scale']),
        "beta_house": float(price['house_fee_vol_scale']),
        "beta_garden": float(price['garden_fee_vol_scale']),
        "beta_terrace": float(price['terrace_fee_vol_scale']),
        "beta_balcony": float(price['balcony_fee_vol_scale']),
        "premium": float(price['agent_premium_vol_scale'])
    }
    
    return c

def generate_property_data(config):
    """Generates structural features based on config.ini."""
    if config['seed'] is not None:
        np.random.seed(config['seed'])
    
    n = config['num_properties']
    
    # Calculate probabilities from ratio (e.g., ratio 3 means 3/4 Flats, 1/4 Houses)
    p_flat = config['flat_house_ratio'] / (config['flat_house_ratio'] + 1)
    prop_types = np.random.choice(['Flat', 'House'], size=n, p=[p_flat, 1-p_flat])
    
    rooms = np.random.poisson(lam=config['room_exp_num'], size=n) + 1
    distance = np.random.exponential(scale=config['distance_scale'], size=n)
    
    outdoor_options = ['Garden', 'Balcony', 'Terrace', 'Nothing']
    outdoor_space = np.random.choice(outdoor_options, size=n, p=config['outdoor_space_weights'])
    
    underground = np.random.binomial(n=1, p=config['underground_prob'], size=n)
    
    return pd.DataFrame({
        'property_type': prop_types,
        'n_rooms': rooms,
        'dist_centre_km': distance,
        'outdoor_space': outdoor_space,
        'near_underground': underground
    })

def generate_price_fix_params(df, config, model_case='A', seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(df)
    
    # 1. Pull Point Estimates from Config
    intercept = config['base']
    beta_room = config['room_coeff']
    beta_dist = config['distance_coeff']
    
    # 2. Map Categorical Fees from Config
    type_map = {'House': config['house_fee'], 'Flat': 0.0}
    outdoor_map = {
        'Garden': config['garden_fee'], 
        'Terrace': config['terrace_fee'], 
        'Balcony': config['balcony_fee'], 
        'Nothing': 0.0
    }
    
    # 3. Handle Agent Premium Case
    agent_premium = config['agent_premium'] if model_case == 'B' else 0.0
    
    # 4. Calculate Vectorized Log-Price
    beta_prop = df['property_type'].map(type_map).values
    beta_outdoor = df['outdoor_space'].map(outdoor_map).values
    beta_under = (df['near_underground'] * config['underground_fee']).values
    
    log_price = (
        intercept + 
        (df['n_rooms'] * beta_room) + 
        (df['dist_centre_km'] * beta_dist) + 
        beta_under + 
        beta_prop + 
        beta_outdoor + 
        agent_premium
    )
    
    # Add noise using noise_scale from config
    noise = np.random.normal(0, config['noise_scale'], size=n)
    prices = np.exp(log_price + noise)

    # Package for consistency
    fixed_params = {
        "intercept": np.full(n, intercept),
        "beta_prop": beta_prop,
        "beta_room": np.full(n, beta_room),
        "beta_dist": np.full(n, beta_dist),
        "beta_outdoor": beta_outdoor,
        "beta_under": beta_under,
        "premium": np.full(n, agent_premium)
    }

    return prices, fixed_params

def generate_price_rand_params(df, config, model_case='A', seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n = len(df)
    vol = config['market_vol']
    
    # 1. Generate Random Pools using config means and global volatility
    intercepts = np.random.normal(config['base'], vol, size=n)
    betas_room  = np.random.laplace(config['room_coeff'], vol, size=n)
    betas_dist  = np.random.normal(config['distance_coeff'], vol, size=n)
    
    # Pools for conditional features
    under_pool   = np.random.normal(config['underground_fee'], vol, size=n)
    house_pool   = np.random.normal(config['house_fee'], vol, size=n)
    garden_pool  = np.random.normal(config['garden_fee'], vol, size=n)
    terrace_pool = np.random.normal(config['terrace_fee'], vol, size=n)
    balcony_pool = np.random.normal(config['balcony_fee'], vol, size=n)
    
    # Agent Premium logic (using Gamma for positive-only constraint)
    if model_case == 'B':
        m = config['agent_premium']
        alpha = (m / vol)**2
        beta = m / (vol**2)
        agent_premium = np.random.gamma(alpha, 1/beta, size=n)
    else:
        agent_premium = np.zeros(n)

    # 2. Masking logic (applying pool only where feature exists)
    beta_under_final = np.where(df['near_underground'] == 1, under_pool, 0.0)
    beta_prop = np.where(df['property_type'] == 'House', house_pool, 0.0)
    
    beta_outdoor = np.zeros(n)
    beta_outdoor = np.where(df['outdoor_space'] == 'Garden', garden_pool, beta_outdoor)
    beta_outdoor = np.where(df['outdoor_space'] == 'Terrace', terrace_pool, beta_outdoor)
    beta_outdoor = np.where(df['outdoor_space'] == 'Balcony', balcony_pool, beta_outdoor)
    
    # 3. Final calculation
    log_price = (
        intercepts + (df['n_rooms'] * betas_room) + (df['dist_centre_km'] * betas_dist) + 
        beta_under_final + beta_prop + beta_outdoor + agent_premium
    )
    
    noise = np.random.standard_t(df=5, size=n) * config['noise_scale']
    prices = np.exp(log_price + noise)

    sampled_params = {
        "intercept": intercepts, "beta_prop": beta_prop, "beta_room": betas_room,
        "beta_dist": betas_dist, "beta_outdoor": beta_outdoor,
        "beta_under": beta_under_final, "premium": agent_premium
    }
    
    return prices, sampled_params

def generate_hierarchical_data(df, config, model_case='A', seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n = len(df)
    
    # 1. Market-Level Means
    market_means = {
        "intercept": config['base'], 
        "beta_room": config['room_coeff'], 
        "beta_dist": config['distance_coeff'], 
        "beta_under": config['underground_fee'], 
        "beta_house": config['house_fee'],
        "beta_garden": config['garden_fee'], 
        "beta_terrace": config['terrace_fee'], 
        "beta_balcony": config['balcony_fee'],
        "premium": config['agent_premium'] if model_case == 'B' else 0.0
    }
    
    # 2. Market-Level Sigmas
    market_sigmas = {k: config['market_vol'] * config['vol_scales'][k] for k in market_means.keys()}

    # 3. Individual Property Betas
    intercepts = np.random.normal(market_means["intercept"], market_sigmas["intercept"], size=n)
    betas_room  = np.random.laplace(market_means["beta_room"], market_sigmas["beta_room"], size=n)
    betas_dist  = np.random.normal(market_means["beta_dist"], market_sigmas["beta_dist"], size=n)
    
    under_pool   = np.random.normal(market_means["beta_under"], market_sigmas["beta_under"], size=n)
    house_pool   = np.random.normal(market_means["beta_house"], market_sigmas["beta_house"], size=n)
    garden_pool  = np.random.normal(market_means["beta_garden"], market_sigmas["beta_garden"], size=n)
    terrace_pool = np.random.normal(market_means["beta_terrace"], market_sigmas["beta_terrace"], size=n)
    balcony_pool = np.random.normal(market_means["beta_balcony"], market_sigmas["beta_balcony"], size=n)
    
    if model_case == 'B':
        # Ensure alpha/beta for Gamma are valid (premium must be > 0)
        alpha_p = (market_means["premium"] / market_sigmas["premium"])**2
        beta_p  = market_means["premium"] / (market_sigmas["premium"]**2)
        agent_premium = np.random.gamma(alpha_p, 1/beta_p, size=n)
    else:
        agent_premium = np.zeros(n)

    # 4. Masking & Calculation
    beta_under_final = np.where(df['near_underground'] == 1, under_pool, 0.0)
    beta_prop = np.where(df['property_type'] == 'House', house_pool, 0.0)
    
    beta_outdoor = np.zeros(n)
    beta_outdoor = np.where(df['outdoor_space'] == 'Garden', garden_pool, beta_outdoor)
    beta_outdoor = np.where(df['outdoor_space'] == 'Terrace', terrace_pool, beta_outdoor)
    beta_outdoor = np.where(df['outdoor_space'] == 'Balcony', balcony_pool, beta_outdoor)
    
    log_price_mu = (
        intercepts + 
        (df['n_rooms'] * betas_room) + 
        (df['dist_centre_km'] * betas_dist) + 
        beta_under_final + beta_prop + beta_outdoor + agent_premium
    )
    
    # Residual noise from config
    noise = np.random.standard_t(df=5, size=n) * config['noise_scale'] 
    prices = np.exp(log_price_mu + noise)

    return prices, {"individual": {"intercept": intercepts, "beta_room": betas_room, "beta_dist": betas_dist, 
                                 "beta_under": beta_under_final, "beta_prop": beta_prop, 
                                 "beta_outdoor": beta_outdoor, "premium": agent_premium},
                    "market": {"means": market_means, "sigmas": market_sigmas}}

if __name__ == "__main__":
    os.makedirs("data/datasets/", exist_ok=True)

    # 1. Load configuration from .ini
    config = load_config()
    mode = config['mode'].lower()
    
    # 2. Generate structural 'Physical' features
    data = generate_property_data(config)
    half = len(data) // 2

    # 3. Branching Logic based on Mode
    if mode == 'hierarchical':
        # 1. Determine the seeds for both cases defensively
        seed_a = config['seed']
        seed_b = config['seed'] + 1 if config['seed'] is not None else None

        # 2. Advanced Multi-level generation
        prices_a, params_a = generate_hierarchical_data(
            data.iloc[:half], config, model_case='A', seed=seed_a
        )
        prices_b, params_b = generate_hierarchical_data(
            data.iloc[half:], config, model_case='B', seed=seed_b
        )
        
        # Map Individual Betas to dataframe columns
        ind_a, ind_b = params_a['individual'], params_b['individual']
        for key in ind_a.keys():
            data[key] = np.concatenate([ind_a[key], ind_b[key]])
            
        data['monthly_rent_gbp'] = np.concatenate([prices_a, prices_b])
        data['listing_type'] = ['Owner'] * half + ['Agent'] * (len(data) - half)

        shutil.copy2("app/config.ini", "data/datasets/london_rentals_hierarchical.ini")

        # Package Individual + Market Truth into JSON
        filename = "data/datasets/london_rentals_hierarchical.json"
        full_output = {
            "metadata": {
                "mode": mode,
                "owner_market_truth": params_a['market'],
                "agent_market_truth": params_b['market']
            },
            "properties": data.to_dict(orient='records')
        }
        with open(filename, 'w') as f:
            json.dump(full_output, f, indent=4)
        
    else:
        # Fixed or Random modes (Standard 2D Logic)
        if mode == 'fixed':
            prices_a, params_a = generate_price_fix_params(data.iloc[:half], config, model_case='A', seed=config['seed'])
            prices_b, params_b = generate_price_fix_params(data.iloc[half:], config, model_case='B', seed=config['seed'] + 1)
            filename = "data/datasets/london_rentals_fixed.csv"
            shutil.copy2("app/config.ini", "data/datasets/london_rentals_fixed.ini")
        elif mode == 'random':
            prices_a, params_a = generate_price_rand_params(data.iloc[:half], config, model_case='A', seed=config['seed'])
            prices_b, params_b = generate_price_rand_params(data.iloc[half:], config, model_case='B', seed=config['seed'] + 1)
            filename = "data/datasets/london_rentals_random.csv"
            shutil.copy2("app/config.ini", "data/datasets/london_rentals_random.ini")
        else:
            raise ValueError(f"Invalid mode '{mode}' detected in config.ini. Choose 'fixed', 'random', or 'hierarchical'.")

        # Map flat parameters to columns
        for key in params_a.keys():
            data[key] = np.concatenate([params_a[key], params_b[key]])
            
        data['monthly_rent_gbp'] = np.concatenate([prices_a, prices_b])
        data['listing_type'] = ['Owner'] * half + ['Agent'] * (len(data) - half)
        
        # Save as standard flat CSV
        data.to_csv(filename, index=False)

    # --- CONSOLE SUMMARY ---
    print("\n" + "="*50, flush=True)
    print(f"DATA GENERATION SUMMARY", flush=True)
    print("="*50, flush=True)
    print(f"Seed:         {config['seed']}", flush=True)
    print(f"Mode:         {mode}", flush=True)
    print(f"Observations: {len(data)}", flush=True)
    print(f"File Saved:   {filename}", flush=True)
    print("-" * 50, flush=True)

    mean_owner = data[data['listing_type'] == 'Owner']['monthly_rent_gbp'].mean()
    mean_agent = data[data['listing_type'] == 'Agent']['monthly_rent_gbp'].mean()
    print(f"Mean Rent (Owner): £{mean_owner:,.2f}", flush=True)
    print(f"Mean Rent (Agent): £{mean_agent:,.2f}", flush=True)
    print(f"Agent Markup:      {((mean_agent/mean_owner)-1)*100:+.2f}%", flush=True)
    print("="*50 + "\n", flush=True)

    # Data Preview
    print("-"*50, flush=True)
    print(f"PREVIEW (First 10 Properties):", flush=True)
    print(data.head(10), flush=True)
    print("="*50 + "\n", flush=True)
