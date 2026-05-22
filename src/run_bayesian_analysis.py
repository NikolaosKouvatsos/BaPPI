import os
os.environ["PYTENSOR_FLAGS"] = "device=cpu,cxx="

import json
import pandas as pd
import numpy as np
import pymc as pm
import configparser
import sys
import argparse
import pytensor.tensor as pt
import ast
import pickle
import shutil
import filecmp
import arviz as az
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
TRACE_DIR = BASE_DIR / "results/trace"
PPC_DIR = BASE_DIR / "results/ppc"
RESULTS_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)
PPC_DIR.mkdir(exist_ok=True)

data_config_path = BASE_DIR / "data/datasets/london_rentals_hierarchical.ini"
config_path = BASE_DIR / "app/config.ini"
are_identical = filecmp.cmp(data_config_path, config_path, shallow=False)
if not are_identical:
    print('WARNING: app/config.ini has been edited since the data generation; please ensure that they are identical and then rerun.')
    sys.exit(1)

filename = BASE_DIR / "data/datasets/london_rentals_hierarchical.json"

parser = argparse.ArgumentParser()
# In terminal: python script.py --is_agent_hyp
parser.add_argument("--is_agent_hyp", action="store_true")
parser.add_argument("--consider_both_modes", action="store_true")
args = parser.parse_args()

def load_config(filename=BASE_DIR / "app/config.ini"):
    """Parses the config.ini file into a usable dictionary."""
    # Use inline_comment_prefixes to ignore everything after '#'
    config_parser = configparser.ConfigParser(inline_comment_prefixes=('#',))
    config_parser.read(filename)

    def parse_optional_int(val):
        if val is None:
            return None
        val = str(val).strip()
        if val == "" or val.lower() == "none":
            return None
        return int(val)
    
    # Helper to remove quotes and whitespace
    def clean_val(val):
        return val.strip().strip('"').strip("'")

    c = {}
    
    # 1. GENERAL Section
    gen_section = config_parser['GENERAL']
    c['mode'] = clean_val(gen_section['mode'])
    c['seed'] = parse_optional_int(gen_section.get('seed'))

    # 2. PRICE ARGUMENTS Section
    price = config_parser['PRICE_ARGUMENTS']
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
    
    # 3. ANALYSIS Section
    # Note: Use exact string 'DATA GENERATION' as per your .ini file
    da = config_parser['ANALYSIS']
    c['num_prop_owner'] = int(da['num_prop_owner'])
    c['num_prop_agent'] = int(da['num_prop_agent'])
    c['market_prior_scale'] = float(da['market_prior_scale'])
    c['mc_draws'] = int(da['mc_draws'])
    c['chains'] = int(da['chains'])
    # Define the master list of all possible price arguments considered by fixed_price_args and known_market_price_args.
    all_possible = [
        "base", "room_coeff", "distance_coeff", "garden_fee", 
        "terrace_fee", "balcony_fee", "underground_fee", "house_fee"
    ]
    # Parse the lists from the config (using json.loads to convert string representation to Python list)
    c['fixed_price_args'] = ast.literal_eval(da['fixed_price_args'])
    if "agent_premium" in c['fixed_price_args']:
        print('The "agent_premium" should not be included in the "fixed_price_args".')
        sys.exit(1)
    c['known_market_price_args'] = ast.literal_eval(da['known_market_price_args'])
    if "agent_premium" in c['known_market_price_args']:
        print('The "agent_premium" should not be included in the "known_market_price_args".')
        sys.exit(1)
    # Sanity Check: Ensure no overlap
    overlap = set(c['fixed_price_args']) & set(c['known_market_price_args'])
    if overlap:
        sys.exit(f"Configuration Error: {overlap} found in both fixed and known_market lists.")
    # Identify unknown_price_args via set subtraction
    # Unknown = All - (Fixed + Known Market)
    accounted_for = set(c['fixed_price_args']) | set(c['known_market_price_args'])
    c['unknown_price_args'] = [arg for arg in all_possible if arg not in accounted_for]

    return c

with open(filename, 'r') as f:
    raw_data = json.load(f)

# Convert properties back to a DataFrame
df = pd.DataFrame(raw_data['properties'])
df['log_rent'] = np.log(df['monthly_rent_gbp'])

config = load_config()

print(f"Hierarchical Dataset Loaded: {len(df)} properties.")
    
def run_hierarchical_model_selection(df_subset, config, is_agent_hyp=False):
    # Pre-calculate feature presence to avoid boolean checks inside the loop
    n_props = len(df_subset)
    rooms = df_subset['n_rooms'].values
    dist = df_subset['dist_centre_km'].values
    is_under = df_subset['near_underground'].values
    is_house = (df_subset['property_type'] == 'House').astype(int)
    is_garden = (df_subset['outdoor_space'] == 'Garden').astype(int)
    is_terrace = (df_subset['outdoor_space'] == 'Terrace').astype(int)
    is_balcony = (df_subset['outdoor_space'] == 'Balcony').astype(int)
    y_obs = df_subset['log_rent'].values

    # Extract property-specific "True" coefficients.
    # These are used when a price argument is marked as 'fixed'
    true_coefs = {
        "base": df_subset['intercept'].values,
        "room_coeff": df_subset['beta_room'].values,
        "distance_coeff": df_subset['beta_dist'].values,
        "underground_fee": df_subset['beta_under'].values,
        "house_fee": df_subset['beta_prop'].values,
        "garden_fee": df_subset['beta_outdoor'].values,
        "terrace_fee": df_subset['beta_outdoor'].values,
        "balcony_fee": df_subset['beta_outdoor'].values,
        "agent_premium": df_subset['premium'].values
    }

    # Market-Level Means ---
    mt_mu_inter        = config['base']
    mt_mu_beta_room    = config['room_coeff']
    mt_mu_beta_dist    = config['distance_coeff']
    mt_mu_beta_under   = config['underground_fee']
    mt_mu_beta_house   = config['house_fee']
    mt_mu_beta_garden  = config['garden_fee']
    mt_mu_beta_terrace = config['terrace_fee']
    mt_mu_beta_balcony = config['balcony_fee']
    mt_mu_premium      = config['agent_premium']
    # Market-Level Sigmas ---
    mv = config['market_vol']
    vs = config['vol_scales']
    mt_sig_inter        = mv * vs["intercept"]
    mt_sig_beta_room    = mv * vs["beta_room"]
    mt_sig_beta_dist    = mv * vs["beta_dist"]
    mt_sig_beta_under   = mv * vs["beta_under"]
    mt_sig_beta_house   = mv * vs["beta_house"]
    mt_sig_beta_garden  = mv * vs["beta_garden"]
    mt_sig_beta_terrace = mv * vs["beta_terrace"]
    mt_sig_beta_balcony = mv * vs["beta_balcony"]
    mt_sig_premium      = mv * vs["premium"]

    # Mapping internal argument keys to their respective Market Truth (mt) constants
    market_truth_map = {
        "base": (mt_mu_inter, mt_sig_inter),
        "room_coeff": (mt_mu_beta_room, mt_sig_beta_room),
        "distance_coeff": (mt_mu_beta_dist, mt_sig_beta_dist),
        "underground_fee": (mt_mu_beta_under, mt_sig_beta_under),
        "house_fee": (mt_mu_beta_house, mt_sig_beta_house),
        "garden_fee": (mt_mu_beta_garden, mt_sig_beta_garden),
        "terrace_fee": (mt_mu_beta_terrace, mt_sig_beta_terrace),
        "balcony_fee": (mt_mu_beta_balcony, mt_sig_beta_balcony),
        "agent_premium": (mt_mu_premium, mt_sig_premium)
    }
    # Map for data vectors
    data_vectors = {
        "base": np.ones(n_props),
        "room_coeff": rooms,
        "distance_coeff": dist,
        "underground_fee": is_under,
        "house_fee": is_house,
        "garden_fee": is_garden,
        "terrace_fee": is_terrace,
        "balcony_fee": is_balcony
    }

    with pm.Model() as model:
        # --- LEVEL 2: MARKET HYPER-PRIORS (Unknown Market Parameters) ---
        # We define hyper-priors only for arguments that are not fixed or known at the market level.
        # These represent our uncertainty about the global market parameters themselves.
        
        # 'market_prior_scale' (k) represents our confidence in the market truth.
        k = config['market_prior_scale']
        
        # For simplicity, we assume the error in the market truth mean is equal to k * market truth sigma.
        # The error in the market truth sigma is equal to k/sqrt(2) * market truth sigma.
        # This is motivated by the standard errors (SEs) of the mean and standard deviation, respectively.
        
        mu_m = {}
        sig_m = {}

        for arg in config['unknown_price_args']:
            mt_mu, mt_sig = market_truth_map[arg]

            # 1. Market Means (mu_m): Sigma is scaled by k
            mu_m[arg] = pm.Normal(f"mu_m_{arg}", mu=mt_mu, sigma=k * mt_sig)
            # 2. Market Sigmas (sig_m): Sigma is scaled by k/sqrt(2)
            s_err = (k / np.sqrt(2)) * mt_sig
            # Boundary logic to keep the sampler within physically plausible limits
            lower_bound = s_err / 100
            upper_bound = mt_sig + (mt_sig - lower_bound)
            
            sig_m[arg] = pm.TruncatedNormal(
                f"sig_m_{arg}", 
                mu=mt_sig, 
                sigma=s_err, 
                lower=lower_bound,
                upper=upper_bound
            )

        # --- LEVEL 1: INDIVIDUAL PROPERTY PARAMETERS ---
        prop_params = {}

        # 1. Unknown Arguments (Estimated with Hyper-priors)
        for arg in config['unknown_price_args']:
            # Using Laplace for rooms for thicker tails, Normal for others
            if arg == "room_coeff":
                prop_params[arg] = pm.Laplace(
                    f"p_{arg}",
                    mu=mu_m[arg],
                    b=sig_m[arg],
                    shape=n_props
                )
            else:
                prop_params[arg] = pm.Normal(
                    f"p_{arg}",
                    mu=mu_m[arg],
                    sigma=sig_m[arg],
                    shape=n_props
                )

        # 2. Market-Known Arguments (Estimated with Config Truth)
        for arg in config['known_market_price_args']:
            mt_mu, mt_sig = market_truth_map[arg]
            if arg == "room_coeff":
                prop_params[arg] = pm.Laplace(
                    f"p_{arg}",
                    mu=mt_mu,
                    b=mt_sig,
                    shape=n_props
                )
            else:
                prop_params[arg] = pm.Normal(
                    f"p_{arg}",
                    mu=mt_mu,
                    sigma=mt_sig,
                    shape=n_props
                )

        # Build Linear Predictor (mu)
        mu = pt.zeros(n_props)
        active_args = list(dict.fromkeys(config['unknown_price_args']+config['known_market_price_args']))
        
        for arg in active_args:
            if arg in data_vectors: # Skip premium here
                mu += prop_params[arg] * data_vectors[arg]

        # Add Fixed Arguments
        for arg in config['fixed_price_args']:
            if arg in data_vectors:
                actual_values = true_coefs[arg] 
                mu += actual_values * data_vectors[arg]

        # --- HYPOTHESIS: AGENT PREMIUM ---
        if is_agent_hyp:
            # Hyper-priors for the Premium
            mu_m_prem = pm.TruncatedNormal("mu_m_prem", mu=mt_mu_premium, sigma=k * mt_sig_premium,
                                           lower=k*mt_sig_premium/100,
                                           upper=mt_mu_premium + (mt_mu_premium - k*mt_sig_premium/100))
            s_err_prem = (k / np.sqrt(2)) * mt_sig_premium
            sig_m_prem = pm.TruncatedNormal("sig_m_prem", mu=mt_sig_premium, sigma=s_err_prem, 
                                            lower=s_err_prem/100, 
                                            upper=mt_sig_premium + (mt_sig_premium - s_err_prem/100))
            
            # Gamma parameters: alpha = (mu/sigma)^2, beta = mu/sigma^2
            alpha_p = (mu_m_prem / sig_m_prem)**2
            beta_p  = mu_m_prem / (sig_m_prem**2)
            
            p_premium = pm.Gamma("p_premium", alpha=alpha_p, beta=beta_p, shape=n_props)
            mu += p_premium

        # --- LIKELIHOOD & SAMPLING ---
        pm.StudentT("obs", nu=5, mu=mu, sigma=config['noise_scale'], observed=y_obs)

        # --- SAMPLING (SMC for Marginal Likelihood) ---
        traces = []

        hypothesis_name = "agent" if is_agent_hyp else "owner"

        for chain_id in range(config['chains']):

            checkpoint_file = TRACE_DIR / f"trace_{hypothesis_name}_chain{chain_id}.pkl"

            # Skip already completed chains
            if checkpoint_file.exists():
                print(f"Loading existing checkpoint: {checkpoint_file.name}")

                with open(checkpoint_file, "rb") as f:
                    chain_trace = pickle.load(f)

                traces.append(chain_trace)
                continue

            print(f"Running chain {chain_id+1}/{config['chains']}...")

            chain_trace = pm.sample_smc(
                draws=config['mc_draws'],
                chains=1,
                random_seed=None if config['seed'] is None else config['seed'] + chain_id,
                progressbar=True,
                return_inferencedata=True,
            )

            chain_trace = chain_trace.assign_coords(chain=[chain_id])
            # Save checkpoint immediately
            with open(checkpoint_file, "wb") as f:
                pickle.dump(chain_trace, f)

            print(f"Saved checkpoint: {checkpoint_file.name}")

            traces.append(chain_trace)

        if not traces:
            raise RuntimeError("No valid traces available.")
        combined_trace = az.concat(*traces, dim="chain")

        ppc = pm.sample_posterior_predictive(
            combined_trace,
            model=model,
            var_names=["obs"],
            random_seed=config['seed']
        )

    return combined_trace, ppc
    
# --- EXECUTION ---
# Filter for each type and take the specified number of rows
owner_subset = df[df['listing_type'] == 'Owner'].head(config['num_prop_owner'])
agent_subset = df[df['listing_type'] == 'Agent'].head(config['num_prop_agent'])
# Combine them into a single test batch
test_batch = pd.concat([owner_subset, agent_subset]).reset_index(drop=True)

# Safety check: print the resulting composition
print(f"Batch created with {len(test_batch)} properties.")
print(test_batch['listing_type'].value_counts())
print('')

# Determine which modes need to be run based on the flag
if args.consider_both_modes:
    print('Considering both property provider modes...\n')
    modes_to_run = ['agent', 'owner']
else:
    modes_to_run = ['agent' if args.is_agent_hyp else 'owner']

# Run the pipeline steps for each required mode
for mode in modes_to_run:
    # Set a localized boolean for the model selector function
    current_is_agent_hyp = (mode == 'agent')

    print(f"Running mode: {mode.upper()} (Hypothesis mapping: {current_is_agent_hyp})")
    
    results_config_path = BASE_DIR / f"results/london_rentals_{mode}.ini"
    
    if results_config_path.exists():
        are_also_identical = filecmp.cmp(results_config_path, config_path, shallow=False)
        if config['seed'] is None or not are_also_identical:
            print(f'WARNING: The "seed" in app/config.ini is set to None or the file has been edited since the last time the hierarchical Bayesian analysis was run for {mode.upper()}.')
            print('Are you sure that you want to proceed? This will result in any previously stored results being lost.')
            while True:
                user_choice = input("Proceed? (y/n): ").strip().lower()
                if user_choice in ['y', 'yes']:
                    print(f"Proceeding with the updated configuration for {mode.upper()}...\n")
                    break
                elif user_choice in ['n', 'no']:
                    print("Execution halted by user. Exiting script safely.")
                    sys.exit(0)

            # Flush old matching traces
            matching_files = list(TRACE_DIR.glob(f"*{mode}*"))
            for matching_file in matching_files:
                matching_file.unlink()
        
    # Copy configuration backup
    shutil.copy2(
        BASE_DIR / "app/config.ini",
        RESULTS_DIR / f"london_rentals_{mode}.ini"
    )

    # Save data split
    test_batch.to_csv(RESULTS_DIR / f"test_batch_{mode}.csv", index=False)

    print("Initiating hierarchical analysis...")

    # Execute Bayesian inference
    trace, ppc = run_hierarchical_model_selection(test_batch, config, is_agent_hyp=current_is_agent_hyp)

    # File designations
    final_trace_file = TRACE_DIR / f"final_trace_{mode}.pkl"
    final_ppc_file = PPC_DIR / f"ppc_{mode}.pkl"

    with open(final_trace_file, "wb") as f:
        pickle.dump(trace, f)

    with open(final_ppc_file, "wb") as f:
        pickle.dump(ppc, f)

    print(f"Final trace saved to {final_trace_file}")
    print(f"PPC saved to {final_ppc_file}")

    log_z = trace.sample_stats.log_marginal_likelihood.values[:, -1].mean()
    print(f"Analysis for {mode.upper()} finished successfully!\n" + "-"*50 + "\n")

print("All requested Bayesian pipeline tasks completed successfully!\n" + "="*50 + "\n")
