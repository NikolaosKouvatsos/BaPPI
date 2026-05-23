"""
================================================================================
BaPPI: Bayesian Property Provider Identifier
================================================================================
A hierarchical Bayesian framework designed to analyze multi-property real estate
markets in London. By considering structural property parameters, hidden agent
premiums, and pricing behaviors across a localized micro-market, BaPPI infers
and predicts whether individual rental listings are arranged directly by the
property owners or managed by third-party agents.

Author: Nikolaos Kouvatsos
Date: May 2026
================================================================================
"""

BAPPI_LOGO = r"""
🏠   ____        _____  _____ ____  📈
    |  _ \      |  __ \|  __ \   _|
    | |_) | __ _| |__) | |__) | |  
    |  _ < / _` |  ___/|  ___/| |  
    | |_) | (_| | |    | |   _| |_ 
    |____/ \__,_|_|    |_|  |_____|
📊                                  🚀
"""

if __name__ == "__main__":
    print(BAPPI_LOGO)
    print("Initializing Bayesian Property Provider Identifier (BaPPI)...")
    print(f"{'='*80}\n")

import configparser
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_config(filename=BASE_DIR / "app/config.ini"):
    """Parses the config.ini file into a usable dictionary."""
    # Use inline_comment_prefixes to ignore everything after '#'
    config_parser = configparser.ConfigParser(inline_comment_prefixes=('#',))
    config_parser.read(filename)
    
    # Helper to remove quotes and whitespace
    def clean_val(val):
        return val.strip().strip('"').strip("'")

    c = {}

    # GENERAL Section
    gen_section = config_parser['GENERAL']
    c['mode'] = clean_val(gen_section['mode'])

    return c

config = load_config()

print('\nGenerating new property data...')
data_script_path = str(BASE_DIR / "data/gen_prop_data.py")
    
result = subprocess.run(
    [sys.executable, data_script_path], 
    capture_output=False, 
    text=True
)

mode = config['mode'].lower()
if mode!='hierarchical':
    print('\nThe Bayesian analysis and the post-analysis scripts only accept the "hierarchical mode" - please switch to that in app/config.ini.')
    sys.exit(1)

print('\nProceeding to the hierarchical Bayesian analysis...\n')

analysis_script_path = str(BASE_DIR / "src/run_bayesian_analysis.py")
        
result = subprocess.run(
    [sys.executable, analysis_script_path, "--consider_both_modes"], 
    capture_output=False, 
    text=True
)

print('Proceeding to the post-analysis...')

post_analysis_script_path = str(BASE_DIR / "src/run_post_analysis.py")
        
result = subprocess.run(
    [sys.executable, post_analysis_script_path], 
    capture_output=False, 
    text=True
)
