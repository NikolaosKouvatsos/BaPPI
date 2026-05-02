import sys
import os

# Get the directory of the current file (app/)
# Then go up one level to the project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from models.inference import get_experiment_results

st.set_page_config(page_title="Bayesian Decision Engine", layout="wide")

st.title("🚀 Bayesian A/B Testing Engine")
st.markdown("Calculating the **Expected Loss** to mitigate the risk of switching.")

# Sidebar for Inputs
with st.sidebar:
    st.header("Experiment Data")
    n_a = st.number_input("Group A Total", value=1000)
    k_a = st.number_input("Group A Conversions", value=100)
    st.divider()
    n_b = st.number_input("Group B Total", value=1000)
    k_b = st.number_input("Group B Conversions", value=120)

# Run Analysis
try:
    res = get_experiment_results(n_a, k_a, n_b, k_b)
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Prob(B > A)", f"{res['prob_better']:.2%}")
    col2.metric("Expected Loss", f"{res['expected_loss']:.5f}")
    col3.metric("Status", res['status'])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(res['samples_a'], bins=50, alpha=0.5, label='Group A', color='blue')
    ax.hist(res['samples_b'], bins=50, alpha=0.5, label='Group B', color='orange')
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")