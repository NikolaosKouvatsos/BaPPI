import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import configparser
import ast
import arviz as az
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
TRACE_DIR = BASE_DIR / "results/trace"
PPC_DIR = BASE_DIR / "results/ppc"
PLOT_DIR = RESULTS_DIR / "plots/"

os.makedirs(PLOT_DIR, exist_ok=True)

with open(TRACE_DIR / "final_trace_owner.pkl", "rb") as f:
    fin_tr_owner = pickle.load(f)
    
with open(TRACE_DIR / "final_trace_agent.pkl", "rb") as f:
    fin_tr_agent = pickle.load(f)

log_z_owner = fin_tr_owner.sample_stats.log_marginal_likelihood.mean().item()
log_z_agent = fin_tr_agent.sample_stats.log_marginal_likelihood.mean().item()

# Bayes Factor Computation
log_BF = log_z_agent - log_z_owner
BF = np.exp(log_BF)

print("\n--- BATCH RESULTS ---")
print(f"Log Z (Owner Model): {log_z_owner:.4f}")
print(f"Log Z (Agent Model): {log_z_agent:.4f}")
print(f"Bayes Factor:  {BF:.4e}")

if BF > 1000:
    print("Conclusion: Decisive Evidence for the AGENT model.")
elif BF > 10:
    print("Conclusion: Strong Evidence for the AGENT model.")
elif BF > 1:
    print("Conclusion: Moderate Evidence for the AGENT model.")
elif BF > 0.1:
    print("Conclusion: Moderate Evidence for the OWNER model.")
elif BF > 0.01:
    print("Conclusion: Strong Evidence for the OWNER model.")
else:
    print("Conclusion: Decisive Evidence for the OWNER model.")

# Access the log_marginal_likelihood per chain
# This shows if all chains reached roughly the same conclusion
log_z_per_chain_owner = fin_tr_owner.sample_stats.log_marginal_likelihood.values[:,-1]
log_z_per_chain_agent = fin_tr_agent.sample_stats.log_marginal_likelihood.values[:,-1]

plt.figure(figsize=(8, 6))
plt.plot(log_z_per_chain_owner, 'o-', label='Log Z per Chain (Owner)')
plt.axhline(log_z_per_chain_owner.mean(), color='purple', linestyle='--', label='Mean Log Z (Owner)')
plt.plot(log_z_per_chain_agent, 'o-', color='orange', label='Log Z per Chain (Agent)')
plt.axhline(log_z_per_chain_agent.mean(), color='r', linestyle='--', label='Mean Log Z (Agent)')
plt.xlabel('Chain Number')
plt.ylabel('Log Marginal Likelihood')
plt.title('Evidence Consistency Across Chains')
plt.xticks([0, 1, 2, 3])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=9, ncol=4, frameon=True)
plt.tight_layout() 
plt.savefig(PLOT_DIR / "Evidence Consistency Across Chains.png")

accept_rates_owner = fin_tr_owner.sample_stats["accept_rate"].values
betas_owner = fin_tr_owner.sample_stats["beta"].values
accept_rates_agent = fin_tr_agent.sample_stats["accept_rate"].values
betas_agent = fin_tr_agent.sample_stats["beta"].values

plt.figure(figsize=(8, 6))

# Create color gradients
owner_colors = plt.cm.Blues(np.linspace(0.4, 0.9, accept_rates_owner.shape[0]))
agent_colors = plt.cm.OrRd(np.linspace(0.4, 0.9, accept_rates_agent.shape[0]))

# Owner chains (blue shades)
for i in range(accept_rates_owner.shape[0]):
    plt.plot(
        betas_owner[i],
        accept_rates_owner[i],
        'o-',
        color=owner_colors[i],
        alpha=0.9,
        label=f'Owner Chain {i}'
    )
# Agent chains (orange/red shades)
for i in range(accept_rates_agent.shape[0]):
    plt.plot(
        betas_agent[i],
        accept_rates_agent[i],
        'v-',
        color=agent_colors[i],
        alpha=0.9,
        label=f'Agent Chain {i}'
    )

plt.axhline(y=0.234, color='k', linestyle='--', label='Target Acceptance')
plt.xlabel('Beta (Prior 0.0 → Likelihood 1.0)')
plt.ylabel('Acceptance Rate')
plt.title('Sampler Health (Acceptance Rate) during Evidence Integration')
plt.grid(True, alpha=0.3)
plt.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.15), 
    ncol=4, 
    frameon=True
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "Acceptance Rate.png")


