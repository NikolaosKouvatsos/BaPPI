# 🧮 Bayesian Inference Engine (Implementation)

This directory contains the core implementation of a **hierarchical Bayesian model for rental price inference and model comparison**, split into:

1. `run_bayesian_analysis.py` → inference + model comparison  
2. `run_post_analysis.py` → diagnostics + evaluation  

---

# 1. Core Inference (`run_bayesian_analysis.py`)

This script performs full Bayesian inference over two competing hypotheses:

- **Owner model** (no agent markup)
- **Agent model** (includes latent agent premium)

The goal is to compute posterior model probabilities and compare model evidence:

\[
P(M \mid X, y), \quad Z_M = P(y \mid M)
\]

---

## 🔷 Generative model

Each observed log-rent is generated as:

\[
y_i \sim \text{StudentT}(\nu=5, \mu_i, \sigma)
\]

where the linear predictor is:

\[
\mu_i = \sum_k p_{k,i} x_{k,i}
\]

- \(x_{k,i}\): property features (rooms, distance, amenities)
- \(p_{k,i}\): latent price coefficients (property-specific)

This defines a noisy structural relationship between features and log price.

---

## 🔷 Hierarchical structure

The model is fully hierarchical:

### 1. Market-level priors
Each coefficient has a market distribution:

\[
\mu_k \sim \mathcal{N}(\mu_k^{market}, k \cdot \sigma_k^{market})
\]
\[
\sigma_k \sim \text{TruncatedNormal}(\sigma_k^{market}, \cdots)
\]

This encodes uncertainty about the true global market structure.

---

### 2. Property-level parameters

Each property has its own coefficient:

\[
p_{k,i} \sim \mathcal{N}(\mu_k, \sigma_k)
\]

Some parameters (e.g. room effects) use Laplace distributions for heavier tails.

---

### 3. Fixed effects

Some coefficients are treated as known and directly injected:

\[
\mu_i \;+=\; \theta^{fixed} x_i
\]

These are not inferred.

---

## 🔷 Agent hypothesis extension

Under the **Agent model only**, an additional latent term is introduced:

\[
p_{premium,i} \sim \text{Gamma}(\alpha, \beta)
\]

and:

\[
\mu_i \;+=\; p_{premium,i}
\]

This represents a property-level **agent markup effect**.

---

## 🔷 Bayesian inference

Inference is performed using **Sequential Monte Carlo (SMC)**:

- approximates posterior distributions
- estimates marginal likelihood \(Z_M\)
- supports model comparison via evidence

Each run produces:
- posterior samples
- log marginal likelihood (log Z)
- posterior predictive samples

---

## 🔷 Model comparison

Model evidence is compared via:

\[
BF = \frac{Z_{agent}}{Z_{owner}}, \quad \log BF = \log Z_{agent} - \log Z_{owner}
\]

Interpretation:
- BF > 1 → Agent model preferred
- BF < 1 → Owner model preferred

---

# 2. Post-analysis (`run_post_analysis.py`)

This script evaluates inference quality, calibration, and parameter recovery.

---

## 🔷 Purpose

It answers:

- Do both models agree internally (across chains)?
- Are posterior predictions accurate?
- Are parameters correctly recovered?
- Is the model well-calibrated?

---

## 🔷 Evidence analysis

- extracts per-chain log marginal likelihoods
- computes:
  - mean log Z
  - Bayes Factor
- checks convergence consistency across chains

---

## 🔷 Posterior predictive checks (PPC)

Compares:

\[
y_{observed} \sim y_{posterior}
\]

for both models using posterior predictive simulations.

This evaluates whether the model reproduces observed price distributions.

---

## 🔷 Hierarchical parameter recovery

For each market-level parameter:

- compares posterior vs true injected values
- computes:
  - Z-score
  - two-tailed Bayesian p-value

This measures whether the hierarchical structure correctly learns the simulated market.

---

## 🔷 Posterior rank calibration

For each property-level coefficient:

\[
\text{rank}_i = P(p_i < p_i^{true})
\]

If calibration is correct:

- ranks should be approximately uniform on \([0,1]\)

This detects:
- bias
- over-shrinkage
- mis-specification

---

## 🔷 Agent premium analysis

Specifically evaluates:

- injected vs inferred agent premium
- degree of hierarchical shrinkage
- population-wide distortion effects

---

## 🔷 Summary of diagnostics

The script produces:

- Evidence consistency plots
- PPC comparisons
- Corner plots of posterior structure
- Parameter recovery tables
- Rank calibration histograms
- Shrinkage diagnostics for agent premium

---

# 3. Conceptual summary

### `run_bayesian_analysis.py`
Implements:

> A hierarchical Bayesian generative model of rental prices  
> and performs full posterior + model evidence inference via SMC.

---

### `run_post_analysis.py`
Evaluates:

> whether the inferred model is statistically valid, well-calibrated, and structurally correct.

---

# 4. Overall pipeline

\[
\text{Data} \rightarrow \text{Hierarchical Bayesian Model} \rightarrow \text{SMC Inference} \rightarrow \text{Model Evidence}
\]

\[
\rightarrow \text{Diagnostics + Calibration Checks}
\]

---

This system is designed to test whether an **agent pricing effect can be statistically identified under hierarchical uncertainty**.
