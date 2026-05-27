# 🧮 Bayesian Inference Engine (Implementation)

This directory contains the core implementation of BaPPI, split into:

1. `src/run_bayesian_analysis.py` → hierarchical Bayesian inference (posterior + PPC)
2. `src/run_post_analysis.py` → model comparison + diagnostics + evaluation

---

# 1. Core Inference (`src/run_bayesian_analysis.py`)

This script performs full Bayesian inference over two competing hypotheses:

- **Owner model** (no agent markup)
- **Agent model** (includes latent agent premium)

The goal is to learn posterior parameter distributions and evaluate model fit.

Importantly:

> ❗ This script does **not compute the Bayes Factor**.  
> It only produces posterior traces and posterior predictive samples (PPC).  
> Model comparison is done later in `src/run_post_analysis.py`.

---

## 🔷 Generative model

Each observed log-rent is generated as:

$$
y_i \sim \text{StudentT}(\nu=5, \mu_i, \sigma)
$$

where:

$$
\mu_i = \sum_k p_{k,i} x_{k,i}
$$

- $x_{k,i}$: property features (rooms, distance, amenities)
- $p_{k,i}$: latent property-level price coefficients

This defines a noisy structural relationship between features and log price.

---

## 🔷 Hierarchical structure

The model is structured around three different levels of prior knowledge about parameters.

### 1. Fixed parameters (fully known)

Some parameters are treated as completely known in advance.

These values come directly from the simulated data-generating process and are assumed to be correct.

They are not inferred by the model and are used as fixed inputs when constructing the likelihood.

---

### 2. Market-known structure (partially known)

For some parameters, we assume that the overall market structure is known.

This means:
- we assume we know the correct global distribution of the parameter at the market level
- but we still estimate how each individual property deviates from that structure

In other words, the model does not learn the global pattern, but it does learn property-specific realizations of it.

---

### 3. Fully unknown parameters (completely inferred)

For the remaining parameters, we assume no knowledge at either level.

Both the market-level distribution and the property-level values must be inferred entirely from the data.

This represents the most uncertain case, where the model must learn both global structure and individual effects simultaneously.

---

## 🔷 Agent hypothesis extension

In the agent version of the model only, an additional latent component is included:

- a property-level agent premium is introduced
- this term is fully learned from the data
- it represents an additional markup applied per property

In the owner model, this component is not present.

---

## 🔷 Bayesian inference (SMC)

Inference is performed using **Sequential Monte Carlo (SMC)**:

- approximates posterior distributions
- produces marginal likelihood estimates (used later)
- generates posterior predictive samples

Each run outputs:

- posterior traces (InferenceData)
- posterior predictive checks (PPC)
- log marginal likelihood samples (stored in trace, used later)

---

## 🔷 Output of this script

This script produces:

- `trace_owner / trace_agent`
- posterior distributions of all parameters
- posterior predictive samples (PPC)

🚫 It does **NOT compute or compare models**

---

# 2. Model comparison + diagnostics (`src/run_post_analysis.py`)

This script performs **all model comparison and evaluation**, including the Bayes Factor.

---

## 🔷 Bayes Factor computation

The Bayes Factor is computed here:

$$
BF = \frac{Z_{agent}}{Z_{owner}}, \quad
\log BF = \log Z_{agent} - \log Z_{owner}
$$

where:

- $Z_M$ = marginal likelihood (log evidence)

---

## 🔷 Evidence analysis

- extracts per-chain log marginal likelihoods
- computes:
  - mean log Z per model
  - Bayes Factor
  - cross-chain consistency checks

---

## 🔷 Posterior predictive checks (PPC)

Compares:

$$
y_{observed} \sim y_{posterior}
$$

for both models using posterior predictive samples.

This evaluates whether the model reproduces observed price distributions.

---

## 🔷 Hierarchical parameter recovery

For each market-level parameter:

- compares posterior vs true injected values
- computes:
  - Z-score
  - two-tailed Bayesian p-value

This tests whether the hierarchical structure correctly recovers the simulated market.

---

## 🔷 Posterior rank calibration

For each property-level parameter:

$$
\text{rank}_i = P(p_i < p_i^{true})
$$

If calibration is correct:

- ranks should be approximately uniform on $[0,1]$

This detects:
- shrinkage
- bias
- misspecification

---

## 🔷 Agent premium analysis

Evaluates:

- injected vs inferred agent premium
- shrinkage effects
- population-level distortion

---

## 🔷 Summary outputs

This script generates:

- Bayes Factor + evidence comparison
- PPC plots (Owner vs Agent)
- posterior corner plots
- parameter recovery diagnostics
- rank calibration histograms
- shrinkage visualizations

---

# 3. Conceptual summary

### `src/run_bayesian_analysis.py`

Implements:

> A hierarchical Bayesian generative model and performs full posterior inference via SMC.

It outputs:
- posterior distributions
- posterior predictive samples

---

### `src/run_post_analysis.py`

Implements:

> All inference validation, diagnostics, and model comparison.

It computes:
- Bayes Factor
- posterior calibration quality
- parameter recovery accuracy

---

# 4. Full pipeline

$$
\text{Data}
\rightarrow
\text{Hierarchical Bayesian Model}
\rightarrow
\text{SMC Inference (Posterior + PPC)}
$$

$$
\rightarrow
\text{Model Evidence (Log Z)}
\rightarrow
\text{Bayes Factor + Diagnostics}
$$

---

This system is designed to test whether an **agent pricing effect can be statistically identified under hierarchical uncertainty**, while keeping inference and evaluation separated.
