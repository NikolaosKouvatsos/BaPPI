# 🧮 Bayesian Inference Engine (Implementation)

This directory contains the core implementation of BaPPI, split into:

1. `src/run_bayesian_analysis.py` → hierarchical Bayesian inference (posterior trace + PPC)
2. `src/run_post_analysis.py` → model comparison + diagnostics + evaluation

---

# 1. Core Inference (`src/run_bayesian_analysis.py`)

This script performs hierarchical Bayesian inference over two competing hypotheses:

- **Owner model** → no agent markup
- **Agent model** → includes latent agent premium

The goal is to infer posterior parameter distributions and generate posterior predictive simulations.

Importantly:

> ❗ This script does **not compute the Bayes Factor**.  
> It only produces traces and posterior predictive checks (PPC).  
> Model comparison is performed later in `src/run_post_analysis.py`.

---

## 🔷 Bayesian framework

The analysis follows Bayes' theorem:

$$P(\theta \mid y) = \frac{P(y \mid \theta) P(\theta)}{P(y)}$$

where:

- $P(\theta \mid y)$ → posterior distribution
- $P(y \mid \theta)$ → likelihood
- $P(\theta)$ → prior distribution
- $P(y)$ → marginal likelihood (model evidence)

The model uses:
- prior assumptions about the housing market
- observed rental prices
- hierarchical latent parameters

to infer posterior distributions over pricing coefficients.

---

## 🔷 Observed likelihood model

Observed log-rents are modeled using a Student-t likelihood:

$$
y_i
\sim
\text{StudentT}(\nu=5,\mu_i,\sigma)
$$

Where:

- $y_i$ → observed log-rent
- $\mu_i$ → latent structural log-price
- $\sigma$ → residual noise scale

The Student-t distribution is used for robustness against outliers.

---

## 🔷 Structural linear predictor

The latent log-price is computed as:

$$
\mu_i=\sum_k p_{k,i} x_{k,i}
$$

Where:

- $x_{k,i}$ → observed property features
- $p_{k,i}$ → latent property-specific pricing coefficients

Concretely:

$$
\mu_i =
\text{intercept}_i
+
\beta_{\text{room},i} \cdot n_{\text{rooms},i}
+
\beta_{\text{dist},i} \cdot \text{dist-centre-km}_i
+
\beta_{\text{under},i}
+
\beta_{\text{prop},i}
+
\beta_{\text{outdoor},i}
+
\text{premium}_i
$$

---

## 🔷 Parameter mapping

The configuration arguments map to the internal Bayesian parameters as follows:

| Config argument | Internal parameter |
|---|---|
| `base` | `intercept` |
| `room_coeff` | `beta_room` |
| `distance_coeff` | `beta_dist` |
| `underground_fee` | `beta_under` |
| `house_fee` | `beta_prop` |
| `garden_fee` | `beta_outdoor` |
| `terrace_fee` | `beta_outdoor` |
| `balcony_fee` | `beta_outdoor` |
| `agent_premium` | `premium` |

---

## 🔷 Hierarchical structure

The model supports three different levels of parameter knowledge.

### 1. Fixed parameters

Some parameters are treated as fully known.

Their true values are directly injected from the synthetic data-generating process and are not inferred.

These act as fixed coefficients inside the likelihood model.

---

### 2. Market-known parameters

For some parameters:

- the global market distribution is assumed known
- but property-level realizations are still inferred

This means the model learns:

$$
p_{k,i}
\sim
\mathcal{N}(\mu_k^{market},\sigma_k^{market})
$$

while treating the market-level values themselves as fixed.

---

### 3. Fully unknown parameters

For fully unknown parameters, both:

- the market-level distribution
- and the property-level coefficients

must be inferred from the data.

---

## 🔷 Market-level hyper-priors

For fully unknown parameters, the model infers market-level distributions:

$$
\mu_{m,k}
\sim
\mathcal{N}
\left(
\mu_k^{market},
k \cdot \sigma_k^{market}
\right)
$$

$$
\sigma_{m,k}
\sim
\text{TruncatedNormal}
\left(
\sigma_k^{market},
\frac{k}{\sqrt{2}} \cdot \sigma_k^{market}
\right)
$$

Where:

- $\mu_k^{market}$ → injected market mean
- $\sigma_k^{market}$ → injected market volatility
- $k$ → uncertainty scaling factor (`market_prior_scale`)

These hyper-priors encode uncertainty about the global market itself.

---

## 🔷 Property-level parameters

Each property receives its own latent coefficient.

Most coefficients are sampled using Gaussian distributions:

$$
p_{k,i}
\sim
\mathcal{N}(\mu_{m,k},\sigma_{m,k})
$$

The room coefficient uses a Laplace distribution to allow heavier tails:

$$
p_{room,i}
\sim
\text{Laplace}(\mu_{m,room},\sigma_{m,room})
$$

---

## 🔷 Agent premium model

In the **Agent hypothesis only**, an additional latent premium is introduced.

First, market-level premium parameters are inferred:

$$
\mu_{prem}
\sim
\text{TruncatedNormal}
\left(
\mu_{\text{premium}}^{market},
\;
k \cdot \sigma_{\text{premium}}^{market}
\right)
$$

$$
\sigma_{prem}
\sim
\text{TruncatedNormal}
\left(
\sigma_{\text{premium}}^{market},
\;
\frac{k}{\sqrt{2}} \cdot \sigma_{\text{premium}}^{market}
\right)
$$

Then property-level premiums are sampled:

$$
p_{premium,i}
\sim
\text{Gamma}(\alpha,\beta)
$$

with:

$$
\alpha
=
\left(
\frac{\mu_{prem}}{\sigma_{prem}}
\right)^2
$$

$$
\beta
=
\frac{\mu_{prem}}{\sigma_{prem}^2}
$$

Finally:

$$
\mu_i \rightarrow \mu_i + p_{premium,i}
$$

This premium exists only under the Agent hypothesis.

---

## 🔷 Sequential Monte Carlo (SMC)

Inference is performed using Sequential Monte Carlo:

$$
P(\theta \mid y)
\approx
\sum_{j=1}^{N}
w_j \delta(\theta-\theta_j)
$$

SMC:
- approximates posterior distributions
- propagates weighted particles
- estimates marginal likelihoods
- supports Bayesian model comparison

Each chain is independently sampled and later concatenated.

---

## 🔷 Output of this script

The script outputs:

- posterior traces
- posterior predictive samples (PPC)
- chain-wise marginal likelihood estimates

Saved outputs include:

- `results/trace/final_trace_owner.pkl`
- `results/trace/final_trace_agent.pkl`
- `results/ppc/ppc_owner.pkl`
- `results/ppc/ppc_agent.pkl`

🚫 No Bayes Factor is computed here.

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
- traces
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
