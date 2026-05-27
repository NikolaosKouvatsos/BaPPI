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

$$P(\theta \mid y, M) = \frac{P(y \mid \theta, M) P(\theta \mid M)}{P(y \mid M)}$$

where:

- $P(\theta \mid y, M)$ → posterior distribution under model $M$
- $P(y \mid \theta, M)$ → likelihood of the data under parameters $\theta$ and model $M$
- $P(\theta \mid M)$ → prior distribution under model $M$
- $P(y \mid M)$ → marginal likelihood (model evidence)

The model uses:
- prior assumptions about the housing market
- observed rental prices
- hierarchical latent parameters

to infer posterior distributions over pricing coefficients.

In this framework, inference is performed separately under two competing models:

- $M_{\text{owner}}$
- $M_{\text{agent}}$

Each model induces a different likelihood due to the presence or absence of the agent premium term.

---

## 🔷 Observed likelihood model

Observed (natural) log-rents are modeled using a Student-t likelihood:

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

The latent (natural) log-price is computed as:

$$
\mu_i=\sum_l p_{l,i} x_{l,i}
$$

where $l$ describes some specific parameter and:

- $x_{l,i}$ → observed property features
- $p_{l,i}$ → latent property-specific pricing coefficients

Concretely:

$$
\mu_i =
\text{intercept}_i
+
\beta_{\text{room},i} \cdot n_{\text{rooms},i}
+
\beta_{\text{dist},i} \cdot \text{dist-centre-km}_i
(+
\beta_{\text{under},i}
+
\beta_{\text{prop},i}
+
\beta_{\text{outdoor},i}
+
\text{premium}_i)
$$

---

## 🔷 Joint likelihood over all properties

The model assumes conditional independence between properties given the latent parameters and the model structure $M$.

Therefore, the total likelihood is computed as the product of all individual property likelihoods:

$$
P(y \mid \theta, M)=\prod_{i=1}^{N}
P(y_i \mid \theta_i, M)
$$

Under the Student-t observation model:

$$
P(y \mid \theta, M)=\prod_{i=1}^{N}
\text{StudentT}
\left(
y_i
\mid
\nu=5,
\mu_i(M),
\sigma
\right)
$$

Equivalently, the total log-likelihood becomes:

$$
\log P(y \mid \theta,M)=\sum_{i=1}^{N}
\log
\text{StudentT}
\left(
y_i
\mid
\nu=5,
\mu_i(M),
\sigma
\right)
$$

This is the quantity used internally during Bayesian inference and SMC sampling.

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
- but property-level realisations are still inferred

This means the model learns:

$$
p_{l,i}
\sim
\mathcal{N}(\mu_l^\text{market},\sigma_l^\text{market})
$$

while treating the market-level values themselves as fixed.

---

### 3. Fully unknown parameters

For fully unknown parameters, both:

- the market-level distribution
- and the property-level coefficients

must be inferred from the data.

The model infers market-level distributions:

$$
\mu_{\text{m},l}
\sim
\mathcal{N}
\left(
\mu_l^\text{market},
k \cdot \sigma_l^\text{market}
\right)
$$

$$
\sigma_{\text{m},l}
\sim
\text{TruncatedNormal}
\left(
\sigma_l^{market},
\frac{k}{\sqrt{2}} \cdot \sigma_l^\text{market}
\right)
$$

where:

- $\mu_l^\text{market}$ → injected market mean
- $\sigma_l^\text{market}$ → injected market volatility
- $k$ → uncertainty scaling factor (`market_prior_scale`)

These hyper-priors encode uncertainty about the global market itself.

Each property receives its own latent coefficient.

Most coefficients are sampled using Gaussian distributions:

$$
p_{l,i}
\sim
\mathcal{N}(\mu_{\text{m},l},\sigma_{\text{m},l})
$$

The room coefficient uses a Laplace distribution to allow heavier tails:

$$
p_{\text{room},i}
\sim
\text{Laplace}(\mu_\text{m,room},\sigma_\text{m,room})
$$

---

## 🔷 Agent premium model

In the **Agent hypothesis only**, an additional latent premium is introduced.

First, market-level premium parameters are inferred:

$$
\mu_\text{m,prem}
\sim
\text{TruncatedNormal}
\left(
\mu_{\text{premium}}^\text{market},
k \cdot \sigma_{\text{premium}}^\text{market}
\right)
$$

$$
\sigma_\text{m,prem}
\sim
\text{TruncatedNormal}
\left(
\sigma_{\text{premium}}^\text{market},
\frac{k}{\sqrt{2}} \cdot \sigma_{\text{premium}}^\text{market}
\right)
$$

Then property-level premiums are sampled:

$$
p_{\text{premium},i}
\sim
\text{Gamma}(\alpha,\beta)
$$

with:

$$
\alpha=\left(
\frac{\mu_\text{m,prem}}{\sigma_\text{m,prem}}
\right)^2
$$

$$
\beta=\frac{\mu_\text{m,prem}}{\sigma_\text{m,prem}^2}
$$

Finally:

$$
\mu_i \rightarrow \mu_i + p_{\text{premium},i}
$$

This premium exists only under the Agent hypothesis.

---

## 🔷 Sequential Monte Carlo (SMC)

Inference is performed using Sequential Monte Carlo (SMC).

SMC:
- approximates posterior distributions
- propagates weighted particles
- estimates marginal likelihoods
- supports Bayesian model comparison

Each chain is independently sampled and later concatenated.

---

## 🔷 Output of this script

The script outputs:

- traces
- posterior predictive samples (PPC)

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

Model comparison is performed by comparing the marginal likelihood (model evidence) of each model.

For a model $M$, the evidence is:

$$
Z_M = P(y \mid M) = \int P(y \mid \theta, M) P(\theta \mid M) d\theta
$$

This quantity integrates out all latent parameters $\theta$, and therefore measures how well the entire model explains the observed data.

In practice, $Z_M$ is not computed analytically. Instead, it is estimated using Sequential Monte Carlo (SMC), which provides an approximation to the log marginal likelihood for each chain.

Once the marginal likelihoods have been estimated for both models, we obtain the Bayes Factor as:

$$
BF = \frac{Z_{\text{agent}}}{Z_{\text{owner}}}
$$

and in (natural) log-space:

$$
\log BF = \log Z_{\text{agent}} - \log Z_{\text{owner}}
$$

Interpretation:
- $BF > 1$ → evidence favours the agent model  
- $BF < 1$ → evidence favours the owner model  
- The magnitude reflects the strength of evidence  

Note:

The Bayes Factor is not computed during inference. Instead:

- each model is fit independently using SMC
- SMC produces an estimate of the marginal likelihood $Z_M$
- the Bayes Factor is computed after sampling from these estimates

Thus, the Bayes Factor depends entirely on the estimated model evidence produced during sampling.

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
y_\text{observed} \sim y_\text{posterior}
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
\text{rank}_i = P(p_i < p_i^\text{true})
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

These outputs can be found in `results/plots/`.

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
