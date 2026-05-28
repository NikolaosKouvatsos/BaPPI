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

- $P(\theta \mid y, M)$ → posterior distribution of parameters $\theta$ under model $M$
- $P(y \mid \theta, M)$ → likelihood of the data under parameters $\theta$ and model $M$
- $P(\theta \mid M)$ → prior distribution of parameters $\theta$ under model $M$
- $P(y \mid M)$ → marginal likelihood (model $M$ evidence)

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

where:

- $y_i$ → observed log-rent
- $\mu_i$ → latent structural log-price
- $\sigma$ → residual noise scale

---

## 🔷 Structural linear predictor

The latent (natural) log-price is computed as:

$$
\mu_i=\sum_l p_{l,i} x_{l,i}
$$

where:

- $l$ → some specific parameter
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

The model supports three different levels of initial parameter knowledge.

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

The room coefficient is an exception, as it uses a Laplace distribution to allow heavier tails:

$$
p_{\text{room},i}
\sim
\mathcal{N}(\mu_\text{room}^\text{market},\sigma_\text{room}^\text{market})
$$

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

> 💡 The factor of $1/\sqrt{2}$ is motivated by how the Standard Error of the Standard Deviation typically scales in relation to the Standard Error of the Mean.
> Ultimately, this ratio preserves a structured statistical balance while leaving absolute calibration entirely in the user's hands via $k$.

These hyper-priors encode uncertainty about the global market itself.

Each property receives its own latent coefficient.

Most coefficients are sampled using Gaussian distributions:

$$
p_{l,i}
\sim
\mathcal{N}(\mu_{\text{m},l},\sigma_{\text{m},l})
$$

Again, the room coefficient uses a Laplace distribution:

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

Then, property-level premiums are sampled:

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

# 2. Model comparison + diagnostics + evaluation (`src/run_post_analysis.py`)

This script performs all post-inference model evaluation and comparison tasks, including computation of the Bayes Factor, posterior predictive checks, hierarchical recovery diagnostics, and posterior calibration analysis.

---

## 🔷 Bayes Factor computation

Model comparison is performed using the marginal likelihood (model evidence) of each model.

For a model $M$, the evidence is:

$$
Z_M = P(y \mid M) = \int P(y \mid \theta, M) P(\theta \mid M) d\theta
$$

where:

- $y$ denotes the observed data
- $\theta$ denotes the latent model parameters
- $P(y \mid \theta, M)$ is the likelihood
- $P(\theta \mid M)$ is the prior distribution

The marginal likelihood integrates over the entire parameter space and therefore measures how well the model explains the observed data while accounting for model complexity.

In practice, $Z_M$ is not computed analytically. Instead, it is estimated using Sequential Monte Carlo (SMC), which provides an estimate of the log marginal likelihood for each independent sampling chain.

The script extracts the per-chain values of:

$$
\log Z_M
$$

and computes the mean evidence estimate across chains for each model.

Once both model evidences have been estimated, the Bayes Factor is computed as:

$$
BF = \frac{Z_{\mathrm{agent}}}{Z_{\mathrm{owner}}}
$$

and equivalently in (natural) log-space:

$$
\log BF = \log Z_{\mathrm{agent}} - \log Z_{\mathrm{owner}}
$$

Interpretation:

- $BF > 1$ → evidence favours the Agent model
- $BF < 1$ → evidence favours the Owner model
- larger magnitudes indicate stronger evidence

The script also visualizes the estimated log evidence across chains to assess consistency of the SMC evidence estimates.

Note that the Bayes Factor is not computed during posterior sampling itself. Instead:

- each model is fit independently using SMC
- SMC estimates the marginal likelihood $Z_M$
- the Bayes Factor is computed afterward from these evidence estimates

Thus, the Bayes Factor depends entirely on the estimated model evidence produced during sampling.

> ❗ Strictly speaking, Bayesian model comparison is based on the posterior odds ratio:
>
> $$
> \frac{P(M_{\mathrm{agent}} \mid y)}
> {P(M_{\mathrm{owner}} \mid y)}
> $$
>
> which measures which model is more probable given the observed data $y$.
>
> Using Bayes' theorem:
>
> $$
> \frac{P(M_{\mathrm{agent}} \mid y)} {P(M_{\mathrm{owner}} \mid y)} = BF \times \frac{P(M_{\mathrm{agent}})} {P(M_{\mathrm{owner}})}
> $$
>
> where:
>
> - $BF$ is the Bayes Factor
> - the second term represents the prior odds between the two models
>
> In this analysis, the Agent and Owner models are assigned equal prior probability:
>
> $$
> P(M_{\mathrm{agent}}) = P(M_{\mathrm{owner}})
> $$
>
> so the prior odds ratio is 1. Under this assumption, the posterior odds ratio is determined entirely by the Bayes Factor.

---

## 🔷 Evidence consistency analysis

The script:

- extracts per-chain log marginal likelihood estimates
- computes:

  - mean $\log Z$ per model
  - Bayes Factor
  - cross-chain evidence consistency diagnostics
- visualizes evidence agreement across chains

This helps assess whether independent SMC chains reached similar evidence estimates.

---

## 🔷 Acceptance-rate diagnostics

The script visualizes SMC acceptance rates across inverse-temperature values ($\beta$):

$$
\beta : 0 \rightarrow 1
$$

where:

- $\beta = 0$ corresponds to the prior distribution
- $\beta = 1$ corresponds to the full posterior distribution

This diagnostic helps assess the stability and efficiency of the SMC tempering procedure during evidence integration.

---

## 🔷 Posterior predictive checks (PPC)

The script compares:

$$
y_{\mathrm{observed}} \sim y_{\mathrm{posterior}}
$$

for both models using posterior predictive samples.

This evaluates whether the inferred models reproduce the observed log-price distributions.

---

## 🔷 Hierarchical parameter recovery

For each market-level parameter, the script compares posterior estimates against the injected ground-truth simulation values.

Computed diagnostics include:

- posterior median
- posterior standard deviation
- recovery Z-score
- two-tailed posterior tail probability diagnostic

This analysis evaluates whether the hierarchical model successfully recovers the underlying simulated market structure.

---

## 🔷 Posterior rank calibration

For each property-level parameter, the script computes the empirical posterior rank of the injected truth value:

$$
\mathrm{rank}_i = P(p_i < p_i^{\mathrm{true}})
$$

If calibration is correct and the posterior is well specified, the resulting rank histogram should be approximately uniform on $[0,1]$.

Deviations from uniformity may indicate:

- posterior bias
- excessive shrinkage
- under/over-dispersion
- model misspecification

For sparse binary-style features (e.g. balcony, terrace, garden, premium), calibration is evaluated only on active properties with non-zero injected effects.

---

## 🔷 Agent-premium shrinkage analysis

The script compares:

- injected property-level agent premiums
- posterior median estimates

across all properties.

This visualization highlights:

- hierarchical shrinkage effects
- population-level regularization
- distortion toward the inferred market mean

---

## 🔷 Summary outputs

This script generates:

- Bayes Factor + evidence comparison
- evidence consistency plots
- SMC acceptance-rate diagnostics
- PPC plots (Owner vs Agent)
- posterior corner plots
- parameter recovery diagnostics
- posterior rank calibration histograms
- shrinkage visualizations

All generated outputs are saved to:

`results/plots/`

---

# 3. Conceptual summary

### `src/run_bayesian_analysis.py`

Implements:

> A hierarchical Bayesian generative model and performs posterior inference using Sequential Monte Carlo (SMC).

It produces:

- posterior traces (containing marginal likelihood $\log Z$ estimates)
- posterior predictive samples (PPC)

while keeping model comparison separate from inference itself.

---

### `src/run_post_analysis.py`

Implements:

> Post-inference evaluation, diagnostics, and Bayesian model comparison.

It performs:

- Bayes Factor computation
- evidence consistency analysis
- posterior predictive validation
- parameter recovery diagnostics
- posterior calibration analysis
- shrinkage evaluation

---

# 4. Full pipeline

$$
\text{Observed Data}
\rightarrow
\text{Hierarchical Bayesian Model}
\rightarrow
\text{SMC Posterior Inference}
$$

$$
\rightarrow
\text{Posterior + PPC + Log Evidence}
\rightarrow
\text{Bayes Factor + Diagnostics + Recovery Analysis}
$$

---

This framework is designed to test whether an agent-specific pricing effect can be statistically identified under hierarchical uncertainty, while maintaining a strict separation between inference, model comparison, and diagnostic evaluation.
