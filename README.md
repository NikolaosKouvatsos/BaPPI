# 🏠 BaPPI — Bayesian Property Provider Identifier

```text
🏠  ____        _____  _____ _____  📈
    |  _ \      |  __ \|  __ \   _|
    | |_) | __ _| |__) | |__) | |  
    |  _ < / _` |  ___/|  ___/| |  
    | |_) | (_| | |    | |   _| |_ 
    |____/ \__,_|_|    |_|  |_____|
📊                                  🚀
```

## 📋 Overview

**BaPPI** is a hierarchical Bayesian inference framework designed to analyse simulated real estate markets in London and determine whether rental listings are more likely managed directly by property owners or by third-party agents.

The engine models how structural property characteristics influence pricing whilst accounting for the additional pricing effects introduced by intermediaries. Using probabilistic inference and Sequential Monte Carlo sampling, BaPPI estimates posterior distributions over pricing parameters and evaluates competing generative explanations for observed property datasets.

The framework is particularly focused on:

- Bayesian market modelling
- Hierarchical market uncertainty
- Agent commission effects
- Bayesian model comparison
- Synthetic real estate market simulation

---

# 🔄 Pipeline Overview

BaPPI executes a full end-to-end Bayesian analysis pipeline:

## 1. Property Market Generation

Synthetic real estate data is generated with configurable:

- Property types (houses/flats)
- Room counts
- Distance from city centre
- Outdoor amenities
- Underground station proximity
- Market volatility
- Agent premiums

The simulation produces both owner-managed and agent-managed listings.

---

## 2. Hierarchical Bayesian Analysis

The engine performs hierarchical Bayesian inference over the pricing parameters governing the simulated market using:

- Sequential Monte Carlo (SMC)
- Market-level priors
- Hierarchical uncertainty modelling
- Property-level structural information

Rather than analysing properties independently, BaPPI jointly analyses a collection of properties simultaneously under two competing market hypotheses:

- **Owner Model** — prices are generated without agent intervention
- **Agent Model** — prices include an additional agent premium contribution

BaPPI runs its analysis while accounting for structural variables such as:

- Number of rooms
- Property type
- Distance from the city centre
- Outdoor amenities
- Underground station proximity

---

## 3. Posterior Inference & Model Comparison

For a jointly analysed collection of properties, BaPPI estimates:

- Posterior distributions over the pricing parameters
- Posterior uncertainty in the pricing model
- Log marginal likelihoods for competing hypotheses
- A Bayes factor comparing the Owner and Agent models

The analysis is performed collectively across multiple properties rather than independently for each listing.

---

## 4. Diagnostics & Evaluation

The bulk of the post-analysis stage includes:

- Diagnostic outputs
- Posterior summaries
- Bayesian convergence information
- Market parameter estimates
- Model comparison statistics

---

# ⚙️ Configuration

Before running the project, the user should first configure the simulation and analysis settings in:

```bash
app/config.ini
```

This file controls all major aspects of the framework.

---

## Important Configuration Sections

### `[GENERAL]`

Controls execution mode and reproducibility.

```ini
mode="hierarchical"
seed=14
```

Currently, the full pipeline only supports:

```ini
mode="hierarchical"
```

---

### `[DATA_GENERATION]`

Controls synthetic market generation:

- Number of properties
- Flat/house ratios
- Room distributions
- Distance distributions
- Amenity probabilities
- Underground proximity

Example:

```ini
num_properties=1000
flat_house_ratio=3
room_exp_num=2.5
```

---

### `[PRICE_ARGUMENTS]`

Defines the pricing model used during simulation and inference.

Includes:

- Base property value
- Room pricing coefficients
- Distance penalties
- House premiums
- Garden/Balcony/Terrace fees
- Agent premium
- Market volatility

Example:

```ini
agent_premium=0.15
market_vol=0.02
```

---

### `[ANALYSIS]`

Controls the Bayesian inference stage:

- Number of analysed properties
- Prior scaling
- Fixed parameters
- Known market-level parameters
- Monte Carlo draws
- Number of chains

Example:

```ini
mc_draws=1000
chains=4
```

---

# 🛠️ Tech Stack

## Core Libraries

- Python
- PyMC
- ArviZ
- NumPy
- SciPy
- Pandas
- Statsmodels

## Visualisation

- Matplotlib
- Seaborn

## Inference Methods

- Hierarchical Bayesian Modelling
- Sequential Monte Carlo (SMC)
- Posterior Probability Estimation
- Bayesian Model Comparison

---

# 📦 Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 How To Run

## Step 1 — Configure the Engine

Edit:

```bash
app/config.ini
```

Adjust the simulation and Bayesian analysis parameters according to your experiment.

---

## Step 2 — Run the Full Pipeline

Launch the framework with:

```bash
python app/main.py
```

---

# ▶️ What `main.py` does

The main execution script automatically performs:

1. Property data generation
2. Hierarchical Bayesian analysis
3. Post-analysis diagnostics

Execution flow:

```text
Generate synthetic market →
Run Bayesian inference →
Perform post-analysis
```

---

# 🧠 Bayesian Modelling Philosophy

Unlike traditional regression or binary classification systems, BaPPI treats market behaviour probabilistically and evaluates competing market-wide pricing hypotheses.

Rather than independently classifying individual properties, the framework jointly analyses a collection of listings under two competing generative models:

- An **Owner Model**
- An **Agent Model** containing an additional pricing premium

The Bayesian framework then evaluates which model better explains the observed dataset using posterior inference and Bayes factor comparison.

This approach allows the engine to:

- Explicitly model uncertainty
- Incorporate market-level volatility
- Compare competing market hypotheses probabilistically
- Measure how strongly the observed data support one market model relative to another through the computation of the Bayes factor

---

# 📊 Example Features Used

Each property may include:

| Feature | Type |
|---|---|
| Property Type | Categorical |
| Number of Rooms | Numerical |
| Distance to Centre | Numerical |
| Underground Access | Binary |
| Garden | Binary |
| Balcony | Binary |
| Terrace | Binary |
| Price | Numerical |

---

# 🔬 Research Applications

BaPPI can be extended towards:

- Rental market anomaly detection
- Agent markup estimation
- Urban economics simulations
- Housing market forecasting
- Probabilistic pricing systems
- Bayesian market microstructure analysis

---

# 🤝 Support & Contact

If you are interested in using, extending, or experimenting with BaPPI, the author is happy to help with setup, methodology, or implementation questions.

Researchers, students, and developers working on Bayesian modelling, probabilistic inference, or real estate market analysis are welcome to get in touch.

---

# 👤 Author

**Nikolaos Kouvatsos**  
May 2026

---

# 📜 License

This project is licensed under the MIT License.