# 📊 Data Generation — `data/gen_prop_data.py`

## 🧾 Overview

This script generates synthetic real estate datasets used by BaPPI. It defines a controlled market where property features determine rental prices through a known generative process.

The output serves as the **ground-truth dataset** for the Bayesian inference pipeline.

---

## ⚙️ What it does

- Generates synthetic property features (type, rooms, location, amenities)
- Builds a **log-linear pricing model** (natural logarithm)
- Converts log-prices into observed rents
- Supports three modes:
  - `fixed`
  - `random`
  - `hierarchical` (main mode used by BaPPI)
- In hierarchical mode, splits data into:
  - Owner-generated listings
  - Agent-generated listings

---

## 📐 Generative pricing model

Prices are generated from a (natural) log-rent model:

$$
\log(y_i) = \mu_i + \epsilon_i
$$

- $y_i$: monthly rent
- $\mu_i$: structural log-price
- $\epsilon_i$: noise term

Noise is:
- Gaussian in `fixed` mode
- Student-t in `random` and `hierarchical` modes

Final observed price is obtained via:

$$
y_i = \exp(\log(y_i))
$$

---

## 🧮 Structural price model

The log-price is computed as:

$$\mu_i = \text{intercept}_i + \beta_{\text{room},i} \cdot n_{\text{rooms},i} + \beta_{\text{dist},i} \cdot \text{dist-centre-km}_i (+ \beta_{\text{under},i} + \beta_{\text{prop},i} + \beta_{\text{outdoor},i} + \text{premium}_i)$$

where:

- `base` (i.e. intercept)  → baseline rent level 
- `room_coeff` (i.e. beta_room)  → effect of number of rooms  
- `distance_coeff` (i.e. beta_dist)  → effect of distance from centre  
- `underground_fee` (i.e. beta_under)  → proximity to underground stations  
- `house_fee` (i.e. beta_prop)  → House vs Flat adjustment  
- `garden_fee / terrace_fee / balcony_fee` (i.e. beta_outdoor)  → outdoor space effects  
- `agent_premium` (i.e. premium)  → agent markup (only in Agent case)

---

## 📉 Parameter distributions

Each structural coefficient is generated from a specific distribution:

- **Gaussian (Normal)**
  - $\beta_0$
  - $\beta_{dist}$
  - $\beta_{under}$
  - $\beta_{type}$
  - $\beta_{outdoor}$

- **Laplace (heavy-tailed)**
  - $\beta_{room}$

- **Gamma (positive-only)**
  - $\beta_{agent}$ (agent premium, only in Agent case)

---

## 🏗️ Hierarchical mode

In the `hierarchical` mode:

- Each coefficient is first sampled from a **market-level distribution**
- Then each property draws its own coefficient from that market distribution

This creates:
- a shared global market structure
- property-level variation around it

---

## 📦 Output (`data/datasets/`)

Depending on mode:

- `london_rentals_hierarchical.json` (main mode)
- `london_rentals_fixed.csv`
- `london_rentals_random.csv`
- `.ini` snapshot of config for reproducibility

---

## 📌 Summary

This module defines a controlled synthetic market:

> Property features → log-linear pricing model → exponential transform → observed rents

This ground truth is then recovered (or not) by the Bayesian inference pipeline in `src/`.