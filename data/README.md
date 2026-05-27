# 📊 Data Generation — `data/gen_prop_data.py`

## 🧾 Overview

This script generates synthetic real estate datasets used by BaPPI. It simulates property features and corresponding rental prices under configurable market conditions defined in `app/config.ini`.

The generated data is used as the ground-truth input for the Bayesian inference pipeline.

---

## ⚙️ What it does

- Generates synthetic property features (type, rooms, location, amenities)
- Simulates rental prices based on configurable pricing rules
- Supports three modes:
  - `fixed`
  - `random`
  - `hierarchical` (main mode used by BaPPI)
- In hierarchical mode, the dataset is split into:
  - Owner-generated listings
  - Agent-generated listings

---

## 📦 Output (`data/datasets/`)

Depending on the selected mode, the script saves:

- `london_rentals_hierarchical.json` (main mode)
- `london_rentals_fixed.csv`
- `london_rentals_random.csv`
- `.ini` snapshot of the config used for reproducibility

---

## 📌 Summary

This module defines the **synthetic market environment** that BaPPI later analyses using Bayesian inference.
