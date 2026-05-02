# 🏗️ Multivariate Bayesian Discriminator

## 📋 Project Overview
This engine performs **Bayesian Model Comparison** to classify the generative source of real estate data. Given a dataset of property attributes, the system determines whether the observed prices are more likely governed by a **Direct Owner (Model A)** or an **External Agent (Model B)**.

Unlike simple A/B tests, this engine accounts for multiple confounding variables (rooms, location, amenities) to isolate the "Agent Premium" through probabilistic modeling.

---

## 🔄 The Workflow

1. **Multivariate Input**: The system ingests property-level data including:
    * **Categorical**: Property type (House/Flat), Amenities (Garden/Subway).
    * **Numerical**: Number of rooms, distance from city center, and price.
2. **Likelihood Modeling**:
    * **Model A**: Defines a price distribution based on pure property value.
    * **Model B**: Incorporates a "Middleman Bias" (higher mean price or increased variance) representing agent commissions.
3. **Posterior Inference**: Utilizing the **Bayes Factor**, the engine calculates the posterior probability $P(\text{Model} | \text{Data})$. It asks: *"Which model's physics better explains this specific price point given these attributes?"*
4. **Risk-Adjusted Decisioning**: The engine outputs a classification (Owner vs. Agent) alongside a **Certainty Score**, allowing for automated detection of high-markup listings.

---

## 🛠️ Tech Stack

* **Engine**: Python (SciPy/NumPy for Monte Carlo integration).
* **Logic**: Bayesian Inference via Likelihood Ratio testing.
* **UI**: Streamlit for real-time property "fingerprinting."

---

## 🚀 How to Run
1. Install dependencies:  
   `pip install -r requirements.txt`
2. Launch the Decision Engine:  
   `streamlit run app/main_ui.py`