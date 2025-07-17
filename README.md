# Data Generator use

python data/generate_enhanced_datasets.py --hotels 500 --offers 1000 --users 200

# Mathematical Approach & Core Formulas (2024 Update)

This project uses a multi-objective, simulation-driven approach for hotel offer ranking and market analysis. Below are the core formulas and their roles in the backend and UI.

## 1. Composite Market Demand Index
A weighted index to characterize market demand by combining normalized proxies for price, booking urgency, volatility, and competition:

$$
\text{Market\_Demand\_Index} = w_1 \cdot \text{norm\_avg\_price} + w_2 \cdot (1 - \text{norm\_days\_to\_go}) + w_3 \cdot \text{norm\_price\_variance} + w_4 \cdot \text{norm\_competition\_density}
$$

Where:
- $\text{norm\_avg\_price}$: Normalized average price for the location
- $\text{norm\_days\_to\_go}$: Normalized average days to go
- $\text{norm\_price\_variance}$: Normalized price variance for the location
- $\text{norm\_competition\_density}$: Normalized competition density (unique hotels × unique partners)
- $w_1, w_2, w_3, w_4$: Weights (default 0.25 each, configurable)

## 2. Dynamic Price Sensitivity
Each user's dynamic price sensitivity is modeled as:

$$
\text{dynamic\_sensitivity} = 0.5 \cdot S_{\text{base}} + 0.2 \cdot \left(1 - \frac{\min(D, 180)}{180}\right) + 0.3 \cdot \left(\frac{\sigma_p}{\mu_p + 1}\right)
$$

Where:
- $S_{\text{base}}$: User's basic price sensitivity (from profile)
- $D$: Days to go for check-in (from offers)
- $\mu_p$, $\sigma_p$: Mean and std of all offered prices in the destination

## 3. Multi-Objective Utility Function
Defines the total expected utility $U(i, j)$ of placing offer $j$ at rank $i$:

$$
U(i, j) = w_{\text{rev}} \cdot R_{ij} + w_{\text{rel}} \cdot Q_{ij} + w_{\text{trust}} \cdot T_{ij}
$$

Where:
- $R_{ij}$: Revenue for offer $j$ at position $i$
- $Q_{ij}$: Relevance (quality) score
- $T_{ij}$: Trust score
- $w_{\text{rev}}, w_{\text{rel}}, w_{\text{trust}}$: Tunable weights

## 4. Price Competitiveness CTR Model
Predicts click-through rate (CTR) as a function of price competitiveness:

$$
\text{CTR}_{ij} = \text{base\_CTR}_j \cdot e^{-\beta (\text{price}_j - \text{min\_price})}
$$

Where $\beta$ is a user or market price sensitivity parameter.

## 5. Price Volatility Penalty on Trust
Penalizes trust for offers with unstable prices:

$$
T_{ij} = \text{base\_trust}_j - \gamma \cdot \sigma^2(\text{price}_j, t-24h)
$$

Where $\gamma$ is a volatility penalty weight and $\sigma^2$ is the variance of price over the last 24h.

---

# How These Formulas Are Used
- **Backend:**
  - Market demand index is computed for each location and saved in `user_market_state.csv`.
  - Dynamic price sensitivity is computed for each user and saved in `user_dynamic_price_sensitivity.csv`.
  - Utility, CTR, and trust penalty functions are available for ranking and simulation logic.
- **Frontend (Shiny App):**
  - Displays dynamic price sensitivity, market demand index, and other metrics for users and offers.
  - Uses backend outputs for scenario analysis, ranking, and market intelligence.

---

# Probability of click

This is the simulated (learned) probability that a user will click on a given offer when it is shown at a specific rank.
It is estimated by simulating many (e.g., 1000) user interactions (clicks/no clicks) for each (user, offer, rank) combination, using a recursive average formula.
The value reflects both the offer’s inherent attractiveness (preference_score) and the effect of its position (rank) in the list.

# true click prob

This is the theoretical or base probability that a user would click on an offer at a given rank, before any simulation noise is added.
It is calculated as:

$$
\text{true\_click\_prob} = \min(0.95, \max(0.05, \text{preference\_score} \times \text{rank\_factor}))
$$

where

$$
\text{rank\_factor} = \frac{1}{\text{rank}}
$$

This value is used as the “ground truth” probability for simulating clicks in the bandit simulation.

# probability_of_conversion

This is the estimated probability that a click on an offer will convert to a booking. It is calculated as a product of interpretable factors:

$$
\text{probability\_of\_conversion} = S_{\text{price}} \times S_{\text{rating}} \times S_{\text{amenities}} \times S_{\text{loyalty}} \times S_{\text{brand}}
$$

Where:
- $S_{\text{price}} = \max\left(0.1, 1 - \frac{|\text{trivago\_price} - \text{partner\_price}|}{\text{user\_budget\_max} + 1}\right)$
- $S_{\text{rating}} = 0.5 + 0.1 \times (\text{hotel\_rating} - 3)$
- $S_{\text{amenities}} = 0.5 + 0.1 \times \min(\text{amenities\_match}, 5)$
- $S_{\text{loyalty}} = 1.0$ if user is Gold/Platinum, $0.8$ if Silver, $0.6$ otherwise
- $S_{\text{brand}} = 1.0$ if partner is Booking.com/Expedia, $0.9$ otherwise

The result is clipped to $[0.01, 0.99]$.

# normalized_probability_of_click

This is an optional, research-oriented column. It is computed using softmax normalization (per user, per rank) on the simulated probability_of_click values:

$$
\text{normalized\_probability\_of\_click}_i = \frac{\exp(\text{probability\_of\_click}_i)}{\sum_j \exp(\text{probability\_of\_click}_j)}
$$

This gives a relative likelihood of click among competing offers at the same rank for a user, but is not required for practical modeling or optimization. In real-world ranking, probability_of_click and true_click_prob are independent signals and do not need to sum to 1.


Summary Table:
Column                Meaning
preference_score      How well the offer matches the user’s preferences (composite score, 0–1).
probability_of_click  Simulated/learned probability of click for (user, offer, rank) after 1000 trials.
true_click_prob       Theoretical probability of click for (user, offer, rank) used as the “ground truth”.
normalized_probability_of_click  (Optional) Softmax-normalized probability for relative comparison among offers at the same rank for a user.

## Where to Find Key Metrics in the UI

- **Dynamic Price Sensitivity**: Displayed in the User/Market table and Offers table. Updated automatically after you click 'Consider Scenario'.
- **Market Demand Index**: Displayed in the User/Market table and as a plot in the Market Analysis tab. Updated after scenario consideration.
- **Fallback Logic**: If backend data is missing, the UI will show 'NA' or hide the metric until new data is available.

See tooltips in the UI for explanations of each metric. For formulas, see the Mathematical Approach & Core Formulas section above.
