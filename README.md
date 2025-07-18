# trivago Hotel Offer Ranking Simulator

## Project Overview & Business Context

trivago, as a leading hotel metasearch engine, faces a unique challenge: it does not set hotel prices, but must rank competing offers for the same hotel from multiple partners (e.g., Booking.com, Expedia, HotelDirect). The core business problem is to design a ranking system that:
- Maximizes trivago's commission revenue (by promoting higher-commission offers)
- Maintains high user conversion (by showing relevant, attractive offers)
- Preserves long-term user trust (by avoiding manipulative or inconsistent rankings)

This project simulates and optimizes this multi-objective ranking problem, providing a realistic, data-driven environment for strategy development and experimentation.

## Features & Modeling Approach

### Market Demand Characterization
- **Market Demand Index (MDI):** We construct a composite index for each destination using proxies such as price trends (from 24h price history), competition density (unique hotels × partners), and booking urgency (days to go). This index dynamically segments the market into low, medium, or high demand states.

### Dynamic Price Sensitivity
- **User-Specific Modeling:** Each user's price sensitivity is not static. It is dynamically adjusted based on their days_to_go, the volatility and level of prices in their destination, and their base profile. This allows the system to simulate realistic user behavior under different market conditions.

### Predictive Models (Forecasting)
- **pCTR (Predicted Click-Through Rate):** ML models estimate the probability a user will click an offer, based on price competitiveness, brand, and user context.
- **pCVR (Predicted Conversion Rate):** ML models estimate the probability a click will convert to a booking, incorporating price difference, hotel quality, amenities match, and user/brand factors.
- These predictions serve as key inputs to the optimization layer.

### Optimization Strategies
- **Stochastic Linear Programming (LP):**
  - Maximizes expected revenue subject to business constraints (e.g., minimum trust score, partner diversity).
  - Mathematical Formulation:
    ```math
    \max_{x} \sum_{i,j} x_{ij} (w_{rev} R_{ij} + w_{rel} Q_{ij} + w_{trust} T_{ij})
    ```
    where $x_{ij}$ is the assignment of offer $j$ to rank $i$, and $R_{ij}$, $Q_{ij}$, $T_{ij}$ are revenue, relevance, and trust scores.
- **Reinforcement Learning (DQN):**
  - Acts as a strategic layer, learning the optimal policy for configuring the LP solver's constraints based on observed market state.
  - Enables adaptive, context-aware ranking strategies that evolve over time.
- **Multi-Armed Bandit (MAB):**
  - Used for visualizing and experimenting with the exploration-exploitation trade-off between different ranking strategies.
  - Provides empirical feedback on which strategy performs best under varying market/user conditions.

## Architecture

- **Frontend (R/Shiny):**
  - Serves as the interactive UI layer for scenario setup, strategy selection, and results visualization.
  - Handles user input, displays tables/plots, and provides a seamless analyst experience.
- **Backend (Python/FastAPI):**
  - Contains all business logic, including data simulation, ML predictions, and optimization models (LP, RL, MAB).
  - Exposes a REST API for all operations, making the system modular and scalable.
  - All data generation, scenario saving, and advanced modeling happens here.
- **Communication:**
  - The frontend and backend communicate exclusively via REST API calls, ensuring clear separation of concerns and easy extensibility.

---

# Backend Data File Management

## Data Directory and File Visibility

All backend-generated data files (CSVs) are written to the `/data/` directory inside the container. This directory is mapped to the `./data/` directory on the host via Docker Compose:

```yaml
volumes:
  - ./data:/data
```

**Important:**
- All backend code must use absolute paths (e.g., `/data/filename.csv`) for reading and writing data files.
- Do **not** use relative paths like `data/filename.csv` or `/app/data/filename.csv`.
- This ensures that all files are visible both inside the container and on the host machine in the `data/` directory.

## Key Data Files and Their Purpose

| File                                 | Location         | Purpose                                                      | Key Columns / Notes                                  |
|--------------------------------------|------------------|--------------------------------------------------------------|------------------------------------------------------|
| enhanced_user_profiles.csv           | /data/           | User profiles for simulation                                 | user_id, price_sensitivity, preferred_amenities, ... |
| enhanced_hotels.csv                  | /data/           | Hotel metadata                                               | hotel_id, name, location, ...                        |
| enhanced_partner_offers.csv          | /data/           | Partner offer details                                        | offer_id, hotel_id, partner_name, price_per_night... |
| trial_sampled_offers.csv             | /data/           | Sampled offers for current scenario                          | user_id, offer_id, location, days_to_go, ...         |
| user_dynamic_price_sensitivity.csv   | /data/           | Dynamic price sensitivity per user/destination               | user_id, destination, base_price_sensitivity, dynamic_price_sensitivity, ... |
| user_market_state.csv                | /data/           | Market demand index and state per user/destination           | location, demand_index, market_state_label, ...       |
| bandit_simulation_results.csv        | /data/           | Results of bandit simulation                                 | user_id, offer_id, rank, probability_of_click, ...    |
| conversion_probabilities.csv         | /data/           | Conversion probability for each user-offer                   | user_id, offer_id, destination, conversion_probability|

## Workflow for Generating and Accessing Data Files

- **Dynamic Price Sensitivity**
  - Triggered via the UI or API endpoint `/user_dynamic_price_sensitivity_csv`.
  - Output: `/data/user_dynamic_price_sensitivity.csv` (visible on host as `data/user_dynamic_price_sensitivity.csv`).

- **Conversion Probabilities**
  - Triggered via the UI or API endpoint `/conversion_probabilities_csv`.
  - Output: `/data/conversion_probabilities.csv` (visible on host as `data/conversion_probabilities.csv`).

- **Other Data Files**
  - Generated as part of scenario setup, sampling, or simulation.
  - Always written to `/data/`.

## Troubleshooting File Visibility

- If you do not see a file in your host's `data/` directory after triggering its generation:
  1. Ensure the backend writes to `/data/filename.csv` (not a relative or /app/data path).
  2. Check your `docker-compose.yml` for the correct volume mapping (`./data:/data`).
  3. Restart the backend container after any changes to volume mapping or file paths.
  4. Use backend debug logs to confirm the actual file path and working directory.

- You can inspect files inside the container with:
  ```sh
  docker compose exec backend ls -l /data
  ```

- All files in `/data/` inside the container should appear in `./data/` on the host.

---

# Mathematical Approach & Core Formulas

This project uses a multi-objective, simulation-driven approach for hotel offer ranking and market analysis. Below are the core formulas and their roles in the backend and UI.

## 1. Composite Market Demand Index
A weighted index to characterize market demand by combining normalized proxies for price, booking urgency, volatility, and competition:

```math
\text{MarketDemandIndex} = w_1 \cdot \text{normAvgPrice} + w_2 \cdot (1 - \text{normDaysToGo}) + w_3 \cdot \text{normPriceVariance} + w_4 \cdot \text{normCompetitionDensity}
```
Where:
- $\text{normAvgPrice}$: Normalized average price for the location
- $\text{normDaysToGo}$: Normalized average days to go
- $\text{normPriceVariance}$: Normalized price variance for the location
- $\text{normCompetitionDensity}$: Normalized competition density (unique hotels × unique partners)
- $w_1, w_2, w_3, w_4$: Weights (default 0.25 each, configurable)

## 2. Dynamic Price Sensitivity
Each user's dynamic price sensitivity is modeled as:

```math
\text{dynamic\_sensitivity} = 0.5 \cdot S_{\text{base}} + 0.2 \cdot \left(1 - \frac{\min(D, 180)}{180}\right) + 0.3 \cdot \left(\frac{\sigma_p}{\mu_p + 1}\right)
```

Where:
- $S_{\text{base}}$: User's basic price sensitivity (from profile)
- $D$: Days to go for check-in (from offers)
- $\mu_p$, $\sigma_p$: Mean and std of all offered prices in the destination

## 3. Multi-Objective Utility Function
Defines the total expected utility $U(i, j)$ of placing offer $j$ at rank $i$:

```math
U(i, j) = w_{\text{rev}} \cdot R_{ij} + w_{\text{rel}} \cdot Q_{ij} + w_{\text{trust}} \cdot T_{ij}
```

Where:
- $R_{ij}$: Revenue for offer $j$ at position $i$
- $Q_{ij}$: Relevance (quality) score
- $T_{ij}$: Trust score
- $w_{\text{rev}}, w_{\text{rel}}, w_{\text{trust}}$: Tunable weights

## 4. Price Competitiveness CTR Model
Predicts click-through rate (CTR) as a function of price competitiveness:

```math
\text{CTR}_{ij} = \text{base\_CTR}_j \cdot e^{-\beta (\text{price}_j - \text{min\_price})}
```

Where $\beta$ is a user or market price sensitivity parameter.

## 5. Price Volatility Penalty on Trust
Penalizes trust for offers with unstable prices:

```math
T_{ij} = \text{base\_trust}_j - \gamma \cdot \sigma^2(\text{price}_j, t-24h)
```

Where $\gamma$ is a volatility penalty weight and $\sigma^2$ is the variance of price over the last 24h.

## 6. Linear Programming Optimization Model

### Objective Function
The main optimization objective maximizes expected revenue while considering multiple factors:

```math
\max_{x_{ij}} \sum_{i=1}^{n_{\text{offers}}} \sum_{j=1}^{n_{\text{positions}}} x_{ij} \cdot \left(w_{\text{conv}} \cdot C_i + w_{\text{rev}} \cdot R_i + w_{\text{trust}} \cdot T_i\right) \cdot \frac{1}{\log_2(j + 2)}
```

Where:
- $x_{ij}$: Binary decision variable (1 if offer $i$ is placed at position $j$)
- $C_i$: Conversion likelihood score for offer $i$
- $R_i$: Revenue score for offer $i$
- $T_i$: Trust score for offer $i$
- $w_{\text{conv}}, w_{\text{rev}}, w_{\text{trust}}$: Objective weights
- $\frac{1}{\log_2(j + 2)}$: Position weight (higher positions get more weight)

### Constraints

**Position Uniqueness:**
```math
\sum_{i=1}^{n_{\text{offers}}} x_{ij} \leq 1 \quad \forall j \in \{1, \ldots, n_{\text{positions}}\}
```

**Offer Uniqueness:**
```math
\sum_{j=1}^{n_{\text{positions}}} x_{ij} \leq 1 \quad \forall i \in \{1, \ldots, n_{\text{offers}}\}
```

**Price Competitiveness:**
```math
x_{ij} = 0 \quad \forall i, j \text{ where } \text{price}_i > 1.5 \cdot \min_{k} \text{price}_k \text{ and } j \leq 3
```

**Partner Diversity:**
```math
\sum_{i \in \text{Partner}_p} \sum_{j=1}^{5} x_{ij} \leq 2 \quad \forall \text{partner } p
```

## 7. Conversion Probability Model

The conversion probability for an offer is calculated as a product of interpretable factors:

```math
P(\text{conversion}) = S_{\text{price}} \times S_{\text{rating}} \times S_{\text{amenities}} \times S_{\text{loyalty}} \times S_{\text{brand}}
```

Where each factor is defined as:

**Price Factor:**
```math
S_{\text{price}} = \max\left(0.1, 1 - \frac{|\text{trivago\_price} - \text{partner\_price}|}{\text{user\_budget\_max} + 1}\right)
```

**Rating Factor:**
```math
S_{\text{rating}} = 0.5 + 0.1 \times (\text{hotel\_rating} - 3)
```

**Amenities Factor:**
```math
S_{\text{amenities}} = 0.5 + 0.1 \times \min(\text{amenities\_match}, 5)
```

**Loyalty Factor:**
```math
S_{\text{loyalty}} = \begin{cases} 
1.0 & \text{if user is Gold/Platinum} \\ 
0.8 & \text{if Silver} \\ 
0.6 & \text{otherwise} 
\end{cases}
```

**Brand Factor:**
```math
S_{\text{brand}} = \begin{cases} 
1.0 & \text{if partner is Booking.com/Expedia} \\ 
0.9 & \text{otherwise} 
\end{cases}
```

## 8. Click Probability Model

### True Click Probability
The theoretical probability that a user would click on an offer at a given rank:

```math
\text{true\_click\_prob} = \min(0.95, \max(0.05, \text{preference\_score} \times \text{rank\_factor}))
```

Where:
```math
\text{rank\_factor} = \frac{1}{\text{rank}}
```

### Normalized Click Probability
For relative comparison among offers at the same rank:

```math
\text{normalized\_probability\_of\_click}_i = \frac{\exp(\text{probability\_of\_click}_i)}{\sum_j \exp(\text{probability\_of\_click}_j)}
```

## 9. Trust Score Calculation

The user trust score combines multiple factors:

```math
\text{trust\_score} = 0.4 \times \text{price\_score} + 0.3 \times \text{consistency\_score} + 0.3 \times \text{brand\_trust}
```

Where:

**Price Score:**
```math
\text{price\_score} = 100 \times \left(1 - \frac{\text{offer\_price} - \min(\text{prices})}{\max(\text{prices}) - \min(\text{prices})}\right)
```

**Consistency Score:**
```math
\text{consistency\_score} = 100 \times \left(1 - \frac{|\text{trivago\_price} - \text{partner\_price}|}{\text{partner\_price}}\right)
```

## 10. Expected Revenue Calculation

The expected revenue for each offer-position combination:

```math
\text{expected\_revenue}_{ij} = \text{bid\_amount}_i \times \text{conversion\_probability}_i \times \text{user\_preference}_{ij} \times \frac{1}{j + 1}
```

Where $\frac{1}{j + 1}$ represents the rank decay factor.

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
The value reflects both the offer's inherent attractiveness (preference_score) and the effect of its position (rank) in the list.

# true click prob

This is the theoretical or base probability that a user would click on an offer at a given rank, before any simulation noise is added.
It is calculated as:

```math
\text{true\_click\_prob} = \min(0.95, \max(0.05, \text{preference\_score} \times \text{rank\_factor}))
```

where

```math
\text{rank\_factor} = \frac{1}{\text{rank}}
```

This value is used as the "ground truth" probability for simulating clicks in the bandit simulation.

# probability_of_conversion

This is the estimated probability that a click on an offer will convert to a booking. It is calculated as a product of interpretable factors:

```math
\text{probability\_of\_conversion} = S_{\text{price}} \times S_{\text{rating}} \times S_{\text{amenities}} \times S_{\text{loyalty}} \times S_{\text{brand}}
```
Where each $S$ factor is defined as:

```math
S_{\text{price}} = \max\left(0.1, 1 - \frac{|\text{trivago\_price} - \text{partner\_price}|}{\text{user\_budget\_max} + 1}\right)
```
```math
S_{\text{rating}} = 0.5 + 0.1 \times (\text{hotel\_rating} - 3)
```
```math
S_{\text{amenities}} = 0.5 + 0.1 \times \min(\text{amenities\_match}, 5)
```
```math
S_{\text{loyalty}} = \begin{cases} 1.0 & \text{if user is Gold/Platinum} \\ 0.8 & \text{if Silver} \\ 0.6 & \text{otherwise} \end{cases}
```
```math
S_{\text{brand}} = \begin{cases} 1.0 & \text{if partner is Booking.com/Expedia} \\ 0.9 & \text{otherwise} \end{cases}
```

The result is clipped to $[0.01, 0.99]$.

# normalized_probability_of_click

This is an optional, research-oriented column. It is computed using softmax normalization (per user, per rank) on the simulated probability_of_click values:

```math
\text{normalized\_probability\_of\_click}_i = \frac{\exp(\text{probability\_of\_click}_i)}{\sum_j \exp(\text{probability\_of\_click}_j)}
```

This gives a relative likelihood of click among competing offers at the same rank for a user, but is not required for practical modeling or optimization. In real-world ranking, probability_of_click and true_click_prob are independent signals and do not need to sum to 1.


Summary Table:
Column                Meaning
preference_score      How well the offer matches the user's preferences (composite score, 0–1).
probability_of_click  Simulated/learned probability of click for (user, offer, rank) after 1000 trials.
true_click_prob       Theoretical probability of click for (user, offer, rank) used as the "ground truth".
normalized_probability_of_click  (Optional) Softmax-normalized probability for relative comparison among offers at the same rank for a user.

## Where to Find Key Metrics in the UI

- **Dynamic Price Sensitivity**: Displayed in the User/Market table and Offers table. Updated automatically after you click 'Consider Scenario'.
- **Market Demand Index**: Displayed in the User/Market table and as a plot in the Market Analysis tab. Updated after scenario consideration.
- **Fallback Logic**: If backend data is missing, the UI will show 'NA' or hide the metric until new data is available.

See tooltips in the UI for explanations of each metric. For formulas, see the Mathematical Approach & Core Formulas section above.
