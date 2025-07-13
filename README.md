# Data Generator use

python data/generate_enhanced_datasets.py --hotels 500 --offers 1000 --users 200

# Mathematical Approach & Formula Documentation

This project uses interpretable, research-driven formulas to simulate user behavior, offer ranking, and market dynamics. All formulas are provided in LaTeX for clarity and reproducibility.

## Preference Score

The **preference score** quantifies how well an offer matches a user's preferences. It is a weighted sum of several factors:

- **Budget Preference**: The closer the offer’s price is to the user’s preferred budget range, the higher the score (weighted 30%).
- **Star Rating**: Higher hotel star ratings increase the score (weighted 20%).
- **Location Match**: If the offer’s location matches the user’s preferred location, the score increases (weighted 30%).
- **Room Type Match**: If the offer’s room type matches the user’s preferred room type, the score increases (weighted 10%).
- **Amenities Match**: The more the hotel’s amenities overlap with the user’s preferred amenities, the higher the score (weighted 10%).

The formula:

$$
\text{preference\_score} = 0.3 \cdot S_{\text{budget}} + 0.2 \cdot S_{\text{star}} + 0.3 \cdot S_{\text{location}} + 0.1 \cdot S_{\text{room}} + 0.1 \cdot S_{\text{amenities}}
$$

Where each $S$ is a normalized score (0–1) for the respective factor.

## Probability of Click

This is the simulated (learned) probability that a user will click on a given offer when it is shown at a specific rank. It is estimated by simulating many (e.g., 1000) user interactions (clicks/no clicks) for each (user, offer, rank) combination, using a recursive average formula.

The value reflects both the offer’s inherent attractiveness (preference_score) and the effect of its position (rank) in the list.

### Theoretical (True) Click Probability

$$
\text{true\_click\_prob} = \min(0.95, \max(0.05, \text{preference\_score} \times \text{rank\_factor}))
$$

Where:

$$
\text{rank\_factor} = \frac{1}{\text{rank}}
$$

### Simulated (Learned) Probability

For each (user, offer, rank), simulate $N$ trials (e.g., $N=1000$):

- For each trial $i$, generate a click (1) with probability $\text{true\_click\_prob}$, else no click (0).
- Update the running average using the recursive formula:

$$
R_{i} = R_{i-1} + \frac{(\text{click}_i - R_{i-1})}{i}
$$

Where $R_i$ is the average after $i$ trials.

## Probability of Conversion

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

## Normalized Probability of Click (Optional)

This is a research-oriented column. It is computed using softmax normalization (per user, per rank) on the simulated probability_of_click values:

$$
\text{normalized\_probability\_of\_click}_i = \frac{\exp(\text{probability\_of\_click}_i)}{\sum_j \exp(\text{probability\_of\_click}_j)}
$$

This gives a relative likelihood of click among competing offers at the same rank, but is not required for practical modeling or optimization. In real-world ranking, probability_of_click and true_click_prob are independent signals and do not need to sum to 1.

---

# Preference Score 

Budget Preference: The closer the offer’s price is to the user’s preferred budget range, the higher the score (weighted 30%).
Star Rating: Higher hotel star ratings increase the score (weighted 20%).
Location Match: If the offer’s location matches the user’s preferred location, the score increases (weighted 30%).
Room Type Match: If the offer’s room type matches the user’s preferred room type, the score increases (weighted 10%).
Amenities Match: The more the hotel’s amenities overlap with the user’s preferred amenities, the higher the score (weighted 10%).

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
