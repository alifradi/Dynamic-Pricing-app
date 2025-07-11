# Data Generator use

python data/generate_enhanced_datasets.py --hotels 500 --offers 1000 --users 200


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

# true clock prob 

This is the theoretical or base probability that a user would click on an offer at a given rank, before any simulation noise is added.
It is calculated as:
true_click_prob = min(0.95, max(0.05, preference_score * rank_factor))
where rank_factor = 1.0 / rank (so higher ranks get a higher factor).
This value is used as the “ground truth” probability for simulating clicks in the bandit simulation.

# normalized_probability_of_click

  normalized_probability_of_click = probability_of_click / sum(probability_of_click for all offers at this rank for this user)


Summary Table:
Column	            Meaning
preference_score	    How well the offer matches the user’s preferences (composite   score, 0–1).
probability_of_click	Simulated/learned probability of click for (user, offer, rank) after 1000 trials.
true_click_prob	        Theoretical probability of click for (user, offer, rank) used as the “ground truth”.
