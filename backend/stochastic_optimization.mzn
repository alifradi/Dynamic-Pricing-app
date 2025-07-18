% Stochastic Hotel Ranking Optimization Model
% MiniZinc model for optimizing hotel offer rankings with multiple objectives

% Parameters
int: n_offers;           % Number of offers to rank
int: n_users;            % Number of users
int: max_rank;           % Maximum rank (usually n_offers)

% Sets
set of int: OFFERS = 1..n_offers;
set of int: USERS = 1..n_users;
set of int: RANKS = 1..max_rank;

% Data arrays
array[OFFERS] of float: conversion_probabilities;  % P(conversion | offer)
array[OFFERS] of float: revenue_per_offer;         % Revenue if converted
array[OFFERS] of float: trust_scores;              % User trust scores
array[OFFERS] of float: price_consistency;         % Price consistency scores
array[USERS, OFFERS] of float: user_preferences;   % User-specific preferences

% Objective weights
float: weight_conversion;    % Weight for conversion probability
float: weight_revenue;       % Weight for revenue
float: weight_trust;         % Weight for trust score

% Decision variables
array[RANKS] of var OFFERS: ranking;  % Which offer gets which rank
array[OFFERS] of var RANKS: offer_rank;  % What rank each offer gets

% Constraints
constraint alldifferent(ranking);  % Each offer can only appear once
constraint alldifferent(offer_rank);  % Each rank can only be assigned once

% Inverse relationship between ranking and offer_rank
constraint forall(r in RANKS, o in OFFERS) (
    (ranking[r] = o) <-> (offer_rank[o] = r)
);

% Objective function: Multi-criteria optimization
var float: total_score;

% Calculate expected conversion value
var float: expected_conversion_value = 
    sum(r in RANKS, o in OFFERS) (
        conversion_probabilities[o] * 
        (1.0 / float(r)) *  % Rank decay factor
        (ranking[r] = o)
    );

% Calculate expected revenue
var float: expected_revenue = 
    sum(r in RANKS, o in OFFERS) (
        revenue_per_offer[o] * 
        conversion_probabilities[o] * 
        (1.0 / float(r)) *  % Rank decay factor
        (ranking[r] = o)
    );

% Calculate average trust score
var float: avg_trust_score = 
    sum(r in RANKS, o in OFFERS) (
        trust_scores[o] * 
        (1.0 / float(r)) *  % Rank decay factor
        (ranking[r] = o)
    ) / float(max_rank);

% Calculate price consistency score
var float: price_consistency_score = 
    sum(r in RANKS, o in OFFERS) (
        price_consistency[o] * 
        (1.0 / float(r)) *  % Rank decay factor
        (ranking[r] = o)
    ) / float(max_rank);

% Multi-objective function
constraint total_score = 
    weight_conversion * expected_conversion_value +
    weight_revenue * expected_revenue +
    weight_trust * (avg_trust_score + price_consistency_score);

% Solve to maximize the total score
solve maximize total_score;

% Output section for JSON/CSV export
output [
    "{\n",
    "  \"optimization_results\": {\n",
    "    \"total_score\": ", show(total_score), ",\n",
    "    \"expected_conversion_value\": ", show(expected_conversion_value), ",\n",
    "    \"expected_revenue\": ", show(expected_revenue), ",\n",
    "    \"avg_trust_score\": ", show(avg_trust_score), ",\n",
    "    \"price_consistency_score\": ", show(price_consistency_score), ",\n",
    "    \"ranking\": [", show(ranking), "],\n",
    "    \"offer_ranks\": [", show(offer_rank), "],\n",
    "    \"parameters\": {\n",
    "      \"n_offers\": ", show(n_offers), ",\n",
    "      \"n_users\": ", show(n_users), ",\n",
    "      \"max_rank\": ", show(max_rank), ",\n",
    "      \"weight_conversion\": ", show(weight_conversion), ",\n",
    "      \"weight_revenue\": ", show(weight_revenue), ",\n",
    "      \"weight_trust\": ", show(weight_trust), "\n",
    "    }\n",
    "  }\n",
    "}"
];

% Alternative CSV output format
output [
    "rank,offer_id,conversion_prob,revenue,trust_score,price_consistency,expected_value\n"
] ++ [
    show(r), ",", show(ranking[r]), ",", 
    show(conversion_probabilities[ranking[r]]), ",", 
    show(revenue_per_offer[ranking[r]]), ",", 
    show(trust_scores[ranking[r]]), ",", 
    show(price_consistency[ranking[r]]), ",", 
    show(conversion_probabilities[ranking[r]] * revenue_per_offer[ranking[r]] / float(r)), "\n"
    | r in RANKS
]; 