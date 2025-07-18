% Deterministic Hotel Offer Ranking Optimization
% Linear programming model for maximizing expected income

% Parameters
int: n_offers;           % Number of offers
int: n_users;            % Number of users
int: max_rank;           % Maximum rank (usually n_offers)

% Sets
set of int: OFFERS = 1..n_offers;
set of int: USERS = 1..n_users;
set of int: RANKS = 1..max_rank;

% Data arrays
array[OFFERS] of float: conversion_probabilities;  % P(click | offer)
array[OFFERS] of float: bid_amounts;               % Bid amount per offer
array[USERS, OFFERS] of float: user_preferences;   % User-specific preferences

% Decision variables
array[USERS, RANKS] of var OFFERS: user_rankings;  % Which offer gets which rank for each user

% Constraints
constraint forall(u in USERS) (
    alldifferent([user_rankings[u, r] | r in RANKS])  % Each offer can only appear once per user
);

% Objective function: Maximize expected income
var float: total_expected_income;

% Calculate expected income for each user
var float: expected_income = 
    sum(u in USERS, r in RANKS, o in OFFERS) (
        bid_amounts[o] * 
        conversion_probabilities[o] * 
        user_preferences[u, o] * 
        (1.0 / float(r)) *  % Rank decay factor
        (user_rankings[u, r] = o)
    );

constraint total_expected_income = expected_income;

% Solve to maximize expected income
solve maximize total_expected_income;

% Output section for JSON export
output [
    "{\n",
    "  \"optimization_results\": {\n",
    "    \"total_expected_income\": ", show(total_expected_income), ",\n",
    "    \"user_rankings\": [\n"
] ++ [
    "      [", show([user_rankings[u, r] | r in RANKS]), "]", if u < n_users then "," else "" endif, "\n"
    | u in USERS
] ++ [
    "    ],\n",
    "    \"parameters\": {\n",
    "      \"n_offers\": ", show(n_offers), ",\n",
    "      \"n_users\": ", show(n_users), ",\n",
    "      \"max_rank\": ", show(max_rank), "\n",
    "    }\n",
    "  }\n",
    "}"
];

% Alternative CSV output format
output [
    "user_id,rank,offer_id,conversion_prob,bid_amount,expected_value\n"
] ++ [
    show(u), ",", show(r), ",", show(user_rankings[u, r]), ",", 
    show(conversion_probabilities[user_rankings[u, r]]), ",", 
    show(bid_amounts[user_rankings[u, r]]), ",", 
    show(bid_amounts[user_rankings[u, r]] * conversion_probabilities[user_rankings[u, r]] / float(r)), "\n"
    | u in USERS, r in RANKS
]; 