import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import uuid
import json
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pulp

# Import our data models
from data_models.models import (
    UserProfile, Hotel, PartnerOffer, MarketConditions, CustomerBehavior,
    StrategyConfig, RankedOffer, PerformanceMetrics,
    ScenarioRequest, ScenarioResponse, RankingRequest, RankingResponse,
    MABSimulationRequest, MABSimulationResponse,
    UserType, MarketDemand, RoomType
)
from data_models.data_generator import DataGenerator

from collections import defaultdict

app = FastAPI(
    title="Enhanced Hotel Ranking Simulation API",
    description="Comprehensive API for trivago-style hotel ranking simulation with realistic data models",
    version="3.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data generator
data_generator = DataGenerator()

# Global storage for scenarios (in production, use a database)
scenarios_cache = {}

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Hotel Ranking Simulation API",
        "version": "3.0.0",
        "features": [
            "Realistic hotel and partner data generation",
            "Multi-objective optimization",
            "Customer behavior modeling",
            "Market condition analysis",
            "Advanced ranking strategies"
        ],
        "endpoints": [
            "/generate_scenario",
            "/rank_offers",
            "/mab_simulation",
            "/user_profiles",
            "/market_analysis"
        ]
    }

@app.get("/user_profiles")
def get_sample_user_profiles():
    """Get 10 random user profiles for selection from enhanced_user_profiles.csv"""
    csv_path = os.path.join('data', 'enhanced_user_profiles.csv')
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=10)
    # Replace NaN, inf, -inf with None for JSON compatibility
    sample_df = sample_df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(sample_df), None)
    profiles = sample_df.to_dict(orient="records")
    return profiles

@app.get("/user_profile/{user_id}")
def get_user_profile(user_id: str):
    """Get a user profile by user_id from enhanced_user_profiles.csv"""
    try:
        file_path = os.path.join('data', 'enhanced_user_profiles.csv')
        if not os.path.exists(file_path):
            return {"profile": None}
        df = pd.read_csv(file_path)
        user_row = df[df['user_id'] == user_id]
        if user_row.empty:
            return {"profile": None}
        profile = user_row.iloc[0].to_dict()
        
        # Clean the profile data to handle infinite and NaN values
        cleaned_profile = {}
        for key, value in profile.items():
            if pd.isna(value) or (isinstance(value, float) and (np.isinf(value) or np.isnan(value))):
                cleaned_profile[key] = None
            else:
                cleaned_profile[key] = value
        
        return {"profile": cleaned_profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading user profile: {str(e)}")

@app.get("/user_offers/{user_id}")
def get_user_offers(user_id: str, num_hotels: int = 10, num_partners: int = 5, days_to_go: int = 30, days_var: int = 5):
    """Get offers for a selected user, with hotel and offer info joined, and days_to_go column."""
    try:
        # Load user
        users = pd.read_csv(os.path.join('data', 'enhanced_user_profiles.csv'))
        user = users[users['user_id'] == user_id]
        if user.empty:
            return []
        user = user.iloc[0]
        
        # Load hotels and offers
        hotels = pd.read_csv(os.path.join('data', 'enhanced_hotels.csv'))
        offers = pd.read_csv(os.path.join('data', 'enhanced_partner_offers.csv'))
        
        # Filter hotels based on user preferences
        filtered_hotels = hotels.copy()
        
        # Filter by user location (exact match)
        if 'location' in user and pd.notna(user['location']):
            location_filtered = hotels[hotels['location'] == user['location']]
            if len(location_filtered) > 0:
                filtered_hotels = location_filtered
        
        # Filter by budget range
        budget_min = user['budget_min']
        budget_max = user['budget_max']
        
        # Get average prices for hotels from offers
        avg_prices = offers.groupby('hotel_id')['price_per_night'].mean().reset_index()
        avg_prices.columns = ['hotel_id', 'avg_price']
        
        # Filter hotels within budget
        hotels_with_prices = filtered_hotels.merge(avg_prices, on='hotel_id', how='left')
        budget_hotels = hotels_with_prices[
            (hotels_with_prices['avg_price'] >= budget_min) & 
            (hotels_with_prices['avg_price'] <= budget_max)
        ]
        
        if len(budget_hotels) > 0:
            filtered_hotels = budget_hotels
        
        # Sample hotels (prefer budget-friendly ones)
        if len(filtered_hotels) > num_hotels:
            # Weight sampling by price (prefer cheaper hotels)
            weights = 1 / (filtered_hotels['avg_price'] + 1)  # Add 1 to avoid division by zero
            sampled_hotels = filtered_hotels.sample(n=num_hotels, weights=weights)
        else:
            sampled_hotels = filtered_hotels
        
        # Get offers for sampled hotels
        hotel_offers = offers[offers['hotel_id'].isin(sampled_hotels['hotel_id'])]
        
        # Sample partners per hotel
        sampled_offers = []
        for hotel_id in sampled_hotels['hotel_id']:
            hotel_offers_subset = hotel_offers[hotel_offers['hotel_id'] == hotel_id]
            if len(hotel_offers_subset) > 0:
                # Sample partners for this hotel
                num_partners_for_hotel = min(num_partners, len(hotel_offers_subset))
                sampled_hotel_offers = hotel_offers_subset.sample(n=num_partners_for_hotel)
                sampled_offers.append(sampled_hotel_offers)
        
        if sampled_offers:
            offers_df = pd.concat(sampled_offers, ignore_index=True)
        else:
            return []
        
        # Join hotel info
        offers_df = offers_df.merge(sampled_hotels, on='hotel_id', suffixes=('_offer', '_hotel'))
        
        # Add user_id
        offers_df['user_id'] = user_id
        
        # Add days_to_go column with normal distribution
        offers_df['days_to_go'] = np.random.normal(days_to_go, days_var, size=len(offers_df)).astype(int)
        offers_df['days_to_go'] = offers_df['days_to_go'].clip(lower=1, upper=365)  # Ensure reasonable range
        
        # Add user preference matching score
        offers_df['preference_score'] = offers_df.apply(
            lambda row: _calculate_preference_score(row, user), axis=1
        )
        
        # Handle NaN and infinite values
        offers_df['preference_score'] = offers_df['preference_score'].fillna(0.0)
        offers_df['preference_score'] = offers_df['preference_score'].replace([np.inf, -np.inf], 0.0)
        
        # Sort by preference score
        offers_df = offers_df.sort_values('preference_score', ascending=False)
        
        # Save as trial CSV
        offers_df.to_csv(os.path.join('data', 'trial_sampled_offers.csv'), index=False)
        
        # Convert to dict and handle any remaining NaN/inf values
        result_dict = offers_df.to_dict(orient='records')
        
        # Clean any remaining problematic values
        for record in result_dict:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = 0.0
                elif value is None:
                    record[key] = ""
        
        return result_dict
        
    except Exception as e:
        print(f"Error in get_user_offers: {str(e)}")
        return []

def _calculate_preference_score(offer_row, user):
    """Calculate how well an offer matches user preferences"""
    score = 0.0
    
    # Budget preference (closer to user's preferred range is better)
    user_budget_mid = (user['budget_min'] + user['budget_max']) / 2
    price_diff = abs(offer_row['price_per_night'] - user_budget_mid)
    budget_score = max(0, 1 - (price_diff / user_budget_mid))
    score += budget_score * 0.3
    
    # Star rating preference (assume users prefer higher ratings)
    if 'star_rating' in offer_row:
        star_score = min(1.0, offer_row['star_rating'] / 5.0)
        score += star_score * 0.2
    
    # Location preference (if available)
    if 'location' in offer_row and 'location' in user:
        if offer_row['location'] == user['location']:
            score += 0.3
    
    # Room type preference (if available)
    if 'room_type' in offer_row and 'room_type_preference' in user:
        if offer_row['room_type'] == user['room_type_preference']:
            score += 0.1
    
    # Amenities preference (check if hotel has user's preferred amenities)
    if 'amenities' in offer_row and 'preferred_amenities' in user:
        hotel_amenities = str(offer_row['amenities']).lower()
        user_amenities = str(user['preferred_amenities']).lower()
        
        # Split amenities and check for matches
        hotel_amenity_list = [amenity.strip() for amenity in hotel_amenities.split(',')]
        user_amenity_list = [amenity.strip() for amenity in user_amenities.split(',')]
        
        # Count matching amenities
        matching_amenities = sum(1 for user_amenity in user_amenity_list 
                               for hotel_amenity in hotel_amenity_list 
                               if user_amenity in hotel_amenity or hotel_amenity in user_amenity)
        
        if len(user_amenity_list) > 0:
            amenities_score = min(1.0, matching_amenities / len(user_amenity_list))
            score += amenities_score * 0.1
    
    return score

@app.post("/generate_scenario", response_model=ScenarioResponse)
def generate_scenario(request: ScenarioRequest):
    """Generate a comprehensive hotel booking scenario"""
    try:
        # Use provided user profile or generate one
        user_profile = request.user_profile
        
        # Generate hotels for the destination
        hotels = []
        for i in range(request.num_hotels):
            hotel = data_generator.generate_hotel(
                destination=request.destination,
                hotel_id=f"hotel_{i+1}"
            )
            hotels.append(hotel)
        
        # Generate partner offers for each hotel
        partner_offers = []
        partners_to_use = data_generator.partners[:request.num_partners]
        
        for hotel in hotels:
            # Base price for this hotel
            base_price = random.uniform(80, 400) * (hotel.star_rating / 3.0)
            
            for partner in partners_to_use:
                offer = data_generator.generate_partner_offer(
                    hotel=hotel,
                    partner_name=partner,
                    base_price=base_price
                )
                partner_offers.append(offer)
        
        # Generate market conditions
        market_conditions = data_generator.generate_market_conditions(
            destination=request.destination,
            travel_date=user_profile.travel_date_start
        )
        
        # Create scenario response
        scenario = ScenarioResponse(
            user_profile=user_profile,
            hotels=hotels,
            partner_offers=partner_offers,
            market_conditions=market_conditions
        )
        
        # Cache the scenario
        scenario_id = str(uuid.uuid4())
        scenarios_cache[scenario_id] = scenario
        
        return scenario
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating scenario: {str(e)}")

@app.post("/rank_offers", response_model=RankingResponse)
def rank_offers(request: RankingRequest):
    """Rank hotel offers using specified strategy and optimization"""
    try:
        scenario = request.scenario
        strategy_config = request.strategy_config
        customer_behavior = request.customer_behavior
        
        # Filter offers based on user budget and preferences
        filtered_offers = []
        user_budget_min, user_budget_max = scenario.user_profile.budget_range
        
        for offer in scenario.partner_offers:
            if user_budget_min <= offer.price_per_night <= user_budget_max:
                filtered_offers.append(offer)
        
        if not filtered_offers:
            # If no offers in budget, take cheapest options
            filtered_offers = sorted(scenario.partner_offers, key=lambda x: x.price_per_night)[:10]
        
        # Apply ranking strategy
        ranked_offers = []
        
        if strategy_config.strategy_name == "Greedy (Highest Commission)":
            ranked_offers = _rank_by_commission(filtered_offers, customer_behavior)
            
        elif strategy_config.strategy_name == "User-First (Lowest Price)":
            ranked_offers = _rank_by_price(filtered_offers, customer_behavior)
            
        elif strategy_config.strategy_name == "Stochastic LP":
            ranked_offers = _rank_by_optimization(filtered_offers, customer_behavior, strategy_config)
            
        elif strategy_config.strategy_name == "RL Optimized Policy":
            ranked_offers = _rank_by_rl_policy(filtered_offers, customer_behavior, scenario.market_conditions)
            
        else:
            raise HTTPException(status_code=400, detail="Unknown ranking strategy")
        
        # Calculate performance metrics
        performance_metrics = _calculate_performance_metrics(ranked_offers, strategy_config.strategy_name)
        
        # Optimization details
        optimization_details = {
            "strategy": strategy_config.strategy_name,
            "optimization_method": strategy_config.optimization_method,
            "offers_considered": len(filtered_offers),
            "offers_ranked": len(ranked_offers),
            "market_demand": scenario.market_conditions.market_demand,
            "user_type": scenario.user_profile.user_type
        }
        
        return RankingResponse(
            ranked_offers=ranked_offers,
            performance_metrics=performance_metrics,
            optimization_details=optimization_details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ranking offers: {str(e)}")

def _rank_by_commission(offers: List[PartnerOffer], customer_behavior: CustomerBehavior) -> List[RankedOffer]:
    """Rank offers by commission revenue"""
    ranked = []
    
    # Sort by commission revenue
    sorted_offers = sorted(offers, 
                          key=lambda x: x.price_per_night * x.commission_rate, 
                          reverse=True)
    
    for rank, offer in enumerate(sorted_offers[:10], 1):
        conversion_prob = _calculate_conversion_probability(offer, customer_behavior)
        expected_revenue = offer.price_per_night * offer.commission_rate * conversion_prob
        user_trust = _calculate_user_trust_score(offer, sorted_offers)
        price_consistency = _calculate_price_consistency(offer)
        bid_competitiveness = _calculate_bid_competitiveness(offer, sorted_offers)
        
        ranked_offer = RankedOffer(
            rank=rank,
            offer=offer,
            score=expected_revenue,
            conversion_probability=conversion_prob,
            expected_revenue=expected_revenue,
            user_trust_score=user_trust,
            price_consistency_score=price_consistency,
            bid_competitiveness=bid_competitiveness,
            explanation=f"Ranked #{rank} by commission revenue (${expected_revenue:.2f})"
        )
        ranked.append(ranked_offer)
    
    return ranked

def _rank_by_price(offers: List[PartnerOffer], customer_behavior: CustomerBehavior) -> List[RankedOffer]:
    """Rank offers by lowest price (user-first)"""
    ranked = []
    
    # Sort by price
    sorted_offers = sorted(offers, key=lambda x: x.price_per_night)
    
    for rank, offer in enumerate(sorted_offers[:10], 1):
        conversion_prob = _calculate_conversion_probability(offer, customer_behavior)
        expected_revenue = offer.price_per_night * offer.commission_rate * conversion_prob
        user_trust = _calculate_user_trust_score(offer, sorted_offers)
        price_consistency = _calculate_price_consistency(offer)
        bid_competitiveness = _calculate_bid_competitiveness(offer, sorted_offers)
        
        ranked_offer = RankedOffer(
            rank=rank,
            offer=offer,
            score=100 - rank,  # Higher score for lower rank (better price)
            conversion_probability=conversion_prob,
            expected_revenue=expected_revenue,
            user_trust_score=user_trust,
            price_consistency_score=price_consistency,
            bid_competitiveness=bid_competitiveness,
            explanation=f"Ranked #{rank} by lowest price (${offer.price_per_night:.2f})"
        )
        ranked.append(ranked_offer)
    
    return ranked

def _rank_by_optimization(offers: List[PartnerOffer], customer_behavior: CustomerBehavior, 
                         strategy_config: StrategyConfig) -> List[RankedOffer]:
    """Rank offers using linear programming optimization"""
    try:
        # Create optimization problem
        prob = pulp.LpProblem("HotelRanking", pulp.LpMaximize)
        
        n_offers = len(offers)
        n_positions = min(10, n_offers)
        
        # Decision variables: x[i][j] = 1 if offer i is at position j
        x = {}
        for i in range(n_offers):
            for j in range(n_positions):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
        
        # Objective weights (can be configured)
        w_conversion = strategy_config.objective_weights.get("conversion", 0.4)
        w_revenue = strategy_config.objective_weights.get("revenue", 0.4)
        w_trust = strategy_config.objective_weights.get("trust", 0.2)
        
        # Calculate metrics for each offer
        conversion_probs = [_calculate_conversion_probability(offer, customer_behavior) for offer in offers]
        revenues = [offer.price_per_night * offer.commission_rate for offer in offers]
        trust_scores = [_calculate_user_trust_score(offer, offers) for offer in offers]
        
        # Position weights (higher positions get more weight)
        position_weights = [1.0 / (j + 1) for j in range(n_positions)]
        
        # Objective function
        objective = 0
        for i in range(n_offers):
            for j in range(n_positions):
                score = (w_conversion * conversion_probs[i] + 
                        w_revenue * revenues[i] / max(revenues) +
                        w_trust * trust_scores[i] / 100.0)
                objective += score * position_weights[j] * x[i, j]
        
        prob += objective
        
        # Constraints
        # Each position can have at most one offer
        for j in range(n_positions):
            prob += pulp.lpSum([x[i, j] for i in range(n_offers)]) <= 1
        
        # Each offer can be in at most one position
        for i in range(n_offers):
            prob += pulp.lpSum([x[i, j] for j in range(n_positions)]) <= 1
        
        # Price constraint: top offer shouldn't be too expensive
        min_price = min(offer.price_per_night for offer in offers)
        max_allowed_price = min_price * 1.5
        
        for i in range(n_offers):
            if offers[i].price_per_night > max_allowed_price:
                prob += x[i, 0] == 0  # Don't put expensive offers at top
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        ranked = []
        solution = {}
        
        for i in range(n_offers):
            for j in range(n_positions):
                if x[i, j].varValue == 1:
                    solution[j] = i
        
        # Create ranked offers
        for position in sorted(solution.keys()):
            offer_idx = solution[position]
            offer = offers[offer_idx]
            rank = position + 1
            
            conversion_prob = conversion_probs[offer_idx]
            expected_revenue = revenues[offer_idx] * conversion_prob
            user_trust = trust_scores[offer_idx]
            price_consistency = _calculate_price_consistency(offer)
            bid_competitiveness = _calculate_bid_competitiveness(offer, offers)
            
            ranked_offer = RankedOffer(
                rank=rank,
                offer=offer,
                score=expected_revenue,
                conversion_probability=conversion_prob,
                expected_revenue=expected_revenue,
                user_trust_score=user_trust,
                price_consistency_score=price_consistency,
                bid_competitiveness=bid_competitiveness,
                explanation=f"Optimized ranking #{rank} (LP solution)"
            )
            ranked.append(ranked_offer)
        
        return ranked
        
    except Exception as e:
        # Fallback to price ranking if optimization fails
        return _rank_by_price(offers, customer_behavior)

def _rank_by_rl_policy(offers: List[PartnerOffer], customer_behavior: CustomerBehavior,
                      market_conditions: MarketConditions) -> List[RankedOffer]:
    """Rank offers using RL-based policy"""
    # Simple RL policy: adapt strategy based on market conditions
    if market_conditions.market_demand == MarketDemand.HIGH:
        # In high demand, prioritize revenue
        return _rank_by_commission(offers, customer_behavior)
    elif market_conditions.days_until_travel < 7:
        # Last minute bookings, balance price and availability
        strategy_config = StrategyConfig(
            strategy_name="RL Optimized Policy",
            optimization_method="pulp",
            objective_weights={"conversion": 0.5, "revenue": 0.3, "trust": 0.2}
        )
        return _rank_by_optimization(offers, customer_behavior, strategy_config)
    else:
        # Normal conditions, prioritize user satisfaction
        return _rank_by_price(offers, customer_behavior)

def _calculate_conversion_probability(offer: PartnerOffer, customer_behavior: CustomerBehavior) -> float:
    """Calculate probability of conversion for an offer"""
    base_prob = customer_behavior.conversion_likelihood
    
    # Adjust for brand preference
    brand_factor = customer_behavior.brand_preference.get(offer.partner_name, 0.5)
    
    # Adjust for price sensitivity
    price_factor = 1.0 - (customer_behavior.price_sensitivity_calculated * 0.3)
    
    # Adjust for special offers
    special_factor = 1.0 + (len(offer.special_offers) * 0.05)
    
    # Adjust for cancellation policy
    cancellation_factor = 1.0
    if offer.cancellation_policy == "Free cancellation":
        cancellation_factor = 1.1
    elif offer.cancellation_policy == "Non-refundable":
        cancellation_factor = 0.9
    
    prob = base_prob * brand_factor * price_factor * special_factor * cancellation_factor
    return min(1.0, max(0.0, prob))

def _calculate_user_trust_score(offer: PartnerOffer, all_offers: List[PartnerOffer]) -> float:
    """Calculate user trust score based on price competitiveness and transparency"""
    min_price = min(o.price_per_night for o in all_offers)
    max_price = max(o.price_per_night for o in all_offers)
    
    # Price competitiveness (lower price = higher trust)
    if max_price > min_price:
        price_score = 100 * (1 - (offer.price_per_night - min_price) / (max_price - min_price))
    else:
        price_score = 100
    
    # Price consistency (trivago price vs partner price)
    consistency_score = 100 * (1 - abs(offer.trivago_displayed_price - offer.price_per_night) / offer.price_per_night)
    
    # Brand trust
    brand_trust = {
        "Booking.com": 90, "Expedia": 85, "Hotels.com": 80,
        "Agoda": 75, "HotelDirect": 70
    }.get(offer.partner_name, 65)
    
    # Weighted average
    trust_score = (price_score * 0.4 + consistency_score * 0.3 + brand_trust * 0.3)
    return min(100.0, max(0.0, trust_score))

def _calculate_price_consistency(offer: PartnerOffer) -> float:
    """Calculate price consistency score"""
    if offer.price_per_night == 0:
        return 0.0
    
    consistency = 1.0 - abs(offer.trivago_displayed_price - offer.price_per_night) / offer.price_per_night
    return max(0.0, min(1.0, consistency))

def _calculate_bid_competitiveness(offer: PartnerOffer, all_offers: List[PartnerOffer]) -> float:
    """Calculate bid competitiveness score"""
    all_bids = [o.cost_per_click_bid for o in all_offers]
    if not all_bids or max(all_bids) == min(all_bids):
        return 0.5
    
    # Normalize bid to 0-1 scale
    min_bid = min(all_bids)
    max_bid = max(all_bids)
    normalized_bid = (offer.cost_per_click_bid - min_bid) / (max_bid - min_bid)
    
    return normalized_bid

def _calculate_performance_metrics(ranked_offers: List[RankedOffer], strategy_name: str) -> PerformanceMetrics:
    """Calculate overall performance metrics for the ranking"""
    if not ranked_offers:
        return PerformanceMetrics(
            strategy_name=strategy_name,
            total_expected_revenue=0.0,
            average_user_trust=0.0,
            conversion_rate=0.0,
            click_through_rate=0.0,
            price_consistency=0.0,
            partner_satisfaction=0.0,
            user_satisfaction=0.0,
            profit_margin=0.0
        )
    
    total_revenue = sum(offer.expected_revenue for offer in ranked_offers)
    avg_trust = sum(offer.user_trust_score for offer in ranked_offers) / len(ranked_offers)
    avg_conversion = sum(offer.conversion_probability for offer in ranked_offers) / len(ranked_offers)
    avg_consistency = sum(offer.price_consistency_score for offer in ranked_offers) / len(ranked_offers)
    avg_bid_comp = sum(offer.bid_competitiveness for offer in ranked_offers) / len(ranked_offers)
    
    # Simulate click-through rate (higher for top positions)
    ctr = sum((1.0 / offer.rank) * 0.1 for offer in ranked_offers) / len(ranked_offers)
    
    return PerformanceMetrics(
        strategy_name=strategy_name,
        total_expected_revenue=total_revenue,
        average_user_trust=avg_trust,
        conversion_rate=avg_conversion,
        click_through_rate=ctr,
        price_consistency=avg_consistency,
        partner_satisfaction=avg_bid_comp,
        user_satisfaction=avg_trust / 100.0,
        profit_margin=total_revenue / sum(offer.offer.price_per_night for offer in ranked_offers) if ranked_offers else 0.0
    )

@app.post("/mab_simulation", response_model=MABSimulationResponse)
def mab_simulation(request: MABSimulationRequest):
    """Run Multi-Armed Bandit simulation comparing strategies"""
    try:
        strategies = request.strategies
        n_iterations = request.num_iterations
        scenario = request.scenario
        
        # Initialize bandit arms (strategies)
        n_arms = len(strategies)
        arm_counts = [0] * n_arms
        arm_rewards = [0.0] * n_arms
        cumulative_rewards = []
        total_reward = 0.0
        
        # Strategy performance tracking
        strategy_performance = {}
        for strategy in strategies:
            strategy_performance[strategy] = {
                "total_reward": 0.0,
                "selections": 0,
                "average_reward": 0.0
            }
        
        # Epsilon-greedy parameters
        epsilon = 0.1
        
        # Generate customer behavior for simulation
        customer_behavior = data_generator.generate_customer_behavior(
            scenario.user_profile, scenario.market_conditions
        )
        
        convergence_data = []
        
        for iteration in range(n_iterations):
            # Epsilon-greedy arm selection
            if random.random() < epsilon or iteration < n_arms:
                # Explore: select random arm
                selected_arm = random.randint(0, n_arms - 1)
            else:
                # Exploit: select best arm
                avg_rewards = [arm_rewards[i] / max(1, arm_counts[i]) for i in range(n_arms)]
                selected_arm = avg_rewards.index(max(avg_rewards))
            
            selected_strategy = strategies[selected_arm]
            
            # Simulate reward for selected strategy
            reward = _simulate_strategy_reward(selected_strategy, scenario, customer_behavior)
            
            # Update arm statistics
            arm_counts[selected_arm] += 1
            arm_rewards[selected_arm] += reward
            total_reward += reward
            cumulative_rewards.append(total_reward)
            
            # Update strategy performance
            strategy_performance[selected_strategy]["total_reward"] += reward
            strategy_performance[selected_strategy]["selections"] += 1
            strategy_performance[selected_strategy]["average_reward"] = (
                strategy_performance[selected_strategy]["total_reward"] / 
                strategy_performance[selected_strategy]["selections"]
            )
            
            # Record convergence data every 10 iterations
            if iteration % 10 == 0:
                convergence_data.append({
                    "iteration": iteration,
                    "total_reward": total_reward,
                    "best_strategy": max(strategy_performance.keys(), 
                                       key=lambda x: strategy_performance[x]["average_reward"]),
                    "exploration_rate": epsilon
                })
        
        # Determine best strategy
        best_strategy = max(strategy_performance.keys(), 
                           key=lambda x: strategy_performance[x]["average_reward"])
        
        return MABSimulationResponse(
            cumulative_rewards=cumulative_rewards,
            strategy_performance=strategy_performance,
            best_strategy=best_strategy,
            convergence_data=convergence_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in MAB simulation: {str(e)}")

def _simulate_strategy_reward(strategy_name: str, scenario: ScenarioResponse, 
                            customer_behavior: CustomerBehavior) -> float:
    """Simulate reward for a strategy (simplified)"""
    base_rewards = {
        "Greedy (Highest Commission)": 1.2,
        "User-First (Lowest Price)": 0.8,
        "Stochastic LP": 1.0,
        "RL Optimized Policy": 1.1
    }
    
    base_reward = base_rewards.get(strategy_name, 0.5)
    
    # Add noise and market condition adjustments
    market_factor = 1.0
    if scenario.market_conditions.market_demand == MarketDemand.HIGH:
        if strategy_name == "Greedy (Highest Commission)":
            market_factor = 1.3
    elif scenario.market_conditions.market_demand == MarketDemand.LOW:
        if strategy_name == "User-First (Lowest Price)":
            market_factor = 1.2
    
    # Add customer behavior influence
    behavior_factor = 1.0 + (customer_behavior.conversion_likelihood - 0.5)
    
    reward = base_reward * market_factor * behavior_factor * random.uniform(0.8, 1.2)
    return max(0.0, reward)

@app.get("/market_analysis/{destination}")
def get_market_analysis(destination: str, travel_date: str = None):
    """Get market analysis for a destination"""
    try:
        if travel_date:
            travel_date_obj = datetime.strptime(travel_date, "%Y-%m-%d").date()
        else:
            travel_date_obj = date.today() + timedelta(days=30)
        
        market_conditions = data_generator.generate_market_conditions(destination, travel_date_obj)
        
        # Additional market insights
        insights = {
            "destination": destination,
            "market_conditions": market_conditions,
            "recommendations": _get_market_recommendations(market_conditions),
            "price_trends": _get_price_trends(destination, travel_date_obj),
            "competitor_analysis": _get_competitor_analysis(destination)
        }
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in market analysis: {str(e)}")

def _get_market_recommendations(market_conditions: MarketConditions) -> List[str]:
    """Get strategy recommendations based on market conditions"""
    recommendations = []
    
    if market_conditions.market_demand == MarketDemand.HIGH:
        recommendations.append("Consider revenue-focused strategies due to high demand")
        recommendations.append("Users may be less price-sensitive")
    elif market_conditions.market_demand == MarketDemand.LOW:
        recommendations.append("Focus on user-first strategies to attract bookings")
        recommendations.append("Competitive pricing is crucial")
    
    if market_conditions.days_until_travel < 7:
        recommendations.append("Last-minute booking behavior - urgency over price")
    elif market_conditions.days_until_travel > 90:
        recommendations.append("Early planners - price sensitivity is high")
    
    if market_conditions.price_volatility > 0.3:
        recommendations.append("High price volatility - monitor competitor pricing closely")
    
    return recommendations

def _get_price_trends(destination: str, travel_date: date) -> Dict[str, Any]:
    """Get simulated price trends for destination"""
    # Simulate historical price data
    days_back = 30
    price_history = []
    base_price = random.uniform(100, 300)
    
    for i in range(days_back):
        date_point = travel_date - timedelta(days=i)
        # Add seasonal and random variation
        seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * date_point.timetuple().tm_yday / 365)
        price = base_price * seasonal_factor * random.uniform(0.9, 1.1)
        price_history.append({
            "date": date_point.isoformat(),
            "average_price": round(price, 2)
        })
    
    return {
        "historical_prices": price_history,
        "trend_direction": random.choice(["increasing", "decreasing", "stable"]),
        "volatility": random.uniform(0.1, 0.4)
    }

def _get_competitor_analysis(destination: str) -> Dict[str, Any]:
    """Get competitor analysis for destination"""
    partners = data_generator.partners[:5]
    analysis = {}
    
    for partner in partners:
        analysis[partner] = {
            "market_share": random.uniform(0.1, 0.3),
            "average_commission": random.uniform(0.08, 0.20),
            "price_competitiveness": random.uniform(0.6, 0.9),
            "user_preference": random.uniform(0.5, 0.8)
        }
    
    return analysis

@app.post("/save_scenario")
def save_scenario_data(scenario_data: dict):
    """Save scenario data for all users with days_to_go column"""
    try:
        all_offers = scenario_data.get('all_offers', [])
        if not all_offers:
            return {"message": "No offers data provided"}
        df = pd.DataFrame(all_offers)
        if 'days_to_go' not in df.columns:
            df['days_to_go'] = np.random.normal(30, 5, size=len(df)).astype(int)
        filename = "trial_sampled_offers.csv"
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        return {
            "message": f"Scenario data saved successfully",
            "filename": filename,
            "records_saved": len(df),
            "filepath": filepath
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving scenario data: {str(e)}")

@app.get("/trial_sampled_offers")
def get_trial_sampled_offers():
    """Return the contents of trial_sampled_offers.csv as JSON, handling NaN/inf."""
    try:
        file_path = os.path.join('data', 'trial_sampled_offers.csv')
        if not os.path.exists(file_path):
            return {"message": "No trial_sampled_offers.csv found", "data": []}
        df = pd.read_csv(file_path)
        df = df.fillna(0.0)
        df = df.replace([np.inf, -np.inf], 0.0)
        data = df.to_dict(orient='records')
        for record in data:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = 0.0
                elif value is None:
                    record[key] = ""
        return {"data": data, "records": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading trial_sampled_offers: {str(e)}")

@app.get("/trial_user_ids")
def get_trial_user_ids():
    """Get unique user IDs from trial_sampled_offers.csv"""
    try:
        file_path = os.path.join('data', 'trial_sampled_offers.csv')
        if not os.path.exists(file_path):
            return {"user_ids": []}
        
        df = pd.read_csv(file_path)
        if 'user_id' not in df.columns:
            return {"user_ids": []}
        
        unique_user_ids = df['user_id'].unique().tolist()
        return {"user_ids": unique_user_ids}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading user IDs: {str(e)}")

@app.get("/trial_offers_by_user/{user_id}")
def get_trial_offers_by_user(user_id: str):
    """Get offers from trial_sampled_offers.csv filtered by user_id"""
    try:
        file_path = os.path.join('data', 'trial_sampled_offers.csv')
        if not os.path.exists(file_path):
            return {"data": [], "records": 0}
        
        df = pd.read_csv(file_path)
        if 'user_id' not in df.columns:
            return {"data": [], "records": 0}
        
        # Filter by user_id
        user_offers = df[df['user_id'] == user_id]
        
        # Handle NaN and infinite values
        user_offers = user_offers.fillna(0.0)
        user_offers = user_offers.replace([np.inf, -np.inf], 0.0)
        
        # Convert to list of dictionaries
        data = user_offers.to_dict(orient='records')
        
        # Clean any remaining problematic values
        for record in data:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = 0.0
                elif value is None:
                    record[key] = ""
        
        return {"data": data, "records": len(data)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error filtering offers by user: {str(e)}")

@app.post("/sample_offers_for_users")
def sample_offers_for_users(
    num_users: int = Query(1, ge=1, le=100),
    num_hotels: int = Query(10, ge=1, le=20),
    num_partners: int = Query(5, ge=1, le=10),
    days_to_go: int = Query(30, ge=1, le=365),
    days_var: int = Query(5, ge=1, le=30)
):
    """Sample offers for N unique users and save all to trial_sampled_offers.csv"""
    try:
        users_df = pd.read_csv(os.path.join('data', 'enhanced_user_profiles.csv'))
        if len(users_df) < num_users:
            num_users = len(users_df)
        sampled_users = users_df.sample(n=num_users, replace=False)
        all_offers = []
        for _, user in sampled_users.iterrows():
            user_id = user['user_id']
            offers = get_user_offers(
                user_id,
                num_hotels=num_hotels,
                num_partners=num_partners,
                days_to_go=days_to_go,
                days_var=days_var
            )
            if offers:
                all_offers.extend(offers)
        if all_offers:
            offers_df = pd.DataFrame(all_offers)
            offers_df.to_csv(os.path.join('data', 'trial_sampled_offers.csv'), index=False)
            return {"message": f"Sampled offers for {num_users} users.", "num_records": len(offers_df)}
        else:
            return {"message": "No offers generated.", "num_records": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling offers for users: {str(e)}")

@app.get("/latest_scenario")
def get_latest_scenario():
    """Get the latest saved scenario data"""
    try:
        # Try to read the latest scenario file
        latest_file = os.path.join('data', 'latest_scenario_data.csv')
        trial_file = os.path.join('data', 'trial_sampled_offers.csv')
        
        # Check which file exists and is more recent
        if os.path.exists(latest_file) and os.path.exists(trial_file):
            latest_time = os.path.getmtime(latest_file)
            trial_time = os.path.getmtime(trial_file)
            file_to_read = latest_file if latest_time > trial_time else trial_file
        elif os.path.exists(latest_file):
            file_to_read = latest_file
        elif os.path.exists(trial_file):
            file_to_read = trial_file
        else:
            return {"message": "No scenario data found", "data": []}
        
        # Read the CSV file
        df = pd.read_csv(file_to_read)
        
        # Handle NaN and infinite values
        df = df.fillna(0.0)
        df = df.replace([np.inf, -np.inf], 0.0)
        
        # Convert to list of dictionaries
        data = df.to_dict(orient='records')
        
        # Clean any remaining problematic values
        for record in data:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = 0.0
                elif value is None:
                    record[key] = ""
        
        return {
            "message": f"Latest scenario data loaded from {os.path.basename(file_to_read)}",
            "data": data,
            "records": len(data),
            "filename": os.path.basename(file_to_read)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading latest scenario: {str(e)}")

@app.post("/calculate_market_conditions")
def calculate_market_conditions():
    """Calculate enhanced market conditions using four signals from trial_sampled_offers.csv"""
    try:
        # Load the trial sampled offers data
        file_path = os.path.join('data', 'trial_sampled_offers.csv')
        if not os.path.exists(file_path):
            return {"error": "No trial data available"}
        
        df = pd.read_csv(file_path)
        
        if len(df) == 0:
            return {"error": "No offers data available"}
        
        # Step 1: Calculate the four signals
        
        # 1. Price Level (avg_price) - average price across all offers
        avg_price = df['price_per_night'].mean()
        
        # 2. Booking Urgency (days_to_go) - average days to go
        avg_days_to_go = df['days_to_go'].mean()
        
        # 3. Price Volatility - analyze price_history_24h for trend and volatility
        price_trends = []
        price_volatilities = []
        
        for _, row in df.iterrows():
            try:
                # Parse the 24-hour price history
                price_history = json.loads(row['price_history_24h'])
                if len(price_history) >= 2:
                    # Calculate trend (positive = increasing, negative = decreasing)
                    trend = (price_history[-1] - price_history[0]) / price_history[0]
                    price_trends.append(trend)
                    
                    # Calculate volatility (standard deviation of price changes)
                    price_changes = [price_history[i] - price_history[i-1] for i in range(1, len(price_history))]
                    volatility = np.std(price_changes) if price_changes else 0
                    price_volatilities.append(volatility)
            except (json.JSONDecodeError, KeyError, TypeError):
                # Fallback to variance if price_history_24h is not available
                price_trends.append(0)
                price_volatilities.append(row.get('price_fluctuation_variance', 0))
        
        avg_price_trend = np.mean(price_trends) if price_trends else 0
        avg_price_volatility = np.mean(price_volatilities) if price_volatilities else 0
        
        # 4. Competition Density - calculate unique hotels and partners
        unique_hotels = df['hotel_id'].nunique()
        unique_partners = df['partner_name'].nunique()
        competition_density = unique_hotels * unique_partners  # Total possible combinations
        
        # Step 2: Normalize all variables to 0-1 scale
        # For price level: normalize based on typical hotel price ranges (50-500)
        norm_avg_price = min(1.0, max(0.0, (avg_price - 50) / (500 - 50)))
        
        # For booking urgency: normalize days_to_go (1-365) and invert (closer = higher urgency)
        norm_days_to_go = 1 - min(1.0, max(0.0, (avg_days_to_go - 1) / (365 - 1)))
        
        # For price volatility: normalize based on typical volatility ranges (0-50)
        norm_price_volatility = min(1.0, max(0.0, avg_price_volatility / 50))
        
        # For price trend: normalize trend percentage (-0.5 to 0.5) to 0-1 scale
        norm_price_trend = min(1.0, max(0.0, (avg_price_trend + 0.5) / 1.0))
        
        # For competition density: normalize based on typical ranges (1-100)
        norm_competition_density = min(1.0, max(0.0, (competition_density - 1) / (100 - 1)))
        
        # Step 3: Calculate Market Demand Index with weights
        w1, w2, w3, w4 = 0.25, 0.25, 0.2, 0.3  # Weights for each signal
        market_demand_index = (
            w1 * norm_avg_price +
            w2 * norm_days_to_go +
            w3 * norm_price_volatility +
            w4 * norm_competition_density
        )
        
        # Step 4: Discretize into categories using terciles
        if market_demand_index < 0.33:
            demand_category = "Low"
        elif market_demand_index < 0.67:
            demand_category = "Medium"
        else:
            demand_category = "High"
        
        # Additional market insights
        market_insights = {
            "price_trend": "Increasing" if avg_price_trend > 0.05 else "Decreasing" if avg_price_trend < -0.05 else "Stable",
            "price_volatility": "High" if norm_price_volatility > 0.7 else "Low" if norm_price_volatility < 0.3 else "Moderate",
            "competition_level": "Low" if norm_competition_density < 0.3 else "High" if norm_competition_density > 0.7 else "Medium",
            "booking_pressure": "Low" if norm_days_to_go < 0.3 else "High" if norm_days_to_go > 0.7 else "Medium"
        }
        
        return {
            "market_demand_index": round(market_demand_index, 3),
            "demand_category": demand_category,
            "signals": {
                "price_level": {
                    "value": round(avg_price, 2),
                    "normalized": round(norm_avg_price, 3),
                    "weight": w1
                },
                "booking_urgency": {
                    "value": round(avg_days_to_go, 1),
                    "normalized": round(norm_days_to_go, 3),
                    "weight": w2
                },
                "price_volatility": {
                    "value": round(avg_price_volatility, 2),
                    "normalized": round(norm_price_volatility, 3),
                    "weight": w3
                },
                "price_trend": {
                    "value": round(avg_price_trend * 100, 1),  # Convert to percentage
                    "normalized": round(norm_price_trend, 3),
                    "direction": "Increasing" if avg_price_trend > 0.05 else "Decreasing" if avg_price_trend < -0.05 else "Stable"
                },
                "competition_density": {
                    "unique_hotels": unique_hotels,
                    "unique_partners": unique_partners,
                    "total_combinations": competition_density,
                    "normalized": round(norm_competition_density, 3),
                    "weight": w4
                }
            },
            "market_insights": market_insights,
            "data_summary": {
                "total_offers": len(df),
                "unique_hotels": unique_hotels,
                "unique_partners": unique_partners,
                "price_range": f"${df['price_per_night'].min():.2f} - ${df['price_per_night'].max():.2f}",
                "days_range": f"{df['days_to_go'].min()} - {df['days_to_go'].max()} days"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating market conditions: {str(e)}")

from collections import defaultdict

@app.get("/market_state/{location}")
def get_market_state(location: str):
    """Aggregate real offer/user data to compute market state for a location (was destination)"""
    import json
    try:
        offers_path = os.path.join('data', 'trial_sampled_offers.csv')
        users_path = os.path.join('data', 'enhanced_user_profiles.csv')
        if not os.path.exists(offers_path) or not os.path.exists(users_path):
            return {"error": "Required data files not found"}
        offers_df = pd.read_csv(offers_path)
        users_df = pd.read_csv(users_path)
        # Filter offers by location
        offers = offers_df[offers_df['location'] == location]
        if offers.empty:
            return {"error": f"No offers found for location {location}"}
        # Users for this location
        user_ids = offers['user_id'].unique().tolist()
        users = users_df[users_df['user_id'].isin(user_ids)]
        # Calculate market state as before, but for this location
        # Step 1: Calculate the four signals for this location
        avg_price = offers['price_per_night'].mean()
        avg_days_to_go = offers['days_to_go'].mean()
        price_trends = []
        price_volatilities = []
        for _, row in offers.iterrows():
            try:
                price_history = json.loads(row['price_history_24h'])
                if len(price_history) >= 2:
                    trend = (price_history[-1] - price_history[0]) / price_history[0]
                    price_trends.append(trend)
                    price_changes = [price_history[i] - price_history[i-1] for i in range(1, len(price_history))]
                    volatility = np.std(price_changes) if price_changes else 0
                    price_volatilities.append(volatility)
            except (json.JSONDecodeError, KeyError, TypeError):
                price_trends.append(0)
                price_volatilities.append(row.get('price_fluctuation_variance', 0))
        avg_price_trend = np.mean(price_trends) if price_trends else 0
        avg_price_volatility = np.mean(price_volatilities) if price_volatilities else 0
        # Competition density: unique hotels x unique partners
        competition_density = offers['hotel_id'].nunique() * offers['partner_name'].nunique()
        # Demand index: composite
        norm_price = min(1.0, max(0.0, avg_price / 1000))
        norm_days_to_go = min(1.0, max(0.0, avg_days_to_go / 180))
        norm_volatility = min(1.0, max(0.0, avg_price_volatility / 50))
        norm_competition = min(1.0, max(0.0, competition_density / 100))
        demand_index = 0.25 * norm_price + 0.25 * norm_days_to_go + 0.2 * norm_volatility + 0.3 * norm_competition
        return {
            "location": location,
            "avg_price": avg_price,
            "avg_days_to_go": avg_days_to_go,
            "avg_price_trend": avg_price_trend,
            "avg_price_volatility": avg_price_volatility,
            "competition_density": competition_density,
            "demand_index": demand_index,
            "user_count": len(users),
            "offer_count": len(offers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing market state: {str(e)}")

@app.get("/dynamic_price_sensitivity/{user_id}")
def get_dynamic_price_sensitivity(user_id: str):
    """Return dynamic price sensitivity for a user, combining base sensitivity, market state, and days to go."""
    import json
    import logging
    try:
        users_path = os.path.join('data', 'enhanced_user_profiles.csv')
        offers_path = os.path.join('data', 'trial_sampled_offers.csv')
        if not os.path.exists(users_path) or not os.path.exists(offers_path):
            return {"error": "Required data files not found"}
        users_df = pd.read_csv(users_path)
        offers_df = pd.read_csv(offers_path)
        user_row = users_df[users_df['user_id'] == user_id]
        user_offers = offers_df[offers_df['user_id'] == user_id]
        logging.info(f"[DPS] user_id={user_id}, user_row_empty={user_row.empty}, user_offers_empty={user_offers.empty}")
        if user_row.empty:
            return {"error": f"User {user_id} not found in enhanced_user_profiles.csv"}
        if user_offers.empty:
            return {"error": f"No offers found for user {user_id} in trial_sampled_offers.csv"}
        # Get location from the first offer
        location = user_offers.iloc[0].get('location', None)
        logging.info(f"[DPS] user_id={user_id}, location={location}")
        if location is None or str(location).strip() == "":
            return {"error": f"No location found for user {user_id}'s offers"}
        # Query market state for this location
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get(f"/market_state/{location}")
        if resp.status_code != 200:
            return {"error": f"Market state not found for location {location}"}
        market_state = resp.json()
        # Use days_to_go from offers
        avg_days_to_go = user_offers['days_to_go'].mean()
        # Base sensitivity from user profile
        base_sensitivity = float(user_row.iloc[0].get('price_sensitivity', 0.5))
        # Combine into dynamic price sensitivity
        volatility = market_state.get('avg_price_volatility', 0)
        trend = market_state.get('avg_price_trend', 0)
        dynamic_sensitivity = 0.4 * base_sensitivity + 0.2 * volatility + 0.2 * trend + 0.2 * (1 - avg_days_to_go / 180)
        return {
            "user_id": user_id,
            "location": location,
            "base_sensitivity": base_sensitivity,
            "market_volatility": volatility,
            "market_trend": trend,
            "avg_days_to_go": avg_days_to_go,
            "dynamic_price_sensitivity": dynamic_sensitivity
        }
    except Exception as e:
        logging.exception(f"[DPS] Exception for user_id={user_id}")
        return {"error": f"Exception: {str(e)}"}

@app.get("/offer_probabilities/{user_id}")
def get_offer_probabilities(user_id: str):
    """Return click and booking probabilities for all offers for a user."""
    import json
    try:
        users_path = os.path.join('data', 'enhanced_user_profiles.csv')
        offers_path = os.path.join('data', 'trial_sampled_offers.csv')
        if not os.path.exists(users_path) or not os.path.exists(offers_path):
            return {"error": "Required data files not found"}
        users_df = pd.read_csv(users_path)
        offers_df = pd.read_csv(offers_path)
        user_row = users_df[users_df['user_id'] == user_id]
        user_offers = offers_df[offers_df['user_id'] == user_id]
        if user_row.empty or user_offers.empty:
            return {"error": f"No data for user {user_id}"}
        user = user_row.iloc[0]
        location = user_offers.iloc[0]['location']
        # Query market state for this location
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get(f"/market_state/{location}")
        if resp.status_code != 200:
            return {"error": f"Market state not found for location {location}"}
        market_state = resp.json()
        # Fetch dynamic price sensitivity for this user
        dps_resp = client.get(f"/dynamic_price_sensitivity/{user_id}")
        if dps_resp.status_code == 200:
            dps_data = dps_resp.json()
            dynamic_sensitivity = dps_data.get('dynamic_price_sensitivity', 0.5)
        else:
            dynamic_sensitivity = 0.5
        # Calculate probabilities as before, but use location-based market state
        # For each offer, calculate click and booking probability
        results = []
        for idx, offer in user_offers.iterrows():
            # Click probability factors
            loyalty = 1.0 if user['loyalty_status'] in ['Gold', 'Platinum'] else 0.7 if user['loyalty_status'] == 'Silver' else 0.5
            price_score = max(0.1, 1 - dynamic_sensitivity * (offer['price_per_night'] / (user['budget_max'] + 1)))
            days_to_go_score = max(0.1, 1 - abs(offer['days_to_go'] - 30) / 365)
            # Service match (simple: 1 if offer includes any preferred amenity, else 0.7)
            preferred_amenities = str(user['preferred_amenities']).split(',') if pd.notna(user['preferred_amenities']) else []
            offer_services = str(offer.get('special_offers', '')).split(',') if pd.notna(offer.get('special_offers', '')) else []
            service_match = 1.0 if any(a.strip() in offer_services for a in preferred_amenities) else 0.7
            # Market state (use demand index from dynamic price sensitivity call)
            market_factor = 1 - abs(market_state.get('demand_index', 0.5) - 0.5)
            # Combine (simple weighted product)
            click_prob = loyalty * price_score * days_to_go_score * service_match * market_factor
            click_prob = min(max(click_prob, 0.01), 0.99)
            # Booking probability factors
            price_diff = offer['price_per_night'] - offer.get('trivago_price', offer['price_per_night'])
            price_diff_score = max(0.1, 1 - abs(price_diff) / (user['budget_max'] + 1))
            partner_policy = 1.0 if offer.get('cancellation_policy', '') == 'Free' else 0.8
            brand_score = 1.0 if offer.get('partner_name', '') in ['Booking.com', 'Expedia'] else 0.9
            hotel_rating = float(offer.get('star_rating', 3)) / 5.0
            booking_prob = price_diff_score * partner_policy * brand_score * hotel_rating
            booking_prob = min(max(booking_prob, 0.01), 0.99)
            results.append({
                "user_id": user_id,
                "offer_id": offer['offer_id'] if 'offer_id' in offer else idx,
                "offer_rank": offer['rank'] if 'rank' in offer else idx+1,
                "click_probability": click_prob,
                "booking_probability": booking_prob,
                "loyalty": loyalty,
                "price_score": price_score,
                "days_to_go_score": days_to_go_score,
                "service_match": service_match,
                "market_factor": market_factor,
                "price_diff_score": price_diff_score,
                "partner_policy": partner_policy,
                "brand_score": brand_score,
                "hotel_rating": hotel_rating,
                "dynamic_sensitivity": dynamic_sensitivity
            })
        return {"user_id": user_id, "location": location, "offers": results}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Error computing offer probabilities: {str(e)}")

@app.get("/test_pandas")
def test_pandas():
    """Test endpoint to verify pandas is working"""
    try:
        import pandas as pd
        import numpy as np
        return {"message": "Pandas is working", "pd_version": pd.__version__, "np_version": np.__version__}
    except Exception as e:
        return {"error": f"Pandas test failed: {str(e)}"}

@app.post("/run_bandit_simulation")
def run_bandit_simulation():
    """
    Run bandit simulation according to specifications:
    1. Consider offers in trial_sampled_offers.csv
    2. For each user, for each offer, simulate all possible ranks (rank i among all possible offers for user)
    3. Generate 1000 clicks for each arm-bandit (offer j at rank i for user U)
    4. Use recursive average online formula for reward learning
    5. Return table with user_id, offer_id, rank, probability of click, and list of rewards learnt
    """
    try:
        # Explicit imports to avoid scoping issues
        import pandas as pd
        import numpy as np
        import os
        
        # Read the offers data
        offers_path = os.path.join('data', 'trial_sampled_offers.csv')
        if not os.path.exists(offers_path):
            return {"error": "trial_sampled_offers.csv not found"}
        
        # Load data using pandas
        offers_df = pd.read_csv(offers_path)
        if offers_df.empty:
            return {"error": "No offers found in trial_sampled_offers.csv"}
        
        results = []
        clicks_per_arm = 1000
        
        for user_id in offers_df['user_id'].unique():
            user_offers = offers_df[offers_df['user_id'] == user_id].copy()
            n_ranks = len(user_offers)
            for _, offer in user_offers.iterrows():
                offer_id = offer['offer_id']
                base_prob = float(offer['preference_score'])
                for rank in range(1, n_ranks + 1):
                    # Simulate as if this offer is shown at this rank
                    rank_factor = 1.0 / rank  # Higher rank (lower number) = higher probability
                    true_click_prob = min(0.95, max(0.05, base_prob * rank_factor))
                    # Simulate 1000 clicks using recursive average online formula
                    rewards = []
                    cumulative_avg = 0.0
                    for click_num in range(1, clicks_per_arm + 1):
                        click = 1 if np.random.random() < true_click_prob else 0
                        rewards.append(click)
                        cumulative_avg = cumulative_avg + (click - cumulative_avg) / click_num
                    learned_probability = cumulative_avg
                    results.append({
                        'user_id': user_id,
                        'offer_id': offer_id,
                        'rank': rank,
                        'probability_of_click': learned_probability,
                        'rewards_learnt': rewards,
                        'true_click_prob': true_click_prob,
                        'preference_score': base_prob
                    })
        # Normalize probabilities per user per rank
        import copy
        user_rank_to_total_prob = {}
        for r in results:
            key = (r['user_id'], r['rank'])
            user_rank_to_total_prob.setdefault(key, 0.0)
            user_rank_to_total_prob[key] += r['probability_of_click']
        normalized_results = []
        for r in results:
            key = (r['user_id'], r['rank'])
            norm_prob = r['probability_of_click'] / user_rank_to_total_prob[key] if user_rank_to_total_prob[key] > 0 else 0.0
            r_copy = copy.deepcopy(r)
            r_copy['normalized_probability_of_click'] = norm_prob
            normalized_results.append(r_copy)
        # Save results to CSV
        results_df = pd.DataFrame(normalized_results)
        if not results_df.empty:
            results_df['user_id'] = results_df['user_id'].astype(str)
            results_df['offer_id'] = results_df['offer_id'].astype(str)
            results_df['rank'] = results_df['rank'].astype(int)
            results_df['probability_of_click'] = results_df['probability_of_click'].astype(float)
            results_df['true_click_prob'] = results_df['true_click_prob'].astype(float)
            results_df['preference_score'] = results_df['preference_score'].astype(float)
            results_df['normalized_probability_of_click'] = results_df['normalized_probability_of_click'].astype(float).round(2)
        csv_path = '/data/bandit_simulation_results.csv'
        results_df.drop(columns=['rewards_learnt','probability_of_click','true_click_prob']).to_csv(csv_path, index=False)
        # Calculate summary statistics
        total_arms = len(normalized_results)
        total_users = len(offers_df['user_id'].unique())
        total_offers = len(offers_df['offer_id'].unique())
        # Return results (including a sample of the normalized table for preview)
        return {
            'total_users': total_users,
            'total_offers': total_offers,
            'total_arms': total_arms,
            'clicks_per_arm': clicks_per_arm,
            'csv_path': csv_path,
            'message': f"Bandit simulation completed. Generated {total_arms} arm-bandit results with {clicks_per_arm} clicks each.",
            'sample_table': normalized_results[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running bandit simulation: {str(e)}")

@app.get("/bandit_simulation_results")
def get_bandit_simulation_results():
    """Return the contents of bandit_simulation_results.csv as JSON, handling NaN/inf/null values."""
    import os
    import pandas as pd
    import numpy as np
    try:
        file_path = os.path.join('data', 'bandit_simulation_results.csv')
        if not os.path.exists(file_path):
            return {"message": "No bandit_simulation_results.csv found", "data": []}
        df = pd.read_csv(file_path)
        df = df.fillna(0.0)
        df = df.replace([np.inf, -np.inf], 0.0)
        data = df.to_dict(orient='records')
        for record in data:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = 0.0
                elif value is None:
                    record[key] = ""
        return {"data": data, "records": len(data)}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Error reading bandit_simulation_results: {str(e)}")


    try:
        import pandas as pd
        import numpy as np
        import random
        
        # Load offers data
        offers_df = pd.read_csv('data/trial_sampled_offers.csv')
        
        # Group offers by user_id and assign ranks
        user_offers = {}
        for user_id in offers_df['user_id'].unique():
            user_offers[user_id] = offers_df[offers_df['user_id'] == user_id].copy()
            # Sort by preference_score (higher is better) and assign ranks
            user_offers[user_id] = user_offers[user_id].sort_values('preference_score', ascending=False)
            user_offers[user_id]['rank'] = range(1, len(user_offers[user_id]) + 1)
        
        results = []
        
        # For each user and their offers
        for user_id, user_df in user_offers.items():
            # For each offer (arm) at each rank
            for _, offer in user_df.iterrows():
                offer_id = offer['offer_id']
                rank = offer['rank']
                
                # Initialize reward learning with recursive average online formula
                # R_new = R_old + (reward - R_old) / n
                cumulative_reward = 0
                n_clicks = 0
                rewards_history = []
                
                # Simulate 1000 clicks for this arm-bandit
                for click in range(1000):
                    # Generate reward based on offer characteristics
                    # Higher preference_score, lower price, better reviews = higher reward probability
                    base_reward_prob = min(0.8, max(0.1, offer['preference_score']))
                    
                    # Add rank penalty (higher rank = lower probability)
                    rank_penalty = 1.0 / rank
                    reward_prob = base_reward_prob * rank_penalty
                    
                    # Add some randomness
                    reward_prob = min(0.9, max(0.05, reward_prob + random.uniform(-0.1, 0.1)))
                    
                    # Generate reward (1 for click, 0 for no click)
                    reward = 1 if random.random() < reward_prob else 0
                    
                    # Update cumulative reward using recursive average online formula
                    n_clicks += 1
                    cumulative_reward = cumulative_reward + (reward - cumulative_reward) / n_clicks
                    rewards_history.append(cumulative_reward)
                
                # Calculate final click probability
                click_probability = cumulative_reward
                
                results.append({
                    'user_id': user_id,
                    'offer_id': offer_id,
                    'rank': rank,
                    'click_probability': round(click_probability, 4),
                    'rewards_learned': [round(r, 4) for r in rewards_history[::100]]  # Sample every 100th reward
                })
        
        # Normalize probabilities to sum to 1 for each user
        for user_id in user_offers.keys():
            user_results = [r for r in results if r['user_id'] == user_id]
            total_prob = sum(r['click_probability'] for r in user_results)
            
            if total_prob > 0:
                for result in user_results:
                    if result['user_id'] == user_id:
                        result['click_probability'] = round(result['click_probability'] / total_prob, 4)
        
        return {
            "status": "success",
            "message": f"Bandit simulation completed for {len(user_offers)} users",
            "results": results
        }
        
    except Exception as e:
        print(f"Error in bandit simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running bandit simulation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

