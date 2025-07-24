import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import uuid
import json
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import pulp
import subprocess
import tempfile
from itertools import combinations
from typing import Dict, List, Tuple

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

# Define the data directory path - use the Docker volume mount path
DATA_DIR = '/data'  # This matches the Docker Compose volume mount

# Update all file path references to use DATA_DIR
def get_data_path(filename):
    """Get the full path for a data file"""
    return os.path.join(DATA_DIR, filename)

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
        ],
        "data_files": [
            "enhanced_user_profiles.csv",
            "enhanced_hotels.csv", 
            "enhanced_partner_offers.csv",
            "trial_sampled_offers.csv",
            "bandit_simulation_results.csv",
            "conversion_probabilities.csv",
            "user_dynamic_price_sensitivity.csv",
            "user_market_state.csv",
            "market_state_by_location.csv",
            "policy_heatmap_debug.json",
            "dqn_model.pth"
        ]
    }

@app.get("/user_profiles")
def get_sample_user_profiles():
    """Get 10 random user profiles for selection from enhanced_user_profiles.csv"""
    csv_path = get_data_path('enhanced_user_profiles.csv')
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
        file_path = get_data_path('enhanced_user_profiles.csv')
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
        users = pd.read_csv(get_data_path('enhanced_user_profiles.csv'))
        user = users[users['user_id'] == user_id]
        if user.empty:
            return []
        user = user.iloc[0]
        
        # Load hotels and offers
        hotels = pd.read_csv(get_data_path('enhanced_hotels.csv'))
        offers = pd.read_csv(get_data_path('enhanced_partner_offers.csv'))
        
        # Filter hotels based on user preferences
        filtered_hotels = hotels.copy()
        
        # Filter by user location (handle format differences)
        if 'location' in user and pd.notna(user['location']):
            user_location = user['location']
            # Extract city name from hotel locations (e.g., "Sydney, Australia" -> "Sydney")
            hotels['city_name'] = hotels['location'].str.split(',').str[0].str.strip()
            location_filtered = hotels[hotels['city_name'] == user_location]
            if len(location_filtered) > 0:
                filtered_hotels = location_filtered
            else:
                print(f"No hotels found for user location: {user_location}")
                # Fallback: try exact match
                location_filtered = hotels[hotels['location'] == user_location]
            if len(location_filtered) > 0:
                filtered_hotels = location_filtered
        
        # Get average prices for hotels from offers
        avg_prices = offers.groupby('hotel_id')['price_per_night'].mean().reset_index()
        avg_prices.columns = ['hotel_id', 'avg_price']
        
        # Filter by budget range
        budget_min = user['budget_min']
        budget_max = user['budget_max']
        
        # Filter hotels within budget
        hotels_with_prices = filtered_hotels.merge(avg_prices, on='hotel_id', how='left')
        budget_hotels = hotels_with_prices[
            (hotels_with_prices['avg_price'] >= budget_min) & 
            (hotels_with_prices['avg_price'] <= budget_max)
        ]
        
        # If budget filtering results in no hotels, try with relaxed budget constraints
        if len(budget_hotels) == 0:
            print(f"No hotels found for user {user_id} with budget {budget_min}-{budget_max}, trying relaxed budget")
            # Expand budget range by 50%
            budget_range = budget_max - budget_min
            expanded_min = max(0, budget_min - budget_range * 0.5)
            expanded_max = budget_max + budget_range * 0.5
            budget_hotels = hotels_with_prices[
                (hotels_with_prices['avg_price'] >= expanded_min) & 
                (hotels_with_prices['avg_price'] <= expanded_max)
            ]
        
        # If still no hotels, use all hotels with prices
        if len(budget_hotels) == 0:
            print(f"Still no hotels found for user {user_id}, using all available hotels")
            budget_hotels = hotels_with_prices.dropna(subset=['avg_price'])
        
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
        
        # Sample partners per hotel to respect num_partners parameter
        sampled_offers = []
        total_offers_available = 0
        for hotel_id in sampled_hotels['hotel_id']:
            hotel_offers_subset = hotel_offers[hotel_offers['hotel_id'] == hotel_id]
            if len(hotel_offers_subset) > 0:
                if len(hotel_offers_subset) > num_partners:
                    # Sample num_partners from available partners for this hotel
                    sampled_hotel_offers = hotel_offers_subset.sample(n=num_partners)
                    print(f"Hotel {hotel_id}: {len(sampled_hotel_offers)} offers sampled (requested {num_partners})")
                else:
                    # Take all available offers if less than num_partners
                    sampled_hotel_offers = hotel_offers_subset
                    print(f"Hotel {hotel_id}: {len(sampled_hotel_offers)} offers taken (less than requested {num_partners})")
                
                sampled_offers.append(sampled_hotel_offers)
                total_offers_available += len(sampled_hotel_offers)
        
        print(f"Total offers available for user {user_id}: {total_offers_available}")
        
        if sampled_offers:
            offers_df = pd.concat(sampled_offers, ignore_index=True)
        else:
            return []
        
        # Join hotel info
        offers_df = offers_df.merge(sampled_hotels, on='hotel_id', suffixes=('_offer', '_hotel'))
        
        # Add user_id
        offers_df['user_id'] = user_id
        
        # Add days_to_go column - same value for all offers of the same user
        user_days_to_go = int(np.random.normal(days_to_go, days_var))
        user_days_to_go = max(1, min(365, user_days_to_go))  # Ensure reasonable range
        offers_df['days_to_go'] = user_days_to_go
        
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
        offers_df.to_csv(get_data_path('trial_sampled_offers.csv'), index=False)
        
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
        # Handle location format differences (e.g., "Sydney" vs "Sydney, Australia")
        offer_location = offer_row['location']
        user_location = user['location']
        
        # Extract city name from offer location
        offer_city = offer_location.split(',')[0].strip() if ',' in offer_location else offer_location
        
        if offer_city == user_location:
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
def save_scenario_data(scenario_data: dict, request: Request = None):
    """Save scenario data for all users with days_to_go column and save market state CSV, bandit simulation results, and user market state CSV."""
    try:
        all_offers = scenario_data.get('all_offers', [])
        if not all_offers:
            return {"message": "No offers data provided"}
        df = pd.DataFrame(all_offers)
        if 'days_to_go' not in df.columns:
            df['days_to_go'] = np.random.normal(30, 5, size=len(df)).astype(int)
        filename = "trial_sampled_offers.csv"
        filepath = get_data_path(filename)
        df.to_csv(filepath, index=False)
        # Save market state CSV after saving offers
        save_market_state_csv()
        # Save bandit simulation results
        run_bandit_simulation_and_save_csv()
        # Save user market state CSV
        user_market_state_csv_function()
        # NEW: Save user dynamic price sensitivity CSV
        user_dynamic_price_sensitivity_csv()
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
        file_path = get_data_path('trial_sampled_offers.csv')
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
        file_path = get_data_path('trial_sampled_offers.csv')
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
        file_path = get_data_path('trial_sampled_offers.csv')
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
    days_var: int = Query(5, ge=1, le=30),
    min_users_per_destination: int = Query(1, ge=1, le=20)
):
    # Ensure parameters are integers, not Query objects
    if hasattr(min_users_per_destination, 'annotation'):
        min_users_per_destination = 1  # Default value if Query object
    if hasattr(num_users, 'annotation'):
        num_users = 1
    if hasattr(num_hotels, 'annotation'):
        num_hotels = 10
    if hasattr(num_partners, 'annotation'):
        num_partners = 5
    if hasattr(days_to_go, 'annotation'):
        days_to_go = 30
    if hasattr(days_var, 'annotation'):
        days_var = 5
    """Sample offers for N unique users by sampling hotels per region/destination, enforcing a minimum number of users per destination."""
    try:
        # Load data
        users_df = pd.read_csv(get_data_path('enhanced_user_profiles.csv'))
        hotels_df = pd.read_csv(get_data_path('enhanced_hotels.csv'))
        offers_df = pd.read_csv(get_data_path('enhanced_partner_offers.csv'))
        
        if len(users_df) < num_users:
            num_users = len(users_df)
        
        print(f"[DEBUG] Requested {num_users} users with minimum {min_users_per_destination} users per destination")
        print(f"[DEBUG] Total users available: {len(users_df)}")
        
        # Count users per destination in the full dataset
        dest_counts_full = users_df['location'].value_counts()
        print(f"[DEBUG] Users per destination in full dataset: {dest_counts_full.to_dict()}")
        
        # Find destinations that have enough users to satisfy the constraint
        valid_destinations = dest_counts_full[dest_counts_full >= min_users_per_destination].index.tolist()
        print(f"[DEBUG] Valid destinations (>= {min_users_per_destination} users): {valid_destinations}")
        
        if not valid_destinations:
            return {
                "message": f"No destinations have at least {min_users_per_destination} users. Available destinations and their user counts: {dest_counts_full.to_dict()}",
                "num_records": 0,
                "error": "constraint_violation",
                "available_destinations": dest_counts_full.to_dict()
            }
        
        # Calculate how many users to sample per destination
        # Try to distribute users evenly across valid destinations
        users_per_dest = max(min_users_per_destination, num_users // len(valid_destinations))
        print(f"[DEBUG] Target users per destination: {users_per_dest}")
        
        # Sample users destination by destination
        sampled_users_list = []
        total_sampled = 0
        
        for destination in valid_destinations:
            if total_sampled >= num_users:
                break
                
            # Get users for this destination
            dest_users = users_df[users_df['location'] == destination]
            
            # Calculate how many users to sample from this destination
            remaining_users_needed = num_users - total_sampled
            users_to_sample = min(users_per_dest, len(dest_users), remaining_users_needed)
            
            if users_to_sample >= min_users_per_destination:
                # Sample users from this destination
                dest_sampled = dest_users.sample(n=users_to_sample, replace=False)
                sampled_users_list.append(dest_sampled)
                total_sampled += users_to_sample
                print(f"[DEBUG] Sampled {users_to_sample} users from {destination}")
            else:
                print(f"[DEBUG] Skipping {destination} - not enough users to satisfy constraint")
        
        if not sampled_users_list:
            return {
                "message": f"Could not sample enough users to satisfy the constraint. Requested: {num_users}, minimum per destination: {min_users_per_destination}",
                "num_records": 0,
                "error": "constraint_violation",
                "available_destinations": dest_counts_full.to_dict()
            }
        
        # Combine all sampled users
        sampled_users = pd.concat(sampled_users_list, ignore_index=True)
        final_destinations = sampled_users['location'].unique()
        final_dest_counts = sampled_users['location'].value_counts()
        
        print(f"[DEBUG] Final sampling result:")
        print(f"- Total users sampled: {len(sampled_users)}")
        print(f"- Destinations selected: {final_destinations}")
        print(f"- Users per destination: {final_dest_counts.to_dict()}")
        print(f"- Constraint satisfied: All destinations have >= {min_users_per_destination} users")
        
        # Sample hotels per destination (not per user)
        destination_hotels = {}
        all_offers = []
        
        for destination in final_destinations:
            print(f"Processing destination: {destination}")
            
            # Filter hotels for this destination
            if ',' in destination:
                # Handle "City, Country" format
                city_name = destination.split(',')[0].strip()
                destination_hotels_filtered = hotels_df[hotels_df['location'].str.contains(city_name, case=False, na=False)]
            else:
                # Handle "City" format
                destination_hotels_filtered = hotels_df[hotels_df['location'].str.contains(destination, case=False, na=False)]
            
            if len(destination_hotels_filtered) == 0:
                print(f"No hotels found for destination: {destination}")
                continue
            
            # Sample hotels for this destination
            if len(destination_hotels_filtered) > num_hotels:
                sampled_hotels_for_destination = destination_hotels_filtered.sample(n=num_hotels)
            else:
                sampled_hotels_for_destination = destination_hotels_filtered
            
            destination_hotels[destination] = sampled_hotels_for_destination
            print(f"Sampled {len(sampled_hotels_for_destination)} hotels for {destination}")
        
        # Generate offers for each user based on their destination
        users_with_offers = 0
        
        for _, user in sampled_users.iterrows():
            user_id = user['user_id']
            user_destination = user['location']
            
            # Get hotels for this user's destination
            if user_destination in destination_hotels:
                hotels_for_user = destination_hotels[user_destination]
                
                # Get all offers for these hotels
                hotel_ids = hotels_for_user['hotel_id'].tolist()
                offers_for_hotels = offers_df[offers_df['hotel_id'].isin(hotel_ids)]
                
                if len(offers_for_hotels) > 0:
                    # Sample partners per hotel to respect num_partners parameter
                    sampled_offers = []
                    for hotel_id in hotel_ids:
                        hotel_offers = offers_for_hotels[offers_for_hotels['hotel_id'] == hotel_id]
                        print(f"Hotel {hotel_id}: {len(hotel_offers)} total offers available")
                        
                        if len(hotel_offers) > num_partners:
                            # Sample num_partners from available partners for this hotel
                            sampled_hotel_offers = hotel_offers.sample(n=num_partners)
                            print(f"  Sampled {len(sampled_hotel_offers)} offers (requested {num_partners})")
                        else:
                            # Take all available offers if less than num_partners
                            sampled_hotel_offers = hotel_offers
                            print(f"  Took all {len(sampled_hotel_offers)} offers (less than requested {num_partners})")
                        
                        sampled_offers.append(sampled_hotel_offers)
                    
                    # Combine all sampled offers
                    offers_for_hotels = pd.concat(sampled_offers, ignore_index=True)
                    print(f"Total offers after partner sampling: {len(offers_for_hotels)} (expected: {len(hotel_ids)} * {num_partners} = {len(hotel_ids) * num_partners})")
                    
                    # Join with hotel information
                    offers_with_hotels = offers_for_hotels.merge(hotels_for_user, on='hotel_id', suffixes=('_offer', '_hotel'))
                    
                    # Add user information
                    offers_with_hotels['user_id'] = user_id
                    offers_with_hotels['location'] = user_destination
                    
                    # Add days_to_go - same value for all offers of the same user
                    user_days_to_go = int(np.random.normal(days_to_go, days_var))
                    user_days_to_go = max(1, min(365, user_days_to_go))  # Ensure reasonable range
                    offers_with_hotels['days_to_go'] = user_days_to_go
                    
                    # Add preference score
                    offers_with_hotels['preference_score'] = offers_with_hotels.apply(
                        lambda row: _calculate_preference_score(row, user), axis=1
                    )
                    
                    # Handle NaN and infinite values
                    offers_with_hotels['preference_score'] = offers_with_hotels['preference_score'].fillna(0.0)
                    offers_with_hotels['preference_score'] = offers_with_hotels['preference_score'].replace([np.inf, -np.inf], 0.0)
                    
                    # Sort by preference score
                    offers_with_hotels = offers_with_hotels.sort_values('preference_score', ascending=False)
                    
                    # Convert to dict and clean values
                    user_offers = offers_with_hotels.to_dict(orient='records')
                    for record in user_offers:
                        for key, value in record.items():
                            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                                record[key] = 0.0
                            elif value is None:
                                record[key] = ""
                    
                    all_offers.extend(user_offers)
                    users_with_offers += 1
                    print(f"Generated {len(user_offers)} offers for user {user_id} in {user_destination}")
                else:
                    print(f"No offers found for user {user_id} in {user_destination}")
            else:
                print(f"No hotels available for user {user_id} in {user_destination}")
        
        if all_offers:
            # Create final DataFrame
            final_offers_df = pd.DataFrame(all_offers)
            final_offers_df.to_csv(get_data_path('trial_sampled_offers.csv'), index=False)
            
            # Calculate metrics
            unique_locations_with_offers = final_offers_df['location'].nunique()
            total_offers = len(final_offers_df)
            
            # Calculate demand per destination (number of users per destination)
            demand_per_destination = final_offers_df.groupby('location')['user_id'].nunique().to_dict()
            
            # Calculate offers per destination (number of rooms × number of partners)
            offers_per_destination = final_offers_df.groupby('location').size().to_dict()
            
            print(f"Demand per destination: {demand_per_destination}")
            print(f"Offers per destination: {offers_per_destination}")
            
            # Update market state CSV after generating new offers
            save_market_state_csv()
            
            # Update bandit simulation results with parameters
            run_bandit_simulation_and_save_csv(num_users=num_users, num_hotels=num_hotels, num_partners=num_partners, days_to_go=days_to_go, days_var=days_var)
            
            # Update user market state CSV
            user_market_state_csv_function()
            
            # Update user dynamic price sensitivity CSV with parameters
            user_dynamic_price_sensitivity_csv(num_users=num_users, num_hotels=num_hotels, num_partners=num_partners, days_to_go=days_to_go, days_var=days_var)
            
            # Update conversion probabilities CSV with parameters
            conversion_probabilities_csv(num_users=num_users, num_hotels=num_hotels, num_partners=num_partners, days_to_go=days_to_go, days_var=days_var)
            
            return {
                "message": f"Generated scenario with {users_with_offers} users out of {num_users} requested. Total offers: {total_offers}, Offered rooms by location: {unique_locations_with_offers}. Demand per destination: {demand_per_destination}",
                "users_with_offers": users_with_offers,
                "total_offers": total_offers,
                "offered_rooms_by_location": unique_locations_with_offers,
                "demand_per_destination": demand_per_destination,
                "offers_per_destination": offers_per_destination,
                "parameters": {
                    "num_users": num_users,
                    "num_hotels": num_hotels,
                    "num_partners": num_partners,
                    "days_to_go": days_to_go,
                    "days_var": days_var
                }
            }
        else:
            return {"message": "No offers generated for any users.", "num_records": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling offers for users: {str(e)}")

@app.get("/latest_scenario")
def get_latest_scenario():
    """Get the latest saved scenario data"""
    try:
        # Try to read the latest scenario file
        latest_file = get_data_path('latest_scenario_data.csv')
        trial_file = get_data_path('trial_sampled_offers.csv')
        
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
        file_path = get_data_path('trial_sampled_offers.csv')
        if not os.path.exists(file_path):
            return {"error": "No trial data available"}
        
        df = pd.read_csv(file_path)
        
        if len(df) == 0:
            return {"error": "No offers data available"}
        
        # Step 1: Calculate the four signals
        
        # 1. Price Level (avg_price) - average price across all offers
        avg_price = df['price_per_night'].mean()
        
        # 2. Booking Urgency (days_to_go) - average across all users
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
                    "value": int(round(avg_days_to_go)),
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
    """Aggregate real offer/user data to compute market state for a location (was destination) and label it."""
    import json
    try:
        offers_path = get_data_path('trial_sampled_offers.csv')
        users_path = get_data_path('enhanced_user_profiles.csv')
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
        avg_price = offers['price_per_night'].mean()
        avg_days_to_go = offers['days_to_go'].mean()  # Average across users in this location
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
        competition_density = offers['hotel_id'].nunique() * offers['partner_name'].nunique()
        norm_price = min(1.0, max(0.0, avg_price / 1000))
        norm_days_to_go = min(1.0, max(0.0, avg_days_to_go / 180))
        norm_volatility = min(1.0, max(0.0, avg_price_volatility / 50))
        norm_competition = min(1.0, max(0.0, competition_density / 100))
        demand_index = 0.25 * norm_price + 0.25 * norm_days_to_go + 0.2 * norm_volatility + 0.3 * norm_competition
        # Compute medians for user_count and offer_count across all locations
        all_user_counts = offers_df.groupby('location')['user_id'].nunique()
        all_offer_counts = offers_df.groupby('location')['offer_id'].nunique()
        median_user_count = all_user_counts.median() if not all_user_counts.empty else 1
        median_offer_count = all_offer_counts.median() if not all_offer_counts.empty else 1
        user_count = len(users)
        offer_count = len(offers)
        # Label logic
        if (demand_index > 0.66 or avg_price_trend > 0.05) and user_count >= median_user_count and offer_count >= median_offer_count:
            market_state_label = 'high'
        elif (demand_index < 0.33 or avg_price_trend < -0.05) and user_count <= median_user_count and offer_count <= median_offer_count:
            market_state_label = 'low'
        else:
            market_state_label = 'medium'
        return {
            "location": location,
            "avg_price": avg_price,
            "avg_days_to_go": int(round(avg_days_to_go)),
            "avg_price_trend": avg_price_trend,
            "avg_price_volatility": avg_price_volatility,
            "competition_density": competition_density,
            "demand_index": demand_index,
            "user_count": user_count,
            "offer_count": offer_count,
            "market_state_label": market_state_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing market state: {str(e)}")

@app.post("/user_market_state_csv")
def user_market_state_csv():
    """Create a user_market_state.csv file mapping each user to their destination and the computed market state label."""
    import pandas as pd
    import os
    import requests
    offers_path = get_data_path('trial_sampled_offers.csv')
    out_path = get_data_path('user_market_state.csv')
    if not os.path.exists(offers_path):
        return {"error": "trial_sampled_offers.csv not found"}
    offers_df = pd.read_csv(offers_path)
    # Get unique (user_id, location) pairs
    user_locs = offers_df[['user_id', 'location']].drop_duplicates()
    # For each location, get market state label
    from fastapi.testclient import TestClient
    client = TestClient(app)
    market_labels = {}
    for loc in user_locs['location'].unique():
        resp = client.get(f"/market_state/{loc}")
        if resp.status_code == 200:
            market_labels[loc] = resp.json().get('market_state_label', 'unknown')
        else:
            market_labels[loc] = 'unknown'
    user_locs['market_state_label'] = user_locs['location'].map(market_labels)
    user_locs.to_csv(out_path, index=False)
    return {"message": f"user_market_state.csv created with {len(user_locs)} rows.", "csv_path": out_path}

@app.get("/dynamic_price_sensitivity/{user_id}")
def get_dynamic_price_sensitivity(user_id: str):
    """Return dynamic price sensitivity for a user, combining base sensitivity, market state, and days to go."""
    import json
    import logging
    try:
        users_path = get_data_path('enhanced_user_profiles.csv')
        offers_path = get_data_path('trial_sampled_offers.csv')
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
        # Use days_to_go from offers (all offers have the same value for a user)
        user_days_to_go = user_offers['days_to_go'].iloc[0]
        # Base sensitivity from user profile
        base_sensitivity = float(user_row.iloc[0].get('price_sensitivity', 0.5))
        # Combine into dynamic price sensitivity
        volatility = market_state.get('avg_price_volatility', 0)
        trend = market_state.get('avg_price_trend', 0)
        dynamic_sensitivity = 0.4 * base_sensitivity + 0.2 * volatility + 0.2 * trend + 0.2 * (1 - user_days_to_go / 180)
        return {
            "user_id": user_id,
            "location": location,
            "base_sensitivity": base_sensitivity,
            "market_volatility": volatility,
            "market_trend": trend,
            "days_to_go": user_days_to_go,
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
        users_path = get_data_path('enhanced_user_profiles.csv')
        offers_path = get_data_path('trial_sampled_offers.csv')
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
            # Calculate probability of conversion (click to booking)
            # Example factors: price difference, hotel rating, amenities match, user profile, partner brand
            trivago_price = offer.get('trivago_price', offer['price_per_night']) if 'trivago_price' in offer else offer['price_per_night']
            partner_price = offer['price_per_night']
            price_diff = trivago_price - partner_price
            price_diff_score = max(0.1, 1 - abs(price_diff) / (user['budget_max'] + 1)) if 'budget_max' in offer else 0.5
            # Hotel rating (normalized 0.5 to 1)
            hotel_rating = float(offer.get('star_rating', 3))
            hotel_rating_score = 0.5 + 0.1 * (hotel_rating - 3)  # 3-star = 0.5, 5-star = 0.7
            # Amenities match (fraction of preferred amenities present)
            preferred_amenities = str(user['preferred_amenities']).split(',') if 'preferred_amenities' in user and pd.notna(user['preferred_amenities']) else []
            offer_amenities = str(offer.get('amenities', '')).split(',') if 'amenities' in offer and pd.notna(offer.get('amenities', '')) else []
            amenities_match = sum(1 for a in preferred_amenities if a.strip() in [b.strip() for b in offer_amenities])
            amenities_score = 0.5 + 0.1 * min(amenities_match, 5)  # up to 1.0
            # User profile: loyalty status
            loyalty = user['loyalty_status'] if 'loyalty_status' in user else ''
            loyalty_score = 1.0 if loyalty in ['Gold', 'Platinum'] else 0.8 if loyalty == 'Silver' else 0.6
            # Partner brand
            partner = offer.get('partner_name', '')
            brand_score = 1.0 if partner in ['Booking.com', 'Expedia'] else 0.9
            # Combine (simple weighted product)
            probability_of_conversion = price_diff_score * hotel_rating_score * amenities_score * loyalty_score * brand_score
            probability_of_conversion = min(max(probability_of_conversion, 0.01), 0.99)
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
                "dynamic_sensitivity": dynamic_sensitivity,
                "probability_of_conversion": probability_of_conversion
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
        import copy
        # Read the offers data
        offers_path = get_data_path('trial_sampled_offers.csv')
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
                        'probability_of_click': round(learned_probability, 4),
                        'rewards_learnt': rewards,
                        'true_click_prob': round(true_click_prob, 4),
                        'preference_score': round(base_prob, 4)
                    })
        # Save results to CSV (remove softmax normalization and normalized_probability_of_click)
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df['user_id'] = results_df['user_id'].astype(str)
            results_df['offer_id'] = results_df['offer_id'].astype(str)
            results_df['rank'] = results_df['rank'].astype(int)
            results_df['probability_of_click'] = results_df['probability_of_click'].astype(float)
            results_df['true_click_prob'] = results_df['true_click_prob'].astype(float)
            results_df['preference_score'] = results_df['preference_score'].astype(float)
        csv_path = get_data_path('bandit_simulation_results.csv')
        results_df.drop(columns=['rewards_learnt']).to_csv(csv_path, index=False)
        # Calculate summary statistics
        total_arms = len(results)
        total_users = len(offers_df['user_id'].unique())
        total_offers = len(offers_df['offer_id'].unique())
        # Return results (including a sample of the table for preview)
        return {
            'total_users': total_users,
            'total_offers': total_offers,
            'total_arms': total_arms,
            'clicks_per_arm': clicks_per_arm,
            'csv_path': csv_path,
            'message': f"Bandit simulation completed. Generated {total_arms} arm-bandit results with {clicks_per_arm} clicks each.",
            'sample_table': results[:5]
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
        file_path = get_data_path('bandit_simulation_results.csv')
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
        offers_df = pd.read_csv(get_data_path('trial_sampled_offers.csv'))
        
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

def compute_and_save_user_market_state():
    """Compute market state for each destination and save user_id, location, market_state to CSV."""
    import pandas as pd
    import numpy as np
    import json
    import os
    offers_path = get_data_path('trial_sampled_offers.csv')
    out_path = get_data_path('user_market_state.csv')
    if not os.path.exists(offers_path):
        print("trial_sampled_offers.csv not found")
        return
    offers_df = pd.read_csv(offers_path)
    # Aggregate signals per location
    def get_volatility(x):
        vals = []
        for h in x:
            try:
                arr = json.loads(h) if isinstance(h, str) else []
                if len(arr) > 1:
                    vals.append(float(np.std(np.diff(arr))))
            except Exception:
                continue
        return np.mean(vals) if vals else 0.0
    def get_trend(x):
        vals = []
        for h in x:
            try:
                arr = json.loads(h) if isinstance(h, str) else []
                if len(arr) > 1 and arr[0] != 0:
                    vals.append((arr[-1] - arr[0]) / arr[0])
            except Exception:
                continue
        return np.mean(vals) if vals else 0.0
    agg = offers_df.groupby('location').agg(
        user_demand=('user_id', 'nunique'),
        offer_supply=('offer_id', 'nunique'),
        price_volatility=('price_history_24h', get_volatility),
        price_trend=('price_history_24h', get_trend)
    ).reset_index()
    # Normalize each column
    for col in ['user_demand', 'offer_supply', 'price_volatility', 'price_trend']:
        min_val, max_val = agg[col].min(), agg[col].max()
        if max_val > min_val:
            agg[f'norm_{col}'] = (agg[col] - min_val) / (max_val - min_val)
        else:
            agg[f'norm_{col}'] = 0.0
    # Compute MDI
    agg['MDI'] = 0.25 * agg['norm_user_demand'] + 0.25 * agg['norm_offer_supply'] + 0.25 * agg['norm_price_volatility'] + 0.25 * agg['norm_price_trend']
    # Assign label
    def label_mdi(mdi):
        if mdi < 0.33:
            return 'Low'
        elif mdi < 0.67:
            return 'Medium'
        else:
            return 'High'
    agg['market_state'] = agg['MDI'].apply(label_mdi)
    # Map location to market_state
    loc_to_label = dict(zip(agg['location'], agg['market_state']))
    # For each user, get their location and assign label
    user_locs = offers_df[['user_id', 'location']].drop_duplicates()
    user_locs['market_state'] = user_locs['location'].map(loc_to_label)
    user_locs.to_csv(out_path, index=False)
    print(f"user_market_state.csv created with {len(user_locs)} rows.")

@app.post("/user_dynamic_price_sensitivity_csv")
def user_dynamic_price_sensitivity_csv(
    num_users: int = Query(None, ge=1, le=100),
    num_hotels: int = Query(None, ge=1, le=20),
    num_partners: int = Query(None, ge=1, le=10),
    days_to_go: int = Query(None, ge=1, le=365),
    days_var: int = Query(None, ge=1, le=30),
    min_users_per_destination: int = Query(None, ge=1, le=20)
):
    # Ensure parameters are integers, not Query objects
    if hasattr(min_users_per_destination, 'annotation'):
        min_users_per_destination = None
    if hasattr(num_users, 'annotation'):
        num_users = None
    if hasattr(num_hotels, 'annotation'):
        num_hotels = None
    if hasattr(num_partners, 'annotation'):
        num_partners = None
    if hasattr(days_to_go, 'annotation'):
        days_to_go = None
    if hasattr(days_var, 'annotation'):
        days_var = None
    """Generate user dynamic price sensitivity CSV with minimum users per destination constraint."""
    try:
        import pandas as pd
        import numpy as np
        import json
        import os
        import traceback
        
        offers_path = get_data_path('trial_sampled_offers.csv')
        users_path = get_data_path('enhanced_user_profiles.csv')
        out_path = get_data_path('user_dynamic_price_sensitivity.csv')
        
        print(f"[DEBUG] Called user_dynamic_price_sensitivity_csv. Offers path: {offers_path}, Users path: {users_path}, Output path: {out_path}")
        print(f"[DEBUG] Parameters: num_users={num_users}, num_hotels={num_hotels}, num_partners={num_partners}, days_to_go={days_to_go}, days_var={days_var}, min_users_per_destination={min_users_per_destination}")
        
        # Handle case where parameters are None (called without parameters)
        if num_users is None and num_hotels is None and num_partners is None:
            print(f"[DEBUG] No parameters provided, using existing data")
            # Use existing data without filtering
        
        if not os.path.exists(offers_path) or not os.path.exists(users_path):
            print(f"[ERROR] Required data files not found: {offers_path}, {users_path}")
            return {"error": "Required data files not found"}
        
        offers_df = pd.read_csv(offers_path)
        users_df = pd.read_csv(users_path)
        
        # Apply minimum users per destination constraint
        if min_users_per_destination is not None and isinstance(min_users_per_destination, int):
            # Count users per destination
            dest_counts = offers_df.groupby('location')['user_id'].nunique()
            valid_destinations = dest_counts[dest_counts >= min_users_per_destination].index.tolist()
            
            # Filter offers to only include valid destinations
            offers_df = offers_df[offers_df['location'].isin(valid_destinations)]
            
            print(f"[DEBUG] After constraint filtering: {len(valid_destinations)} destinations with >= {min_users_per_destination} users")
            print(f"[DEBUG] Valid destinations: {valid_destinations}")
            print(f"[DEBUG] Users per destination: {dest_counts.to_dict()}")
        
        # Filter offers based on parameters if provided
        if num_users is not None:
            # Ensure we only process the expected number of users
            unique_users = offers_df['user_id'].unique()[:num_users]
            offers_df = offers_df[offers_df['user_id'].isin(unique_users)]
        
        user_rows = []
        for user_id in offers_df['user_id'].unique():
            user_offers = offers_df[offers_df['user_id'] == user_id]
            if user_offers.empty:
                continue
            location = user_offers.iloc[0]['location']
            days_to_go = user_offers['days_to_go'].iloc[0]  # All offers have the same value
            
            # Get all offers for this location
            location_offers = offers_df[offers_df['location'] == location]
            
            # Calculate price statistics using 24-hour price history
            all_24h_prices = []
            for _, offer in location_offers.iterrows():
                try:
                    # Parse the 24-hour price history
                    price_history = json.loads(offer['price_history_24h'])
                    if isinstance(price_history, list) and len(price_history) > 0:
                        all_24h_prices.extend(price_history)
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Fallback to current price if history is not available
                    all_24h_prices.append(offer['price_per_night'])
            
            # Calculate statistics from all 24-hour prices
            if all_24h_prices:
                price_mean = np.mean(all_24h_prices)
                price_std = np.std(all_24h_prices)
            else:
                # Fallback to current prices if no history available
                offered_prices = location_offers['price_per_night']
                price_mean = offered_prices.mean()
                price_std = offered_prices.std()
            
            n_offers = len(location_offers)
            n_users = location_offers['user_id'].nunique()
            user_row = users_df[users_df['user_id'] == user_id]
            if user_row.empty:
                continue
            base_sensitivity = float(user_row.iloc[0].get('price_sensitivity', 0.5))
            
            # Adjust dynamic sensitivity based on parameters
            days_factor = 0.2 * (1 - min(days_to_go, 180)/180) if days_to_go is not None else 0.1
            price_factor = 0.3 * (price_std/(price_mean+1) if price_mean > 0 else 0)
            dynamic_sensitivity = 0.5 * base_sensitivity + days_factor + price_factor
            
            user_rows.append({
                'user_id': user_id,
                'destination': location,
                'base_price_sensitivity': round(base_sensitivity, 4),
                'dynamic_price_sensitivity': round(dynamic_sensitivity, 4),
                'num_offers': n_offers,
                'num_users': n_users,
                'price_mean': round(price_mean, 2),
                'price_std': round(price_std, 2),
                'days_to_go': days_to_go
            })
        
        out_df = pd.DataFrame(user_rows)
        out_df.to_csv(out_path, index=False)
        print(f"[DEBUG] user_dynamic_price_sensitivity.csv written to {out_path} with {len(out_df)} rows.")
        return {"message": f"user_dynamic_price_sensitivity.csv created with {len(out_df)} rows.", "csv_path": out_path}
    except Exception as e:
        print(f"[ERROR] Exception in user_dynamic_price_sensitivity_csv: {e}")
        traceback.print_exc()
        return {"error": f"Exception: {str(e)}"}

def save_market_state_csv():
    """Compute and save market state for each location in trial_sampled_offers.csv as /data/market_state_by_location.csv."""
    import pandas as pd
    import numpy as np
    import json
    import os
    offers_path = get_data_path('trial_sampled_offers.csv')
    users_path = get_data_path('enhanced_user_profiles.csv')
    out_path = get_data_path('market_state_by_location.csv')
    if not os.path.exists(offers_path) or not os.path.exists(users_path):
        return
    offers_df = pd.read_csv(offers_path)
    users_df = pd.read_csv(users_path)
    all_user_counts = offers_df.groupby('location')['user_id'].nunique()
    all_offer_counts = offers_df.groupby('location')['offer_id'].nunique()
    median_user_count = all_user_counts.median() if not all_user_counts.empty else 1
    median_offer_count = all_offer_counts.median() if not all_offer_counts.empty else 1
    rows = []
    # --- Normalization helpers ---
    def norm(val, minv, maxv):
        if maxv > minv:
            return (val - minv) / (maxv - minv)
        else:
            return 0.0
    # Precompute global min/max for normalization
    price_min, price_max = offers_df['price_per_night'].min(), offers_df['price_per_night'].max()
    days_min, days_max = offers_df['days_to_go'].min(), offers_df['days_to_go'].max()
    price_var_min, price_var_max = offers_df['price_fluctuation_variance'].min(), offers_df['price_fluctuation_variance'].max()
    comp_min, comp_max = 1, 100
    for location in offers_df['location'].unique():
        offers = offers_df[offers_df['location'] == location]
        user_ids = offers['user_id'].unique().tolist()
        users = users_df[users_df['user_id'].isin(user_ids)]
        
        # Calculate price statistics using 24-hour price history
        all_24h_prices = []
        for _, offer in offers.iterrows():
            try:
                # Parse the 24-hour price history
                price_history = json.loads(offer['price_history_24h'])
                if isinstance(price_history, list) and len(price_history) > 0:
                    all_24h_prices.extend(price_history)
            except (json.JSONDecodeError, KeyError, TypeError):
                # Fallback to current price if history is not available
                all_24h_prices.append(offer['price_per_night'])
        
        # Calculate statistics from all 24-hour prices
        if all_24h_prices:
            avg_price = np.mean(all_24h_prices)
            price_var = np.var(all_24h_prices)  # Use variance from 24h history
        else:
            # Fallback to current prices if no history available
            avg_price = offers['price_per_night'].mean()
            price_var = offers['price_fluctuation_variance'].mean()
        
        avg_days_to_go = offers['days_to_go'].mean()  # Average across users in this location
        unique_hotels = offers['hotel_id'].nunique()
        unique_partners = offers['partner_name'].nunique()
        competition_density = unique_hotels * unique_partners
        # --- Normalized signals ---
        norm_avg_price = norm(avg_price, price_min, price_max)
        norm_days_to_go = norm(avg_days_to_go, days_min, days_max)
        norm_price_var = norm(price_var, price_var_min, price_var_max)
        norm_competition_density = norm(competition_density, comp_min, comp_max)
        # --- Composite Market Demand Index ---
        w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
        demand_index = (
            w1 * norm_avg_price +
            w2 * (1 - norm_days_to_go) +
            w3 * norm_price_var +
            w4 * norm_competition_density
        )
        # --- Label logic (unchanged) ---
        user_count = len(users)
        offer_count = len(offers)
        if (demand_index > 0.66) and user_count >= median_user_count and offer_count >= median_offer_count:
            market_state_label = 'high'
        elif (demand_index < 0.33) and user_count <= median_user_count and offer_count <= median_offer_count:
            market_state_label = 'low'
        else:
            market_state_label = 'medium'
        rows.append({
            'location': location,
            'avg_price': avg_price,
            'avg_days_to_go': int(round(avg_days_to_go)),
            'avg_price_trend': price_var,  # Using price variance as trend
            'user_count': user_count,
            'offer_count': offer_count,
            'demand_index': demand_index,
            'market_state_label': market_state_label
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)

def run_bandit_simulation_and_save_csv(num_users: int = None, num_hotels: int = None, num_partners: int = None, days_to_go: int = None, days_var: int = None):
    """Run bandit simulation and save results to /data/bandit_simulation_results.csv."""
    # This is a refactor of the /run_bandit_simulation endpoint for direct call
    import pandas as pd
    import numpy as np
    import os
    import copy
    offers_path = get_data_path('trial_sampled_offers.csv')
    if not os.path.exists(offers_path):
        return
    offers_df = pd.read_csv(offers_path)
    if offers_df.empty:
        return
    
    # Use parameters to adjust simulation behavior
    clicks_per_arm = 1000 if num_users is None else max(100, 1000 // max(1, num_users // 10))
    
    results = []
    for user_id in offers_df['user_id'].unique():
        user_offers = offers_df[offers_df['user_id'] == user_id].copy()
        n_ranks = len(user_offers)
        for _, offer in user_offers.iterrows():
            offer_id = offer['offer_id']
            base_prob = float(offer['preference_score'])
            for rank in range(1, n_ranks + 1):
                rank_factor = 1.0 / rank
                true_click_prob = min(0.95, max(0.05, base_prob * rank_factor))
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
                    'probability_of_click': round(learned_probability, 4),
                    'rewards_learnt': rewards,
                    'true_click_prob': round(true_click_prob, 4),
                    'preference_score': round(base_prob, 4)
                })
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['user_id'] = results_df['user_id'].astype(str)
        results_df['offer_id'] = results_df['offer_id'].astype(str)
        results_df['rank'] = results_df['rank'].astype(int)
        results_df['probability_of_click'] = results_df['probability_of_click'].astype(float)
        results_df['true_click_prob'] = results_df['true_click_prob'].astype(float)
        results_df['preference_score'] = results_df['preference_score'].astype(float)
    csv_path = get_data_path('bandit_simulation_results.csv')
    results_df.drop(columns=['rewards_learnt']).to_csv(csv_path, index=False)

def user_market_state_csv_function():
    """Create a user_market_state.csv file mapping each user to their destination and the computed market state label."""
    import pandas as pd
    import os
    import json
    offers_path = get_data_path('trial_sampled_offers.csv')
    out_path = get_data_path('user_market_state.csv')
    if not os.path.exists(offers_path):
        return
    offers_df = pd.read_csv(offers_path)
    user_locs = offers_df[['user_id', 'location']].drop_duplicates()
    # For each location, get market state label
    from fastapi.testclient import TestClient
    client = TestClient(app)
    market_labels = {}
    for loc in user_locs['location'].unique():
        resp = client.get(f"/market_state/{loc}")
        if resp.status_code == 200:
            market_labels[loc] = resp.json().get('market_state_label', 'unknown')
        else:
            market_labels[loc] = 'unknown'
    user_locs['market_state_label'] = user_locs['location'].map(market_labels)
    user_locs.to_csv(out_path, index=False)

@app.post("/conversion_probabilities_csv")
def conversion_probabilities_csv(
    num_users: int = Query(None, ge=1, le=100),
    num_hotels: int = Query(None, ge=1, le=20),
    num_partners: int = Query(None, ge=1, le=10),
    days_to_go: int = Query(None, ge=1, le=365),
    days_var: int = Query(None, ge=1, le=30),
    min_users_per_destination: int = Query(None, ge=1, le=20)
):
    # Ensure parameters are integers, not Query objects
    if hasattr(min_users_per_destination, 'annotation'):
        min_users_per_destination = None
    if hasattr(num_users, 'annotation'):
        num_users = None
    if hasattr(num_hotels, 'annotation'):
        num_hotels = None
    if hasattr(num_partners, 'annotation'):
        num_partners = None
    if hasattr(days_to_go, 'annotation'):
        days_to_go = None
    if hasattr(days_var, 'annotation'):
        days_var = None
    """Compute and save conversion probabilities for all user-offer pairs to data/conversion_probabilities.csv."""
    import pandas as pd
    import numpy as np
    import os
    import traceback
    offers_path = get_data_path('trial_sampled_offers.csv')
    users_path = get_data_path('enhanced_user_profiles.csv')
    dps_path = get_data_path('user_dynamic_price_sensitivity.csv')
    out_path = get_data_path('conversion_probabilities.csv')
    try:
        if not (os.path.exists(offers_path) and os.path.exists(users_path) and os.path.exists(dps_path)):
            return {"error": "Required data files not found"}
        offers_df = pd.read_csv(offers_path)
        users_df = pd.read_csv(users_path)
        dps_df = pd.read_csv(dps_path)
        
        # Handle case where parameters are None (called without parameters)
        if num_users is None and num_hotels is None and num_partners is None:
            print(f"[DEBUG] No parameters provided, using existing data")
            # Use existing data without filtering
        
        # Apply minimum users per destination constraint
        if min_users_per_destination is not None and isinstance(min_users_per_destination, int):
            # Count users per destination
            dest_counts = offers_df.groupby('location')['user_id'].nunique()
            valid_destinations = dest_counts[dest_counts >= min_users_per_destination].index.tolist()
            
            # Filter offers to only include valid destinations
            offers_df = offers_df[offers_df['location'].isin(valid_destinations)]
            
            print(f"[DEBUG] Conversion CSV: After constraint filtering: {len(valid_destinations)} destinations with >= {min_users_per_destination} users")
            print(f"[DEBUG] Conversion CSV: Valid destinations: {valid_destinations}")
            print(f"[DEBUG] Conversion CSV: Users per destination: {dest_counts.to_dict()}")
        
        # Filter offers based on parameters if provided
        if num_users is not None:
            # Ensure we only process the expected number of users
            unique_users = offers_df['user_id'].unique()[:num_users]
            offers_df = offers_df[offers_df['user_id'].isin(unique_users)]
        
        results = []
        for idx, offer in offers_df.iterrows():
            user_id = offer['user_id']
            offer_id = offer['offer_id'] if 'offer_id' in offer else idx
            destination = offer['location'] if 'location' in offer else ''
            user_row = users_df[users_df['user_id'] == user_id]
            if user_row.empty:
                continue
            user = user_row.iloc[0]
            # Get dynamic price sensitivity for this user
            dps_row = dps_df[dps_df['user_id'] == user_id]
            if dps_row.empty:
                dynamic_sensitivity = 0.5
            else:
                dynamic_sensitivity = float(dps_row.iloc[0].get('dynamic_price_sensitivity', 0.5))
            # --- Conversion probability model ---
            trivago_price = offer.get('trivago_price', offer['price_per_night']) if 'trivago_price' in offer else offer['price_per_night']
            partner_price = offer['price_per_night']
            price_diff = trivago_price - partner_price
            # Hotel rating (normalized 0.5 to 1)
            hotel_rating = float(offer.get('star_rating', 3))
            hotel_rating_score = 0.5 + 0.1 * (hotel_rating - 3)  # 3-star = 0.5, 5-star = 0.7
            # Amenities match (fraction of preferred amenities present)
            preferred_amenities = str(user.get('preferred_amenities', '')).split(',') if pd.notna(user.get('preferred_amenities', '')) else []
            offer_amenities = str(offer.get('amenities', '')).split(',') if pd.notna(offer.get('amenities', '')) else []
            amenities_match = sum(1 for a in preferred_amenities if a.strip() in [b.strip() for b in offer_amenities])
            amenities_score = 0.5 + 0.1 * min(amenities_match, 5)  # up to 1.0
            # User profile: loyalty status
            loyalty = user.get('loyalty_status', '')
            loyalty_score = 1.0 if loyalty in ['Gold', 'Platinum'] else 0.8 if loyalty == 'Silver' else 0.6
            # Partner brand
            partner = offer.get('partner_name', '')
            brand_score = 1.0 if partner in ['Booking.com', 'Expedia'] else 0.9
            # --- Logistic regression style logit ---
            logit = (
                -0.5 * price_diff +
                1.5 * hotel_rating_score +
                0.8 * amenities_score +
                1.2 * brand_score +
                0.5 * loyalty_score -
                partner_price * dynamic_sensitivity * 0.01
            )
            probability_of_conversion = 1 / (1 + np.exp(-logit))
            results.append({
                'user_id': user_id,
                'offer_id': offer_id,
                'destination': destination,
                'conversion_probability': round(probability_of_conversion, 4)
            })
        out_df = pd.DataFrame(results)
        # Use relative path for data directory
        out_path = get_data_path('conversion_probabilities.csv')
        out_df.to_csv(out_path, index=False)
        print(f"[DEBUG] conversion_probabilities.csv written to {out_path}, exists: {os.path.exists(out_path)}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Parameters used: num_users={num_users}, num_hotels={num_hotels}, num_partners={num_partners}, days_to_go={days_to_go}, days_var={days_var}")
        if not os.path.exists(out_path):
            print(f"[WARNING] conversion_probabilities.csv NOT FOUND at {out_path} after writing!")
        return {"message": f"conversion_probabilities.csv created with {len(out_df)} rows.", "csv_path": out_path}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Exception: {str(e)}"}

@app.get("/conversion_probability/{user_id}/{offer_id}")
def get_conversion_probability(user_id: str, offer_id: str):
    """Return conversion probability for a user-offer pair from conversion_probabilities.csv."""
    import pandas as pd
    import os
    csv_path = get_data_path('conversion_probabilities.csv')
    if not os.path.exists(csv_path):
        return {"error": "conversion_probabilities.csv not found. Please generate it first."}
    df = pd.read_csv(csv_path)
    row = df[(df['user_id'] == user_id) & (df['offer_id'].astype(str) == str(offer_id))]
    if row.empty:
        return {"error": f"No conversion probability found for user {user_id} and offer {offer_id}"}
    return row.iloc[0].to_dict()

@app.post("/run_deterministic_optimization")
def run_deterministic_optimization():
    """Run deterministic optimization using unified optimizer"""
    try:
        # Import the unified optimizer
        from unified_optimizer import UnifiedOptimizer
        
        # Initialize optimizer
        optimizer = UnifiedOptimizer(DATA_DIR)
        
        # Load data
        if not optimizer.load_data():
            raise HTTPException(status_code=400, detail="Failed to load optimization data")
        
        # Prepare optimization data
        optimization_data = optimizer.prepare_optimization_data(max_offers=50, max_users=20)
        
        # Run deterministic optimization
        results = optimizer.run_deterministic_optimization(
            optimization_data,
            alpha=0.4,  # Revenue weight
            beta=0.3,   # User satisfaction weight
            gamma=0.3,  # Partner value weight
            num_positions=10
        )
        
        # Save results
        optimizer.save_results(results, 'deterministic_optimization_results.json')
        
        return {
            "status": "success",
            "message": "Deterministic optimization completed",
            "results": {
                "json": results,
                "csv": results.get('user_rankings', {})
            }
        }
        
    except Exception as e:
        print(f"Error in deterministic optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Error running optimization: {str(e)}")

@app.post("/run_stochastic_optimization")
def run_stochastic_optimization():
    """Run stochastic optimization using unified optimizer"""
    try:
        # Import the unified optimizer
        from unified_optimizer import UnifiedOptimizer
        
        # Initialize optimizer
        optimizer = UnifiedOptimizer(DATA_DIR)
        
        # Load data
        if not optimizer.load_data():
            raise HTTPException(status_code=400, detail="Failed to load optimization data")
        
        # Prepare optimization data
        optimization_data = optimizer.prepare_optimization_data(max_offers=50, max_users=20)
        
        # Run stochastic optimization
        results = optimizer.run_stochastic_optimization(
            optimization_data,
            num_selected=10
        )
        
        # Save results
        optimizer.save_results(results, 'stochastic_optimization_results.json')
        
        return {
            "status": "success",
            "message": "Stochastic optimization completed",
            "results": {
                "json": results,
                "csv": results.get('selected_offers', [])
            }
        }
        
    except Exception as e:
        print(f"Error in stochastic optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Error running optimization: {str(e)}")

@app.get("/optimization_results")
def get_optimization_results():
    """Get the latest optimization results"""
    try:
        # Check if optimization results exist
        results_file = get_data_path('optimization_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        else:
            return {"message": "No optimization results available. Run optimization first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading optimization results: {str(e)}")

# New endpoints for final implementation

@app.post("/get_scenario_inputs")
def get_scenario_inputs():
    """Load trial_sampled_offers.csv and return unique user_ids from the generated scenario for dropdown"""
    try:
        csv_path = get_data_path('trial_sampled_offers.csv')
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Trial sampled offers CSV not found. Please generate a scenario first.")
        
        df = pd.read_csv(csv_path)
        if df.empty:
            raise HTTPException(status_code=404, detail="No scenario data available. Please generate a scenario first.")
        
        user_ids = df['user_id'].unique().tolist()
        
        return {"user_ids": user_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scenario data: {str(e)}")

@app.post("/get_user_scenario")
def get_user_scenario(request: dict):
    """Get comprehensive scenario data for a specific user"""
    try:
        user_id = request.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Load all CSV files
        user_profiles_path = get_data_path('enhanced_user_profiles.csv')
        market_state_path = get_data_path('market_state_by_location.csv')
        bandit_results_path = get_data_path('bandit_simulation_results.csv')
        conversion_probs_path = get_data_path('conversion_probabilities.csv')
        
        # Check if files exist
        for path in [user_profiles_path, market_state_path, bandit_results_path, conversion_probs_path]:
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail=f"Required CSV file not found: {path}")
        
        # Load data
        user_profiles = pd.read_csv(user_profiles_path)
        market_state = pd.read_csv(market_state_path)
        bandit_results = pd.read_csv(bandit_results_path)
        conversion_probs = pd.read_csv(conversion_probs_path)
        
        # Get user profile
        user_profile = user_profiles[user_profiles['user_id'] == user_id]
        if user_profile.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        user_location = user_profile.iloc[0]['location']
        
        # Get market state for user's location
        market_data = market_state[market_state['location'] == user_location]
        if market_data.empty:
            # Use default market data if location not found
            market_data = market_state.iloc[0:1]
        
        # Filter bandit results for user
        user_bandit_data = bandit_results[bandit_results['user_id'] == user_id]
        
        # Filter conversion probabilities for user
        user_conversion_data = conversion_probs[conversion_probs['user_id'] == user_id]
        
        # Merge data
        merged_data = user_bandit_data.merge(
            user_conversion_data[['offer_id', 'conversion_probability']], 
            on='offer_id', 
            how='left'
        )
        
        # Add market state information
        market_info = market_data.iloc[0]
        merged_data['location'] = market_info['location']
        merged_data['market_state_label'] = market_info['market_state_label']
        merged_data['avg_price'] = market_info['avg_price']
        merged_data['avg_days_to_go'] = market_info['avg_days_to_go']
        
        # Select and order columns as specified
        result_columns = [
            'location', 'market_state_label', 'avg_price', 'avg_days_to_go',
            'offer_id', 'rank', 'probability_of_click', 'conversion_probability'
        ]
        
        # Ensure all required columns exist
        for col in result_columns:
            if col not in merged_data.columns:
                merged_data[col] = None
        
        result_data = merged_data[result_columns].fillna(0)
        
        # Convert to JSON-serializable format
        result_records = result_data.to_dict(orient='records')
        
        return {
            "user_id": user_id,
            "scenario_data": result_records,
            "market_context": {
                "location": market_info['location'],
                "market_state": market_info['market_state_label'],
                "avg_price": float(market_info['avg_price']),
                "avg_days_to_go": float(market_info['avg_days_to_go']),
                "demand_index": float(market_info['demand_index'])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing user scenario: {str(e)}")

@app.post("/rank_user")
def rank_offers_for_user(request: dict):
    """Rank offers using different strategies for a specific user"""
    try:
        user_id = request.get("user_id")
        strategy = request.get("strategy", "Greedy")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Load required data
        user_profiles_path = get_data_path('enhanced_user_profiles.csv')
        offers_path = get_data_path('enhanced_partner_offers.csv')
        hotels_path = get_data_path('enhanced_hotels.csv')
        bandit_results_path = get_data_path('bandit_simulation_results.csv')
        conversion_probs_path = get_data_path('conversion_probabilities.csv')
        market_state_path = get_data_path('market_state_by_location.csv')
        
        # Check if files exist
        for path in [user_profiles_path, offers_path, hotels_path, bandit_results_path, conversion_probs_path, market_state_path]:
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail=f"Required CSV file not found: {path}")
        
        # Load data
        user_profiles = pd.read_csv(user_profiles_path)
        offers = pd.read_csv(offers_path)
        hotels = pd.read_csv(hotels_path)
        bandit_results = pd.read_csv(bandit_results_path)
        conversion_probs = pd.read_csv(conversion_probs_path)
        market_state = pd.read_csv(market_state_path)
        
        # Get user profile
        user_profile = user_profiles[user_profiles['user_id'] == user_id]
        if user_profile.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        user_location = user_profile.iloc[0]['location']
        
        # Get user's offers and related data
        user_bandit_data = bandit_results[bandit_results['user_id'] == user_id]
        user_conversion_data = conversion_probs[conversion_probs['user_id'] == user_id]
        
        # Merge offer data
        offer_data = user_bandit_data.merge(
            user_conversion_data[['offer_id', 'conversion_probability']], 
            on='offer_id', 
            how='left'
        )
        
        # Get full offer details
        offer_details = offers[offers['offer_id'].isin(offer_data['offer_id'])]
        hotel_details = hotels[hotels['hotel_id'].isin(offer_details['hotel_id'])]
        
        # Merge all data
        complete_data = offer_data.merge(offer_details, on='offer_id', how='left')
        complete_data = complete_data.merge(hotel_details, on='hotel_id', how='left')
        
        # Get market state
        market_info = market_state[market_state['location'] == user_location]
        if market_info.empty:
            market_info = market_state.iloc[0:1]
        
        # Apply ranking strategies
        results = {}
        
        # Strategy 1: Greedy (maximize commission revenue)
        if strategy == "Greedy" or strategy == "all":
            greedy_ranking = _apply_greedy_strategy(complete_data)
            results["Greedy"] = greedy_ranking
        
        # Strategy 2: User-First (prioritize user satisfaction)
        if strategy == "User-First" or strategy == "all":
            user_first_ranking = _apply_user_first_strategy(complete_data)
            results["User-First"] = user_first_ranking
        
        # Strategy 3: Stochastic LP (optimization-based)
        if strategy == "Stochastic LP" or strategy == "all":
            stochastic_lp_ranking = _apply_stochastic_lp_strategy(complete_data)
            results["Stochastic LP"] = stochastic_lp_ranking
        
        # Strategy 4: RL Policy (adaptive based on market conditions)
        if strategy == "RL Policy" or strategy == "all":
            rl_policy_ranking = _apply_rl_policy_strategy(complete_data, market_info.iloc[0])
            results["RL Policy"] = rl_policy_ranking
        
        return {
            "user_id": user_id,
            "strategy": strategy,
            "results": results,
            "market_context": {
                "location": market_info.iloc[0]['location'],
                "market_state": market_info.iloc[0]['market_state_label'],
                "demand_index": float(market_info.iloc[0]['demand_index'])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ranking offers: {str(e)}")

def _apply_greedy_strategy(data: pd.DataFrame) -> dict:
    """Greedy strategy: maximize commission_rate * total_price * probability_of_click * conversion_probability"""
    try:
        # Calculate greedy score
        data['greedy_score'] = (
            data['commission_rate'] * 
            data['price_per_night'] * 
            data['probability_of_click'] * 
            data['conversion_probability']
        )
        
        # Sort by greedy score
        ranked_data = data.sort_values('greedy_score', ascending=False)
        
        # Calculate metrics
        expected_revenue = ranked_data['greedy_score'].sum()
        user_trust_score = _calculate_trust_score_for_ranking(ranked_data)
        
        return {
            "ranked_list": ranked_data[['offer_id', 'partner_name', 'price_per_night', 'greedy_score']].head(10).to_dict('records'),
            "expected_revenue": float(expected_revenue),
            "user_trust_score": float(user_trust_score)
        }
    except Exception as e:
        return {"error": str(e)}

def _apply_user_first_strategy(data: pd.DataFrame) -> dict:
    """User-First strategy: prioritize low price and high review score"""
    try:
        # Calculate user-first score (inverse price + review score)
        max_price = data['price_per_night'].max()
        data['user_first_score'] = (
            (1 - data['price_per_night'] / max_price) * 0.7 +  # Price factor (70% weight)
            (data['review_score'] / 10) * 0.3  # Review factor (30% weight)
        )
        
        # Sort by user-first score
        ranked_data = data.sort_values('user_first_score', ascending=False)
        
        # Calculate metrics
        expected_revenue = (ranked_data['commission_rate'] * 
                          ranked_data['price_per_night'] * 
                          ranked_data['probability_of_click'] * 
                          ranked_data['conversion_probability']).sum()
        user_trust_score = _calculate_trust_score_for_ranking(ranked_data)
        
        return {
            "ranked_list": ranked_data[['offer_id', 'partner_name', 'price_per_night', 'user_first_score']].head(10).to_dict('records'),
            "expected_revenue": float(expected_revenue),
            "user_trust_score": float(user_trust_score)
        }
    except Exception as e:
        return {"error": str(e)}

def _apply_stochastic_lp_strategy(data: pd.DataFrame) -> dict:
    """Stochastic LP strategy: optimization-based ranking"""
    try:
        # Prepare data for optimization
        n_offers = len(data)
        n_positions = min(10, n_offers)
        
        # Create optimization problem
        prob = pulp.LpProblem("Hotel_Ranking_Optimization", pulp.LpMaximize)
        
        # Decision variables: x[i][j] = 1 if offer i is at position j
        x = {}
        for i in range(n_offers):
            for j in range(n_positions):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
        
        # Objective function: maximize expected revenue with position weights
        objective = 0
        for i in range(n_offers):
            for j in range(n_positions):
                position_weight = 1.0 / (j + 1)  # Higher positions get more weight
                expected_value = (
                    data.iloc[i]['commission_rate'] * 
                    data.iloc[i]['price_per_night'] * 
                    data.iloc[i]['probability_of_click'] * 
                    data.iloc[i]['conversion_probability'] * 
                    position_weight
                )
                objective += expected_value * x[i, j]
        
        prob += objective
        
        # Constraints
        # Each position can have at most one offer
        for j in range(n_positions):
            prob += pulp.lpSum([x[i, j] for i in range(n_offers)]) <= 1
        
        # Each offer can be in at most one position
        for i in range(n_offers):
            prob += pulp.lpSum([x[i, j] for j in range(n_positions)]) <= 1
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        if prob.status == pulp.LpStatusOptimal:
            ranking = {}
            for i in range(n_offers):
                for j in range(n_positions):
                    if x[i, j].varValue == 1:
                        ranking[j] = i
            
            # Create ranked list
            ranked_offers = []
            for pos in sorted(ranking.keys()):
                offer_idx = ranking[pos]
                offer_data = data.iloc[offer_idx]
                ranked_offers.append({
                    'offer_id': offer_data['offer_id'],
                    'partner_name': offer_data['partner_name'],
                    'price_per_night': float(offer_data['price_per_night']),
                    'position': pos + 1
                })
            
            # Calculate metrics
            expected_revenue = pulp.value(prob.objective)
            user_trust_score = _calculate_trust_score_for_ranking(data.iloc[list(ranking.values())])
            
            return {
                "ranked_list": ranked_offers,
                "expected_revenue": float(expected_revenue),
                "user_trust_score": float(user_trust_score)
            }
        else:
            return {"error": "Optimization failed"}
            
    except Exception as e:
        return {"error": str(e)}

def _apply_rl_policy_strategy(data: pd.DataFrame, market_info: pd.Series) -> dict:
    """RL Policy strategy: adaptive based on market conditions"""
    try:
        market_state = market_info['market_state_label']
        
        if market_state == 'high':
            # High demand: use greedy strategy
            return _apply_greedy_strategy(data)
        elif market_state == 'medium':
            # Medium demand: use balanced approach
            data['rl_score'] = (
                data['commission_rate'] * 0.4 +
                (1 - data['price_per_night'] / data['price_per_night'].max()) * 0.4 +
                (data['review_score'] / 10) * 0.2
            )
        else:
            # Low demand: prioritize user satisfaction
            return _apply_user_first_strategy(data)
        
        # For medium demand, use custom scoring
        ranked_data = data.sort_values('rl_score', ascending=False)
        
        # Calculate metrics
        expected_revenue = (ranked_data['commission_rate'] * 
                          ranked_data['price_per_night'] * 
                          ranked_data['probability_of_click'] * 
                          ranked_data['conversion_probability']).sum()
        user_trust_score = _calculate_trust_score_for_ranking(ranked_data)
        
        return {
            "ranked_list": ranked_data[['offer_id', 'partner_name', 'price_per_night', 'rl_score']].head(10).to_dict('records'),
            "expected_revenue": float(expected_revenue),
            "user_trust_score": float(user_trust_score)
        }
        
    except Exception as e:
        return {"error": str(e)}

def _calculate_trust_score_for_ranking(data: pd.DataFrame) -> float:
    """Calculate user trust score for a ranking"""
    try:
        if len(data) == 0:
            return 0.0
        
        # Price competitiveness (lower price = higher trust)
        min_price = data['price_per_night'].min()
        max_price = data['price_per_night'].max()
        if max_price > min_price:
            price_score = 100 * (1 - (data['price_per_night'].mean() - min_price) / (max_price - min_price))
        else:
            price_score = 100
        
        # Brand trust
        brand_trust_scores = {
            "Booking.com": 90, "Expedia": 85, "Hotels.com": 80,
            "Agoda": 75, "HotelDirect": 70
        }
        brand_score = data['partner_name'].map(lambda x: brand_trust_scores.get(x, 65)).mean()
        
        # Review score
        review_score = data['review_score'].mean() * 10
        
        # Weighted average
        trust_score = (price_score * 0.4 + brand_score * 0.3 + review_score * 0.3)
        return min(100.0, max(0.0, trust_score))
        
    except Exception as e:
        return 50.0  # Default trust score

@app.get("/user_dynamic_price_sensitivity_data")
def get_user_dynamic_price_sensitivity_data():
    """Return the contents of user_dynamic_price_sensitivity.csv as JSON"""
    import os
    import pandas as pd
    import numpy as np
    try:
        file_path = get_data_path('user_dynamic_price_sensitivity.csv')
        if not os.path.exists(file_path):
            return {"message": "No user_dynamic_price_sensitivity.csv found", "data": []}
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
        raise HTTPException(status_code=500, detail=f"Error reading user_dynamic_price_sensitivity.csv: {str(e)}")

@app.get("/conversion_probabilities_data")
def get_conversion_probabilities_data():
    """Return the contents of conversion_probabilities.csv as JSON"""
    import os
    import pandas as pd
    import numpy as np
    try:
        file_path = get_data_path('conversion_probabilities.csv')
        if not os.path.exists(file_path):
            return {"message": "No conversion_probabilities.csv found", "data": []}
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
        raise HTTPException(status_code=500, detail=f"Error reading conversion_probabilities.csv: {str(e)}")

@app.get("/conversion_probability/{user_id}/{offer_id}")
def get_conversion_probability(user_id: str, offer_id: str):
    """Return conversion probability for a user-offer pair from conversion_probabilities.csv."""
    import pandas as pd
    import os
    csv_path = get_data_path('conversion_probabilities.csv')
    if not os.path.exists(csv_path):
        return {"error": "conversion_probabilities.csv not found. Please generate it first."}
    df = pd.read_csv(csv_path)
    row = df[(df['user_id'] == user_id) & (df['offer_id'].astype(str) == str(offer_id))]
    if row.empty:
        return {"error": f"No conversion probability found for user {user_id} and offer {offer_id}"}
    return row.iloc[0].to_dict()

@app.post("/rank")
def rank_offers_with_budget_constraints(
    alpha: float = Query(0.4, ge=0.0, le=1.0, description="Weight for trivago income"),
    beta: float = Query(0.3, ge=0.0, le=1.0, description="Weight for user satisfaction"),
    gamma: float = Query(0.3, ge=0.0, le=1.0, description="Weight for partner conversion value"),
    num_positions: int = Query(10, ge=1, le=50, description="Number of ranking positions")
):
    """Rank offers using multi-objective optimization with budget constraints."""
    try:
        print(f"[DEBUG] rank_offers_with_budget_constraints called with: alpha={alpha}, beta={beta}, gamma={gamma}, num_positions={num_positions}")
        print(f"[DEBUG] Parameter types: alpha={type(alpha)}, beta={type(beta)}, gamma={type(gamma)}, num_positions={type(num_positions)}")
        import pandas as pd
        import numpy as np
        
        # Load current offer data
        offers_path = get_data_path('trial_sampled_offers.csv')
        if not os.path.exists(offers_path):
            return {"error": "No offers data available. Please run sampling first."}
        
        offers_df = pd.read_csv(offers_path)
        
        # Limit number of offers for optimization
        n_offers = min(len(offers_df), 50)  # Limit for performance
        offers_subset = offers_df.head(n_offers).copy()
        
        # Calculate user satisfaction scores based on hotel quality and price
        offers_subset['user_satisfaction_score'] = (
            offers_subset['star_rating'] * 0.3 +
            (10 - offers_subset['price_per_night'] / 50) * 0.4 +  # Lower price = higher satisfaction
            offers_subset['review_score'] * 0.3
        )
        
        # Calculate conversion probabilities
        offers_subset['conversion_probability'] = (
            0.1 +  # Base conversion rate
            offers_subset['star_rating'] * 0.05 +
            offers_subset['review_score'] * 0.02 -
            offers_subset['price_per_night'] * 0.0001
        ).clip(0.01, 0.95)
        
        # Add missing columns that the optimization expects
        import random
        import subprocess
        if 'remaining_budget' not in offers_subset.columns:
            # Create a reasonable budget based on partner and hotel characteristics
            offers_subset['remaining_budget'] = offers_subset['price_per_night'] * random.uniform(10, 50)
        
        if 'commission_rate' not in offers_subset.columns:
            # Use a default commission rate
            offers_subset['commission_rate'] = 0.15
        
        if 'cost_per_click_bid' not in offers_subset.columns:
            # Calculate cost per click based on price
            offers_subset['cost_per_click_bid'] = offers_subset['price_per_night'] * 0.01
        
        # Position-based click-through rates (decreasing with rank)
        position_ctr = [1.0 / (1 + 0.5 * i) for i in range(num_positions)]
        
        # Create optimization data
        mzn_data = f"""
n_offers = {n_offers};
n_positions = {num_positions};
alpha = {alpha};
beta = {beta};
gamma = {gamma};

commission_rates = {list(offers_subset['commission_rate'])};
cost_per_click_bids = {list(offers_subset['cost_per_click_bid'])};
user_satisfaction_scores = {list(offers_subset['user_satisfaction_score'])};
conversion_probabilities = {list(offers_subset['conversion_probability'])};
partner_budgets = {list(offers_subset['remaining_budget'])};
partner_ids = {list(range(1, n_offers + 1))};  % Simplified: each offer is from different partner
position_ctr = {position_ctr};
"""
        
        # Use PuLP for linear programming optimization
        print(f"[DEBUG] Using PuLP optimization solver")
        return _pulp_optimization(offers_subset, alpha, beta, gamma, num_positions)
            
    except Exception as e:
        print(f"[ERROR] Exception in rank_offers_with_budget_constraints: {e}")
        traceback.print_exc()
        return {"error": f"Exception: {str(e)}"}

@app.post("/calculate_shapley_values")
def calculate_shapley_values():
    """Calculate Shapley values for each partner's contribution to total revenue."""
    try:
        import pandas as pd
        import numpy as np
        
        # Load current offer data
        offers_path = get_data_path('trial_sampled_offers.csv')
        if not os.path.exists(offers_path):
            return {"error": "No offers data available. Please run sampling first."}
        
        offers_df = pd.read_csv(offers_path)
        
        # Add missing columns if they don't exist
        if 'commission_rate' not in offers_df.columns:
            offers_df['commission_rate'] = 0.15
        if 'cost_per_click_bid' not in offers_df.columns:
            offers_df['cost_per_click_bid'] = offers_df['price_per_night'] * 0.01
        if 'remaining_budget' not in offers_df.columns:
            offers_df['remaining_budget'] = offers_df['price_per_night'] * random.uniform(10, 50)
        
        # Group offers by partner
        partner_offers = offers_df.groupby('partner_name').agg({
            'commission_rate': 'mean',
            'cost_per_click_bid': 'sum',
            'remaining_budget': 'first',
            'offer_id': 'count'
        }).reset_index()
        
        partner_offers.columns = ['partner_name', 'avg_commission_rate', 'total_cpc_bid', 'budget', 'num_offers']
        
        # Calculate expected revenue for each partner
        partner_offers['expected_revenue'] = (
            partner_offers['avg_commission_rate'] * 
            partner_offers['total_cpc_bid'] * 
            partner_offers['num_offers']
        )
        
        # Calculate Shapley values using Monte Carlo approximation
        n_partners = len(partner_offers)
        shapley_values = {partner: 0.0 for partner in partner_offers['partner_name']}
        
        # Monte Carlo sampling for Shapley value calculation
        n_samples = min(1000, 2**n_partners)  # Limit computational complexity
        
        for _ in range(n_samples):
            # Random coalition
            coalition_size = np.random.randint(1, n_partners + 1)
            coalition = np.random.choice(partner_offers['partner_name'], coalition_size, replace=False)
            
            # Calculate coalition value
            coalition_value = partner_offers[
                partner_offers['partner_name'].isin(coalition)
            ]['expected_revenue'].sum()
            
            # Calculate marginal contribution for each partner in coalition
            for partner in coalition:
                coalition_without_partner = [p for p in coalition if p != partner]
                if coalition_without_partner:
                    value_without_partner = partner_offers[
                        partner_offers['partner_name'].isin(coalition_without_partner)
                    ]['expected_revenue'].sum()
                else:
                    value_without_partner = 0
                
                marginal_contribution = coalition_value - value_without_partner
                shapley_values[partner] += marginal_contribution / n_samples
        
        # Normalize Shapley values
        total_shapley = sum(shapley_values.values())
        if total_shapley > 0:
            shapley_values = {k: v / total_shapley for k, v in shapley_values.items()}
        
        # Prepare results
        results = []
        for _, partner in partner_offers.iterrows():
            results.append({
                "partner_name": partner['partner_name'],
                "num_offers": int(partner['num_offers']),
                "avg_commission_rate": round(partner['avg_commission_rate'], 4),
                "total_cpc_bid": round(partner['total_cpc_bid'], 2),
                "budget": round(partner['budget'], 2),
                "expected_revenue": round(partner['expected_revenue'], 2),
                "shapley_value": round(shapley_values[partner['partner_name']], 4),
                "contribution_percentage": round(shapley_values[partner['partner_name']] * 100, 2)
            })
        
        # Sort by Shapley value
        results.sort(key=lambda x: x['shapley_value'], reverse=True)
        
        return {
            "shapley_values": results,
            "total_expected_revenue": round(partner_offers['expected_revenue'].sum(), 2),
            "total_shapley_value": round(sum(shapley_values.values()), 4),
            "parameters": {
                "n_partners": n_partners,
                "n_samples": n_samples
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Exception in calculate_shapley_values: {e}")
        traceback.print_exc()
        return {"error": f"Exception: {str(e)}"}

# Import DQN agent functions
from dqn_agent import get_dqn_agent, calculate_reward

# Add these new endpoints after the existing ones

@app.post("/select_strategic_policy")
def select_strategic_policy():
    """Select optimal policy using DQN based on current market state"""
    try:
        import pandas as pd
        import numpy as np
        
        # Get current market state
        offers_path = get_data_path('trial_sampled_offers.csv')
        if not os.path.exists(offers_path):
            return {"error": "No offers data available. Please run sampling first."}
        
        offers_df = pd.read_csv(offers_path)
        
        # Calculate market state metrics
        market_demand = len(offers_df)
        days_to_go = int(round(offers_df['days_to_go'].mean())) if 'days_to_go' in offers_df.columns else 30  # Average across all users
        competition_density = len(offers_df['partner_name'].unique()) if 'partner_name' in offers_df.columns else 5
        
        # Calculate price volatility
        if 'price_per_night' in offers_df.columns:
            price_volatility = offers_df['price_per_night'].std() / offers_df['price_per_night'].mean()
        else:
            price_volatility = 0.1
        
        # Calculate satisfaction trend (simplified)
        satisfaction_trend = 0.0  # Would be calculated from historical data
        
        # Calculate budget utilization
        if 'remaining_budget' in offers_df.columns and 'partner_marketing_budget' in offers_df.columns:
            total_budget = offers_df['partner_marketing_budget'].sum()
            used_budget = total_budget - offers_df['remaining_budget'].sum()
            budget_utilization = (used_budget / total_budget) * 100 if total_budget > 0 else 0
        else:
            budget_utilization = 0
        
        # Create market state
        market_state = {
            'market_demand': market_demand,
            'days_to_go': days_to_go,
            'competition_density': competition_density,
            'price_volatility': price_volatility,
            'satisfaction_trend': satisfaction_trend,
            'budget_utilization': budget_utilization
        }
        
        # Get DQN agent and select policy
        dqn_agent = get_dqn_agent()
        state_vector = dqn_agent.get_state_vector(market_state)
        action = dqn_agent.select_action(state_vector, training=True)
        policy_weights = dqn_agent.get_policy_weights(action)
        
        return {
            "selected_policy": {
                "action_id": action,
                "policy_name": policy_weights["name"],
                "weights": policy_weights
            },
            "market_state": market_state,
            "state_vector": state_vector.tolist(),
            "epsilon": dqn_agent.epsilon
        }
        
    except Exception as e:
        print(f"[ERROR] Exception in select_strategic_policy: {e}")
        traceback.print_exc()
        return {"error": f"Exception: {str(e)}"}

@app.post("/train_rl_agent")
def train_rl_agent():
    """Train the RL agent using current optimization results"""
    try:
        print(f"[DEBUG] Starting RL agent training...")
        import pandas as pd
        import numpy as np
        
        # Get current market state
        offers_path = get_data_path('trial_sampled_offers.csv')
        if not os.path.exists(offers_path):
            return {"error": "No offers data available. Please run sampling first."}
        
        offers_df = pd.read_csv(offers_path)
        
        # Calculate market state (same as in select_strategic_policy)
        market_demand = len(offers_df)
        days_to_go = int(round(offers_df['days_to_go'].mean())) if 'days_to_go' in offers_df.columns else 30  # Average across all users
        competition_density = len(offers_df['partner_name'].unique()) if 'partner_name' in offers_df.columns else 5
        
        if 'price_per_night' in offers_df.columns:
            price_volatility = offers_df['price_per_night'].std() / offers_df['price_per_night'].mean()
        else:
            price_volatility = 0.1
        
        satisfaction_trend = 0.0
        
        if 'remaining_budget' in offers_df.columns and 'partner_marketing_budget' in offers_df.columns:
            total_budget = offers_df['partner_marketing_budget'].sum()
            used_budget = total_budget - offers_df['remaining_budget'].sum()
            budget_utilization = (used_budget / total_budget) * 100 if total_budget > 0 else 0
        else:
            budget_utilization = 0
        
        market_state = {
            'market_demand': market_demand,
            'days_to_go': days_to_go,
            'competition_density': competition_density,
            'price_volatility': price_volatility,
            'satisfaction_trend': satisfaction_trend,
            'budget_utilization': budget_utilization
        }
        
        # Get DQN agent
        dqn_agent = get_dqn_agent()
        current_state = dqn_agent.get_state_vector(market_state)
        
        # Select action
        action = dqn_agent.select_action(current_state, training=True)
        
        # Run optimization with selected policy
        policy_weights = dqn_agent.get_policy_weights(action)
        
        # Call the ranking function directly with policy weights
        # Create a mock request object for the function call
        class MockRequest:
            def __init__(self, alpha, beta, gamma, num_positions):
                self.query_params = {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "num_positions": num_positions
                }
        
        mock_request = MockRequest(
            policy_weights["alpha"],
            policy_weights["beta"],
            policy_weights["gamma"],
            10
        )
        
        # Call the ranking function directly
        print(f"[DEBUG] Calling optimization with weights: alpha={policy_weights['alpha']}, beta={policy_weights['beta']}, gamma={policy_weights['gamma']}")
        try:
            ranking_result = rank_offers_with_budget_constraints(
                alpha=policy_weights["alpha"],
                beta=policy_weights["beta"],
                gamma=policy_weights["gamma"],
                num_positions=10
            )
            print(f"[DEBUG] Optimization result: {ranking_result}")
        except Exception as e:
            print(f"[ERROR] Optimization failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Optimization failed with exception: {str(e)}"}
        
        if "error" in ranking_result:
            return {"error": f"Optimization failed: {ranking_result['error']}"}
        
        # Calculate reward
        reward = calculate_reward(ranking_result.get("objectives", {}))
        
        # Simulate next state (in practice, this would be the actual next state)
        next_market_state = market_state.copy()
        next_market_state['market_demand'] = max(0, market_demand - 1)  # Simplified
        next_state = dqn_agent.get_state_vector(next_market_state)
        
        # Train the agent with multiple episodes for realistic training
        print(f"[DEBUG] Starting realistic RL training with multiple episodes...")
        
        # Run multiple training episodes
        num_episodes = 50  # More realistic training
        total_loss = 0
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Simulate different market conditions for each episode
            episode_market_state = market_state.copy()
            episode_market_state['market_demand'] = max(10, market_demand + np.random.randint(-20, 20))
            episode_market_state['days_to_go'] = max(1, days_to_go + np.random.randint(-10, 10))
            episode_market_state['competition_density'] = max(1, competition_density + np.random.randint(-2, 2))
            
            episode_state = dqn_agent.get_state_vector(episode_market_state)
            episode_action = dqn_agent.select_action(episode_state, training=True)
            
            # Run optimization for this episode
            episode_policy_weights = dqn_agent.get_policy_weights(episode_action)
            episode_ranking_result = rank_offers_with_budget_constraints(
                alpha=episode_policy_weights["alpha"],
                beta=episode_policy_weights["beta"],
                gamma=episode_policy_weights["gamma"],
                num_positions=10
            )
            
            if "error" not in episode_ranking_result:
                episode_reward = calculate_reward(episode_ranking_result.get("objectives", {}))
                episode_rewards.append(episode_reward)
                
                # Simulate next state
                episode_next_state = episode_market_state.copy()
                episode_next_state['market_demand'] = max(0, episode_market_state['market_demand'] - 1)
                episode_next_state_vector = dqn_agent.get_state_vector(episode_next_state)
                
                # Train the agent
                episode_loss = dqn_agent.step(episode_state, episode_action, episode_reward, episode_next_state_vector, done=False)
                if episode_loss is not None:
                    total_loss += episode_loss
        
        # Save model
        model_path = get_data_path('dqn_model.pth')
        dqn_agent.save_model(model_path)
        
        avg_loss = total_loss / num_episodes if num_episodes > 0 else 0
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        print(f"[DEBUG] RL Training completed successfully!")
        print(f"[DEBUG] Episodes trained: {num_episodes}")
        print(f"[DEBUG] Final action selected: {action}")
        print(f"[DEBUG] Policy: {policy_weights['name']}")
        print(f"[DEBUG] Average reward: {avg_reward:.4f}")
        print(f"[DEBUG] Average loss: {avg_loss:.4f}")
        print(f"[DEBUG] Final epsilon: {dqn_agent.epsilon:.4f}")
        print(f"[DEBUG] Model saved to: {model_path}")
        
        return {
            "training_result": {
                "action_selected": action,
                "policy_name": policy_weights["name"],
                "reward": avg_reward,
                "loss": avg_loss,
                "epsilon": dqn_agent.epsilon
            },
            "optimization_results": ranking_result,
            "market_state": market_state
        }
        
    except Exception as e:
        print(f"[ERROR] Exception in train_rl_agent: {e}")
        traceback.print_exc()
        return {"error": f"Exception: {str(e)}"}

def _pulp_optimization(offers_df, alpha, beta, gamma, num_positions):
    """Use PuLP for linear programming optimization"""
    try:
        import pulp
        
        # Create optimization problem
        prob = pulp.LpProblem("Hotel_Ranking_Optimization", pulp.LpMaximize)
        
        # Decision variables: x[i] = 1 if offer i is selected, 0 otherwise
        n_offers = len(offers_df)
        x = pulp.LpVariable.dicts("offer", range(n_offers), cat='Binary')
        
        # Objective function: maximize weighted sum of objectives
        trivago_income = pulp.lpSum([
            alpha * offers_df.iloc[i]['commission_rate'] * offers_df.iloc[i]['cost_per_click_bid'] * x[i]
            for i in range(n_offers)
        ])
        
        user_satisfaction = pulp.lpSum([
            beta * offers_df.iloc[i]['user_satisfaction_score'] * x[i]
            for i in range(n_offers)
        ])
        
        partner_conversion_value = pulp.lpSum([
            gamma * offers_df.iloc[i]['conversion_probability'] * offers_df.iloc[i]['price_per_night'] * x[i]
            for i in range(n_offers)
        ])
        
        prob += trivago_income + user_satisfaction + partner_conversion_value
        
        # Constraints
        # 1. Select exactly num_positions offers
        prob += pulp.lpSum([x[i] for i in range(n_offers)]) == num_positions
        
        # 2. Budget constraints (if available)
        if 'remaining_budget' in offers_df.columns:
            prob += pulp.lpSum([
                offers_df.iloc[i]['cost_per_click_bid'] * x[i]
                for i in range(n_offers)
            ]) <= offers_df['remaining_budget'].sum()
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if prob.status != pulp.LpStatusOptimal:
            print(f"[DEBUG] PuLP optimization failed, using fallback")
            return _simple_optimization_fallback(offers_df, alpha, beta, gamma, num_positions)
        
        # Extract solution
        selected_offers = []
        for i in range(n_offers):
            if x[i].value() == 1:
                selected_offers.append(i)
        
        # Calculate objectives
        trivago_income_val = sum([
            alpha * offers_df.iloc[i]['commission_rate'] * offers_df.iloc[i]['cost_per_click_bid']
            for i in selected_offers
        ])
        
        user_satisfaction_val = sum([
            beta * offers_df.iloc[i]['user_satisfaction_score']
            for i in selected_offers
        ]) / len(selected_offers) if selected_offers else 0
        
        partner_conversion_val = sum([
            gamma * offers_df.iloc[i]['conversion_probability'] * offers_df.iloc[i]['price_per_night']
            for i in selected_offers
        ])
        
        total_objective = trivago_income_val + user_satisfaction_val + partner_conversion_val
        
        # Create ranked offers list
        ranked_offers = []
        for pos, offer_idx in enumerate(selected_offers):
            offer = offers_df.iloc[offer_idx]
            ranked_offers.append({
                "position": pos + 1,
                "offer_id": offer['offer_id'],
                "hotel_id": offer.get('hotel_id', f"hotel_{offer_idx}"),
                "partner_name": offer['partner_name'],
                "price_per_night": offer['price_per_night'],
                "commission_rate": offer['commission_rate'],
                "cost_per_click_bid": offer['cost_per_click_bid'],
                "user_satisfaction_score": offer['user_satisfaction_score'],
                "conversion_probability": offer['conversion_probability'],
                "remaining_budget": offer.get('remaining_budget', 0)
            })
        
        return {
            "ranking": ranked_offers,
            "objectives": {
                "trivago_income": trivago_income_val,
                "user_satisfaction": user_satisfaction_val,
                "partner_conversion_value": partner_conversion_val,
                "total_objective": total_objective
            },
            "weights": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            },
            "parameters": {
                "n_offers": n_offers,
                "n_positions": num_positions
            }
        }
        
    except Exception as e:
        print(f"[ERROR] PuLP optimization failed: {e}")
        return _simple_optimization_fallback(offers_df, alpha, beta, gamma, num_positions)

def _simple_optimization_fallback(offers_df, alpha, beta, gamma, num_positions):
    """Simple optimization fallback when PuLP optimization fails"""
    try:
        # Sort offers by weighted objective
        offers_df['weighted_score'] = (
            alpha * offers_df['commission_rate'] * offers_df['cost_per_click_bid'] +
            beta * offers_df['user_satisfaction_score'] +
            gamma * offers_df['conversion_probability'] * offers_df['price_per_night']
        )
        
        # Get top offers
        top_offers = offers_df.nlargest(num_positions, 'weighted_score')
        
        # Calculate objectives
        trivago_income = (top_offers['commission_rate'] * top_offers['cost_per_click_bid']).sum()
        user_satisfaction = top_offers['user_satisfaction_score'].mean()
        partner_conversion_value = (top_offers['conversion_probability'] * top_offers['price_per_night']).sum()
        total_objective = alpha * trivago_income + beta * user_satisfaction + gamma * partner_conversion_value
        
        # Create ranking
        ranking = list(range(1, len(top_offers) + 1))
        
        # Create ranked offers list
        ranked_offers = []
        for i, (_, offer) in enumerate(top_offers.iterrows()):
            ranked_offers.append({
                "position": i + 1,
                "offer_id": offer['offer_id'],
                "hotel_id": offer.get('hotel_id', f"hotel_{i}"),
                "partner_name": offer['partner_name'],
                "price_per_night": offer['price_per_night'],
                "commission_rate": offer['commission_rate'],
                "cost_per_click_bid": offer['cost_per_click_bid'],
                "user_satisfaction_score": offer['user_satisfaction_score'],
                "conversion_probability": offer['conversion_probability'],
                "remaining_budget": offer['remaining_budget']
            })
        
        return {
            "ranking": ranked_offers,
            "objectives": {
                "trivago_income": trivago_income,
                "user_satisfaction": user_satisfaction,
                "partner_conversion_value": partner_conversion_value,
                "total_objective": total_objective
            },
            "weights": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            },
            "parameters": {
                "n_offers": len(offers_df),
                "n_positions": num_positions
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Fallback optimization failed: {e}")
        return {"error": f"Fallback optimization failed: {str(e)}"}

@app.get("/data_status")
def get_data_status():
    """Get status of all data files in the /data directory"""
    import os
    from datetime import datetime
    
    data_files = [
        "enhanced_user_profiles.csv",
        "enhanced_hotels.csv", 
        "enhanced_partner_offers.csv",
        "trial_sampled_offers.csv",
        "bandit_simulation_results.csv",
        "conversion_probabilities.csv",
        "user_dynamic_price_sensitivity.csv",
        "user_market_state.csv",
        "market_state_by_location.csv",
        "policy_heatmap_debug.json",
        "dqn_model.pth",
        "deterministic_optimization_results.json",
        "stochastic_optimization_results.json",
        "optimization_results.json"
    ]
    
    status = {}
    for filename in data_files:
        file_path = get_data_path(filename)
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            status[filename] = {
                "exists": True,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_path": file_path
            }
        else:
            status[filename] = {
                "exists": False,
                "size_bytes": 0,
                "size_mb": 0,
                "last_modified": None,
                "file_path": file_path
            }
    
    return {
        "data_directory": DATA_DIR,
        "total_files": len(data_files),
        "existing_files": sum(1 for info in status.values() if info["exists"]),
        "total_size_mb": sum(info["size_mb"] for info in status.values() if info["exists"]),
        "files": status
    }

@app.get("/get_policy_heatmap")
def get_policy_heatmap():
    """Get policy heatmap data for visualization"""
    try:
        # Generate sample market scenarios
        market_scenarios = []
        
        for demand in [10, 30, 50, 70, 90]:
            for days in [7, 30, 90, 180]:
                for competition in [3, 5, 8, 12]:
                    market_scenarios.append({
                        'market_demand': demand,
                        'days_to_go': days,
                        'competition_density': competition,
                        'price_volatility': 0.1,
                        'satisfaction_trend': 0.0,
                        'budget_utilization': 50.0
                    })
        
        # Get DQN agent and generate heatmap
        dqn_agent = get_dqn_agent()
        heatmap_data = dqn_agent.get_policy_heatmap(market_scenarios)
        
        # Save heatmap data to file for debugging
        import json
        heatmap_file = get_data_path('policy_heatmap_debug.json')
        with open(heatmap_file, 'w') as f:
            json.dump(heatmap_data, f, indent=2, default=str)
        print(f"[DEBUG] Policy heatmap data saved to: {heatmap_file}")
        print(f"[DEBUG] Heatmap data contains {len(heatmap_data.get('heatmap_data', []))} scenarios")
        
        return heatmap_data
        
    except Exception as e:
        print(f"[ERROR] Exception in get_policy_heatmap: {e}")
        traceback.print_exc()
        return {"error": f"Exception: {str(e)}"}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compute_user_market_state":
        compute_and_save_user_market_state()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

