import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import uuid
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

