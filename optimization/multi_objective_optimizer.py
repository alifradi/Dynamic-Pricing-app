"""
Multi-Objective Optimization Module for Hotel Ranking
Implements various optimization strategies for hotel offer ranking
"""

import numpy as np
import pulp
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class OptimizationObjective:
    """Represents a single optimization objective"""
    name: str
    weight: float
    maximize: bool = True
    
@dataclass
class OptimizationConstraint:
    """Represents an optimization constraint"""
    name: str
    constraint_type: str  # 'equality', 'inequality_le', 'inequality_ge'
    value: float

class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for hotel ranking using linear programming
    
    Objectives:
    1. Likelihood of conversion to book after an offer has been clicked
    2. Price consistency between partner and trivago showed price  
    3. Expected bid of collaborators
    """
    
    def __init__(self):
        self.objectives = []
        self.constraints = []
        
    def add_objective(self, name: str, weight: float, maximize: bool = True):
        """Add an optimization objective"""
        self.objectives.append(OptimizationObjective(name, weight, maximize))
        
    def add_constraint(self, name: str, constraint_type: str, value: float):
        """Add an optimization constraint"""
        self.constraints.append(OptimizationConstraint(name, constraint_type, value))
        
    def optimize_hotel_ranking(self, offers: List[Dict], customer_behavior: Dict, 
                             market_conditions: Dict, max_positions: int = 10) -> Dict[str, Any]:
        """
        Optimize hotel ranking using multi-objective linear programming
        
        Args:
            offers: List of hotel offers with pricing and partner information
            customer_behavior: Customer behavior parameters
            market_conditions: Market condition parameters
            max_positions: Maximum number of positions to rank
            
        Returns:
            Dictionary with optimization results
        """
        try:
            n_offers = len(offers)
            n_positions = min(max_positions, n_offers)
            
            # Create the optimization problem
            prob = pulp.LpProblem("HotelRankingOptimization", pulp.LpMaximize)
            
            # Decision variables: x[i][j] = 1 if offer i is placed at position j
            x = {}
            for i in range(n_offers):
                for j in range(n_positions):
                    x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
            
            # Calculate objective components for each offer
            conversion_scores = self._calculate_conversion_likelihood(offers, customer_behavior)
            price_consistency_scores = self._calculate_price_consistency(offers)
            bid_competitiveness_scores = self._calculate_bid_competitiveness(offers)
            
            # Position weights (higher positions get more weight)
            position_weights = [1.0 / np.log2(j + 2) for j in range(n_positions)]
            
            # Multi-objective function
            objective_expr = 0
            
            # Objective 1: Conversion Likelihood
            conversion_weight = next((obj.weight for obj in self.objectives if obj.name == "conversion"), 0.4)
            for i in range(n_offers):
                for j in range(n_positions):
                    objective_expr += conversion_weight * conversion_scores[i] * position_weights[j] * x[i, j]
            
            # Objective 2: Price Consistency
            consistency_weight = next((obj.weight for obj in self.objectives if obj.name == "price_consistency"), 0.3)
            for i in range(n_offers):
                for j in range(n_positions):
                    objective_expr += consistency_weight * price_consistency_scores[i] * position_weights[j] * x[i, j]
            
            # Objective 3: Expected Bid Revenue
            bid_weight = next((obj.weight for obj in self.objectives if obj.name == "bid_revenue"), 0.3)
            for i in range(n_offers):
                for j in range(n_positions):
                    objective_expr += bid_weight * bid_competitiveness_scores[i] * position_weights[j] * x[i, j]
            
            prob += objective_expr
            
            # Constraints
            
            # Constraint 1: Each position can have at most one offer
            for j in range(n_positions):
                prob += pulp.lpSum([x[i, j] for i in range(n_offers)]) <= 1, f"Position_{j}_unique"
            
            # Constraint 2: Each offer can be in at most one position
            for i in range(n_offers):
                prob += pulp.lpSum([x[i, j] for j in range(n_positions)]) <= 1, f"Offer_{i}_unique"
            
            # Constraint 3: Price competitiveness constraint
            min_price = min(offer['price_per_night'] for offer in offers)
            max_allowed_price = min_price * 1.5  # Top offer shouldn't be more than 50% above minimum
            
            for i in range(n_offers):
                if offers[i]['price_per_night'] > max_allowed_price:
                    # Don't allow expensive offers in top 3 positions
                    for j in range(min(3, n_positions)):
                        prob += x[i, j] == 0, f"Price_constraint_offer_{i}_pos_{j}"
            
            # Constraint 4: Partner diversity constraint
            # Limit number of offers from same partner in top 5
            partners = list(set(offer['partner_name'] for offer in offers))
            for partner in partners:
                partner_offers = [i for i, offer in enumerate(offers) if offer['partner_name'] == partner]
                if len(partner_offers) > 1:
                    top_positions = min(5, n_positions)
                    partner_constraint = 0
                    for i in partner_offers:
                        for j in range(top_positions):
                            partner_constraint += x[i, j]
                    prob += partner_constraint <= 2, f"Partner_diversity_{partner}"
            
            # Constraint 5: Quality threshold constraint
            # Ensure offers in top positions meet minimum quality standards
            for i in range(n_offers):
                # Find corresponding hotel quality (simplified)
                hotel_quality = offers[i].get('review_score', 7.0)  # Default quality score
                if hotel_quality < 7.0:  # Below quality threshold
                    # Don't allow low-quality offers in top 3 positions
                    for j in range(min(3, n_positions)):
                        prob += x[i, j] == 0, f"Quality_constraint_offer_{i}_pos_{j}"
            
            # Solve the optimization problem
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Extract results
            if prob.status == pulp.LpStatusOptimal:
                ranking = {}
                objective_value = pulp.value(prob.objective)
                
                # Extract solution
                for i in range(n_offers):
                    for j in range(n_positions):
                        if x[i, j].varValue == 1:
                            ranking[j] = i
                
                # Calculate detailed metrics
                metrics = self._calculate_optimization_metrics(
                    ranking, offers, conversion_scores, price_consistency_scores, 
                    bid_competitiveness_scores, position_weights
                )
                
                return {
                    "status": "optimal",
                    "ranking": ranking,
                    "objective_value": objective_value,
                    "metrics": metrics,
                    "solver_info": {
                        "solver": "CBC",
                        "variables": len(x),
                        "constraints": len(prob.constraints)
                    }
                }
            else:
                return {
                    "status": "infeasible",
                    "ranking": {},
                    "objective_value": 0,
                    "metrics": {},
                    "error": f"Optimization failed with status: {pulp.LpStatus[prob.status]}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "ranking": {},
                "objective_value": 0,
                "metrics": {},
                "error": str(e)
            }
    
    def _calculate_conversion_likelihood(self, offers: List[Dict], customer_behavior: Dict) -> List[float]:
        """Calculate conversion likelihood for each offer"""
        scores = []
        
        for offer in offers:
            base_conversion = customer_behavior.get('conversion_likelihood', 0.5)
            
            # Brand preference factor
            brand_pref = customer_behavior.get('brand_preference', {})
            brand_factor = brand_pref.get(offer['partner_name'], 0.5)
            
            # Price sensitivity factor
            price_sensitivity = customer_behavior.get('price_sensitivity_calculated', 0.5)
            all_prices = [o['price_per_night'] for o in offers]
            price_percentile = (offer['price_per_night'] - min(all_prices)) / (max(all_prices) - min(all_prices)) if max(all_prices) > min(all_prices) else 0.5
            price_factor = 1.0 - (price_sensitivity * price_percentile * 0.5)
            
            # Special offers factor
            special_offers_count = len(offer.get('special_offers', []))
            special_factor = 1.0 + (special_offers_count * 0.05)
            
            # Cancellation policy factor
            cancellation_factor = 1.0
            if offer.get('cancellation_policy') == 'Free cancellation':
                cancellation_factor = 1.1
            elif offer.get('cancellation_policy') == 'Non-refundable':
                cancellation_factor = 0.9
            
            # Instant booking factor
            instant_factor = 1.05 if offer.get('instant_booking', False) else 1.0
            
            # Calculate final conversion score
            conversion_score = (base_conversion * brand_factor * price_factor * 
                              special_factor * cancellation_factor * instant_factor)
            
            scores.append(min(1.0, max(0.0, conversion_score)))
        
        return scores
    
    def _calculate_price_consistency(self, offers: List[Dict]) -> List[float]:
        """Calculate price consistency scores for each offer"""
        scores = []
        
        for offer in offers:
            partner_price = offer['price_per_night']
            trivago_price = offer['trivago_displayed_price']
            
            if partner_price > 0:
                # Calculate consistency as 1 - relative difference
                relative_diff = abs(trivago_price - partner_price) / partner_price
                consistency_score = max(0.0, 1.0 - relative_diff)
            else:
                consistency_score = 0.0
            
            scores.append(consistency_score)
        
        return scores
    
    def _calculate_bid_competitiveness(self, offers: List[Dict]) -> List[float]:
        """Calculate bid competitiveness scores for each offer"""
        scores = []
        all_bids = [offer['cost_per_click_bid'] for offer in offers]
        
        if not all_bids or max(all_bids) == min(all_bids):
            return [0.5] * len(offers)
        
        min_bid = min(all_bids)
        max_bid = max(all_bids)
        
        for offer in offers:
            # Normalize bid to 0-1 scale
            normalized_bid = (offer['cost_per_click_bid'] - min_bid) / (max_bid - min_bid)
            
            # Consider commission rate as well
            commission_factor = offer.get('commission_rate', 0.1) / 0.2  # Normalize assuming max 20%
            
            # Combined competitiveness score
            competitiveness = (normalized_bid * 0.7 + commission_factor * 0.3)
            scores.append(min(1.0, max(0.0, competitiveness)))
        
        return scores
    
    def _calculate_optimization_metrics(self, ranking: Dict, offers: List[Dict], 
                                      conversion_scores: List[float], 
                                      consistency_scores: List[float],
                                      bid_scores: List[float],
                                      position_weights: List[float]) -> Dict[str, Any]:
        """Calculate detailed optimization metrics"""
        metrics = {
            "total_conversion_score": 0.0,
            "total_consistency_score": 0.0,
            "total_bid_score": 0.0,
            "weighted_total_score": 0.0,
            "position_utilization": len(ranking) / len(position_weights),
            "partner_diversity": 0.0,
            "price_range": {"min": 0.0, "max": 0.0, "avg": 0.0}
        }
        
        if not ranking:
            return metrics
        
        # Calculate scores for ranked offers
        ranked_offers = [offers[ranking[pos]] for pos in sorted(ranking.keys())]
        ranked_prices = [offer['price_per_night'] for offer in ranked_offers]
        
        for pos in sorted(ranking.keys()):
            offer_idx = ranking[pos]
            weight = position_weights[pos] if pos < len(position_weights) else 0.1
            
            metrics["total_conversion_score"] += conversion_scores[offer_idx] * weight
            metrics["total_consistency_score"] += consistency_scores[offer_idx] * weight
            metrics["total_bid_score"] += bid_scores[offer_idx] * weight
        
        # Weighted total score
        metrics["weighted_total_score"] = (
            metrics["total_conversion_score"] * 0.4 +
            metrics["total_consistency_score"] * 0.3 +
            metrics["total_bid_score"] * 0.3
        )
        
        # Partner diversity
        partners = set(offers[ranking[pos]]['partner_name'] for pos in ranking.keys())
        metrics["partner_diversity"] = len(partners) / len(ranking) if ranking else 0.0
        
        # Price range
        if ranked_prices:
            metrics["price_range"] = {
                "min": min(ranked_prices),
                "max": max(ranked_prices),
                "avg": sum(ranked_prices) / len(ranked_prices)
            }
        
        return metrics

class AdvancedOptimizationStrategies:
    """Advanced optimization strategies for different market conditions"""
    
    @staticmethod
    def get_strategy_config(strategy_name: str, market_conditions: Dict) -> Dict[str, float]:
        """Get optimization weights based on strategy and market conditions"""
        
        base_configs = {
            "revenue_focused": {"conversion": 0.2, "price_consistency": 0.2, "bid_revenue": 0.6},
            "user_focused": {"conversion": 0.6, "price_consistency": 0.3, "bid_revenue": 0.1},
            "balanced": {"conversion": 0.4, "price_consistency": 0.3, "bid_revenue": 0.3},
            "trust_focused": {"conversion": 0.3, "price_consistency": 0.6, "bid_revenue": 0.1}
        }
        
        # Adjust based on market conditions
        market_demand = market_conditions.get('market_demand', 'Medium')
        days_until_travel = market_conditions.get('days_until_travel', 30)
        
        if strategy_name == "RL Optimized Policy":
            if market_demand == "High":
                config = base_configs["revenue_focused"]
            elif days_until_travel < 7:
                config = base_configs["balanced"]
            else:
                config = base_configs["user_focused"]
        elif strategy_name == "Stochastic LP":
            config = base_configs["balanced"]
        else:
            config = base_configs["balanced"]
        
        # Market condition adjustments
        if market_demand == "High":
            # In high demand, slightly favor revenue
            config["bid_revenue"] = min(1.0, config["bid_revenue"] * 1.2)
            config["conversion"] = max(0.1, config["conversion"] * 0.9)
        elif market_demand == "Low":
            # In low demand, favor user satisfaction
            config["conversion"] = min(1.0, config["conversion"] * 1.2)
            config["bid_revenue"] = max(0.1, config["bid_revenue"] * 0.8)
        
        # Normalize weights to sum to 1
        total_weight = sum(config.values())
        if total_weight > 0:
            config = {k: v / total_weight for k, v in config.items()}
        
        return config

def create_optimizer_for_strategy(strategy_name: str, market_conditions: Dict, 
                                custom_weights: Dict = None) -> MultiObjectiveOptimizer:
    """Create and configure optimizer for a specific strategy"""
    optimizer = MultiObjectiveOptimizer()
    
    # Get strategy configuration
    if custom_weights:
        weights = custom_weights
    else:
        weights = AdvancedOptimizationStrategies.get_strategy_config(strategy_name, market_conditions)
    
    # Add objectives
    optimizer.add_objective("conversion", weights.get("conversion", 0.4), maximize=True)
    optimizer.add_objective("price_consistency", weights.get("price_consistency", 0.3), maximize=True)
    optimizer.add_objective("bid_revenue", weights.get("bid_revenue", 0.3), maximize=True)
    
    # Add standard constraints
    optimizer.add_constraint("price_competitiveness", "inequality_le", 1.5)
    optimizer.add_constraint("partner_diversity", "inequality_ge", 0.3)
    optimizer.add_constraint("quality_threshold", "inequality_ge", 7.0)
    
    return optimizer

