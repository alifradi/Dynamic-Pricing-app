"""
Unified Optimization Module for Dynamic Pricing

This module provides a flexible, modular optimization system that can handle:
- Deterministic optimization (user-specific rankings)
- Stochastic optimization (offer selection)
- Multi-objective optimization with configurable weights
- Proper user-offer data matching

Author: Dynamic Pricing Team
Date: 2024
"""

import pandas as pd
import numpy as np
import pulp
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime


class UnifiedOptimizer:
    """
    Unified optimization system for dynamic pricing scenarios.
    
    This class provides a single interface for different types of optimization:
    - Deterministic: User-specific offer rankings
    - Stochastic: Offer selection with uncertainty
    - Multi-objective: Balancing multiple objectives
    """
    
    def __init__(self, data_dir: str = "/data"):
        """
        Initialize the optimizer.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.offers_data = None
        self.users_data = None
        self.optimization_results = {}
        
    def load_data(self, offers_file: str = "trial_sampled_offers.csv", 
                  users_file: str = "enhanced_user_profiles.csv") -> bool:
        """
        Load and prepare data for optimization.
        
        Args:
            offers_file: Name of offers CSV file
            users_file: Name of users CSV file
            
        Returns:
            bool: True if data loaded successfully
        """
        try:
            # Load offers data
            offers_path = os.path.join(self.data_dir, offers_file)
            if os.path.exists(offers_path):
                self.offers_data = pd.read_csv(offers_path)
                print(f"Loaded {len(self.offers_data)} offers from {offers_file}")
            else:
                print(f"Warning: {offers_file} not found")
                return False
            
            # Load users data (if separate file exists)
            users_path = os.path.join(self.data_dir, users_file)
            if os.path.exists(users_path):
                self.users_data = pd.read_csv(users_path)
                print(f"Loaded {len(self.users_data)} users from {users_file}")
            else:
                print(f"Warning: {users_file} not found, using user data from offers file")
                # Extract unique users from offers data
                if 'user_id' in self.offers_data.columns:
                    unique_users = self.offers_data['user_id'].unique()
                    self.users_data = pd.DataFrame({'user_id': unique_users})
                    print(f"Extracted {len(unique_users)} unique users from offers data")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_optimization_data(self, max_offers: int = None, max_users: int = None) -> Dict[str, Any]:
        """
        Prepare data for optimization using automatic sampling.
        Uses simulated/sampled data with configurable limits.
        
        Args:
            max_offers: Maximum number of offers to use (None = use all)
            max_users: Maximum number of users to use (None = use all)
        
        Returns:
            Dict containing prepared optimization data
        """
        if self.offers_data is None:
            raise ValueError("No offers data loaded. Call load_data() first.")
        
        # Use all available data if limits not specified, but cap at PuLP limits
        if max_offers is None:
            max_offers = min(len(self.offers_data), 100)  # Cap at 100 offers for PuLP
        if max_users is None:
            max_users = min(len(self.offers_data['user_id'].unique()), 100)  # Cap at 100 users for PuLP
        
        print(f"Preparing optimization data: max_offers={max_offers}, max_users={max_users}")
        print(f"Available data: {len(self.offers_data)} offers, {len(self.offers_data['user_id'].unique())} users")
        
        # Validate problem size for PuLP limits
        # Assuming num_positions=5 (from UI), target < 100K decision variables
        estimated_variables = max_offers * max_users * 5  # 5 positions
        if estimated_variables > 100_000:
            print(f"WARNING: Problem size ({estimated_variables:,} variables) may exceed PuLP limits")
            print("Consider reducing max_offers or max_users")
        else:
            print(f"Problem size ({estimated_variables:,} variables) is within PuLP limits")
        
        # First, get unique users to ensure we have multiple users
        all_unique_users = self.offers_data['user_id'].unique()
        if len(all_unique_users) > max_users:
            # Sample users randomly
            selected_users = np.random.choice(all_unique_users, max_users, replace=False)
        else:
            selected_users = all_unique_users
        
        # Filter offers for selected users
        offers_sample = self.offers_data[self.offers_data['user_id'].isin(selected_users)].copy()
        
        # Ensure balanced sampling across users
        offers_per_user = max_offers // len(selected_users)
        balanced_offers = []
        
        for user_id in selected_users:
            user_offers = offers_sample[offers_sample['user_id'] == user_id]
            if len(user_offers) > offers_per_user:
                # Sample offers for this user
                user_sample = user_offers.sample(n=offers_per_user, random_state=42)
            else:
                # Take all offers for this user
                user_sample = user_offers
            
            balanced_offers.append(user_sample)
        
        # Combine all user samples
        if balanced_offers:
            offers_sample = pd.concat(balanced_offers, ignore_index=True)
        else:
            # Fallback: take first max_offers
            offers_sample = offers_sample.head(max_offers)
        
        # Get the actual unique users from the final sample
        unique_users = offers_sample['user_id'].unique()
        
        # Prepare offers data
        offers_list = []
        for idx, offer in offers_sample.iterrows():
            # Ensure offer_id is valid
            try:
                offer_id = int(offer['offer_id'].replace('O', '')) if isinstance(offer['offer_id'], str) else int(offer['offer_id'])
            except (ValueError, TypeError):
                offer_id = idx + 1
            
            # Calculate derived metrics
            conversion_prob = self._calculate_conversion_probability(offer)
            user_satisfaction = self._calculate_user_satisfaction(offer)
            trust_score = self._calculate_trust_score(offer)
            
            offers_list.append({
                'offer_id': offer_id,
                'hotel_id': offer.get('hotel_id', f"H{offer_id:06d}"),
                'partner_name': offer.get('partner_name', 'Unknown'),
                'price_per_night': float(offer.get('price_per_night', 0)),
                'commission_rate': float(offer.get('commission_rate', 0.15)),
                'cost_per_click_bid': float(offer.get('cost_per_click_bid', 1.0)),
                'conversion_probability': conversion_prob,
                'user_satisfaction_score': user_satisfaction,
                'trust_score': trust_score,
                'star_rating': float(offer.get('star_rating', 3.0)),
                'review_score': float(offer.get('review_score', 7.0)),
                'hotel_type': offer.get('property_type', 'standard'),
                'price_level': self._categorize_price_level(offer.get('price_per_night', 0))
            })
        
        # Prepare users data
        users_list = []
        for user_id in unique_users:
            user_offers = offers_sample[offers_sample['user_id'] == user_id]
            if not user_offers.empty:
                # Calculate user preferences from their offers
                avg_price = user_offers['price_per_night'].mean()
                avg_rating = user_offers['star_rating'].mean()
                
                users_list.append({
                    'user_id': str(user_id),
                    'user_type': self._categorize_user_type(avg_price, avg_rating),
                    'budget_level': self._categorize_budget_level(avg_price),
                    'preference_score': float(user_offers['preference_score'].mean() if 'preference_score' in user_offers.columns else 0.5),
                    'days_to_go': int(user_offers['days_to_go'].iloc[0]) if 'days_to_go' in user_offers.columns else 30
                })
        
        return {
            'offers': offers_list,
            'users': users_list,
            'metadata': {
                'total_offers': len(offers_list),
                'total_users': len(users_list),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def run_deterministic_optimization(self, optimization_data: Dict[str, Any], 
                                     alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
                                     num_positions: int = 10) -> Dict[str, Any]:
        """
        Run deterministic optimization for user-specific offer rankings.
        
        Args:
            optimization_data: Prepared optimization data
            alpha: Weight for revenue objective
            beta: Weight for user satisfaction objective
            gamma: Weight for partner value objective
            num_positions: Number of ranking positions
            
        Returns:
            Dict containing optimization results
        """
        offers = optimization_data['offers']
        users = optimization_data['users']
        
        # Create optimization problem
        prob = pulp.LpProblem("Deterministic_Optimization", pulp.LpMaximize)
        
        # Decision variables: x[i,j,k] = 1 if offer i is ranked at position j for user k
        x = pulp.LpVariable.dicts("x", 
                                 [(i, j, k) for i in range(len(offers)) 
                                  for j in range(num_positions) 
                                  for k in range(len(users))], 
                                 cat='Binary')
        
        # Objective function: Maximize weighted sum of objectives
        objective = 0
        for i, offer in enumerate(offers):
            for j in range(num_positions):
                for k, user in enumerate(users):
                    # Position decay factor
                    position_factor = 1.0 / (1 + 0.5 * j)
                    
                    # Revenue component
                    revenue = offer['price_per_night'] * offer['commission_rate'] * offer['conversion_probability']
                    
                    # User satisfaction component
                    satisfaction = offer['user_satisfaction_score'] * offer['trust_score']
                    
                    # Partner value component
                    partner_value = offer['trust_score'] * offer['conversion_probability']
                    
                    objective += x[i, j, k] * position_factor * (
                        alpha * revenue + 
                        beta * satisfaction + 
                        gamma * partner_value
                    )
        
        prob += objective
        
        # Constraints
        # Each position for each user can have at most one offer
        for j in range(num_positions):
            for k in range(len(users)):
                prob += pulp.lpSum(x[i, j, k] for i in range(len(offers))) <= 1
        
        # Each offer can be assigned to at most one position per user
        for i in range(len(offers)):
            for k in range(len(users)):
                prob += pulp.lpSum(x[i, j, k] for j in range(num_positions)) <= 1
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results
        user_rankings = {}
        for k, user in enumerate(users):
            user_rankings[user['user_id']] = []
            for j in range(num_positions):
                for i, offer in enumerate(offers):
                    if x[i, j, k].value() == 1:
                        user_rankings[user['user_id']].append({
                            'position': j + 1,
                            'offer_id': offer['offer_id'],
                            'hotel_id': offer['hotel_id'],
                            'partner_name': offer['partner_name'],
                            'price_per_night': offer['price_per_night'],
                            'conversion_probability': offer['conversion_probability'],
                            'user_satisfaction_score': offer['user_satisfaction_score'],
                            'trust_score': offer['trust_score']
                        })
        
        return {
            'optimization_type': 'deterministic',
            'status': pulp.LpStatus[prob.status],
            'objective_value': pulp.value(prob.objective),
            'user_rankings': user_rankings,
            'parameters': {
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'num_positions': num_positions
            },
            'metadata': optimization_data['metadata']
        }
    
    def run_stochastic_optimization(self, optimization_data: Dict[str, Any],
                                  num_selected: int = 10) -> Dict[str, Any]:
        """
        Run stochastic optimization for offer selection.
        
        Args:
            optimization_data: Prepared optimization data
            num_selected: Number of offers to select
            
        Returns:
            Dict containing optimization results
        """
        offers = optimization_data['offers']
        
        # Create optimization problem
        prob = pulp.LpProblem("Stochastic_Optimization", pulp.LpMaximize)
        
        # Decision variables: x[i] = 1 if offer i is selected
        x = pulp.LpVariable.dicts("x", range(len(offers)), cat='Binary')
        
        # Objective function: Maximize expected value
        objective = pulp.lpSum(
            x[i] * (
                offers[i]['conversion_probability'] * offers[i]['price_per_night'] * offers[i]['commission_rate'] +
                offers[i]['user_satisfaction_score'] * offers[i]['trust_score']
            ) for i in range(len(offers))
        )
        
        prob += objective
        
        # Constraint: Select exactly num_selected offers
        prob += pulp.lpSum(x[i] for i in range(len(offers))) == num_selected
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results
        selected_offers = []
        for i, offer in enumerate(offers):
            if x[i].value() == 1:
                selected_offers.append({
                    'offer_id': offer['offer_id'],
                    'hotel_id': offer['hotel_id'],
                    'partner_name': offer['partner_name'],
                    'price_per_night': offer['price_per_night'],
                    'conversion_probability': offer['conversion_probability'],
                    'user_satisfaction_score': offer['user_satisfaction_score'],
                    'trust_score': offer['trust_score'],
                    'commission_rate': offer['commission_rate']
                })
        
        return {
            'optimization_type': 'stochastic',
            'status': pulp.LpStatus[prob.status],
            'objective_value': pulp.value(prob.objective),
            'selected_offers': selected_offers,
            'num_selected': num_selected,
            'metadata': optimization_data['metadata']
        }
    
    def run_multi_objective_optimization(self, optimization_data: Dict[str, Any],
                                       weights: Dict[str, float] = None,
                                       num_positions: int = 10) -> Dict[str, Any]:
        """
        Run multi-objective optimization with configurable weights.
        
        Args:
            optimization_data: Prepared optimization data
            weights: Dictionary of objective weights
            num_positions: Number of ranking positions
            
        Returns:
            Dict containing optimization results
        """
        if weights is None:
            weights = {
                'revenue': 0.4,
                'user_satisfaction': 0.3,
                'partner_value': 0.3
            }
        
        offers = optimization_data['offers']
        users = optimization_data['users']
        
        # Create optimization problem
        prob = pulp.LpProblem("Multi_Objective_Optimization", pulp.LpMaximize)
        
        # Decision variables: x[i,j,k] = 1 if offer i is ranked at position j for user k
        x = pulp.LpVariable.dicts("x", 
                                 [(i, j, k) for i in range(len(offers)) 
                                  for j in range(num_positions) 
                                  for k in range(len(users))], 
                                 cat='Binary')
        
        # Objective function components
        revenue_obj = 0
        satisfaction_obj = 0
        partner_obj = 0
        
        for i, offer in enumerate(offers):
            for j in range(num_positions):
                for k, user in enumerate(users):
                    position_factor = 1.0 / (1 + 0.5 * j)
                    
                    # Revenue objective
                    revenue_obj += x[i, j, k] * position_factor * (
                        offer['price_per_night'] * offer['commission_rate'] * offer['conversion_probability']
                    )
                    
                    # User satisfaction objective
                    satisfaction_obj += x[i, j, k] * position_factor * (
                        offer['user_satisfaction_score'] * offer['trust_score']
                    )
                    
                    # Partner value objective
                    partner_obj += x[i, j, k] * position_factor * (
                        offer['trust_score'] * offer['conversion_probability']
                    )
        
        # Combined objective
        prob += weights['revenue'] * revenue_obj + \
                weights['user_satisfaction'] * satisfaction_obj + \
                weights['partner_value'] * partner_obj
        
        # Constraints (same as deterministic)
        for j in range(num_positions):
            for k in range(len(users)):
                prob += pulp.lpSum(x[i, j, k] for i in range(len(offers))) <= 1
        
        for i in range(len(offers)):
            for k in range(len(users)):
                prob += pulp.lpSum(x[i, j, k] for j in range(num_positions)) <= 1
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results
        user_rankings = {}
        for k, user in enumerate(users):
            user_rankings[user['user_id']] = []
            for j in range(num_positions):
                for i, offer in enumerate(offers):
                    if x[i, j, k].value() == 1:
                        user_rankings[user['user_id']].append({
                            'position': j + 1,
                            'offer_id': offer['offer_id'],
                            'hotel_id': offer['hotel_id'],
                            'partner_name': offer['partner_name'],
                            'price_per_night': offer['price_per_night'],
                            'conversion_probability': offer['conversion_probability'],
                            'user_satisfaction_score': offer['user_satisfaction_score'],
                            'trust_score': offer['trust_score']
                        })
        
        return {
            'optimization_type': 'multi_objective',
            'status': pulp.LpStatus[prob.status],
            'objective_value': pulp.value(prob.objective),
            'user_rankings': user_rankings,
            'weights': weights,
            'num_positions': num_positions,
            'metadata': optimization_data['metadata']
        }
    
    def run_three_stage_stochastic_optimization(self, optimization_data: Dict[str, Any],
                                              alpha: float = 0.4,
                                              beta: float = 0.3,
                                              gamma: float = 0.3,
                                              confidence_level: float = 0.8,
                                              num_positions: int = 10) -> Dict[str, Any]:
        """
        Three-stage stochastic optimization for Trivago revenue optimization.
        
        Stage 1: Offer exposure decision
        Stage 2: User regret minimization (stochastic)
        Stage 3: Re-conversion maximization
        
        Args:
            optimization_data: Prepared optimization data
            confidence_level: Confidence level for chance constraints (default 0.8 = 80%)
            num_positions: Number of ranking positions
            
        Returns:
            Dict containing optimization results
        """
        offers = optimization_data['offers']
        users = optimization_data['users']
        
        # Create optimization problem
        prob = pulp.LpProblem("Three_Stage_Stochastic_Optimization", pulp.LpMaximize)
        
        # Stage 1: Decision variables for offer exposure
        # x[i,j,k] = 1 if offer i is ranked at position j for user k
        x = pulp.LpVariable.dicts("x", 
                                 [(i, j, k) for i in range(len(offers)) 
                                  for j in range(num_positions) 
                                  for k in range(len(users))], 
                                 cat='Binary')
        
        # Stage 2: Stochastic regret variables (expected regret for each user)
        regret = pulp.LpVariable.dicts("regret", 
                                      [k for k in range(len(users))], 
                                      lowBound=0)
        
        # Stage 3: Re-conversion probability variables
        reconversion = pulp.LpVariable.dicts("reconversion", 
                                            [k for k in range(len(users))], 
                                            lowBound=0, upBound=1)
        
        # Objective function: Maximize Trivago revenue minus expected regret plus re-conversion value
        revenue_obj = 0
        regret_obj = 0
        reconversion_obj = 0
        
        for i, offer in enumerate(offers):
            for j in range(num_positions):
                for k, user in enumerate(users):
                    position_factor = 1.0 / (1 + 0.5 * j)  # Position decay
                    
                    # Stage 1: Revenue from bid fees and conversions
                    bid_fee = offer['cost_per_click_bid']
                    conversion_revenue = offer['price_per_night'] * offer['commission_rate'] * offer['conversion_probability']
                    revenue_obj += x[i, j, k] * position_factor * (bid_fee + conversion_revenue)
        
        # Stage 2: Expected regret (stochastic component)
        for k, user in enumerate(users):
            # Expected regret based on price sensitivity and user preferences
            user_price_sensitivity = user.get('price_sensitivity', 0.5)
            regret_obj += regret[k] * user_price_sensitivity
        
        # Stage 3: Re-conversion value
        for k, user in enumerate(users):
            # Re-conversion value based on user satisfaction and trust
            user_satisfaction = user.get('preference_score', 0.5)
            reconversion_obj += reconversion[k] * user_satisfaction * 100  # Scale factor
        
        # Combined objective using user-defined weights
        prob += alpha * revenue_obj - beta * regret_obj + gamma * reconversion_obj
        
        # Constraints
        
        # 1. Position constraints: Each position for each user can have at most one offer
        for j in range(num_positions):
            for k in range(len(users)):
                prob += pulp.lpSum(x[i, j, k] for i in range(len(offers))) <= 1
        
        # 2. Offer constraints: Each offer can be assigned to at most one position per user
        for i in range(len(offers)):
            for k in range(len(users)):
                prob += pulp.lpSum(x[i, j, k] for j in range(num_positions)) <= 1
        
        # 3. Stage 2: Regret constraints (stochastic)
        for k, user in enumerate(users):
            user_offers = [i for i, offer in enumerate(offers) if offer.get('user_id') == user['user_id']]
            if user_offers:
                # Expected regret based on price differences
                avg_price = sum(offers[i]['price_per_night'] for i in user_offers) / len(user_offers)
                for i in user_offers:
                    for j in range(num_positions):
                        price_diff = abs(offers[i]['price_per_night'] - avg_price) / avg_price
                        prob += regret[k] >= x[i, j, k] * price_diff * 0.1
        
        # 4. Stage 3: Re-conversion constraints
        for k, user in enumerate(users):
            # Re-conversion probability based on ranking quality
            for i, offer in enumerate(offers):
                for j in range(num_positions):
                    position_quality = 1.0 / (1 + 0.3 * j)  # Better positions = higher re-conversion
                    trust_factor = offer['trust_score']
                    prob += reconversion[k] >= x[i, j, k] * position_quality * trust_factor * 0.1
        
        # 5. CHANCE CONSTRAINT: Marketing budget constraint (80% confidence)
        # Group offers by partner
        partner_offers = {}
        for i, offer in enumerate(offers):
            partner = offer['partner_name']
            if partner not in partner_offers:
                partner_offers[partner] = []
            partner_offers[partner].append(i)
        
        # Budget constraint for each partner
        for partner, offer_indices in partner_offers.items():
            total_budget = sum(offers[i]['cost_per_click_bid'] for i in offer_indices)
            # Chance constraint: P(budget_consumed <= 0.8 * total_budget) >= confidence_level
            # This translates to: expected_budget_consumption <= 0.8 * total_budget * confidence_level
            max_budget_consumption = 0.8 * total_budget * confidence_level
            
            budget_consumption = 0
            for i in offer_indices:
                for j in range(num_positions):
                    for k in range(len(users)):
                        # Expected budget consumption considering conversion probability
                        expected_cost = offers[i]['cost_per_click_bid'] * offers[i]['conversion_probability']
                        budget_consumption += x[i, j, k] * expected_cost
            
            prob += budget_consumption <= max_budget_consumption
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results
        user_rankings = {}
        stage_metrics = {}
        
        for k, user in enumerate(users):
            user_rankings[user['user_id']] = []
            user_revenue = 0
            user_regret = regret[k].value() if regret[k].value() is not None else 0
            user_reconversion = reconversion[k].value() if reconversion[k].value() is not None else 0
            
            for j in range(num_positions):
                for i, offer in enumerate(offers):
                    if x[i, j, k].value() == 1:
                        position_factor = 1.0 / (1 + 0.5 * j)
                        offer_revenue = position_factor * (
                            offer['cost_per_click_bid'] + 
                            offer['price_per_night'] * offer['commission_rate'] * offer['conversion_probability']
                        )
                        user_revenue += offer_revenue
                        
                        user_rankings[user['user_id']].append({
                            'position': j + 1,
                            'offer_id': offer['offer_id'],
                            'hotel_id': offer['hotel_id'],
                            'partner_name': offer['partner_name'],
                            'price_per_night': offer['price_per_night'],
                            'cost_per_click_bid': offer['cost_per_click_bid'],
                            'conversion_probability': offer['conversion_probability'],
                            'user_satisfaction_score': offer['user_satisfaction_score'],
                            'trust_score': offer['trust_score'],
                            'expected_revenue': offer_revenue
                        })
            
            stage_metrics[user['user_id']] = {
                'stage1_revenue': user_revenue,
                'stage2_regret': user_regret,
                'stage3_reconversion_prob': user_reconversion
            }
        
        return {
            'optimization_type': 'three_stage_stochastic',
            'status': pulp.LpStatus[prob.status],
            'objective_value': pulp.value(prob.objective),
            'user_rankings': user_rankings,
            'stage_metrics': stage_metrics,
            'confidence_level': confidence_level,
            'num_positions': num_positions,
            'metadata': {
                'total_offers': len(offers),
                'total_users': len(users),
                'total_revenue': sum(metrics['stage1_revenue'] for metrics in stage_metrics.values()),
                'total_regret': sum(metrics['stage2_regret'] for metrics in stage_metrics.values()),
                'avg_reconversion': sum(metrics['stage3_reconversion_prob'] for metrics in stage_metrics.values()) / len(users),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def run_simple_optimization(self) -> Dict[str, Any]:
        """
        Simple optimization model: Maximize 2x + y
        Subject to: x >= 0, y >= 0, x <= 3, y <= 2
        
        Returns:
            Dict containing optimization results with x, y values and objective value
        """
        try:
            # Create optimization problem
            prob = pulp.LpProblem("Simple_Optimization_2x_plus_y", pulp.LpMaximize)
            
            # Decision variables
            x = pulp.LpVariable("x", lowBound=0, upBound=3, cat='Continuous')
            y = pulp.LpVariable("y", lowBound=0, upBound=2, cat='Continuous')
            
            # Objective function: Maximize 2x + y
            prob += 2 * x + y, "Objective"
            
            # Constraints are already defined in variable bounds:
            # x >= 0, y >= 0 (lowBound=0)
            # x <= 3, y <= 2 (upBound=3 and upBound=2)
            
            # Solve the problem
            print("Solving simple optimization: Maximize 2x + y")
            prob.solve()
            
            # Get results
            if prob.status == pulp.LpStatusOptimal:
                x_value = pulp.value(x)
                y_value = pulp.value(y)
                objective_value = pulp.value(prob.objective)
                
                results = {
                    "status": "Optimal",
                    "objective_value": objective_value,
                    "variables": {
                        "x": x_value,
                        "y": y_value
                    },
                    "constraints": {
                        "x_lower": 0,
                        "x_upper": 3,
                        "y_lower": 0,
                        "y_upper": 2
                    },
                    "model_info": {
                        "objective_function": "2x + y",
                        "constraints": "x >= 0, y >= 0, x <= 3, y <= 2",
                        "solver_status": pulp.LpStatus[prob.status]
                    },
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "model_type": "simple_2x_plus_y"
                    }
                }
                
                print(f"Optimization completed successfully!")
                print(f"x = {x_value}")
                print(f"y = {y_value}")
                print(f"Objective value = {objective_value}")
                
                return results
            else:
                print(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
                return {
                    "status": "Failed",
                    "error": f"Optimization failed with status: {pulp.LpStatus[prob.status]}",
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "model_type": "simple_2x_plus_y"
                    }
                }
                
        except Exception as e:
            print(f"Error in simple optimization: {e}")
            return {
                "status": "Error",
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_type": "simple_2x_plus_y"
                }
            }
    
    def save_results(self, results: Dict[str, Any], filename: str) -> bool:
        """
        Save optimization results to file.
        
        Args:
            results: Optimization results
            filename: Output filename
            
        Returns:
            bool: True if saved successfully
        """
        try:
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def _calculate_conversion_probability(self, offer: pd.Series) -> float:
        """Calculate conversion probability for an offer."""
        base_rate = 0.1
        star_bonus = offer.get('star_rating', 3.0) * 0.05
        review_bonus = offer.get('review_score', 7.0) * 0.02
        price_penalty = offer.get('price_per_night', 100) * 0.0001
        
        return np.clip(base_rate + star_bonus + review_bonus - price_penalty, 0.01, 0.95)
    
    def _calculate_user_satisfaction(self, offer: pd.Series) -> float:
        """Calculate user satisfaction score for an offer."""
        star_score = offer.get('star_rating', 3.0) * 0.3
        price_score = (10 - offer.get('price_per_night', 100) / 50) * 0.4
        review_score = offer.get('review_score', 7.0) * 0.3
        
        return np.clip(star_score + price_score + review_score, 0, 10)
    
    def _calculate_trust_score(self, offer: pd.Series) -> float:
        """Calculate trust score for an offer."""
        base_trust = 0.5
        star_bonus = offer.get('star_rating', 3.0) * 0.1
        review_bonus = offer.get('review_score', 7.0) * 0.05
        
        return np.clip(base_trust + star_bonus + review_bonus, 0, 1)
    
    def _categorize_price_level(self, price: float) -> str:
        """Categorize price level."""
        if price > 200:
            return 'high'
        elif price > 100:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_user_type(self, avg_price: float, avg_rating: float) -> str:
        """Categorize user type based on preferences."""
        if avg_price > 200 and avg_rating > 4.0:
            return 'luxury'
        elif avg_price > 100:
            return 'business'
        else:
            return 'leisure'
    
    def _categorize_budget_level(self, avg_price: float) -> str:
        """Categorize budget level."""
        if avg_price > 200:
            return 'high'
        elif avg_price > 100:
            return 'medium'
        else:
            return 'low'


# Convenience functions for easy integration
def run_deterministic_optimization(data_dir: str = "/data", **kwargs) -> Dict[str, Any]:
    """Convenience function to run deterministic optimization."""
    optimizer = UnifiedOptimizer(data_dir)
    if optimizer.load_data():
        opt_data = optimizer.prepare_optimization_data()
        return optimizer.run_deterministic_optimization(opt_data, **kwargs)
    else:
        raise ValueError("Failed to load data")

def run_stochastic_optimization(data_dir: str = "/data", **kwargs) -> Dict[str, Any]:
    """Convenience function to run stochastic optimization."""
    optimizer = UnifiedOptimizer(data_dir)
    if optimizer.load_data():
        opt_data = optimizer.prepare_optimization_data()
        return optimizer.run_stochastic_optimization(opt_data, **kwargs)
    else:
        raise ValueError("Failed to load data")

def run_multi_objective_optimization(data_dir: str = "/data", **kwargs) -> Dict[str, Any]:
    """Convenience function to run multi-objective optimization."""
    optimizer = UnifiedOptimizer(data_dir)
    if optimizer.load_data():
        opt_data = optimizer.prepare_optimization_data()
        return optimizer.run_multi_objective_optimization(opt_data, **kwargs)
    else:
        raise ValueError("Failed to load data") 