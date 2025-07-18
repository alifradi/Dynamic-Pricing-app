#!/usr/bin/env python3
"""
Deterministic Optimization Runner
Runs PuLP model for deterministic hotel offer ranking optimization
"""

import json
import csv
import os
import pandas as pd
import pulp
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeterministicOptimizer:
    def __init__(self):
        self.solver = pulp.PULP_CBC_CMD()  # Default solver
    
    def prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for PuLP optimization"""
        
        # Extract data
        offers_data = data.get('offers', [])
        users_data = data.get('users', [])
        n_offers = len(offers_data)
        n_users = len(users_data)
        max_rank = n_offers
        
        # Extract arrays
        conversion_probs = [offer.get('conversion_probability', 0.1) for offer in offers_data]
        bid_amounts = [offer.get('bid_amount', 10.0) for offer in offers_data]
        
        # Create user preferences matrix
        user_preferences = []
        for user in users_data:
            user_pref = []
            for offer in offers_data:
                # Simple preference based on user type and offer characteristics
                base_pref = 0.5
                if user.get('user_type') == 'business' and offer.get('hotel_type') == 'business':
                    base_pref += 0.3
                if user.get('budget_level') == offer.get('price_level'):
                    base_pref += 0.2
                user_pref.append(min(1.0, base_pref))
            user_preferences.append(user_pref)
        
        return {
            'n_offers': n_offers,
            'n_users': n_users,
            'max_rank': max_rank,
            'conversion_probs': conversion_probs,
            'bid_amounts': bid_amounts,
            'user_preferences': user_preferences,
            'offers_data': offers_data,
            'users_data': users_data
        }
    
    def run_optimization(self, data: Dict[str, Any], output_format: str = "json") -> Dict[str, Any]:
        """Run the PuLP optimization and return results"""
        
        try:
            # Prepare data
            opt_data = self.prepare_data(data)
            
            logger.info("Running PuLP optimization...")
            
            # Create PuLP problem
            prob = pulp.LpProblem("Hotel_Ranking_Optimization", pulp.LpMaximize)
            
            # Decision variables: x[u][r][o] = 1 if user u gets offer o at rank r
            x = {}
            for u in range(opt_data['n_users']):
                for r in range(opt_data['max_rank']):
                    for o in range(opt_data['n_offers']):
                        x[u, r, o] = pulp.LpVariable(f"x_{u}_{r}_{o}", cat='Binary')
            
            # Objective function: maximize expected income
            objective = 0
            for u in range(opt_data['n_users']):
                for r in range(opt_data['max_rank']):
                    for o in range(opt_data['n_offers']):
                        rank_decay = 1.0 / (r + 1)  # Rank decay factor
                        expected_value = (opt_data['bid_amounts'][o] * 
                                        opt_data['conversion_probs'][o] * 
                                        opt_data['user_preferences'][u][o] * 
                                        rank_decay)
                        objective += expected_value * x[u, r, o]
            
            prob += objective
            
            # Constraints
            # Each user can only have one offer at each rank
            for u in range(opt_data['n_users']):
                for r in range(opt_data['max_rank']):
                    prob += pulp.lpSum(x[u, r, o] for o in range(opt_data['n_offers'])) == 1
            
            # Each offer can only appear once per user
            for u in range(opt_data['n_users']):
                for o in range(opt_data['n_offers']):
                    prob += pulp.lpSum(x[u, r, o] for r in range(opt_data['max_rank'])) <= 1
            
            # Solve the problem
            prob.solve(self.solver)
            
            if prob.status != pulp.LpStatusOptimal:
                raise RuntimeError(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
            
            # Extract results
            results = self._extract_pulp_results(prob, opt_data, x)
            
            # Parse results based on output format
            if output_format == "json":
                return self._format_json_results(results, opt_data)
            else:
                return self._format_csv_results(results, opt_data)
                
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            raise
    
    def _extract_pulp_results(self, prob, opt_data, x):
        """Extract results from PuLP solution"""
        results = {
            'user_rankings': [],
            'total_expected_income': pulp.value(prob.objective),
            'ranking_data': []
        }
        
        # Extract user rankings
        for u in range(opt_data['n_users']):
            user_ranking = []
            for r in range(opt_data['max_rank']):
                for o in range(opt_data['n_offers']):
                    if pulp.value(x[u, r, o]) == 1:
                        user_ranking.append(o + 1)  # Convert to 1-based indexing
                        break
            results['user_rankings'].append(user_ranking)
        
        # Create ranking data for CSV
        for u in range(opt_data['n_users']):
            for r in range(opt_data['max_rank']):
                for o in range(opt_data['n_offers']):
                    if pulp.value(x[u, r, o]) == 1:
                        rank_decay = 1.0 / (r + 1)
                        expected_value = (opt_data['bid_amounts'][o] * 
                                        opt_data['conversion_probs'][o] * 
                                        opt_data['user_preferences'][u][o] * 
                                        rank_decay)
                        
                        results['ranking_data'].append({
                            'user_id': u + 1,
                            'rank': r + 1,
                            'offer_id': o + 1,
                            'conversion_prob': opt_data['conversion_probs'][o],
                            'bid_amount': opt_data['bid_amounts'][o],
                            'expected_value': expected_value
                        })
        
        return results
    
    def _format_json_results(self, results, opt_data):
        """Format results as JSON"""
        return {
            'optimization_results': {
                'total_expected_income': results['total_expected_income'],
                'user_rankings': results['user_rankings'],
                'parameters': {
                    'n_offers': opt_data['n_offers'],
                    'n_users': opt_data['n_users'],
                    'max_rank': opt_data['max_rank']
                }
            },
            'original_data': {
                'n_offers': opt_data['n_offers'],
                'n_users': opt_data['n_users'],
                'offers': opt_data['offers_data'],
                'users': opt_data['users_data']
            }
        }
    
    def _format_csv_results(self, results, opt_data):
        """Format results as CSV data"""
        return {
            'ranking_data': results['ranking_data'],
            'summary': {
                'total_expected_value': results['total_expected_income'],
                'n_offers': opt_data['n_offers'],
                'n_users': opt_data['n_users']
            },
            'original_data': {
                'n_offers': opt_data['n_offers'],
                'n_users': opt_data['n_users'],
                'offers': opt_data['offers_data'],
                'users': opt_data['users_data']
            }
        }
    
    def export_results(self, results: Dict[str, Any], output_file: str, format: str = "json"):
        """Export results to file"""
        
        try:
            if format.lower() == "json":
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            elif format.lower() == "csv":
                if 'ranking_data' in results:
                    df = pd.DataFrame(results['ranking_data'])
                    df.to_csv(output_file, index=False)
                else:
                    # Convert JSON results to CSV format
                    ranking_data = []
                    if 'optimization_results' in results:
                        opt_results = results['optimization_results']
                        user_rankings = opt_results.get('user_rankings', [])
                        for user_id, ranking in enumerate(user_rankings):
                            for rank, offer_id in enumerate(ranking, 1):
                                ranking_data.append({
                                    'user_id': user_id + 1,
                                    'rank': rank,
                                    'offer_id': offer_id
                                })
                    
                    df = pd.DataFrame(ranking_data)
                    df.to_csv(output_file, index=False)
            
            logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise

def main():
    """Example usage of the DeterministicOptimizer"""
    
    # Example data
    sample_data = {
        'offers': [
            {'offer_id': 1, 'conversion_probability': 0.8, 'bid_amount': 15.0, 'hotel_type': 'business', 'price_level': 'high'},
            {'offer_id': 2, 'conversion_probability': 0.6, 'bid_amount': 12.0, 'hotel_type': 'leisure', 'price_level': 'medium'},
            {'offer_id': 3, 'conversion_probability': 0.9, 'bid_amount': 20.0, 'hotel_type': 'business', 'price_level': 'high'},
            {'offer_id': 4, 'conversion_probability': 0.5, 'bid_amount': 8.0, 'hotel_type': 'leisure', 'price_level': 'low'},
            {'offer_id': 5, 'conversion_probability': 0.7, 'bid_amount': 18.0, 'hotel_type': 'business', 'price_level': 'medium'}
        ],
        'users': [
            {'user_id': 1, 'user_type': 'business', 'budget_level': 'high'},
            {'user_id': 2, 'user_type': 'leisure', 'budget_level': 'medium'},
            {'user_id': 3, 'user_type': 'business', 'budget_level': 'medium'}
        ]
    }
    
    # Initialize optimizer
    optimizer = DeterministicOptimizer()
    
    try:
        # Run optimization
        logger.info("Running deterministic optimization...")
        results = optimizer.run_optimization(sample_data, output_format="json")
        
        # Export results
        optimizer.export_results(results, "deterministic_optimization_results.json", "json")
        optimizer.export_results(results, "deterministic_optimization_results.csv", "csv")
        
        logger.info("Optimization completed successfully!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 