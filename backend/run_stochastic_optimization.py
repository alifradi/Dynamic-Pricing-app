#!/usr/bin/env python3
"""
Stochastic Optimization Runner
Runs MiniZinc model for hotel ranking optimization and exports results
"""

import json
import csv
import subprocess
import tempfile
import os
import pandas as pd
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StochasticOptimizer:
    def __init__(self, minizinc_model_path: str = "stochastic_optimization.mzn"):
        self.model_path = minizinc_model_path
        self.solver = "gecode"  # Default solver, can be changed to "chuffed", "cbc", etc.
    
    def create_data_file(self, data: Dict[str, Any]) -> str:
        """Create a MiniZinc data file from the input data"""
        
        # Extract data
        offers_data = data.get('offers', [])
        n_offers = len(offers_data)
        n_users = data.get('n_users', 1)
        max_rank = n_offers
        
        # Extract arrays
        conversion_probs = [offer.get('conversion_probability', 0.1) for offer in offers_data]
        revenue_per_offer = [offer.get('revenue', 100.0) for offer in offers_data]
        trust_scores = [offer.get('trust_score', 0.5) for offer in offers_data]
        price_consistency = [offer.get('price_consistency', 0.5) for offer in offers_data]
        
        # Create user preferences matrix (simplified for now)
        user_preferences = [[0.5 for _ in range(n_offers)] for _ in range(n_users)]
        
        # Weights
        weights = data.get('weights', {'conversion': 0.4, 'revenue': 0.4, 'trust': 0.2})
        
        # Create data file content
        data_content = f"""
n_offers = {n_offers};
n_users = {n_users};
max_rank = {max_rank};

conversion_probabilities = {conversion_probs};
revenue_per_offer = {revenue_per_offer};
trust_scores = {trust_scores};
price_consistency = {price_consistency};
user_preferences = {user_preferences};

weight_conversion = {weights['conversion']};
weight_revenue = {weights['revenue']};
weight_trust = {weights['trust']};
"""
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dzn', delete=False) as f:
            f.write(data_content)
            return f.name
    
    def run_optimization(self, data: Dict[str, Any], output_format: str = "json") -> Dict[str, Any]:
        """Run the MiniZinc optimization and return results"""
        
        try:
            # Create data file
            data_file = self.create_data_file(data)
            
            # Prepare MiniZinc command
            cmd = [
                "minizinc",
                "--solver", self.solver,
                "--output-mode", "json" if output_format == "json" else "csv",
                self.model_path,
                data_file
            ]
            
            logger.info(f"Running MiniZinc command: {' '.join(cmd)}")
            
            # Run MiniZinc
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Clean up data file
            os.unlink(data_file)
            
            if result.returncode != 0:
                logger.error(f"MiniZinc failed: {result.stderr}")
                raise RuntimeError(f"MiniZinc optimization failed: {result.stderr}")
            
            # Parse results
            if output_format == "json":
                return self._parse_json_output(result.stdout, data)
            else:
                return self._parse_csv_output(result.stdout, data)
                
        except subprocess.TimeoutExpired:
            logger.error("MiniZinc optimization timed out")
            raise RuntimeError("Optimization timed out")
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            raise
    
    def _parse_json_output(self, output: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON output from MiniZinc"""
        try:
            # Find JSON content in output
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON content found in output")
            
            json_str = output[start_idx:end_idx]
            results = json.loads(json_str)
            
            # Add original data context
            results['original_data'] = {
                'n_offers': len(original_data.get('offers', [])),
                'n_users': original_data.get('n_users', 1),
                'weights': original_data.get('weights', {})
            }
            
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output: {e}")
            logger.error(f"Raw output: {output}")
            raise ValueError(f"Invalid JSON output: {e}")
    
    def _parse_csv_output(self, output: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CSV output from MiniZinc"""
        try:
            # Parse CSV content
            lines = output.strip().split('\n')
            reader = csv.DictReader(lines)
            
            ranking_data = []
            for row in reader:
                ranking_data.append({
                    'rank': int(row['rank']),
                    'offer_id': int(row['offer_id']),
                    'conversion_prob': float(row['conversion_prob']),
                    'revenue': float(row['revenue']),
                    'trust_score': float(row['trust_score']),
                    'price_consistency': float(row['price_consistency']),
                    'expected_value': float(row['expected_value'])
                })
            
            # Calculate summary metrics
            total_expected_value = sum(row['expected_value'] for row in ranking_data)
            avg_trust_score = sum(row['trust_score'] for row in ranking_data) / len(ranking_data)
            
            return {
                'ranking_data': ranking_data,
                'summary': {
                    'total_expected_value': total_expected_value,
                    'avg_trust_score': avg_trust_score,
                    'n_offers': len(ranking_data)
                },
                'original_data': {
                    'n_offers': len(original_data.get('offers', [])),
                    'n_users': original_data.get('n_users', 1),
                    'weights': original_data.get('weights', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to parse CSV output: {e}")
            logger.error(f"Raw output: {output}")
            raise ValueError(f"Invalid CSV output: {e}")
    
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
                        ranking = opt_results.get('ranking', [])
                        for i, offer_id in enumerate(ranking, 1):
                            ranking_data.append({
                                'rank': i,
                                'offer_id': offer_id,
                                'total_score': opt_results.get('total_score', 0),
                                'expected_revenue': opt_results.get('expected_revenue', 0)
                            })
                    
                    df = pd.DataFrame(ranking_data)
                    df.to_csv(output_file, index=False)
            
            logger.info(f"Results exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise

def main():
    """Example usage of the StochasticOptimizer"""
    
    # Example data
    sample_data = {
        'offers': [
            {'offer_id': 1, 'conversion_probability': 0.8, 'revenue': 150.0, 'trust_score': 0.9, 'price_consistency': 0.8},
            {'offer_id': 2, 'conversion_probability': 0.6, 'revenue': 120.0, 'trust_score': 0.7, 'price_consistency': 0.6},
            {'offer_id': 3, 'conversion_probability': 0.9, 'revenue': 200.0, 'trust_score': 0.8, 'price_consistency': 0.9},
            {'offer_id': 4, 'conversion_probability': 0.5, 'revenue': 100.0, 'trust_score': 0.6, 'price_consistency': 0.5},
            {'offer_id': 5, 'conversion_probability': 0.7, 'revenue': 180.0, 'trust_score': 0.85, 'price_consistency': 0.7}
        ],
        'n_users': 1,
        'weights': {
            'conversion': 0.4,
            'revenue': 0.4,
            'trust': 0.2
        }
    }
    
    # Initialize optimizer
    optimizer = StochasticOptimizer()
    
    try:
        # Run optimization
        logger.info("Running stochastic optimization...")
        results = optimizer.run_optimization(sample_data, output_format="json")
        
        # Export results
        optimizer.export_results(results, "stochastic_optimization_results.json", "json")
        optimizer.export_results(results, "stochastic_optimization_results.csv", "csv")
        
        logger.info("Optimization completed successfully!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 