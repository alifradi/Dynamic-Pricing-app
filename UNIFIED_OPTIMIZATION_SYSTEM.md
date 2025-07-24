# Unified Optimization System for Dynamic Pricing

## Overview

The Unified Optimization System provides a modular, flexible approach to solving dynamic pricing optimization problems. It replaces the previous MiniZinc-based system with a pure Python implementation using PuLP for linear programming.

## Key Features

### 🎯 **Unified Interface**
- Single `UnifiedOptimizer` class handles all optimization types
- Consistent data loading and preprocessing
- Standardized result formats

### 🔧 **Modular Design**
- **Deterministic Optimization**: User-specific offer rankings
- **Stochastic Optimization**: Offer selection with uncertainty
- **Multi-Objective Optimization**: Configurable objective weights

### 📊 **Smart Data Handling**
- Automatic user-offer data matching
- Robust data validation and cleaning
- Derived metrics calculation (conversion probability, satisfaction scores, trust scores)

### 🚀 **Performance Optimized**
- Efficient PuLP-based linear programming
- Configurable data sampling for large datasets
- Fast execution times

## Architecture

```
UnifiedOptimizer
├── load_data()                    # Load CSV files
├── prepare_optimization_data()    # Clean and structure data
├── run_deterministic_optimization()    # User-specific rankings
├── run_stochastic_optimization()       # Offer selection
├── run_multi_objective_optimization()  # Multi-objective optimization
└── save_results()                 # Export results
```

## Data Structure

### Input Data
The system expects data in the following format:

**Offers Data** (`trial_sampled_offers.csv`):
```csv
offer_id,hotel_id,partner_name,price_per_night,commission_rate,star_rating,review_score,user_id,days_to_go,preference_score
O010048,H00048,Venere,445.55,0.188,5.0,10.0,U000610,27,0.794
```

**Users Data** (`enhanced_user_profiles.csv`):
```csv
user_id,user_type,budget_level,preference_score,days_to_go
U000610,leisure,medium,0.794,27
```

### Derived Metrics
The system automatically calculates:
- **Conversion Probability**: Based on star rating, review score, and price
- **User Satisfaction Score**: Based on hotel quality and price
- **Trust Score**: Based on hotel ratings and reviews
- **Price Level**: Categorized as low/medium/high
- **User Type**: Categorized as leisure/business/luxury

## Optimization Types

### 1. Deterministic Optimization
**Purpose**: Generate user-specific offer rankings

**Objective Function**:
```
Maximize: Σ(α × Revenue + β × Satisfaction + γ × Partner_Value) × Position_Factor
```

**Decision Variables**:
- `x[i,j,k]`: Binary variable indicating if offer i is ranked at position j for user k

**Constraints**:
- Each position for each user can have at most one offer
- Each offer can be assigned to at most one position per user

**Parameters**:
- `alpha`: Weight for revenue objective (default: 0.4)
- `beta`: Weight for user satisfaction objective (default: 0.3)
- `gamma`: Weight for partner value objective (default: 0.3)
- `num_positions`: Number of ranking positions (default: 10)

### 2. Stochastic Optimization
**Purpose**: Select optimal set of offers considering uncertainty

**Objective Function**:
```
Maximize: Σ(Conversion_Prob × Revenue + Satisfaction × Trust_Score)
```

**Decision Variables**:
- `x[i]`: Binary variable indicating if offer i is selected

**Constraints**:
- Select exactly N offers (default: 10)

### 3. Multi-Objective Optimization
**Purpose**: Balance multiple conflicting objectives

**Objective Function**:
```
Maximize: w1 × Revenue_Obj + w2 × Satisfaction_Obj + w3 × Partner_Value_Obj
```

**Configurable Weights**:
- `revenue`: Weight for revenue objective
- `user_satisfaction`: Weight for user satisfaction objective
- `partner_value`: Weight for partner value objective

## Usage Examples

### Basic Usage
```python
from unified_optimizer import UnifiedOptimizer

# Initialize optimizer
optimizer = UnifiedOptimizer("/data")

# Load data
optimizer.load_data()

# Prepare optimization data
opt_data = optimizer.prepare_optimization_data(max_offers=50, max_users=20)

# Run deterministic optimization
results = optimizer.run_deterministic_optimization(
    opt_data,
    alpha=0.4,
    beta=0.3,
    gamma=0.3,
    num_positions=10
)

# Save results
optimizer.save_results(results, 'optimization_results.json')
```

### Convenience Functions
```python
from unified_optimizer import run_deterministic_optimization, run_stochastic_optimization

# Run deterministic optimization
results = run_deterministic_optimization(
    data_dir="/data",
    alpha=0.4,
    beta=0.3,
    gamma=0.3,
    num_positions=10
)

# Run stochastic optimization
results = run_stochastic_optimization(
    data_dir="/data",
    num_selected=10
)
```

## API Integration

### Deterministic Optimization Endpoint
```http
POST /run_deterministic_optimization
```

**Response**:
```json
{
  "status": "success",
  "message": "Deterministic optimization completed",
  "results": {
    "json": {
      "optimization_type": "deterministic",
      "status": "Optimal",
      "objective_value": 1234.56,
      "user_rankings": {
        "U000610": [
          {
            "position": 1,
            "offer_id": 1,
            "hotel_id": "H00048",
            "partner_name": "Venere",
            "price_per_night": 445.55,
            "conversion_probability": 0.15,
            "user_satisfaction_score": 8.5,
            "trust_score": 0.8
          }
        ]
      }
    },
    "csv": {...}
  }
}
```

### Stochastic Optimization Endpoint
```http
POST /run_stochastic_optimization
```

**Response**:
```json
{
  "status": "success",
  "message": "Stochastic optimization completed",
  "results": {
    "json": {
      "optimization_type": "stochastic",
      "status": "Optimal",
      "objective_value": 567.89,
      "selected_offers": [
        {
          "offer_id": 1,
          "hotel_id": "H00048",
          "partner_name": "Venere",
          "price_per_night": 445.55,
          "conversion_probability": 0.15,
          "user_satisfaction_score": 8.5,
          "trust_score": 0.8,
          "commission_rate": 0.188
        }
      ]
    },
    "csv": [...]
  }
}
```

## Configuration

### Data Sampling
Control performance vs. accuracy trade-offs:
```python
opt_data = optimizer.prepare_optimization_data(
    max_offers=50,    # Maximum offers to process
    max_users=20      # Maximum users to process
)
```

### Objective Weights
Customize optimization objectives:
```python
# Deterministic optimization
results = optimizer.run_deterministic_optimization(
    opt_data,
    alpha=0.5,    # Higher weight for revenue
    beta=0.3,     # Medium weight for satisfaction
    gamma=0.2     # Lower weight for partner value
)

# Multi-objective optimization
results = optimizer.run_multi_objective_optimization(
    opt_data,
    weights={
        'revenue': 0.5,
        'user_satisfaction': 0.3,
        'partner_value': 0.2
    }
)
```

## Performance Characteristics

### Execution Times
- **Small datasets** (< 100 offers, < 20 users): < 1 second
- **Medium datasets** (100-500 offers, 20-50 users): 1-5 seconds
- **Large datasets** (> 500 offers, > 50 users): 5-30 seconds

### Memory Usage
- Linear scaling with dataset size
- Efficient PuLP solver with minimal memory overhead
- Configurable sampling for large datasets

## Error Handling

### Data Validation
- Automatic detection of missing or invalid data
- Graceful fallbacks for missing columns
- Robust handling of data type mismatches

### Optimization Failures
- Clear error messages for infeasible problems
- Fallback to simple ranking when optimization fails
- Detailed logging for debugging

## Migration from MiniZinc

### Benefits of Migration
1. **No external dependencies**: Pure Python implementation
2. **Faster execution**: Optimized PuLP solver
3. **Better integration**: Native Python data structures
4. **Easier maintenance**: Single codebase
5. **Flexible configuration**: Runtime parameter adjustment

### Data Compatibility
- Same input CSV format
- Enhanced data validation
- Better user-offer matching
- Improved derived metrics calculation

## Future Enhancements

### Planned Features
1. **Real-time optimization**: Support for streaming data
2. **Advanced constraints**: Budget limits, capacity constraints
3. **Machine learning integration**: Dynamic parameter tuning
4. **Multi-period optimization**: Time-series optimization
5. **Risk-aware optimization**: Uncertainty quantification

### Extensibility
The modular design allows easy addition of:
- New objective functions
- Additional constraint types
- Custom data preprocessing
- Alternative optimization algorithms

## Troubleshooting

### Common Issues

**1. Data Loading Failures**
```python
# Check file existence
import os
if not os.path.exists('/data/trial_sampled_offers.csv'):
    print("Data file not found")
```

**2. Optimization Timeouts**
```python
# Reduce dataset size
opt_data = optimizer.prepare_optimization_data(
    max_offers=20,  # Reduce from 50
    max_users=10    # Reduce from 20
)
```

**3. Infeasible Problems**
```python
# Check constraint parameters
results = optimizer.run_stochastic_optimization(
    opt_data,
    num_selected=5  # Reduce from 10 if not enough offers
)
```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

optimizer = UnifiedOptimizer("/data")
optimizer.load_data()
```

## Conclusion

The Unified Optimization System provides a robust, scalable solution for dynamic pricing optimization. Its modular design, efficient implementation, and comprehensive error handling make it suitable for both development and production environments.

The system successfully addresses the data matching issues you identified and provides a clean, maintainable foundation for future optimization enhancements. 