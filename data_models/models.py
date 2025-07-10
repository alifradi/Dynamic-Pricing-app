from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime
from enum import Enum

# Enums for better type safety
class UserType(str, Enum):
    BUSINESS = "Business"
    LEISURE = "Leisure"
    FAMILY = "Family"
    BUDGET = "Budget"

class AgeGroup(str, Enum):
    YOUNG = "18-25"
    ADULT = "26-35"
    MIDDLE = "36-50"
    SENIOR = "50+"

class LoyaltyStatus(str, Enum):
    NONE = "None"
    SILVER = "Silver"
    GOLD = "Gold"
    PLATINUM = "Platinum"

class MarketDemand(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class PropertyType(str, Enum):
    HOTEL = "Hotel"
    RESORT = "Resort"
    APARTMENT = "Apartment"
    BNB = "B&B"
    HOSTEL = "Hostel"

class RoomType(str, Enum):
    STANDARD = "Standard"
    DELUXE = "Deluxe"
    SUITE = "Suite"
    FAMILY = "Family"

# Tab 1: Scenario Setup Data Models

class UserProfile(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    location: str = Field(..., description="Current location")
    travel_date_start: date = Field(..., description="Check-in date")
    travel_date_end: date = Field(..., description="Check-out date")
    booking_date: date = Field(..., description="Date of booking")
    user_type: UserType = Field(default=UserType.LEISURE)
    age_group: AgeGroup = Field(default=AgeGroup.ADULT)
    price_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    loyalty_status: LoyaltyStatus = Field(default=LoyaltyStatus.NONE)
    previous_bookings: int = Field(default=0, ge=0)
    preferred_amenities: List[str] = Field(default_factory=list)
    group_size: int = Field(default=1, ge=1)
    room_type_preference: RoomType = Field(default=RoomType.STANDARD)
    budget_range: Tuple[float, float] = Field(default=(50.0, 300.0))

class Hotel(BaseModel):
    hotel_id: str = Field(..., description="Unique hotel identifier")
    name: str = Field(..., description="Hotel name")
    location: str = Field(..., description="Hotel location")
    coordinates: Tuple[float, float] = Field(..., description="Latitude, Longitude")
    star_rating: float = Field(..., ge=1.0, le=5.0)
    review_score: float = Field(..., ge=1.0, le=10.0)
    review_count: int = Field(..., ge=0)
    amenities: List[str] = Field(default_factory=list)
    property_type: PropertyType = Field(default=PropertyType.HOTEL)
    distance_to_center: float = Field(..., ge=0.0, description="Distance to city center in km")
    distance_to_airport: float = Field(..., ge=0.0, description="Distance to airport in km")
    chain_brand: Optional[str] = Field(default=None)
    sustainability_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    accessibility_features: List[str] = Field(default_factory=list)
    check_in_time: str = Field(default="15:00")
    check_out_time: str = Field(default="11:00")
    cancellation_policy_default: str = Field(default="Free cancellation")

class PartnerOffer(BaseModel):
    offer_id: str = Field(..., description="Unique offer identifier")
    hotel_id: str = Field(..., description="Associated hotel ID")
    partner_name: str = Field(..., description="Partner name (Booking.com, Expedia, etc.)")
    partner_brand_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    room_type: RoomType = Field(default=RoomType.STANDARD)
    price_per_night: float = Field(..., gt=0.0)
    total_price: float = Field(..., gt=0.0)
    trivago_displayed_price: float = Field(..., gt=0.0)
    cost_per_click_bid: float = Field(..., ge=0.0)
    commission_rate: float = Field(..., ge=0.0, le=1.0)
    cancellation_policy: str = Field(default="Free cancellation")
    cancellation_deadline: Optional[date] = Field(default=None)
    breakfast_included: bool = Field(default=False)
    refundable: bool = Field(default=True)
    instant_booking: bool = Field(default=False)
    loyalty_points_offered: int = Field(default=0, ge=0)
    special_offers: List[str] = Field(default_factory=list)
    availability_status: str = Field(default="Available")
    price_history: List[Dict] = Field(default_factory=list)
    competitor_prices: Dict[str, float] = Field(default_factory=dict)

# Tab 2: Strategy Selection Data Models

class MarketConditions(BaseModel):
    destination: str = Field(..., description="Travel destination")
    travel_date: date = Field(..., description="Travel date")
    days_until_travel: int = Field(..., ge=0)
    market_demand: MarketDemand = Field(default=MarketDemand.MEDIUM)
    demand_score: float = Field(default=0.5, ge=0.0, le=1.0)
    seasonal_factor: float = Field(default=1.0, ge=0.1, le=3.0)
    event_impact: Optional[str] = Field(default=None)
    competitor_activity: float = Field(default=0.5, ge=0.0, le=1.0)
    price_volatility: float = Field(default=0.2, ge=0.0, le=1.0)
    booking_velocity: float = Field(default=0.5, ge=0.0, le=1.0)

class CustomerBehavior(BaseModel):
    user_id: str = Field(..., description="User identifier")
    price_sensitivity_calculated: float = Field(..., ge=0.0, le=1.0)
    conversion_likelihood: float = Field(..., ge=0.0, le=1.0)
    brand_preference: Dict[str, float] = Field(default_factory=dict)
    amenity_importance: Dict[str, float] = Field(default_factory=dict)
    location_importance: float = Field(default=0.5, ge=0.0, le=1.0)
    review_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    cancellation_preference: str = Field(default="Flexible")
    booking_urgency: float = Field(default=0.5, ge=0.0, le=1.0)

class StrategyConfig(BaseModel):
    strategy_name: str = Field(..., description="Name of the ranking strategy")
    optimization_method: str = Field(default="pulp", description="Optimization method")
    objective_weights: Dict[str, float] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    rl_model_params: Optional[Dict] = Field(default=None)
    bandit_algorithm: str = Field(default="epsilon_greedy")
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)

# Tab 3: Results and Analysis Data Models

class RankedOffer(BaseModel):
    rank: int = Field(..., ge=1)
    offer: PartnerOffer
    score: float = Field(..., description="Overall ranking score")
    conversion_probability: float = Field(..., ge=0.0, le=1.0)
    expected_revenue: float = Field(..., ge=0.0)
    user_trust_score: float = Field(..., ge=0.0, le=100.0)
    price_consistency_score: float = Field(..., ge=0.0, le=1.0)
    bid_competitiveness: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(default="", description="Ranking explanation")

class PerformanceMetrics(BaseModel):
    strategy_name: str = Field(..., description="Strategy name")
    total_expected_revenue: float = Field(..., ge=0.0)
    average_user_trust: float = Field(..., ge=0.0, le=100.0)
    conversion_rate: float = Field(..., ge=0.0, le=1.0)
    click_through_rate: float = Field(..., ge=0.0, le=1.0)
    price_consistency: float = Field(..., ge=0.0, le=1.0)
    partner_satisfaction: float = Field(..., ge=0.0, le=1.0)
    user_satisfaction: float = Field(..., ge=0.0, le=1.0)
    profit_margin: float = Field(..., ge=0.0, le=1.0)

# Simulation and Optimization Models

class PriceVolatility(BaseModel):
    base_price: float = Field(..., gt=0.0)
    volatility_factor: float = Field(default=0.1, ge=0.0, le=1.0)
    time_decay: float = Field(default=0.05, ge=0.0, le=1.0)
    demand_elasticity: float = Field(default=0.3, ge=0.0, le=2.0)
    competitor_influence: float = Field(default=0.2, ge=0.0, le=1.0)
    seasonal_adjustment: float = Field(default=1.0, ge=0.1, le=3.0)

class TimeFactors(BaseModel):
    days_until_travel: int = Field(..., ge=0)
    booking_window_category: str = Field(default="Normal")
    price_sensitivity_multiplier: float = Field(default=1.0, ge=0.1, le=3.0)
    urgency_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    cancellation_risk: float = Field(default=0.1, ge=0.0, le=1.0)

class CompetitiveData(BaseModel):
    hotel_id: str = Field(..., description="Hotel identifier")
    partner_prices: Dict[str, float] = Field(default_factory=dict)
    market_position: Dict[str, int] = Field(default_factory=dict)
    price_gaps: Dict[str, float] = Field(default_factory=dict)
    availability_comparison: Dict[str, bool] = Field(default_factory=dict)
    feature_comparison: Dict[str, Dict] = Field(default_factory=dict)

# Request/Response Models for API

class ScenarioRequest(BaseModel):
    user_profile: UserProfile
    destination: str
    num_hotels: int = Field(default=10, ge=1, le=50)
    num_partners: int = Field(default=5, ge=1, le=10)

class ScenarioResponse(BaseModel):
    user_profile: UserProfile
    hotels: List[Hotel]
    partner_offers: List[PartnerOffer]
    market_conditions: MarketConditions

class RankingRequest(BaseModel):
    scenario: ScenarioResponse
    strategy_config: StrategyConfig
    customer_behavior: CustomerBehavior

class RankingResponse(BaseModel):
    ranked_offers: List[RankedOffer]
    performance_metrics: PerformanceMetrics
    optimization_details: Dict[str, Any]

class MABSimulationRequest(BaseModel):
    strategies: List[str]
    num_iterations: int = Field(default=100, ge=10, le=1000)
    scenario: ScenarioResponse

class MABSimulationResponse(BaseModel):
    cumulative_rewards: List[float]
    strategy_performance: Dict[str, Dict[str, float]]
    best_strategy: str
    convergence_data: List[Dict]

