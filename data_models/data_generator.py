import random
import uuid
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple
from faker import Faker
import numpy as np

from .models import (
    UserProfile, Hotel, PartnerOffer, MarketConditions, CustomerBehavior,
    UserType, AgeGroup, LoyaltyStatus, MarketDemand, PropertyType, RoomType
)

fake = Faker()

class DataGenerator:
    """Generates realistic synthetic data for hotel booking simulation"""
    
    def __init__(self):
        self.partners = [
            "Booking.com", "Expedia", "Hotels.com", "Agoda", "HotelDirect",
            "Priceline", "Kayak", "Orbitz", "Travelocity", "Hotwire"
        ]
        
        self.hotel_chains = [
            "Hilton", "Marriott", "Hyatt", "InterContinental", "Accor",
            "Wyndham", "Choice Hotels", "Best Western", "Radisson", "Starwood"
        ]
        
        self.amenities = [
            "WiFi", "Pool", "Gym", "Spa", "Restaurant", "Bar", "Room Service",
            "Business Center", "Parking", "Pet Friendly", "Airport Shuttle",
            "Concierge", "Laundry", "24h Front Desk", "Air Conditioning"
        ]
        
        self.destinations = [
            "New York", "London", "Paris", "Tokyo", "Barcelona", "Rome",
            "Amsterdam", "Berlin", "Prague", "Vienna", "Dubai", "Singapore"
        ]

    def generate_user_profile(self, user_id: str = None) -> UserProfile:
        """Generate a realistic user profile"""
        if not user_id:
            user_id = str(uuid.uuid4())
        
        # Generate travel dates
        booking_date = fake.date_between(start_date='-30d', end_date='today')
        travel_start = fake.date_between(start_date=booking_date, end_date='+180d')
        travel_end = travel_start + timedelta(days=random.randint(1, 14))
        
        # Calculate price sensitivity based on user type and demographics
        user_type = random.choice(list(UserType))
        age_group = random.choice(list(AgeGroup))
        
        base_sensitivity = {
            UserType.BUDGET: 0.8,
            UserType.BUSINESS: 0.3,
            UserType.LEISURE: 0.6,
            UserType.FAMILY: 0.7
        }[user_type]
        
        # Adjust based on age
        age_adjustment = {
            AgeGroup.YOUNG: 0.1,
            AgeGroup.ADULT: 0.0,
            AgeGroup.MIDDLE: -0.1,
            AgeGroup.SENIOR: -0.2
        }[age_group]
        
        price_sensitivity = max(0.1, min(0.9, base_sensitivity + age_adjustment + random.uniform(-0.1, 0.1)))
        
        # Generate budget range based on user type
        budget_multipliers = {
            UserType.BUDGET: (0.5, 1.2),
            UserType.BUSINESS: (1.5, 3.0),
            UserType.LEISURE: (0.8, 2.0),
            UserType.FAMILY: (1.0, 2.5)
        }
        
        base_budget = random.uniform(80, 250)
        multiplier = budget_multipliers[user_type]
        budget_min = base_budget * multiplier[0]
        budget_max = base_budget * multiplier[1]
        
        return UserProfile(
            user_id=user_id,
            location=fake.city(),
            travel_date_start=travel_start,
            travel_date_end=travel_end,
            booking_date=booking_date,
            user_type=user_type,
            age_group=age_group,
            price_sensitivity=price_sensitivity,
            loyalty_status=random.choice(list(LoyaltyStatus)),
            previous_bookings=random.randint(0, 50),
            preferred_amenities=random.sample(self.amenities, random.randint(2, 6)),
            group_size=random.randint(1, 6),
            room_type_preference=random.choice(list(RoomType)),
            budget_range=(budget_min, budget_max)
        )

    def generate_hotel(self, destination: str, hotel_id: str = None) -> Hotel:
        """Generate a realistic hotel"""
        if not hotel_id:
            hotel_id = str(uuid.uuid4())
        
        # Generate realistic coordinates (simplified)
        base_coords = {
            "New York": (40.7128, -74.0060),
            "London": (51.5074, -0.1278),
            "Paris": (48.8566, 2.3522),
            "Tokyo": (35.6762, 139.6503),
            "Barcelona": (41.3851, 2.1734),
            "Rome": (41.9028, 12.4964)
        }
        
        base_lat, base_lng = base_coords.get(destination, (40.7128, -74.0060))
        lat = base_lat + random.uniform(-0.1, 0.1)
        lng = base_lng + random.uniform(-0.1, 0.1)
        
        # Generate star rating and review score (correlated)
        star_rating = random.choices([1, 2, 3, 4, 5], weights=[2, 8, 25, 40, 25])[0]
        review_score = max(1.0, min(10.0, star_rating * 2 + random.uniform(-1, 1)))
        
        # Review count based on star rating
        review_count = random.randint(
            max(10, star_rating * 50),
            star_rating * 500
        )
        
        # Chain brand probability based on star rating
        has_chain = random.random() < (star_rating / 5.0) * 0.7
        chain_brand = random.choice(self.hotel_chains) if has_chain else None
        
        return Hotel(
            hotel_id=hotel_id,
            name=fake.company() + " Hotel",
            location=destination,
            coordinates=(lat, lng),
            star_rating=float(star_rating),
            review_score=round(review_score, 1),
            review_count=review_count,
            amenities=random.sample(self.amenities, random.randint(3, 10)),
            property_type=random.choice(list(PropertyType)),
            distance_to_center=round(random.uniform(0.5, 15.0), 1),
            distance_to_airport=round(random.uniform(5.0, 50.0), 1),
            chain_brand=chain_brand,
            sustainability_rating=round(random.uniform(1.0, 5.0), 1) if random.random() < 0.6 else None,
            accessibility_features=random.sample(
                ["Wheelchair Access", "Elevator", "Braille", "Hearing Loop"], 
                random.randint(0, 3)
            ),
            check_in_time=random.choice(["14:00", "15:00", "16:00"]),
            check_out_time=random.choice(["10:00", "11:00", "12:00"]),
            cancellation_policy_default=random.choice([
                "Free cancellation", "Partial refund", "Non-refundable"
            ])
        )

    def generate_partner_offer(self, hotel: Hotel, partner_name: str, 
                             base_price: float = None) -> PartnerOffer:
        """Generate a realistic partner offer for a hotel"""
        if not base_price:
            # Base price influenced by hotel star rating and location
            base_price = random.uniform(50, 500) * (hotel.star_rating / 3.0)
        
        # Partner-specific pricing strategy
        partner_multipliers = {
            "Booking.com": (0.95, 1.05),
            "Expedia": (0.90, 1.10),
            "Hotels.com": (0.92, 1.08),
            "Agoda": (0.88, 1.12),
            "HotelDirect": (1.00, 1.15),
            "Priceline": (0.85, 1.20),
            "Kayak": (0.90, 1.10),
            "Orbitz": (0.93, 1.07),
            "Travelocity": (0.91, 1.09),
            "Hotwire": (0.80, 1.25)
        }
        
        multiplier_range = partner_multipliers.get(partner_name, (0.90, 1.10))
        price_multiplier = random.uniform(*multiplier_range)
        partner_price = base_price * price_multiplier
        
        # Trivago displayed price (may include fees or be promotional)
        trivago_price = partner_price * random.uniform(0.98, 1.05)
        
        # Cost per click bid (higher for more expensive hotels and competitive partners)
        base_cpc = (partner_price / 100) * random.uniform(0.5, 2.0)
        partner_strength = {
            "Booking.com": 0.9, "Expedia": 0.8, "Hotels.com": 0.7,
            "Agoda": 0.6, "HotelDirect": 0.4
        }.get(partner_name, 0.5)
        
        cpc_bid = base_cpc * (1 + partner_strength)
        
        # Commission rate varies by partner and hotel
        commission_rate = random.uniform(0.08, 0.25)
        
        # Generate price history (last 30 days)
        price_history = []
        for i in range(30):
            historical_date = date.today() - timedelta(days=i)
            historical_price = partner_price * random.uniform(0.85, 1.15)
            price_history.append({
                "date": historical_date.isoformat(),
                "price": round(historical_price, 2)
            })
        
        # Competitor prices
        competitor_prices = {}
        for competitor in random.sample(self.partners, random.randint(2, 5)):
            if competitor != partner_name:
                competitor_prices[competitor] = round(
                    partner_price * random.uniform(0.90, 1.10), 2
                )
        
        return PartnerOffer(
            offer_id=str(uuid.uuid4()),
            hotel_id=hotel.hotel_id,
            partner_name=partner_name,
            partner_brand_strength=partner_strength,
            room_type=random.choice(list(RoomType)),
            price_per_night=round(partner_price, 2),
            total_price=round(partner_price * random.randint(1, 7), 2),  # Multi-night stay
            trivago_displayed_price=round(trivago_price, 2),
            cost_per_click_bid=round(cpc_bid, 2),
            commission_rate=round(commission_rate, 3),
            cancellation_policy=random.choice([
                "Free cancellation", "Partial refund", "Non-refundable"
            ]),
            cancellation_deadline=fake.date_between(start_date='today', end_date='+30d') if random.random() < 0.7 else None,
            breakfast_included=random.random() < 0.4,
            refundable=random.random() < 0.7,
            instant_booking=random.random() < 0.6,
            loyalty_points_offered=random.randint(0, int(partner_price / 10)),
            special_offers=random.sample([
                "Early Bird", "Last Minute", "Member Deal", "Free Upgrade",
                "Extended Stay", "Weekend Special"
            ], random.randint(0, 2)),
            availability_status=random.choices([
                "Available", "Limited", "Last Room"
            ], weights=[70, 20, 10])[0],
            price_history=price_history,
            competitor_prices=competitor_prices
        )

    def generate_market_conditions(self, destination: str, travel_date: date) -> MarketConditions:
        """Generate realistic market conditions"""
        days_until_travel = (travel_date - date.today()).days
        
        # Seasonal factors
        month = travel_date.month
        seasonal_factor = 1.0
        if month in [6, 7, 8, 12]:  # Peak seasons
            seasonal_factor = 1.4
        elif month in [1, 2, 11]:  # Low seasons
            seasonal_factor = 0.7
        
        # Demand based on days until travel and season
        if days_until_travel < 7:
            base_demand = 0.8  # Last minute
        elif days_until_travel < 30:
            base_demand = 0.6
        elif days_until_travel < 90:
            base_demand = 0.5
        else:
            base_demand = 0.4  # Early booking
        
        demand_score = min(1.0, base_demand * seasonal_factor)
        
        # Categorize demand
        if demand_score > 0.7:
            market_demand = MarketDemand.HIGH
        elif demand_score > 0.4:
            market_demand = MarketDemand.MEDIUM
        else:
            market_demand = MarketDemand.LOW
        
        return MarketConditions(
            destination=destination,
            travel_date=travel_date,
            days_until_travel=max(0, days_until_travel),
            market_demand=market_demand,
            demand_score=demand_score,
            seasonal_factor=seasonal_factor,
            event_impact=random.choice([None, "Conference", "Festival", "Holiday"]) if random.random() < 0.3 else None,
            competitor_activity=random.uniform(0.3, 0.8),
            price_volatility=random.uniform(0.1, 0.5),
            booking_velocity=demand_score * random.uniform(0.8, 1.2)
        )

    def generate_customer_behavior(self, user_profile: UserProfile, 
                                 market_conditions: MarketConditions) -> CustomerBehavior:
        """Generate customer behavior based on user profile and market conditions"""
        
        # Calculate price sensitivity based on multiple factors
        base_sensitivity = user_profile.price_sensitivity
        
        # Adjust for time pressure
        time_factor = 1.0
        if market_conditions.days_until_travel < 7:
            time_factor = 0.7  # Less price sensitive when urgent
        elif market_conditions.days_until_travel > 90:
            time_factor = 1.2  # More price sensitive when planning ahead
        
        # Adjust for market demand
        demand_factor = 1.0
        if market_conditions.market_demand == MarketDemand.HIGH:
            demand_factor = 0.8  # Less price sensitive in high demand
        elif market_conditions.market_demand == MarketDemand.LOW:
            demand_factor = 1.1  # More price sensitive in low demand
        
        calculated_sensitivity = min(1.0, base_sensitivity * time_factor * demand_factor)
        
        # Conversion likelihood based on user type and loyalty
        base_conversion = {
            UserType.BUSINESS: 0.7,
            UserType.LEISURE: 0.4,
            UserType.FAMILY: 0.5,
            UserType.BUDGET: 0.6
        }[user_profile.user_type]
        
        loyalty_boost = {
            LoyaltyStatus.NONE: 0.0,
            LoyaltyStatus.SILVER: 0.1,
            LoyaltyStatus.GOLD: 0.15,
            LoyaltyStatus.PLATINUM: 0.2
        }[user_profile.loyalty_status]
        
        conversion_likelihood = min(1.0, base_conversion + loyalty_boost)
        
        # Brand preferences
        brand_preference = {}
        for partner in self.partners:
            base_pref = random.uniform(0.3, 0.7)
            # Boost for well-known brands
            if partner in ["Booking.com", "Expedia", "Hotels.com"]:
                base_pref += 0.1
            brand_preference[partner] = min(1.0, base_pref)
        
        # Amenity importance based on user type
        amenity_importance = {}
        for amenity in self.amenities:
            base_importance = random.uniform(0.2, 0.8)
            
            # Adjust based on user type
            if user_profile.user_type == UserType.BUSINESS:
                if amenity in ["WiFi", "Business Center", "Room Service"]:
                    base_importance += 0.2
            elif user_profile.user_type == UserType.FAMILY:
                if amenity in ["Pool", "Restaurant", "Parking"]:
                    base_importance += 0.2
            
            amenity_importance[amenity] = min(1.0, base_importance)
        
        return CustomerBehavior(
            user_id=user_profile.user_id,
            price_sensitivity_calculated=calculated_sensitivity,
            conversion_likelihood=conversion_likelihood,
            brand_preference=brand_preference,
            amenity_importance=amenity_importance,
            location_importance=random.uniform(0.3, 0.8),
            review_sensitivity=random.uniform(0.4, 0.9),
            cancellation_preference=random.choice(["Flexible", "Standard", "Strict"]),
            booking_urgency=1.0 - (market_conditions.days_until_travel / 180.0)
        )

