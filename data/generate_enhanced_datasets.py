#!/usr/bin/env python3
"""
Enhanced Hotel Booking Dataset Generator
Generates realistic hotel, partner offers, and user profile datasets
with price fluctuation data for trivago-style simulation
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import json
import argparse
import os

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

class HotelDatasetGenerator:
    def __init__(self):
        self.cities = [
            {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
            {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
            {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
            {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
            {"name": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
            {"name": "Madrid", "country": "Spain", "lat": 40.4168, "lon": -3.7038},
            {"name": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964},
            {"name": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lon": 4.9041},
            {"name": "Barcelona", "country": "Spain", "lat": 41.3851, "lon": 2.1734},
            {"name": "Vienna", "country": "Austria", "lat": 48.2082, "lon": 16.3738},
            {"name": "Prague", "country": "Czech Republic", "lat": 50.0755, "lon": 14.4378},
            {"name": "Dubai", "country": "UAE", "lat": 25.2048, "lon": 55.2708},
            {"name": "Singapore", "country": "Singapore", "lat": 1.3521, "lon": 103.8198},
            {"name": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
            {"name": "Los Angeles", "country": "USA", "lat": 34.0522, "lon": -118.2437},
            {"name": "Miami", "country": "USA", "lat": 25.7617, "lon": -80.1918},
            {"name": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018},
            {"name": "Istanbul", "country": "Turkey", "lat": 41.0082, "lon": 28.9784},
            {"name": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
            {"name": "São Paulo", "country": "Brazil", "lat": -23.5505, "lon": -46.6333}
        ]
        
        self.chain_brands = [
            "Marriott", "Hilton", "Hyatt", "InterContinental", "Radisson",
            "Sheraton", "Westin", "Renaissance", "Courtyard", "Hampton Inn",
            "Holiday Inn", "Best Western", "Novotel", "Ibis", "Accor",
            "Four Seasons", "Ritz-Carlton", "St. Regis", "W Hotels", "Independent"
        ]
        
        self.amenities_list = [
            "WiFi", "Pool", "Gym", "Spa", "Restaurant", "Bar", "Room Service",
            "Business Center", "Conference Rooms", "Parking", "Airport Shuttle",
            "Pet Friendly", "24h Front Desk", "Concierge", "Laundry",
            "Air Conditioning", "Balcony", "Kitchen", "Breakfast", "Beach Access"
        ]
        
        self.property_types = ["Hotel", "Resort", "Apartment", "Hostel", "Villa", "B&B"]
        
        self.partners = [
            "Booking.com", "Expedia", "Hotels.com", "Agoda", "Priceline",
            "Kayak", "Orbitz", "Travelocity", "HotelDirect", "Venere",
            "HRS", "Lastminute.com", "Otel.com", "Getaroom", "Hotwire"
        ]
        
        self.room_types = ["Single", "Double", "Twin", "Suite", "Deluxe", "Executive", "Family"]
        
        self.user_types = ["Leisure", "Business", "Family", "Budget", "Luxury"]
        self.age_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        self.loyalty_statuses = ["None", "Silver", "Gold", "Platinum"]
        
    def generate_hotels(self, n=10000):
        """Generate realistic hotel data"""
        hotels = []
        
        for i in range(n):
            city = random.choice(self.cities)
            star_rating = np.random.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2])
            
            # Generate coordinates near city center
            lat_offset = np.random.normal(0, 0.05)
            lon_offset = np.random.normal(0, 0.05)
            
            # Star rating influences review score and amenities
            base_score = 6.0 + star_rating * 0.8
            review_score = round(np.random.normal(base_score, 0.5), 1)
            review_score = max(5.0, min(10.0, review_score))
            
            review_count = int(np.random.exponential(300) * star_rating)
            review_count = max(10, min(5000, review_count))
            
            # Generate amenities based on star rating
            num_amenities = min(len(self.amenities_list), 
                              int(np.random.normal(star_rating * 3, 2)))
            amenities = random.sample(self.amenities_list, max(3, num_amenities))
            
            hotel = {
                "hotel_id": f"H{i+1:05d}",
                "name": self._generate_hotel_name(city["name"]),
                "location": f"{city['name']}, {city['country']}",
                "coordinates": [round(city["lat"] + lat_offset, 6), 
                              round(city["lon"] + lon_offset, 6)],
                "star_rating": float(star_rating),
                "review_score": review_score,
                "review_count": review_count,
                "property_type": np.random.choice(self.property_types, 
                                                p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]),
                "distance_to_center": round(np.random.exponential(3), 1),
                "distance_to_airport": round(np.random.normal(25, 10), 1),
                "chain_brand": random.choice(self.chain_brands),
                "amenities": ",".join(amenities),
                "sustainability_rating": round(np.random.uniform(1, 5), 1),
                "city": city["name"],
                "country": city["country"]
            }
            hotels.append(hotel)
            
        return pd.DataFrame(hotels)
    
    def _generate_hotel_name(self, city):
        """Generate realistic hotel names"""
        prefixes = ["Grand", "Royal", "Imperial", "Palace", "Plaza", "Central", 
                   "Park", "Garden", "Golden", "Silver", "Crown", "Premier"]
        suffixes = ["Hotel", "Resort", "Inn", "Suites", "Lodge", "Manor", "House"]
        
        if random.random() < 0.3:
            return f"{random.choice(prefixes)} {city} {random.choice(suffixes)}"
        elif random.random() < 0.5:
            return f"{city} {random.choice(prefixes)} {random.choice(suffixes)}"
        else:
            return f"{fake.company().replace(',', '').replace('Inc', '').replace('LLC', '').strip()} {random.choice(suffixes)}"
    
    def generate_partner_offers(self, hotels_df, n=10000):
        """Generate realistic partner offers with price fluctuation"""
        offers = []
        
        # Generate base date for price fluctuation (last 24 hours)
        base_date = datetime.now() - timedelta(hours=24)
        
        for i in range(n):
            hotel = hotels_df.iloc[i % len(hotels_df)]
            partner = random.choice(self.partners)
            room_type = random.choice(self.room_types)
            
            # Base price influenced by star rating and location
            base_price = self._calculate_base_price(hotel)
            
            # Partner-specific pricing strategy
            partner_multiplier = self._get_partner_multiplier(partner)
            price_per_night = round(base_price * partner_multiplier, 2)
            
            # Stay duration (1-7 nights, weighted toward shorter stays)
            nights = np.random.choice([1, 2, 3, 4, 5, 6, 7], 
                                    p=[0.3, 0.25, 0.2, 0.1, 0.08, 0.04, 0.03])
            total_price = round(price_per_night * nights, 2)
            
            # Trivago displayed price (sometimes different for transparency testing)
            price_diff = np.random.normal(0, price_per_night * 0.02)
            trivago_price = round(price_per_night + price_diff, 2)
            
            # Generate price fluctuation data for past 24 hours
            price_history = self._generate_price_fluctuation(price_per_night, base_date)
            
            offer = {
                "offer_id": f"O{i+1:06d}",
                "hotel_id": hotel["hotel_id"],
                "partner_name": partner,
                "room_type": room_type,
                "price_per_night": price_per_night,
                "total_price": total_price,
                "trivago_displayed_price": trivago_price,
                "cost_per_click_bid": round(np.random.uniform(0.5, 3.0), 2),
                "commission_rate": round(np.random.uniform(0.08, 0.20), 3),
                "cancellation_policy": np.random.choice(["Free", "Non-refundable", "Partial"], 
                                                      p=[0.6, 0.25, 0.15]),
                "breakfast_included": random.choice([True, False]),
                "instant_booking": random.choice([True, False]),
                "special_offers": self._generate_special_offers(),
                "nights": nights,
                "price_fluctuation_mean": round(np.mean(price_history), 2),
                "price_fluctuation_variance": round(np.var(price_history), 2),
                "price_history_24h": json.dumps([round(p, 2) for p in price_history]),
                "last_updated": (base_date + timedelta(hours=24)).isoformat()
            }
            offers.append(offer)
            
        return pd.DataFrame(offers)
    
    def _calculate_base_price(self, hotel):
        """Calculate base price based on hotel characteristics"""
        # Base price by star rating
        star_base = {2: 60, 3: 90, 4: 150, 5: 300}
        base = star_base.get(hotel["star_rating"], 100)
        
        # Location premium for expensive cities
        expensive_cities = ["London", "Paris", "New York", "Tokyo", "Dubai", "Singapore"]
        if hotel["city"] in expensive_cities:
            base *= 1.5
        
        # Property type adjustment
        type_multipliers = {"Resort": 1.3, "Villa": 1.4, "Hotel": 1.0, 
                          "Apartment": 0.8, "B&B": 0.7, "Hostel": 0.4}
        base *= type_multipliers.get(hotel["property_type"], 1.0)
        
        # Review score premium
        if hotel["review_score"] > 8.5:
            base *= 1.2
        elif hotel["review_score"] > 7.5:
            base *= 1.1
        
        return base
    
    def _get_partner_multiplier(self, partner):
        """Get partner-specific pricing multiplier"""
        multipliers = {
            "Booking.com": 1.0, "Expedia": 1.02, "Hotels.com": 0.98,
            "Agoda": 0.95, "Priceline": 0.92, "HotelDirect": 0.88
        }
        return multipliers.get(partner, np.random.uniform(0.9, 1.1))
    
    def _generate_price_fluctuation(self, base_price, start_date):
        """Generate hourly price fluctuation for past 24 hours"""
        prices = []
        current_price = base_price
        
        for hour in range(24):
            # Random walk with mean reversion
            change = np.random.normal(0, base_price * 0.01)
            current_price += change
            
            # Mean reversion
            current_price = current_price * 0.95 + base_price * 0.05
            
            # Ensure reasonable bounds
            current_price = max(base_price * 0.8, min(base_price * 1.2, current_price))
            prices.append(current_price)
            
        return prices
    
    def _generate_special_offers(self):
        """Generate special offers text"""
        offers = [
            "Free parking", "Late checkout", "Free WiFi", "Kids stay free",
            "Free breakfast", "Airport shuttle", "Spa discount", "Room upgrade",
            "Free cancellation", "Pay at hotel", "Mobile exclusive"
        ]
        
        if random.random() < 0.3:
            return ""
        elif random.random() < 0.7:
            return random.choice(offers)
        else:
            return ", ".join(random.sample(offers, 2))
    
    def generate_user_profiles(self, n=10000):
        """Generate realistic user profiles"""
        users = []
        
        for i in range(n):
            user_type = random.choice(self.user_types)
            age_group = random.choice(self.age_groups)
            
            # Generate travel dates (next 6 months)
            travel_start = fake.date_between(start_date='+1d', end_date='+180d')
            stay_duration = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14], 
                                           p=[0.2, 0.25, 0.2, 0.1, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01])
            travel_end = travel_start + timedelta(days=int(stay_duration))
            
            # Booking date (1-365 days before travel)
            lead_time = int(np.random.exponential(30))
            lead_time = max(1, min(365, lead_time))
            booking_date = travel_start - timedelta(days=lead_time)
            
            # Group size based on user type
            if user_type == "Family":
                group_size = np.random.choice([3, 4, 5, 6], p=[0.3, 0.4, 0.2, 0.1])
            elif user_type == "Business":
                group_size = 1
            else:
                group_size = np.random.choice([1, 2, 3, 4], p=[0.4, 0.4, 0.15, 0.05])
            
            # Price sensitivity based on user type and age
            price_sensitivity = self._calculate_price_sensitivity(user_type, age_group)
            
            # Budget range
            budget_min, budget_max = self._generate_budget_range(user_type, age_group, price_sensitivity)
            
            # Loyalty status
            loyalty_prob = {"Business": [0.2, 0.3, 0.3, 0.2], "Luxury": [0.1, 0.2, 0.4, 0.3]}
            loyalty_status = np.random.choice(self.loyalty_statuses, 
                                            p=loyalty_prob.get(user_type, [0.5, 0.3, 0.15, 0.05]))
            
            # Previous bookings based on loyalty
            loyalty_bookings = {"None": (0, 2), "Silver": (3, 8), "Gold": (9, 20), "Platinum": (21, 50)}
            min_bookings, max_bookings = loyalty_bookings[loyalty_status]
            previous_bookings = random.randint(min_bookings, max_bookings)
            
            user = {
                "user_id": f"U{i+1:06d}",
                "user_type": user_type,
                "age_group": age_group,
                "location": random.choice(self.cities)["name"],
                "group_size": group_size,
                "loyalty_status": loyalty_status,
                "price_sensitivity": round(price_sensitivity, 3),
                "budget_min": budget_min,
                "budget_max": budget_max,
                "previous_bookings": previous_bookings,
                "room_type_preference": self._get_room_preference(user_type, group_size),
                "preferred_amenities": self._generate_user_amenities(user_type),
                "travel_date_start": travel_start.isoformat(),
                "travel_date_end": travel_end.isoformat(),
                "booking_date": booking_date.isoformat(),
                "lead_time_days": lead_time,
                "stay_duration": stay_duration
            }
            users.append(user)
            
        return pd.DataFrame(users)
    
    def _calculate_price_sensitivity(self, user_type, age_group):
        """Calculate price sensitivity based on user characteristics"""
        base_sensitivity = {
            "Budget": 0.8, "Leisure": 0.6, "Family": 0.7, 
            "Business": 0.3, "Luxury": 0.2
        }
        
        age_adjustment = {
            "18-25": 0.1, "26-35": 0.05, "36-45": 0.0, 
            "46-55": -0.05, "56-65": -0.1, "65+": 0.05
        }
        
        sensitivity = base_sensitivity[user_type] + age_adjustment[age_group]
        sensitivity += np.random.normal(0, 0.1)
        return max(0.0, min(1.0, sensitivity))
    
    def _generate_budget_range(self, user_type, age_group, price_sensitivity):
        """Generate realistic budget ranges"""
        base_budgets = {
            "Budget": (30, 80), "Leisure": (60, 150), "Family": (80, 200),
            "Business": (100, 300), "Luxury": (200, 800)
        }
        
        min_base, max_base = base_budgets[user_type]
        
        # Age group adjustment
        age_multipliers = {
            "18-25": 0.7, "26-35": 0.9, "36-45": 1.1, 
            "46-55": 1.3, "56-65": 1.2, "65+": 0.9
        }
        
        multiplier = age_multipliers[age_group]
        budget_min = round(min_base * multiplier * (1 - price_sensitivity * 0.3))
        budget_max = round(max_base * multiplier * (1 + (1 - price_sensitivity) * 0.5))
        
        return budget_min, budget_max
    
    def _get_room_preference(self, user_type, group_size):
        """Get room type preference based on user characteristics"""
        if group_size == 1:
            return "Single"
        elif group_size == 2:
            if user_type in ["Business", "Luxury"]:
                return random.choice(["Double", "Deluxe", "Suite"])
            else:
                return "Double"
        elif group_size <= 4:
            if user_type == "Family":
                return "Family"
            else:
                return random.choice(["Suite", "Deluxe"])
        else:
            return "Suite"
    
    def _generate_user_amenities(self, user_type):
        """Generate preferred amenities based on user type"""
        type_amenities = {
            "Business": ["WiFi", "Business Center", "Gym", "Room Service", "Parking"],
            "Family": ["Pool", "WiFi", "Breakfast", "Parking", "Restaurant"],
            "Leisure": ["Pool", "Spa", "WiFi", "Bar", "Beach Access"],
            "Budget": ["WiFi", "Breakfast", "Parking"],
            "Luxury": ["Spa", "Concierge", "Room Service", "Bar", "Restaurant"]
        }
        
        preferred = type_amenities.get(user_type, ["WiFi"])
        num_prefs = random.randint(2, min(6, len(preferred) + 2))
        
        # Add some random amenities
        all_amenities = list(set(preferred + random.sample(self.amenities_list, 3)))
        selected = random.sample(all_amenities, min(num_prefs, len(all_amenities)))
        
        return ",".join(selected)

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced hotel datasets.")
    parser.add_argument('--hotels', type=int, default=10000, help='Number of hotels to generate')
    parser.add_argument('--offers', type=int, default=10000, help='Number of partner offers to generate')
    parser.add_argument('--users', type=int, default=10000, help='Number of user profiles to generate')
    args = parser.parse_args()

    # Ensure data directory exists (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("\U0001F3E8 Starting Enhanced Hotel Dataset Generation...")
    
    generator = HotelDatasetGenerator()
    
    print(f"\U0001F4CA Generating Hotels dataset ({args.hotels} rows)...")
    hotels_df = generator.generate_hotels(args.hotels)
    hotels_path = os.path.join(data_dir, "enhanced_hotels.csv")
    hotels_df.to_csv(hotels_path, index=False)
    print(f"✅ Hotels dataset saved: {len(hotels_df)} rows, {len(hotels_df.columns)} columns")
    
    print(f"\U0001F4B0 Generating Partner Offers dataset ({args.offers} rows)...")
    offers_df = generator.generate_partner_offers(hotels_df, args.offers)
    offers_path = os.path.join(data_dir, "enhanced_partner_offers.csv")
    offers_df.to_csv(offers_path, index=False)
    print(f"✅ Partner Offers dataset saved: {len(offers_df)} rows, {len(offers_df.columns)} columns")
    
    print(f"\U0001F465 Generating User Profiles dataset ({args.users} rows)...")
    users_df = generator.generate_user_profiles(args.users)
    users_path = os.path.join(data_dir, "enhanced_user_profiles.csv")
    users_df.to_csv(users_path, index=False)
    print(f"✅ User Profiles dataset saved: {len(users_df)} rows, {len(users_df.columns)} columns")
    
    # Generate summary statistics
    print("\n📈 Dataset Summary:")
    print(f"Hotels: {len(hotels_df)} rows across {len(hotels_df['city'].unique())} cities")
    print(f"Partner Offers: {len(offers_df)} rows from {len(offers_df['partner_name'].unique())} partners")
    print(f"User Profiles: {len(users_df)} rows with {len(users_df['user_type'].unique())} user types")
    
    print("\n🎉 All datasets generated successfully!")
    print("Files created:")
    print("- enhanced_hotels.csv")
    print("- enhanced_partner_offers.csv") 
    print("- enhanced_user_profiles.csv")

if __name__ == "__main__":
    main()