#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Test budget generation
partner_budgets = {
    "Booking.com": (50000, 200000),
    "Expedia": (40000, 180000),
    "Hotels.com": (30000, 150000)
}

offers = []
for i in range(5):
    partner = "Booking.com"
    budget_min, budget_max = partner_budgets.get(partner, (1000, 10000))
    partner_marketing_budget = round(np.random.uniform(budget_min, budget_max), 2)
    
    offer = {
        "offer_id": f"O{i+1:06d}",
        "partner_name": partner,
        "partner_marketing_budget": partner_marketing_budget,
        "remaining_budget": partner_marketing_budget
    }
    offers.append(offer)

df = pd.DataFrame(offers)
print("Test DataFrame columns:", df.columns.tolist())
print("Sample data:")
print(df.head())
print("Budget columns found:", [col for col in df.columns if 'budget' in col.lower()]) 