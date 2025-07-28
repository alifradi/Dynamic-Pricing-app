import urllib.request
import urllib.parse
import json

# Test the three-stage optimization endpoint
url = "http://localhost:8001/run_three_stage_optimization"
data = {
    "alpha": "0.4",
    "beta": "0.3", 
    "gamma": "0.3",
    "confidence_level": "0.8",
    "num_positions": "10"
}

# Encode the data
data = urllib.parse.urlencode(data).encode('utf-8')

try:
    # Create the request
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    
    # Send the request
    with urllib.request.urlopen(req) as response:
        result = response.read().decode('utf-8')
        print(f"Status Code: {response.status}")
        print(f"Response: {result}")
        
        if response.status == 200:
            json_result = json.loads(result)
            print(f"Success! Model type: {json_result.get('model_type')}")
            print(f"Parameters: {json_result.get('parameters')}")
            
except Exception as e:
    print(f"Exception: {e}") 