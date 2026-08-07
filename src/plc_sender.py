import requests

API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "x": 120,
    "y": 80,
    "defect": "crack",
    "confidence": 0.95
}

try:
    response = requests.post(API_URL, json=payload)
    print(response.status_code)
    print(response.text)
except Exception as e:
    print(e)