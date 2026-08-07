import requests

API_URL = "http://127.0.0.1:8000/plc/send"

test_payloads = [
    {
        "x": 120,
        "y": 80,
        "defect": "crazing",
        "confidence": 0.95
    },
    {
        "x": 150,
        "y": 100,
        "defect": "inclusion",
        "confidence": 0.91
    },
    {
        "x": 180,
        "y": 120,
        "defect": "patches",
        "confidence": 0.89
    },
    {
        "x": 200,
        "y": 140,
        "defect": "pitted_surface",
        "confidence": 0.93
    },
    {
        "x": 220,
        "y": 160,
        "defect": "rolled_in_scale",
        "confidence": 0.88
    },
    {
        "x": 240,
        "y": 180,
        "defect": "scratches",
        "confidence": 0.87
    }
]

for payload in test_payloads:

    print("----------------------------------")
    print("Sending defect:", payload["defect"])

    try:
        response = requests.post(
            API_URL,
            json=payload
        )

        print("Status Code:", response.status_code)
        print("Response:", response.text)

    except Exception as e:
        print("Error:", e)