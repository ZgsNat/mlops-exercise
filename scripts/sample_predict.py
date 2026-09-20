import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from src.data import prepare_splits, TARGET_NAMES


def main():
    api_url = "http://localhost:18013/predict"
    if len(sys.argv) > 1:
        api_url = sys.argv[1]

    _, X_test, _, y_test = prepare_splits(random_state=42)

    # Pick sample row 0
    sample_row = X_test.iloc[0].to_dict()
    true_label = TARGET_NAMES[int(y_test.iloc[0])]

    payload = sample_row

    print("=" * 60)
    print("WINE CLASSIFIER SAMPLE REQUEST")
    print(f"Target API Endpoint: {api_url}")
    print(f"Ground Truth Class: {true_label}")
    print("-" * 60)
    print("Payload JSON:")
    print(json.dumps(payload, indent=2))
    print("-" * 60)

    try:
        response = requests.post(api_url, json=payload, timeout=5)
        print(f"Response Status: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
        print("=" * 60)
    except requests.exceptions.RequestException as e:
        print(f"Error querying API: {e}")
        print(f"Make sure the API container is running at {api_url}")


if __name__ == "__main__":
    main()
