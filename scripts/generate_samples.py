#!/usr/bin/env python
"""Generate sample prediction requests for testing the API."""
import json
import random
import argparse
import sys


def generate_sample(positive_bias: bool = False) -> dict:
    """Generate a random patient sample.

    Args:
        positive_bias: If True, generate samples more likely to be positive.
    """
    if positive_bias:
        return {
            "age": random.randint(55, 77),
            "sex": random.choice([1, 1, 0]),  # More males
            "cp": random.choice([2, 3, 3]),  # Higher chest pain types
            "trestbps": random.randint(140, 180),
            "chol": random.randint(240, 400),
            "fbs": random.choice([0, 1, 1]),  # Higher blood sugar
            "restecg": random.randint(0, 2),
            "thalach": random.randint(100, 140),  # Lower max heart rate
            "exang": random.choice([0, 1, 1]),  # Exercise induced angina
            "oldpeak": round(random.uniform(1.5, 4.0), 1),
            "slope": random.choice([0, 1]),
            "ca": random.randint(1, 3),  # More vessels
            "thal": random.choice([2, 3, 3]),  # Thalassemia
        }
    else:
        return {
            "age": random.randint(29, 77),
            "sex": random.randint(0, 1),
            "cp": random.randint(0, 3),
            "trestbps": random.randint(94, 200),
            "chol": random.randint(126, 564),
            "fbs": random.randint(0, 1),
            "restecg": random.randint(0, 2),
            "thalach": random.randint(71, 202),
            "exang": random.randint(0, 1),
            "oldpeak": round(random.uniform(0, 6.2), 1),
            "slope": random.randint(0, 2),
            "ca": random.randint(0, 4),
            "thal": random.randint(0, 3),
        }


def generate_batch(n: int, positive_bias: bool = False) -> dict:
    """Generate a batch of samples."""
    return {"data": [generate_sample(positive_bias) for _ in range(n)]}


def main():
    parser = argparse.ArgumentParser(description="Generate sample prediction requests")
    parser.add_argument("-n", "--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument(
        "-p", "--positive-bias", action="store_true", help="Generate samples biased towards positive prediction"
    )
    parser.add_argument("-o", "--output", type=str, default="-", help="Output file (- for stdout)")
    parser.add_argument("--curl", action="store_true", help="Output as curl command")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API URL for curl command")

    args = parser.parse_args()

    data = generate_batch(args.num_samples, args.positive_bias)
    json_str = json.dumps(data, indent=2)

    if args.curl:
        curl_cmd = f"""curl -X POST {args.url}/predict \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(data)}'"""
        output = curl_cmd
    else:
        output = json_str

    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
