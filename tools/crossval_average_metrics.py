import sys
import json
from pathlib import Path

def calculate_average_metrics(json_files):
    metrics_sum = {}
    count = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if key not in metrics_sum:
                    metrics_sum[key] = 0
                metrics_sum[key] += value
            count += 1

    if count == 0:
        print("No metrics to average.")
        return

    # Calculate average for each metric
    metrics_avg = {key: value / count for key, value in metrics_sum.items()}

    print("Average metrics over 5 folds:")
    for key, value in metrics_avg.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/calculate_average_metrics.py <json_file1> <json_file2> ...")
        sys.exit(1)

    json_files = sys.argv[1:]
    calculate_average_metrics(json_files)
