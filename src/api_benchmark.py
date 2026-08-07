import time
import requests
import concurrent.futures
import numpy as np
import os
from pathlib import Path

# ==================================================
# Configuration
# ==================================================
API_URL = "http://127.0.0.1:8000"
NUM_REQUESTS = 100
CONCURRENCY_LEVELS = [1, 5, 10]
TEST_IMAGE = Path("dataset/test/images/rolled-in_scale_277.jpg")

# ==================================================
# Helper to check API status
# ==================================================
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# ==================================================
# Benchmarking task functions
# ==================================================
def test_health():
    start = time.time()
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        success = r.status_code == 200
    except Exception:
        success = False
    return time.time() - start, success

def test_plc():
    payload = {
        "x": 145.5,
        "y": 92.0,
        "defect": "crazing",
        "confidence": 0.94
    }
    start = time.time()
    try:
        r = requests.post(f"{API_URL}/plc/send", json=payload, timeout=5)
        success = r.status_code == 200
    except Exception:
        success = False
    return time.time() - start, success

def test_predict(image_bytes):
    start = time.time()
    try:
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        r = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        success = r.status_code == 200
    except Exception:
        success = False
    return time.time() - start, success

# ==================================================
# Performance Analysis Suite
# ==================================================
def run_benchmark_for_endpoint(name, test_func, concurrency=1):
    latencies = []
    success_count = 0
    
    start_total = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(test_func) for _ in range(NUM_REQUESTS)]
        for fut in concurrent.futures.as_completed(futures):
            latency, success = fut.result()
            latencies.append(latency)
            if success:
                success_count += 1
                
    total_time = time.time() - start_total
    rps = len(latencies) / total_time
    
    latencies_ms = [l * 1000 for l in latencies]
    avg_lat = np.mean(latencies_ms)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    p50 = np.percentile(latencies_ms, 50)
    p90 = np.percentile(latencies_ms, 90)
    p99 = np.percentile(latencies_ms, 99)
    
    return {
        "concurrency": concurrency,
        "success_rate": (success_count / len(latencies)) * 100,
        "rps": rps,
        "avg": avg_lat,
        "min": min_lat,
        "max": max_lat,
        "p50": p50,
        "p90": p90,
        "p99": p99
    }

def main():
    print("==================================================")
    print("         FastAPI Endpoint Benchmark Suite         ")
    print("==================================================")
    
    if not check_api():
        print(f"ERROR: Cannot connect to FastAPI server at {API_URL}.")
        print("Please ensure uvicorn is running: python -m uvicorn src.app:app --port 8000")
        return
        
    if not TEST_IMAGE.exists():
        print(f"ERROR: Test image not found at {TEST_IMAGE}.")
        return

    with open(TEST_IMAGE, "rb") as f:
        image_bytes = f.read()

    results_report = []
    results_report.append("==================================================")
    results_report.append("      WEEK 4 API PERFORMANCE BENCHMARK REPORT      ")
    results_report.append("==================================================")
    results_report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    results_report.append(f"Total requests per test run: {NUM_REQUESTS}")
    results_report.append(f"Target API base URL: {API_URL}")
    results_report.append("==================================================\n")

    # 1. Health Endpoint GET /health
    results_report.append("--- Endpoint: GET /health ---")
    for c in CONCURRENCY_LEVELS:
        res = run_benchmark_for_endpoint("GET /health", test_health, concurrency=c)
        results_report.append(
            f"Concurrency {res['concurrency']:<2} | Success: {res['success_rate']:.1f}% | RPS: {res['rps']:.2f} | "
            f"Latency: Avg={res['avg']:.2f}ms, P50={res['p50']:.2f}ms, P90={res['p90']:.2f}ms, P99={res['p99']:.2f}ms"
        )
    results_report.append("")

    # 2. PLC Endpoint POST /plc/send
    results_report.append("--- Endpoint: POST /plc/send ---")
    for c in CONCURRENCY_LEVELS:
        res = run_benchmark_for_endpoint("POST /plc/send", test_plc, concurrency=c)
        results_report.append(
            f"Concurrency {res['concurrency']:<2} | Success: {res['success_rate']:.1f}% | RPS: {res['rps']:.2f} | "
            f"Latency: Avg={res['avg']:.2f}ms, P50={res['p50']:.2f}ms, P90={res['p90']:.2f}ms, P99={res['p99']:.2f}ms"
        )
    results_report.append("")

    # 3. Predict Endpoint POST /predict
    results_report.append("--- Endpoint: POST /predict (YOLO ONNX Inference) ---")
    for c in CONCURRENCY_LEVELS:
        res = run_benchmark_for_endpoint("POST /predict", lambda: test_predict(image_bytes), concurrency=c)
        results_report.append(
            f"Concurrency {res['concurrency']:<2} | Success: {res['success_rate']:.1f}% | RPS: {res['rps']:.2f} | "
            f"Latency: Avg={res['avg']:.2f}ms, P50={res['p50']:.2f}ms, P90={res['p90']:.2f}ms, P99={res['p99']:.2f}ms"
        )
    results_report.append("\n==================================================")
    results_report.append("                  End of Report                   ")
    results_report.append("==================================================")

    report_content = "\n".join(results_report)
    print(report_content)
    
    # Save the report to docs
    output_path = Path("docs/week4_performance_metrics.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        out.write(report_content)
    print(f"\nReport successfully saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
