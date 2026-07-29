import os
import time
import numpy as np
from ultralytics import YOLO

def benchmark_model(model_path, device="cpu", num_runs=50, imgsz=640):
    """
    Benchmarks a YOLO model (PyTorch or ONNX) for a given number of runs.
    """
    print(f"Loading model: {model_path} on {device}...")
    model = YOLO(model_path)
    
    # Create dummy image matching target resolution
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    
    # Warmup runs
    print("Warming up model...")
    for _ in range(10):
        _ = model(dummy_img, device=device, verbose=False)
        
    print(f"Running benchmark ({num_runs} iterations)...")
    latencies = []
    
    for i in range(num_runs):
        t0 = time.time()
        _ = model(dummy_img, device=device, verbose=False)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # milliseconds
        
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency
    
    print(f"Results for {os.path.basename(model_path)}:")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    print(f"  Std Dev:     {std_latency:.2f} ms")
    print(f"  Throughput:  {fps:.2f} FPS")
    print("-" * 40)
    
    return avg_latency, fps

def main():
    pt_path = "runs/detect/train/weights/best.pt"
    onnx_path = "runs/detect/train/weights/best.onnx"
    
    # Verify files exist
    if not os.path.exists(pt_path):
        print(f"Error: PyTorch model not found at {pt_path}")
        return
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return
        
    print("==========================================")
    print("        CPU INFERENCE BENCHMARKING        ")
    print("==========================================")
    
    pt_latency, pt_fps = benchmark_model(pt_path, device="cpu")
    onnx_latency, onnx_fps = benchmark_model(onnx_path, device="cpu")
    
    # Read TensorRT results from benchmark_results.txt if available
    trt_latency, trt_fps = None, None
    trt_file = "benchmark_results.txt"
    if os.path.exists(trt_file):
        try:
            with open(trt_file, "r") as f:
                content = f.read()
                # Parse Avg Latency and Throughput
                for line in content.split("\n"):
                    if "Avg Latency:" in line:
                        trt_latency = float(line.split("Avg Latency:")[1].split("ms")[0].strip())
                    if "Throughput:" in line:
                        trt_fps = float(line.split("Throughput:")[1].split("FPS")[0].strip())
            print(f"Loaded TensorRT GPU metrics from '{trt_file}':")
            print(f"  Avg Latency: {trt_latency:.2f} ms")
            print(f"  Throughput:  {trt_fps:.2f} FPS")
        except Exception as e:
            print(f"Warning: Could not parse {trt_file}: {e}")
            
    # Write unified report
    report_path = "docs/performance_metrics.txt"
    os.makedirs("docs", exist_ok=True)
    with open(report_path, "w") as f:
        f.write("==================================================\n")
        f.write("     DEFECT DETECTION MODEL PERFORMANCE REPORT     \n")
        f.write("==================================================\n")
        f.write(f"PyTorch (CPU):\n")
        f.write(f"  Avg Latency: {pt_latency:.2f} ms\n")
        f.write(f"  Throughput:  {pt_fps:.2f} FPS\n\n")
        
        f.write(f"ONNX Runtime (CPU):\n")
        f.write(f"  Avg Latency: {onnx_latency:.2f} ms\n")
        f.write(f"  Throughput:  {onnx_fps:.2f} FPS\n\n")
        
        if trt_latency is not None:
            f.write(f"TensorRT (NVIDIA RTX 3050 GPU):\n")
            f.write(f"  Avg Latency: {trt_latency:.2f} ms\n")
            f.write(f"  Throughput:  {trt_fps:.2f} FPS\n\n")
            
            # Speedup calculations
            cpu_onnx_speedup = pt_latency / onnx_latency
            gpu_trt_speedup = pt_latency / trt_latency
            f.write("Speedup Analysis:\n")
            f.write(f"  ONNX (CPU) vs PyTorch (CPU): {cpu_onnx_speedup:.2f}x faster\n")
            f.write(f"  TensorRT (GPU) vs PyTorch (CPU): {gpu_trt_speedup:.2f}x faster\n")
            f.write(f"  TensorRT (GPU) vs ONNX (CPU): {(onnx_latency / trt_latency):.2f}x faster\n")
        f.write("==================================================\n")
        
    print(f"\nUnified benchmark report saved to '{report_path}'")

if __name__ == "__main__":
    main()
