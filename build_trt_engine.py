import os
import time
import shutil
import subprocess
import numpy as np
from ultralytics import YOLO

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except Exception:
    trt = None
    TRT_AVAILABLE = False

def build_engine_from_onnx(onnx_path="model.onnx", engine_path="model.engine"):
    """Attempt to build a TensorRT engine from an ONNX model.

    This function prefers native TensorRT Python bindings. If they are not
    available it will fall back to invoking the `trtexec` CLI if present on PATH.
    If neither is available it prints instructions for the user.
    """
    if TRT_AVAILABLE:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH') else 0
        network = builder.create_network(flag) if flag else builder.create_network()
        parser = trt.OnnxParser(network, logger)

        print(f"[1/2] Reading '{onnx_path}' and compiling TensorRT Engine using TensorRT Python API...")
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX file:")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return False

        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        else:
            config.max_workspace_size = 1 << 30

        if hasattr(trt.BuilderFlag, 'FP16'):
            config.set_flag(trt.BuilderFlag.FP16)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Error: Engine build failed!")
            return False

        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        print(f" Successfully compiled TensorRT engine: '{engine_path}'")
        return True

    # Fallback: use trtexec CLI if available
    trtexec = shutil.which('trtexec')
    if trtexec:
        print(f"TensorRT Python bindings not found; using trtexec at {trtexec} to build engine...")
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--workspace=4096",
            "--fp16"
        ]
        try:
            res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(res.stdout)
            return os.path.exists(engine_path)
        except subprocess.CalledProcessError as e:
            print("trtexec failed:")
            print(e.stdout)
            return False

    print("TensorRT not available: neither Python bindings nor `trtexec` found on PATH.")
    print("To build a TensorRT engine, either: (a) install TensorRT Python bindings on the host, or (b) run inside an NVIDIA container that includes `trtexec`/TensorRT.")
    return False

def benchmark_engine(engine_path="model.engine"):
    """
    Benchmarks 50 inference runs on the RTX 3050 GPU.
    """
    print("\n[2/2] Benchmarking TensorRT Inference Speed...")
    
    trt_model = YOLO(engine_path, task="detect")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        _ = trt_model(dummy_img, verbose=False)

    num_runs = 50
    start_time = time.time()
    for _ in range(num_runs):
        _ = trt_model(dummy_img, verbose=False)
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency = (total_time / num_runs) * 1000
    fps = num_runs / total_time

    results_str = (
        f"\n==================================================\n"
        f"           TENSORRT BENCHMARK RESULTS           \n"
        f"==================================================\n"
        f" Model Engine:   {engine_path}\n"
        f" GPU Device:     NVIDIA GeForce RTX 3050\n"
        f" Avg Latency:    {avg_latency:.2f} ms per frame\n"
        f" Throughput:     {fps:.2f} FPS\n"
        f"==================================================\n"
    )
    print(results_str)

    with open("benchmark_results.txt", "w") as f:
        f.write(results_str)
    print(" Saved report to 'benchmark_results.txt'")

def main():
    onnx_file = "model.onnx"
    engine_file = "model.engine"

    if not os.path.exists(onnx_file):
        print(f"Exporting '{onnx_file}'...")
        model = YOLO("yolov8n.pt")
        model.export(format="onnx")
        if os.path.exists("yolov8n.onnx"):
            os.replace("yolov8n.onnx", onnx_file)

    if build_engine_from_onnx(onnx_file, engine_file):
        benchmark_engine(engine_file)

if __name__ == "__main__":
    main() 