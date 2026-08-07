# Week 4 API Usage Guide

## Start the service

From the project root, start the containerized API:

```powershell
docker compose up --build
```

The API is then available at `http://127.0.0.1:8000`. Interactive OpenAPI
documentation is available at `http://127.0.0.1:8000/docs`.

For local development without Docker:

```powershell
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## Check service health

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The response confirms that the API and ONNX model are loaded, and lists the
supported defect classes.

## Submit an image for defect detection

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -F "file=@dataset/test/images/crazing_281.jpg"
```

`/predict` accepts JPEG and PNG files. Its response includes the uploaded
filename, detection count, inference time, and a detection list containing the
class ID, class name, confidence, and `[x1, y1, x2, y2]` bounding box.

## Send a PLC/external-system event

```powershell
$payload = @{
  x = 120
  y = 80
  defect = "scratches"
  confidence = 0.87
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/plc/send `
  -Method Post `
  -ContentType "application/json" `
  -Body $payload
```

The endpoint returns the accepted payload with `status: success`. To send the
project's predefined sample events, run:

```powershell
python src/plc_sender.py
```

## Inspect monitoring metrics

```powershell
Invoke-WebRequest http://127.0.0.1:8000/metrics | Select-Object -Expand Content
```

Prometheus metrics include request count, request latency, API uptime, and
inference FPS. Runtime request logs are written to `logs/app.log`.

## Run the benchmark

With the API running, execute:

```powershell
python src/api_benchmark.py
```

The benchmark exercises `/health`, `/plc/send`, and `/predict` at concurrency
levels 1, 5, and 10. It writes the results to
`docs/week4_performance_metrics.txt`.
