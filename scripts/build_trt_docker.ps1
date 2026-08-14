Param(
    [string]$Image = 'nvcr.io/nvidia/tensorrt:23.09-py3',
    [string]$OnnxPath = 'runs/detect/train/weights/best.onnx',
    [string]$EngineOut = 'model.engine'
)

if (-Not (Test-Path $OnnxPath)) {
    Write-Error "ONNX file not found: $OnnxPath"
    exit 2
}

$pwdHost = (Get-Location).Path

Write-Output "Running trtexec inside container $Image"
docker run --rm --gpus all -v "${pwdHost}:/workspace" -w /workspace $Image trtexec --onnx=$OnnxPath --saveEngine=$EngineOut --workspace=4096 --fp16

Write-Output "Engine saved to: $EngineOut"
