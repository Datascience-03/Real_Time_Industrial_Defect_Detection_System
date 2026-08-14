Param(
    [string]$OutDir = 'release'
)

Test-Path $OutDir -PathType Container | Out-Null
if (-Not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Copy-Item -Path 'runs/detect/train/weights/best.onnx' -Destination $OutDir -ErrorAction SilentlyContinue
Copy-Item -Path 'model.engine' -Destination $OutDir -ErrorAction SilentlyContinue
Copy-Item -Path 'requirements-pinned.txt' -Destination $OutDir -ErrorAction SilentlyContinue
Copy-Item -Path 'Dockerfile.gpu.trt' -Destination $OutDir -ErrorAction SilentlyContinue

Push-Location $OutDir
try {
    if (Get-Command sha256sum -ErrorAction SilentlyContinue) {
        sha256sum * > release.sha256
    } else {
        Get-ChildItem -File | ForEach-Object {
            $h = Get-FileHash $_.FullName -Algorithm SHA256
            "{0}  {1}" -f $h.Hash, $_.Name
        } | Out-File -Encoding ascii release.sha256
    }
    Compress-Archive -Path * -DestinationPath release-artifacts.zip -Force
} finally {
    Pop-Location
}

Write-Output "Release artifacts prepared in $OutDir"
