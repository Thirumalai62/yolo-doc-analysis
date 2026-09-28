param(
    [string]$Image = "legal-notice-detector:r7",
    [string]$PythonVersion = "3.13"
)

$ErrorActionPreference = "Stop"
& docker run --rm `
    --platform linux/amd64 `
    --entrypoint /bin/bash `
    --env "PYTHON_VERSION=$PythonVersion" `
    $Image `
    -lc 'source /opt/code-interpreter/code-interpreter-env.sh python "${PYTHON_VERSION}" && python -c "from doc_detector import get_detector; detector = get_detector(); print(detector.warmup())"'
if ($LASTEXITCODE -ne 0) {
    throw "Image smoke test failed with exit code $LASTEXITCODE."
}
"OpenSandbox detector image smoke test passed: $Image"
