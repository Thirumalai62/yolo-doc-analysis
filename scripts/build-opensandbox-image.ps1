param(
    [string]$Image = "legal-notice-detector:r7",
    [string]$BaseImage = "opensandbox/code-interpreter:v1.1.0",
    [string]$PythonVersion = "3.13",
    [string]$ModelSource,
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $ModelSource) {
    $ModelSource = Join-Path $Root "runs/legal_notice_v8_protected_head_cpu_r7_paa_analogue/weights/candidate.pt"
}
$ExpectedSha256 = "2fbbad3b81969beefb1ffd428a5bc89dd83beaa9238f262ce67f25a63e237f76"
$StagingDir = Join-Path $Root ".build/opensandbox"

if ($Release -and $BaseImage -notmatch "@sha256:[0-9a-f]{64}$") {
    throw "Release builds require BaseImage to include an immutable sha256 digest."
}

if (-not (Test-Path -LiteralPath $ModelSource -PathType Leaf)) {
    throw "R7 model does not exist: $ModelSource"
}
$ActualSha256 = (Get-FileHash -LiteralPath $ModelSource -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256) {
    throw "R7 model checksum mismatch. Expected $ExpectedSha256, got $ActualSha256."
}

try {
    New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null
    Copy-Item -LiteralPath $ModelSource -Destination (Join-Path $StagingDir "legal_notice_r7.pt")
    & docker build `
        --platform linux/amd64 `
        --file (Join-Path $Root "Dockerfile.opensandbox") `
        --tag $Image `
        --build-arg "BASE_IMAGE=$BaseImage" `
        --build-arg "PYTHON_VERSION=$PythonVersion" `
        $Root
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed with exit code $LASTEXITCODE."
    }
    "Built $Image"
}
finally {
    Remove-Item -LiteralPath $StagingDir -Recurse -Force -ErrorAction SilentlyContinue
}
