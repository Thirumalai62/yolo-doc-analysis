param(
    [string]$Image = "legal-notice-detector:r7",
    [ValidateSet("slim", "full")]
    [string]$Variant = "slim",
    [string]$BaseImage,
    [string]$PythonVersion = "3.13",
    [string]$ModelVersion = "r7",
    [string]$ModelSource,
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DockerfileName = if ($Variant -eq "full") { "Dockerfile.opensandbox.full" } else { "Dockerfile.opensandbox" }
if ([string]::IsNullOrWhiteSpace($BaseImage)) {
    $BaseImage = if ($Variant -eq "full") {
        "opensandbox/code-interpreter:v1.1.0"
    }
    else {
        "python:3.13.13-slim-bookworm@sha256:355bfa66770995d7e9a0da4b3473b44d0cb451f6b56f5615ad9c39e3c4eca03f"
    }
}
$ManifestPath = Join-Path $Root "model-manifest.json"
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Model manifest does not exist: $ManifestPath"
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$Model = $Manifest.models.$ModelVersion
if (-not $Model) {
    throw "Model version '$ModelVersion' is not defined in $ManifestPath."
}
$ExpectedSha256 = [string]$Model.sha256
if ($ExpectedSha256 -notmatch "^[0-9a-f]{64}$") {
    throw "Model version '$ModelVersion' has an invalid SHA256 in $ManifestPath."
}
if (-not $ModelSource) {
    $ModelSource = Join-Path $Root ([string]$Model.file)
}
$StagingDir = Join-Path $Root ".build/opensandbox"

if ($Release -and $BaseImage -notmatch "@sha256:[0-9a-f]{64}$") {
    throw "Release builds require BaseImage to include an immutable sha256 digest."
}

if (-not (Test-Path -LiteralPath $ModelSource -PathType Leaf)) {
    throw "Model '$ModelVersion' does not exist: $ModelSource. Run 'git lfs pull' in a deployment clone or provide -ModelSource."
}
$ModelStream = [System.IO.File]::OpenRead($ModelSource)
try {
    $HeaderBytes = New-Object byte[] 128
    $HeaderLength = $ModelStream.Read($HeaderBytes, 0, $HeaderBytes.Length)
    $ModelHeader = [System.Text.Encoding]::UTF8.GetString($HeaderBytes, 0, $HeaderLength)
}
finally {
    $ModelStream.Dispose()
}
if ($ModelHeader.StartsWith("version https://git-lfs.github.com/spec/v1")) {
    throw "Model '$ModelVersion' is an unresolved Git LFS pointer. Run 'git lfs pull'."
}
$ActualSha256 = (Get-FileHash -LiteralPath $ModelSource -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256) {
    throw "Model '$ModelVersion' checksum mismatch. Expected $ExpectedSha256, got $ActualSha256."
}

try {
    New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null
    Copy-Item -LiteralPath $ModelSource -Destination (Join-Path $StagingDir "legal_notice_r7.pt")
    & docker build `
        --platform linux/amd64 `
        --file (Join-Path $Root $DockerfileName) `
        --tag $Image `
        --build-arg "BASE_IMAGE=$BaseImage" `
        --build-arg "PYTHON_VERSION=$PythonVersion" `
        $Root
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed with exit code $LASTEXITCODE."
    }
    "Built $Image ($Variant runtime)"
}
finally {
    Remove-Item -LiteralPath $StagingDir -Recurse -Force -ErrorAction SilentlyContinue
}
