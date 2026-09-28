param(
    [Parameter(Mandatory = $true)]
    [string]$PdfUrl,
    [string]$Image = "legal-notice-detector:r7",
    [string]$OutputRoot,
    [string]$JobId,
    [string]$Outputs = "json,crops,annotated_pages,rendered_pages",
    [string]$PythonVersion = "3.13",
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $Root "output"
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$OutputRoot = (Resolve-Path $OutputRoot).Path

& docker image inspect $Image --format "{{.Id}}" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker image does not exist: $Image"
}

$PreviousUrl = $env:DETECTOR_PDF_URL
$PreviousJobId = $env:DETECTOR_JOB_ID
$PreviousOutputs = $env:DETECTOR_OUTPUTS
$env:DETECTOR_PDF_URL = $PdfUrl
$env:DETECTOR_JOB_ID = $JobId
$env:DETECTOR_OUTPUTS = $Outputs

$DockerArguments = @(
    "run", "--rm", "-i",
    "--platform", "linux/amd64",
    "--entrypoint", "/bin/bash",
    "--env", "DETECTOR_PDF_URL",
    "--env", "DETECTOR_JOB_ID",
    "--env", "DETECTOR_OUTPUTS",
    "--env", "PYTHON_VERSION=$PythonVersion",
    "--mount", "type=bind,source=$OutputRoot,target=/tmp/doc-detector/jobs"
)
if ($Dev) {
    $SourceDir = Join-Path $Root "doc_detector"
    $DockerArguments += @(
        "--mount", "type=bind,source=$SourceDir,target=/opt/doc-detector/doc_detector,readonly"
    )
}
$DockerArguments += @(
    $Image,
    "-lc",
    'source /opt/code-interpreter/code-interpreter-env.sh python "${PYTHON_VERSION}" && python -'
)

try {
    Get-Content -LiteralPath (Join-Path $Root "examples/local_detection_task.py") -Raw |
        & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Local URL detection failed with exit code $LASTEXITCODE."
    }
    "Detection outputs are under $OutputRoot"
}
finally {
    $env:DETECTOR_PDF_URL = $PreviousUrl
    $env:DETECTOR_JOB_ID = $PreviousJobId
    $env:DETECTOR_OUTPUTS = $PreviousOutputs
}
