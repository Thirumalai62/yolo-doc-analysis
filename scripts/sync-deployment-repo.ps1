param(
    [string]$Destination = "E:\legal-pdf-detector",
    [string]$ModelVersion = "r7",
    [string]$ModelSource,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ManifestPath = Join-Path $SourceRoot "model-manifest.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$Model = $Manifest.models.$ModelVersion
if (-not $Model) {
    throw "Model version '$ModelVersion' is not defined in $ManifestPath."
}
$ExpectedModelSha256 = [string]$Model.sha256
if ($ExpectedModelSha256 -notmatch "^[0-9a-f]{64}$") {
    throw "Model version '$ModelVersion' has an invalid SHA256."
}
if (-not $ModelSource) {
    $ModelSource = Join-Path $SourceRoot ([string]$Model.file)
}
if (-not (Test-Path -LiteralPath $ModelSource -PathType Leaf)) {
    throw "Approved model does not exist: $ModelSource"
}
$ActualModelSha256 = (Get-FileHash -LiteralPath $ModelSource -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualModelSha256 -ne $ExpectedModelSha256) {
    throw "Model checksum mismatch. Expected $ExpectedModelSha256, got $ActualModelSha256."
}

$Mappings = [System.Collections.Generic.List[object]]::new()
function Add-ManagedFile([string]$Source, [string]$Target) {
    $Mappings.Add([pscustomobject]@{ Source = $Source; Target = $Target })
}
function Add-ManagedTree([string]$SourceDirectory, [string]$TargetDirectory) {
    Get-ChildItem -LiteralPath $SourceDirectory -File -Recurse |
        Where-Object {
            $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and
            $_.Extension -notin @('.pyc', '.pyo')
        } |
        ForEach-Object {
            $Relative = $_.FullName.Substring($SourceDirectory.Length).TrimStart('\', '/')
            Add-ManagedFile $_.FullName (Join-Path $TargetDirectory $Relative)
        }
}

foreach ($Name in @(
    "Dockerfile.opensandbox",
    "Dockerfile.opensandbox.dockerignore",
    "model-manifest.json",
    "requirements-sandbox.in",
    "requirements-sandbox.lock"
)) {
    Add-ManagedFile (Join-Path $SourceRoot $Name) $Name
}
Add-ManagedFile (Join-Path $SourceRoot "deployment/.gitattributes") ".gitattributes"
Add-ManagedFile (Join-Path $SourceRoot "deployment/.gitignore") ".gitignore"
Add-ManagedFile (Join-Path $SourceRoot "deployment/AGENTS.md") "AGENTS.md"
Add-ManagedFile (Join-Path $SourceRoot "deployment/README.md") "README.md"
Add-ManagedFile (Join-Path $SourceRoot "deployment/validation.md") "docs/validation.md"
Add-ManagedFile (Join-Path $SourceRoot "docs/opensandbox-deployment.md") "docs/opensandbox-deployment.md"
Add-ManagedFile (Join-Path $SourceRoot "examples/local_detection_task.py") "examples/local_detection_task.py"
Add-ManagedFile (Join-Path $SourceRoot "examples/sandbox_detection_task.py") "examples/sandbox_detection_task.py"
foreach ($Name in @(
    "build-opensandbox-image.ps1",
    "build-opensandbox-image.sh",
    "run-local-detector.ps1",
    "run-local-detector.sh",
    "smoke-test-opensandbox-image.ps1",
    "smoke-test-opensandbox-image.sh"
)) {
    Add-ManagedFile (Join-Path $SourceRoot "scripts/$Name") "scripts/$Name"
}
Add-ManagedTree (Join-Path $SourceRoot "doc_detector") "doc_detector"
Add-ManagedTree (Join-Path $SourceRoot "tests") "tests"
Add-ManagedFile $ModelSource ([string]$Model.file)

$StatePath = Join-Path $Destination ".deployment-sync-state.json"
$PreviousState = $null
if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
    $PreviousState = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
}
$DesiredTargets = @($Mappings.Target | ForEach-Object { $_.Replace('\', '/') })
$Conflicts = [System.Collections.Generic.List[string]]::new()
if ($PreviousState) {
    foreach ($PreviousFile in $PreviousState.files) {
        $TargetPath = Join-Path $Destination ([string]$PreviousFile.path)
        if (Test-Path -LiteralPath $TargetPath -PathType Leaf) {
            $CurrentHash = (Get-FileHash -LiteralPath $TargetPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($CurrentHash -ne [string]$PreviousFile.sha256) {
                $Conflicts.Add([string]$PreviousFile.path)
            }
        }
    }
}
if ($Conflicts.Count -gt 0 -and -not $Force) {
    throw "Destination managed files changed since the last sync: $($Conflicts -join ', '). Review them or rerun with -Force."
}

$SourceCommit = (& git -C $SourceRoot rev-parse HEAD 2>$null)
$SourceStatus = @(& git -C $SourceRoot status --porcelain 2>$null)
$ReleaseInfo = [ordered]@{
    schema_version = 1
    exported_at_utc = [DateTime]::UtcNow.ToString("o")
    source_repository = $SourceRoot
    source_commit = if ($LASTEXITCODE -eq 0) { $SourceCommit } else { $null }
    source_dirty = $SourceStatus.Count -gt 0
    model_version = $ModelVersion
    model_sha256 = $ExpectedModelSha256
    model_file = [string]$Model.file
}

if ($DryRun) {
    "Deployment sync preview:"
    "  Source:      $SourceRoot"
    "  Destination: $Destination"
    "  Model:       $ModelVersion ($ExpectedModelSha256)"
    "  Files:       $($Mappings.Count + 1)"
    foreach ($Mapping in $Mappings) {
        "  COPY $($Mapping.Target)"
    }
    "  WRITE release-info.json"
    return
}

[System.IO.Directory]::CreateDirectory($Destination) | Out-Null
if ($PreviousState) {
    foreach ($PreviousFile in $PreviousState.files) {
        $PreviousTarget = ([string]$PreviousFile.path).Replace('\', '/')
        if ($PreviousTarget -notin $DesiredTargets -and $PreviousTarget -ne "release-info.json") {
            $ObsoletePath = Join-Path $Destination $PreviousTarget
            if (Test-Path -LiteralPath $ObsoletePath -PathType Leaf) {
                Remove-Item -LiteralPath $ObsoletePath -Force
            }
        }
    }
}
foreach ($Mapping in $Mappings) {
    $TargetPath = Join-Path $Destination $Mapping.Target
    [System.IO.Directory]::CreateDirectory((Split-Path $TargetPath -Parent)) | Out-Null
    [System.IO.File]::Copy($Mapping.Source, $TargetPath, $true)
}
$ReleasePath = Join-Path $Destination "release-info.json"
$ReleaseInfo | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ReleasePath -Encoding UTF8

$StateFiles = [System.Collections.Generic.List[object]]::new()
foreach ($Target in @($DesiredTargets + "release-info.json" | Sort-Object -Unique)) {
    $TargetPath = Join-Path $Destination $Target
    $StateFiles.Add([ordered]@{
        path = $Target
        sha256 = (Get-FileHash -LiteralPath $TargetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}
$State = [ordered]@{
    schema_version = 1
    source_repository = $SourceRoot
    files = $StateFiles
}
$State | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatePath -Encoding UTF8
"Synchronized $($Mappings.Count) managed files to $Destination"
