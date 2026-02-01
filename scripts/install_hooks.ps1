param()

$repoRoot = Split-Path -Parent $PSScriptRoot
$gitDir = Join-Path $repoRoot ".git"
$hooksSource = Join-Path $repoRoot "hooks"
$hooksTarget = Join-Path $gitDir "hooks"

if (-not (Test-Path $gitDir)) {
  Write-Error "No .git directory found. Run this script from inside the repo."
  exit 1
}

if (-not (Test-Path $hooksSource)) {
  Write-Error "Hooks source directory not found: $hooksSource"
  exit 1
}

if (-not (Test-Path $hooksTarget)) {
  New-Item -ItemType Directory -Path $hooksTarget | Out-Null
}

$preCommitSource = Join-Path $hooksSource "pre-commit"
$prePushSource = Join-Path $hooksSource "pre-push"
$preCommitTarget = Join-Path $hooksTarget "pre-commit"
$prePushTarget = Join-Path $hooksTarget "pre-push"

Copy-Item -Force $preCommitSource $preCommitTarget
Copy-Item -Force $prePushSource $prePushTarget

Write-Host "Installed hooks:"
Write-Host "- $preCommitTarget"
Write-Host "- $prePushTarget"
