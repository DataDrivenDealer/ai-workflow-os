param(
  [switch]$Force
)

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

# All hooks to install (exhaustive list)
$hookNames = @(
  "pre-commit",
  "pre-push",
  "pre-spec-change",
  "post-spec-change",
  "post-tag",
  "pre-destructive-op"
)

$installedCount = 0
$skippedCount = 0

foreach ($hookName in $hookNames) {
  $src = Join-Path $hooksSource $hookName
  $dst = Join-Path $hooksTarget $hookName

  if (-not (Test-Path $src)) {
    Write-Host "  [SKIP] $hookName (source not found)"
    $skippedCount++
    continue
  }

  if ((Test-Path $dst) -and -not $Force) {
    Write-Host "  [EXISTS] $hookName (use -Force to overwrite)"
    $skippedCount++
    continue
  }

  Copy-Item -Force $src $dst
  Write-Host "  [OK] $hookName"
  $installedCount++
}

Write-Host ""
Write-Host "Installed: $installedCount hook(s), Skipped: $skippedCount"
Write-Host "Hooks target: $hooksTarget"

# Remove decline state file if it exists (user explicitly ran install)
$declineFile = Join-Path $repoRoot "state\.git_hooks_check"
if (Test-Path $declineFile) {
  Remove-Item $declineFile -Force
  Write-Host "Cleared hooks-check decline state."
}
