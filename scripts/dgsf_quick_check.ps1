# DGSF Quick Check Script
# Purpose: 快速验证 DGSF 项目状态，降低日常迭代摩擦
# Author: Gene Kim (Execution Flow principle)
# Created: 2026-02-03
# Usage: .\scripts\dgsf_quick_check.ps1

$ErrorActionPreference = "SilentlyContinue"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   DGSF Quick Check" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Change to DGSF repo directory
$repo_path = "E:\AI Tools\AI Workflow OS\projects\dgsf\repo"
Push-Location $repo_path

# [1] Git Status
Write-Host "[1] Git Status:" -ForegroundColor Yellow
$git_status = git status --short
if ($git_status) {
    Write-Host $git_status -ForegroundColor White
} else {
    Write-Host "  ✓ Working tree clean" -ForegroundColor Green
}

# [2] Test Summary
Write-Host "`n[2] Test Summary:" -ForegroundColor Yellow
$test_collect = pytest tests/ --collect-only -q 2>$null | Select-Object -Last 3
if ($test_collect) {
    Write-Host $test_collect -ForegroundColor White
} else {
    Write-Host "  ⚠ pytest not available or no tests collected" -ForegroundColor Yellow
}

# [3] Submodule Sync
Write-Host "`n[3] Submodule Sync:" -ForegroundColor Yellow
$last_commit = git log --oneline -1
Write-Host "  $last_commit" -ForegroundColor White

# [4] Branch
Write-Host "`n[4] Branch:" -ForegroundColor Yellow
$current_branch = git branch --show-current
Write-Host "  $current_branch" -ForegroundColor Cyan

# [5] Remote Status
Write-Host "`n[5] Remote Status:" -ForegroundColor Yellow
$remote_status = git status -sb | Select-Object -First 1
Write-Host "  $remote_status" -ForegroundColor White

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Quick Check Complete" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Pop-Location
