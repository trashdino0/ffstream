param(
    [int]$Season = 0,
    [int]$Week = 0,
    [int]$Top = 20,
    [switch]$SkipRefresh
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $env:PYTHONPATH = "src"

    # Season defaults to the current NFL season (Aug-Jan rolls over in August).
    if ($Season -le 0) {
        $now = Get-Date
        $Season = if ($now.Month -ge 8) { $now.Year } else { $now.Year - 1 }
    }

    if (-not $SkipRefresh) {
        Write-Host "Refreshing $Season data..." -ForegroundColor Cyan
        python -m ffstream refresh --seasons $Season --force
    }

    Write-Host "Rebuilding processed tables (2014-$Season)..." -ForegroundColor Cyan
    # Rebuild every season; passing only the current season would drop history.
    python -m ffstream build

    Write-Host "Generating the weekly board..." -ForegroundColor Cyan
    if ($Week -gt 0) {
        python -m ffstream week --season $Season --week $Week --top $Top
    } else {
        python -m ffstream week --season $Season --top $Top
    }

    Write-Host ""
    Write-Host "Done. Open outputs/week_${Season}_*_report.html for the board." -ForegroundColor Green
}
finally {
    Pop-Location
}
