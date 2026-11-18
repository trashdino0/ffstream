param(
    [int]$Season = 0,
    [int]$Week = 0,
    [int]$Top = 20,
    [switch]$SkipRefresh
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$exitCode = 0

function Resolve-Python {
    param([string]$Root)

    # Prefer the project virtualenv; a bare `python` on PATH often points at a
    # global install without pandas/scikit-learn.
    $candidates = @(
        (Join-Path $Root ".venv\Scripts\python.exe"),
        (Join-Path $Root "venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    foreach ($name in @("py", "python")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }
    throw "No Python interpreter found. Create .venv and install requirements first."
}

function Invoke-Ffstream {
    param(
        [string]$Python,
        [string]$Label,
        [string[]]$Arguments
    )

    Write-Host $Label -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Python $($Arguments -join ' ')"
    }
}

try {
    $env:PYTHONPATH = "src"
    $env:PYTHONIOENCODING = "utf-8"
    try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
    $python = Resolve-Python -Root $root

    # Check dependencies up front so the user gets one clear message instead of
    # a traceback from every step.
    & $python -c "import pandas, sklearn, pyarrow, requests, scipy, joblib" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python at '$python' is missing project dependencies. Run: python -m pip install -e `".[dev]`""
    }
    Write-Host "Using Python: $python" -ForegroundColor DarkGray

    # Season defaults to the current NFL season (Aug-Jan rolls over in August).
    if ($Season -le 0) {
        $now = Get-Date
        $Season = if ($now.Month -ge 8) { $now.Year } else { $now.Year - 1 }
    }

    if (-not $SkipRefresh) {
        Invoke-Ffstream -Python $python -Label "Refreshing $Season data..." -Arguments @(
            "-m", "ffstream", "refresh", "--seasons", "$Season", "--force"
        )
    }

    # Rebuild every season; passing only the current season would drop history.
    Invoke-Ffstream -Python $python -Label "Rebuilding processed tables (2014-$Season)..." -Arguments @(
        "-m", "ffstream", "build"
    )

    if ($Week -gt 0) {
        $weekArgs = @("-m", "ffstream", "week", "--season", "$Season", "--week", "$Week", "--top", "$Top")
    } else {
        $weekArgs = @("-m", "ffstream", "week", "--season", "$Season", "--top", "$Top")
    }
    Invoke-Ffstream -Python $python -Label "Generating the weekly board..." -Arguments $weekArgs

    # Grade the most recent completed week that still has a saved board. This is
    # intentionally non-fatal on a fresh install where only this week's board
    # exists: the grader prints "nothing to grade" and exits 0.
    Invoke-Ffstream -Python $python -Label "Grading the most recent completed week..." -Arguments @(
        "-m", "ffstream", "grade", "--season", "$Season", "--latest"
    )

    Write-Host ""
    Write-Host "Done. Open outputs/week_${Season}_*_report.html for the board." -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
}
finally {
    Pop-Location
}

exit $exitCode
