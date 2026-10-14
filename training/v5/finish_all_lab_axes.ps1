$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $root
$runRoot = Join-Path $root 'training_runs/v5_strict_all22_binary_meds_2026-09-30'
$exitCode = & docker wait aecopd-lab22-binary-20260930
& docker logs aecopd-lab22-binary-20260930 *> (Join-Path $runRoot 'training.log')
if ($LASTEXITCODE -ne 0 -or "$exitCode".Trim() -ne '0') {
    throw "Lab-axis batch failed; inspect $runRoot/training.log"
}
$analysis = Join-Path $runRoot 'paired_analysis/paired_results.json'
if (Test-Path -LiteralPath $analysis) {
    $saved = Get-Content -Raw -LiteralPath $analysis | ConvertFrom-Json
    if ($saved.status -ne 'completed' -or $saved.rows.Count -ne 176) { throw 'Incomplete paired analysis' }
    return
}
& docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
    -e MPLCONFIGDIR=/tmp/matplotlib aecopd-training:v1 python -u -m training.v5.compare_all_lab_axes `
    --run-root training_runs/v5_strict_all22_binary_meds_2026-09-30 `
    --output-dir training_runs/v5_strict_all22_binary_meds_2026-09-30/paired_analysis `
    *> (Join-Path $runRoot 'analysis.log')
if ($LASTEXITCODE -ne 0) { throw 'Paired analysis failed; inspect analysis.log' }
