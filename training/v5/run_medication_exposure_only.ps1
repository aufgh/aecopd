param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_medication_effect_2026-09-29'
foreach ($seed in $Seeds) {
    if (-not (Test-Path -LiteralPath (Join-Path $runRoot "active_seed${seed}/report.json"))) {
        throw "Active reference run missing for seed $seed"
    }
    & (Join-Path $PSScriptRoot 'run_medication_effect_ablations.ps1') -Seeds $seed -Modes 'exposure_only'
    if ($LASTEXITCODE -ne 0) { throw "Exposure-only training failed: seed=$seed" }
    $comparison = Join-Path $runRoot "comparison_exposure_only_seed${seed}.json"
    if (-not (Test-Path -LiteralPath $comparison)) {
        & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
            aecopd-training:v1 python -m training.v5.compare_medication_effect_ablations `
            --run-root training_runs/v5_medication_effect_2026-09-29 `
            --output "training_runs/v5_medication_effect_2026-09-29/comparison_exposure_only_seed${seed}.json" `
            --seed $seed --control exposure_only --bootstrap 200
        if ($LASTEXITCODE -ne 0) { throw "Exposure-only comparison failed: seed=$seed" }
    }
}
