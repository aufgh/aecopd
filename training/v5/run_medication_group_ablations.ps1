param([int[]]$Seeds = @(2026))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_medication_effect_2026-09-29'
$groups = @(
    'short_beta_agonist', 'short_muscarinic_antagonist',
    'long_beta_agonist', 'long_muscarinic_antagonist',
    'inhaled_corticosteroid', 'systemic_corticosteroid',
    'common_systemic_antibacterial', 'diuretic'
)
foreach ($seed in $Seeds) {
    if (-not (Test-Path -LiteralPath (Join-Path $runRoot "active_seed${seed}/report.json"))) {
        throw "Active reference run missing for seed $seed"
    }
    foreach ($group in $groups) {
        $mode = "without_$group"
        & (Join-Path $PSScriptRoot 'run_medication_effect_ablations.ps1') -Seeds $seed -Modes $mode
        if ($LASTEXITCODE -ne 0) { throw "Ablation failed: $mode seed=$seed" }
        $comparison = Join-Path $runRoot "comparison_${mode}_seed${seed}.json"
        if (-not (Test-Path -LiteralPath $comparison)) {
            & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
                aecopd-training:v1 python -m training.v5.compare_medication_effect_ablations `
                --run-root training_runs/v5_medication_effect_2026-09-29 `
                --output "training_runs/v5_medication_effect_2026-09-29/comparison_${mode}_seed${seed}.json" `
                --seed $seed --control $mode --bootstrap 200
            if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $mode seed=$seed" }
        }
    }
}
