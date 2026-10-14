param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = 'training_runs/v5_abnormality_only_2026-09-29'
foreach ($seed in $Seeds) {
    foreach ($control in @('core9', 'measurement_only')) {
        $output = "$runRoot/comparison_active_minus_${control}_seed${seed}.json"
        if (Test-Path -LiteralPath $output) {
            $existing = Get-Content -Raw -LiteralPath $output | ConvertFrom-Json
            if ($existing.status -ne 'single_seed_pilot' -or $existing.control_mode -ne $control) {
                throw "Unrecognized comparison already exists: $output"
            }
            Write-Output "SKIP active minus $control seed $seed"
            continue
        }
        & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
            aecopd-training:v1 python -m training.v5.compare_lab_recovery_pilot `
            --run-root $runRoot --output $output --seed $seed --bootstrap 200 `
            --control-mode $control
        if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $control seed $seed" }
        Write-Output "DONE active minus $control seed $seed"
    }
}
