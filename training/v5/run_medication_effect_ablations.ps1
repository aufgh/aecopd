param([int[]]$Seeds = @(2026, 2027, 2028),
      [string[]]$Modes = @('masked', 'active'))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_medication_effect_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_formulation_verified_2026-09-29'
$features = 'patient_data/processed/abnormality_only_2026-09-29'
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', "$private/v5_qualified_copd_labels_2026_09_27/labels.csv.gz",
    '--processed-event-dir', $processed,
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', $features, '--clinical-recovery-mode', 'active',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12'
)
foreach ($seed in $Seeds) {
    foreach ($mode in $Modes) {
        $name = "${mode}_seed${seed}"
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed' -or $report.medication_effect_mode -ne $mode) {
                throw "Inconsistent completed run: $output"
            }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run requires inspection: $output" }
        Write-Output "START $name"
        & docker @common '--medication-effect-mode' $mode '--seed' "$seed" `
            '--output-dir' "training_runs/v5_medication_effect_2026-09-29/$name" `
            *> (Join-Path $runRoot "$name.log")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $runRoot "$name.log") -Tail 70
            throw "Training failed: $name"
        }
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model) {
            throw "Incorrect run contract: $name"
        }
        Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
    if (($Modes -contains 'masked') -and ($Modes -contains 'active')) {
        $comparison = Join-Path $runRoot "comparison_masked_seed${seed}.json"
        if (-not (Test-Path -LiteralPath $comparison)) {
            & docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
                aecopd-training:v1 python -m training.v5.compare_medication_effect_ablations `
                --run-root training_runs/v5_medication_effect_2026-09-29 `
                --output "training_runs/v5_medication_effect_2026-09-29/comparison_masked_seed${seed}.json" `
                --seed $seed --control masked --bootstrap 200
            if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $seed" }
        }
    }
}
