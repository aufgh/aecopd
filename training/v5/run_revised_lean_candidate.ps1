param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$parent = Join-Path $root 'training_runs/v5_revised_labs_meds_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29'
$features = 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29'
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
    '--clinical-recovery-dir', $features,
    '--abnormality-scale-mode', 'training_positive_p95',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12',
    '--clinical-recovery-mode', 'core_risk', '--medication-effect-mode', 'core_acute'
)
foreach ($seed in $Seeds) {
    $full = Join-Path $parent "full_seed${seed}/report.json"
    if (-not (Test-Path -LiteralPath $full)) { throw "Full reference missing: $full" }
    $name = "lean_clinical_seed${seed}"
    $output = Join-Path $parent $name
    if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.clinical_recovery_mode -ne 'core_risk' -or
            $report.medication_effect_mode -ne 'core_acute') { throw "Inconsistent completed run: $output" }
        Write-Output "SKIP $name"
        continue
    }
    if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
    Write-Output "START $name"
    & docker @common '--seed' "$seed" '--output-dir' "training_runs/v5_revised_labs_meds_2026-09-29/$name" `
        *> (Join-Path $parent "$name.log")
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath (Join-Path $parent "$name.log") -Tail 70
        throw "Training failed: $name"
    }
    $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
    if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model) {
        throw "Incorrect run contract: $name"
    }
    Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
}
