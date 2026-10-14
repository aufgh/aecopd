param([int[]]$Seeds = @(2026, 2027, 2028),
      [string[]]$Arms = @('baseline', 'lab_only', 'med_only', 'full'))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_revised_labs_meds_2026-09-29'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29'
$features = 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29'
if (-not (Test-Path -LiteralPath (Join-Path $processed 'manifest.json')) -or
    -not (Test-Path -LiteralPath (Join-Path $features 'manifest.json'))) {
    throw 'Revised input artifacts are missing'
}
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
    '--clinical-recovery-dir', $features,
    '--abnormality-scale-mode', 'training_positive_p95',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12'
)
$modes = @{
    baseline = @('masked', 'masked')
    lab_only = @('active', 'masked')
    med_only = @('masked', 'active')
    full = @('active', 'active')
}
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        if (-not $modes.ContainsKey($arm)) { throw "Unknown arm: $arm" }
        $name = "${arm}_seed${seed}"
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed' -or
                $report.abnormality_scale_mode -ne 'training_positive_p95' -or
                $report.clinical_recovery_mode -ne $modes[$arm][0] -or
                $report.medication_effect_mode -ne $modes[$arm][1]) {
                throw "Inconsistent completed run: $output"
            }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        Write-Output "START $name"
        & docker @common '--clinical-recovery-mode' $modes[$arm][0] `
            '--medication-effect-mode' $modes[$arm][1] '--seed' "$seed" `
            '--output-dir' "training_runs/v5_revised_labs_meds_2026-09-29/$name" `
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
}
