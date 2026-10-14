param(
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string[]]$Arms = @('full', 'selected', 'arterial_hypercapnic_acidosis', 'platelet_low', 'magnesium_low', 'ntprobnp_high')
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_strict_axis_screen_2026-09-30'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$strictLabels = 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz'
$strictSummary = Get-Content -Raw -LiteralPath 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/summary.json' | ConvertFrom-Json
if ($strictSummary.status -ne 'strict_labels_reconstructed_and_checked' -or
    $strictSummary.cohort_admissions -ne 19401 -or -not (Test-Path -LiteralPath $strictLabels)) {
    throw 'Strict endpoint labels are missing or unverified'
}
$validArms = @('full', 'selected', 'arterial_hypercapnic_acidosis', 'platelet_low', 'magnesium_low', 'ntprobnp_high')
foreach ($arm in $Arms) {
    if ($arm -notin $validArms) { throw "Unknown screening arm: $arm" }
}
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$private/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$private/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', $strictLabels,
    '--processed-event-dir', 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29',
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29',
    '--abnormality-scale-mode', 'training_positive_p95',
    '--event-encoder', 'abnormal_medication', '--max-epochs', '35', '--batch-size', '12'
)
if (-not (Test-Path -LiteralPath $runRoot)) { New-Item -ItemType Directory -Path $runRoot | Out-Null }
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        if ($arm -eq 'full') {
            $labMode, $medMode, $name = 'active', 'active', "full_seed${seed}"
        } elseif ($arm -eq 'selected') {
            $labMode, $medMode, $name = 'without_renal_metabolic_perfusion', 'without_diuretic', "selected_seed${seed}"
        } else {
            $labMode, $medMode, $name = "without_renal_metabolic_perfusion_plus_${arm}", 'without_diuretic', "without_axis_${arm}_seed${seed}"
        }
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed' -or $report.clinical_recovery_mode -ne $labMode -or
                $report.medication_effect_mode -ne $medMode -or $report.label_sha256 -ne $strictSummary.label_sha256) {
                throw "Inconsistent completed run: $output"
            }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        Write-Output "START $name"
        & docker @common '--clinical-recovery-mode' $labMode '--medication-effect-mode' $medMode `
            '--seed' "$seed" '--output-dir' "training_runs/v5_strict_axis_screen_2026-09-30/$name" `
            *> (Join-Path $runRoot "$name.log")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $runRoot "$name.log") -Tail 70
            throw "Training failed: $name"
        }
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.raw_events_enter_model -or
            $report.clinical_recovery_mode -ne $labMode -or $report.medication_effect_mode -ne $medMode -or
            $report.label_sha256 -ne $strictSummary.label_sha256) {
            throw "Incorrect run contract: $name"
        }
        Write-Output "DONE $name epoch=$($report.best_epoch) valIBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
