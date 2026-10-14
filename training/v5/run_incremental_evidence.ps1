param(
    [int[]]$Seeds = @(2026, 2027, 2028)
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$outputRoot = Join-Path $root 'training_runs/v5_incremental_history_evidence_2026-09-28'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$dataRoot = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$dataRoot/v5_section_aligned/dataset_v3",
    '--structured-dir', "$dataRoot/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$dataRoot/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', "$dataRoot/v5_section_aligned/field_text_baseline_2026_09_27",
    '--vectors', "$dataRoot/v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
    '--labels-csv', 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz',
    '--processed-event-dir', 'patient_data/processed/mimiciv_copd_events_2026-09-27',
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--event-encoder', 'set'
)
$variants = @(
    @{ Name = 'no_current_static'; Args = @('--disable-current-static') },
    @{ Name = 'no_medication_events'; Args = @('--disable-medications') },
    @{ Name = 'no_lab_time'; Args = @('--lab-process', 'no_time') },
    @{ Name = 'one_lab_per_concept'; Args = @('--lab-process', 'one_per_concept') }
)
$textFields = @('history_of_present_illness', 'past_medical_history', 'hospital_course', 'discharge_condition')
foreach ($field in $textFields) {
    $active = ($textFields | Where-Object { $_ -ne $field }) -join ','
    $variants += @{ Name = "without_text_$field"; Args = @('--active-text-fields', $active); OnlySeed2026 = $true }
}
foreach ($seed in $Seeds) {
    foreach ($variant in $variants) {
        if ($variant.OnlySeed2026 -and $seed -ne 2026) { continue }
        $name = "$($variant.Name)_seed$seed"
        if ($name -eq 'no_current_static_seed2026') {
            $existing = Join-Path $root 'training_runs/v5_history_evidence_no_current_static_seed2026_2026-09-28/report.json'
            if (-not (Test-Path -LiteralPath $existing)) { throw "Initial run is not complete: $existing" }
            $initialReport = Get-Content -Raw -LiteralPath $existing | ConvertFrom-Json
            if ($initialReport.status -ne 'training_completed') { throw "Initial run is incomplete: $existing" }
            Write-Output "SKIP $name (initial run already complete)"
            continue
        }
        $output = Join-Path $outputRoot $name
        $reportPath = Join-Path $output 'report.json'
        if (Test-Path -LiteralPath $reportPath) {
            $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
            if ($report.status -ne 'training_completed') { throw "Incomplete report: $reportPath" }
            Write-Output "SKIP $name (already complete)"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        Write-Output "START $name"
        $log = Join-Path $outputRoot "$name.log"
        $variantArgs = $variant.Args
        & docker @common '--output-dir' "training_runs/v5_incremental_history_evidence_2026-09-28/$name" '--seed' "$seed" @variantArgs *> $log
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath $log -Tail 60
            throw "Training failed: $name (see $log)"
        }
        $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
        if ($report.status -ne 'training_completed') { throw "Missing complete report: $reportPath" }
        Write-Output "DONE $name best_epoch=$($report.best_epoch) validation_IBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
