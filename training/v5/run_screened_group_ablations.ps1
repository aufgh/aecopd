param(
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string[]]$Arms = @('full', 'no_all_events', 'no_all_labs', 'no_all_medications', 'no_text', 'no_noncontext_static',
                       'no_prior_counts', 'no_blood_gas', 'no_routine_labs',
                       'no_sparse_blood_gas', 'no_bronchodilators', 'no_steroids',
                       'no_antibiotics', 'no_diuretics')
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_feature_groups_formulation_verified_2026-09-29'
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_formulation_verified_2026-09-29'
$manifest = Get-Content -Raw -LiteralPath (Join-Path $root "$processed/manifest.json") | ConvertFrom-Json
if ($manifest.status -ne 'preprocessed_and_checked') { throw 'Corrected event data is not validated.' }
$known = @($manifest.concepts)
$groups = @{
    no_all_labs = @($known | Where-Object { $_ -match '^(ph|pco2|po2):' -or $_ -in @('albumin','bicarbonate','creatinine','hemoglobin','wbc') })
    no_blood_gas = @($known | Where-Object { $_ -match '^(ph|pco2|po2):' })
    no_sparse_blood_gas = @($known | Where-Object { $_ -match '^(ph|pco2|po2):(central_venous|mixed|unknown)$' })
    no_routine_labs = @('albumin','bicarbonate','creatinine','hemoglobin','wbc')
    no_bronchodilators = @('saba','sama','laba','lama')
    no_steroids = @('inhaled_steroid','systemic_steroid')
    no_antibiotics = @('macrolide','beta_lactam','fluoroquinolone','other_antibiotic')
    no_diuretics = @('loop_diuretic','other_diuretic')
}
foreach ($entry in $groups.GetEnumerator()) {
    if (-not $entry.Value -or @($entry.Value | Where-Object { $_ -notin $known }).Count) {
        throw "Invalid event concept group: $($entry.Key)"
    }
}
$validArms = @('full','no_all_events','no_all_labs','no_all_medications','no_text','no_noncontext_static',
               'no_prior_counts','no_blood_gas','no_routine_labs',
               'no_sparse_blood_gas','no_bronchodilators','no_steroids','no_antibiotics','no_diuretics')
if (@($Arms | Where-Object { $_ -notin $validArms }).Count) { throw 'Unknown arm requested.' }
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
    '--event-encoder', 'set', '--max-epochs', '35', '--batch-size', '12'
)
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        $name = "${arm}_seed${seed}"
        $output = Join-Path $runRoot $name
        $reportPath = Join-Path $output 'report.json'
        if (Test-Path -LiteralPath $reportPath) {
            $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
            if ($report.status -ne 'training_completed') { throw "Incomplete run: $output" }
            Write-Output "SKIP $name (already complete)"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        $variantArgs = @()
        if ($arm -eq 'no_all_events') { $variantArgs = @('--disable-events') }
        elseif ($arm -eq 'no_all_medications') { $variantArgs = @('--disable-medications') }
        elseif ($arm -eq 'no_text') { $variantArgs = @('--disable-text') }
        elseif ($arm -eq 'no_noncontext_static') { $variantArgs = @('--disable-current-static') }
        elseif ($arm -eq 'no_prior_counts') { $variantArgs = @('--exclude-prior-admission-counts') }
        elseif ($groups.ContainsKey($arm)) { $variantArgs = @('--exclude-event-concepts', ($groups[$arm] -join ',')) }
        $log = Join-Path $runRoot "$name.log"
        Write-Output "START $name"
        & docker @common '--output-dir' "training_runs/v5_feature_groups_formulation_verified_2026-09-29/$name" '--seed' "$seed" @variantArgs *> $log
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath $log -Tail 50
            throw "Training failed: $name (see $log)"
        }
        $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
        if ($report.status -ne 'training_completed') { throw "Missing completed report: $reportPath" }
        Write-Output "DONE $name best_epoch=$($report.best_epoch) validation_IBS=$($report.validation.temporal_full.integrated_brier_7_365)"
    }
}
