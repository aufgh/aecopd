param([int[]]$Seeds = @(2026, 2027, 2028))
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$resultRoot = 'training_runs/v5_incremental_history_evidence_2026-09-28'
$dataRoot = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$dockerBase = @('run', '--rm', '-v', "${root}:/workspace", '-w', '/workspace',
                '-e', 'PYTHONPATH=/workspace:/workspace/training:/workspace/model_loader',
                'aecopd-baselines:v1', 'python', '-m')
$fullDirs = @{
    2026 = 'training_runs/v5_with_prior_counts_strict_aecopd_bert_2026-09-28'
    2027 = 'training_runs/v5_strict_full_seed2027_2026-09-28'
    2028 = 'training_runs/v5_strict_full_seed2028_2026-09-28'
}
foreach ($seed in $Seeds) {
    $comparators = @()
    foreach ($name in @('no_current_static', 'no_medication_events', 'no_lab_time', 'one_lab_per_concept')) {
        $directory = "$resultRoot/${name}_seed$seed"
        if ($seed -eq 2026 -and $name -eq 'no_current_static') {
            $directory = 'training_runs/v5_history_evidence_no_current_static_seed2026_2026-09-28'
        }
        if (-not (Test-Path -LiteralPath (Join-Path $directory 'report.json'))) {
            throw "Missing completed training: $directory"
        }
        $comparators += @('--comparator', $name, $directory)
    }
    if ($seed -eq 2026) {
        foreach ($name in @('history_of_present_illness', 'past_medical_history', 'hospital_course', 'discharge_condition')) {
            $comparators += @('--comparator', "without_text_$name", "$resultRoot/without_text_${name}_seed2026")
        }
        $comparators += @('--comparator', 'no_text', 'training_runs/v5_strict_retrained_without_text_2026-09-28')
        $comparators += @('--comparator', 'no_lab_values', 'training_runs/v5_strict_retrained_without_lab_values_2026-09-28')
        $comparators += @('--comparator', 'no_events', 'training_runs/v5_strict_retrained_without_events_2026-09-28')
    } else {
        $comparators += @('--comparator', 'no_lab_values', "training_runs/v5_strict_no_lab_values_seed${seed}_2026-09-28")
    }
    $paired = "$resultRoot/paired_seed$seed.json"
    if (-not (Test-Path -LiteralPath $paired)) {
        Write-Output "START paired seed=$seed"
        & docker @dockerBase 'training.v5.compare_strict_ablations' '--candidate-dir' $fullDirs[$seed] @comparators '--output-path' $paired
        if ($LASTEXITCODE -ne 0) { throw "Paired comparison failed for seed $seed" }
    }
    $strata = "$resultRoot/strata_seed$seed.json"
    if (-not (Test-Path -LiteralPath $strata)) {
        Write-Output "START strata seed=$seed"
        & docker @dockerBase 'training.v5.analyze_incremental_evidence' `
            '--base-csv' "$dataRoot/v5_mimiciv_2026-09-25/structured/base.csv" `
            '--labels-csv' 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz' `
            '--full-dir' $fullDirs[$seed] @comparators '--output-path' $strata
        if ($LASTEXITCODE -ne 0) { throw "Stratified comparison failed for seed $seed" }
    }
    Write-Output "DONE evidence seed=$seed"
}
