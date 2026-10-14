param([int[]]$Seeds = @(2026, 2027, 2028),
      [string[]]$Groups = @(
          'med_short_beta_agonist', 'med_short_muscarinic_antagonist',
          'med_long_beta_agonist', 'med_long_muscarinic_antagonist',
          'med_inhaled_corticosteroid', 'med_systemic_corticosteroid',
          'med_common_systemic_antibacterial', 'med_diuretic',
          'lab_ventilation_acid_base', 'lab_inflammation_infection',
          'lab_nutrition_hematology', 'lab_renal', 'lab_electrolytes',
          'lab_metabolic_perfusion', 'lab_cardiac_burden'))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$parent = Join-Path $root 'training_runs/v5_revised_labs_meds_2026-09-29'
$runRoot = Join-Path $parent 'groups'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$processed = 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29'
$features = 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29'
$medGroups = @('short_beta_agonist', 'short_muscarinic_antagonist', 'long_beta_agonist',
               'long_muscarinic_antagonist', 'inhaled_corticosteroid',
               'systemic_corticosteroid', 'common_systemic_antibacterial', 'diuretic',
               'short_bronchodilators', 'maintenance_inhalers')
$labGroups = @('ventilation_acid_base', 'inflammation_infection', 'nutrition_hematology',
               'renal', 'electrolytes', 'metabolic_perfusion', 'cardiac_burden')
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
foreach ($seed in $Seeds) {
    $full = Join-Path $parent "full_seed${seed}/report.json"
    if (-not (Test-Path -LiteralPath $full)) { throw "Full reference missing: $full" }
    foreach ($group in $Groups) {
        if ($group.StartsWith('med_')) {
            $groupName = $group.Substring(4)
            if ($groupName -notin $medGroups) { throw "Unknown medication group: $group" }
            $labMode = 'active'
            $medMode = "without_$groupName"
        } elseif ($group.StartsWith('lab_')) {
            $groupName = $group.Substring(4)
            if ($groupName -notin $labGroups) { throw "Unknown laboratory group: $group" }
            $labMode = "without_$groupName"
            $medMode = 'active'
        } else { throw "Unknown group: $group" }
        $name = "without_${group}_seed${seed}"
        $output = Join-Path $runRoot $name
        if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
            $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
            if ($report.status -ne 'training_completed' -or
                $report.clinical_recovery_mode -ne $labMode -or
                $report.medication_effect_mode -ne $medMode -or
                $report.abnormality_scale_mode -ne 'training_positive_p95') {
                throw "Inconsistent completed run: $output"
            }
            Write-Output "SKIP $name"
            continue
        }
        if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
        Write-Output "START $name"
        & docker @common '--clinical-recovery-mode' $labMode `
            '--medication-effect-mode' $medMode '--seed' "$seed" `
            '--output-dir' "training_runs/v5_revised_labs_meds_2026-09-29/groups/$name" `
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
