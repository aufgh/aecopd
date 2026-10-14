param([int[]]$Seeds = @(2026, 2027, 2028))

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath '.').Path
$runRoot = Join-Path $root 'training_runs/v5_strict_qwen_bert_2026-09-30'
$bertRoot = Join-Path $root 'training_runs/v5_strict_axis_screen_2026-09-30'
$private = 'outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'
$fieldRoot = "$private/v5_section_aligned/field_text_baseline_2026_09_27"
$fieldLocal = Join-Path $root ($fieldRoot -replace '/', '\')
$manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $fieldLocal 'shared_text.csv.gz')).Hash.ToLowerInvariant()
$qwenState = Get-Content -Raw -LiteralPath (Join-Path $fieldLocal 'qwen/state.json') | ConvertFrom-Json
$bertState = Get-Content -Raw -LiteralPath (Join-Path $fieldLocal 'bert/state.json') | ConvertFrom-Json
if ($qwenState.status -ne 'complete' -or $bertState.status -ne 'complete' -or
    $qwenState.completed_rows -ne 76974 -or $bertState.completed_rows -ne 76974 -or
    $qwenState.manifest_sha256 -ne $manifestHash -or $bertState.manifest_sha256 -ne $manifestHash -or
    -not (Test-Path -LiteralPath (Join-Path $fieldLocal 'qwen/qwen3_embedding.float16.npy'))) {
    throw 'Qwen and BERT must use the same completed 76,974-row manifest'
}
$strictSummary = Get-Content -Raw -LiteralPath 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/summary.json' | ConvertFrom-Json
$common = @(
    'run', '--rm', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    'aecopd-training:v1', 'python', '-m', 'training.v5.train_event_sequence',
    '--aligned-dir', "$private/v5_section_aligned/dataset_v3",
    '--structured-dir', "$private/v5_section_aligned/section_inputs_sparse_2026_09_26",
    '--base-csv', "$private/v5_mimiciv_2026-09-25/structured/base.csv",
    '--field-dir', $fieldRoot,
    '--vectors', "$fieldRoot/qwen/qwen3_embedding.float16.npy",
    '--labels-csv', 'patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz',
    '--processed-event-dir', 'patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29',
    '--raw-ledger-dir', 'patient_data/mimiciv_copd/snapshot_2026-09-27',
    '--clinical-recovery-dir', 'patient_data/processed/abnormality_clinical_reviewed_2026-09-29',
    '--abnormality-scale-mode', 'training_positive_p95',
    '--clinical-recovery-mode', 'active', '--medication-effect-mode', 'active',
    '--event-encoder', 'abnormal_medication', '--shared-text-init-width', '768',
    '--max-epochs', '35', '--batch-size', '12'
)
if (-not (Test-Path -LiteralPath $runRoot)) { New-Item -ItemType Directory -Path $runRoot | Out-Null }
foreach ($seed in $Seeds) {
    $bertReport = Get-Content -Raw -LiteralPath (Join-Path $bertRoot "full_seed${seed}/report.json") | ConvertFrom-Json
    if ($bertReport.status -ne 'training_completed' -or $bertReport.label_sha256 -ne $strictSummary.label_sha256 -or
        $bertReport.field_manifest_sha256 -ne $manifestHash -or $bertReport.text_vector_width -ne 768 -or
        $bertReport.clinical_recovery_mode -ne 'active' -or $bertReport.medication_effect_mode -ne 'active') {
        throw "BERT reference differs from full strict-endpoint contract: $seed"
    }
    $output = Join-Path $runRoot "qwen_seed${seed}"
    if (Test-Path -LiteralPath (Join-Path $output 'report.json')) {
        $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
        if ($report.status -ne 'training_completed' -or $report.label_sha256 -ne $strictSummary.label_sha256 -or
            $report.field_manifest_sha256 -ne $manifestHash -or $report.text_vector_width -ne 2560 -or
            $report.shared_text_init_width -ne 768) { throw "Inconsistent Qwen run: $seed" }
        Write-Output "SKIP qwen_seed${seed}"
        continue
    }
    if (Test-Path -LiteralPath $output) { throw "Partial run needs inspection: $output" }
    Write-Output "START qwen_seed${seed}"
    & docker @common '--seed' "$seed" '--output-dir' "training_runs/v5_strict_qwen_bert_2026-09-30/qwen_seed${seed}" `
        *> (Join-Path $runRoot "qwen_seed${seed}.log")
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath (Join-Path $runRoot "qwen_seed${seed}.log") -Tail 70
        throw "Qwen training failed: $seed"
    }
    $report = Get-Content -Raw -LiteralPath (Join-Path $output 'report.json') | ConvertFrom-Json
    if ($report.status -ne 'training_completed' -or $report.label_sha256 -ne $strictSummary.label_sha256 -or
        $report.field_manifest_sha256 -ne $manifestHash -or $report.text_vector_width -ne 2560 -or
        $report.shared_text_init_width -ne 768) { throw "Qwen report contract failed: $seed" }
    Write-Output "DONE qwen_seed${seed} epoch=$($report.best_epoch)"
}
