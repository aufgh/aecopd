$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $root
$runRoot = Join-Path $root 'training_runs/v5_strict_all22_binary_meds_one_seed_2026-09-30'
if (-not (Test-Path -LiteralPath $runRoot)) { New-Item -ItemType Directory -Path $runRoot | Out-Null }
$firstAxes = @('arterial_low_ph', 'arterial_high_pco2', 'bicarbonate_low', 'bicarbonate_high',
    'wbc_high', 'albumin_low', 'hemoglobin_low', 'creatinine_high', 'arterial_hypercapnic_acidosis',
    'wbc_low', 'sodium_low')
$secondAxes = @('sodium_high', 'potassium_low', 'potassium_high', 'urea_nitrogen_high',
    'glucose_low', 'glucose_high', 'platelet_low', 'magnesium_low', 'lactate_high', 'crp_high', 'ntprobnp_high')
$common = @('run', '-d', '--gpus', 'all', '-v', "${root}:/workspace", '-w', '/workspace',
    '-e', 'PYTHONPATH=/workspace', '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
    '-e', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', '-e', 'OMP_NUM_THREADS=2', '-e', 'MKL_NUM_THREADS=2')
& docker @common '--name' 'aecopd-lab22-single-a' 'aecopd-training:v1' `
    'python' '-u' '-m' 'training.v5.run_all_lab_axes' '--seeds' '2026' '--worker-name' 'worker_a' '--axes' @firstAxes
if ($LASTEXITCODE -ne 0) { throw 'First worker could not start' }
& docker @common '--name' 'aecopd-lab22-single-b' 'aecopd-training:v1' `
    'python' '-u' '-m' 'training.v5.run_all_lab_axes' '--seeds' '2026' '--worker-name' 'worker_b' `
    '--skip-baseline' '--axes' @secondAxes
if ($LASTEXITCODE -ne 0) { throw 'Second worker could not start; first worker may still be running' }
$failed = @()
foreach ($name in @('aecopd-lab22-single-a', 'aecopd-lab22-single-b')) {
    $exitCode = & docker wait $name
    & docker logs $name *> (Join-Path $runRoot "$name.log")
    if ($LASTEXITCODE -ne 0 -or "$exitCode".Trim() -ne '0') { $failed += $name }
}
if ($failed.Count -gt 0) { throw "Training failed: $failed; completed artifacts are preserved" }
& docker run --rm -v "${root}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
    -e MPLCONFIGDIR=/tmp/matplotlib aecopd-training:v1 python -u -m training.v5.compare_all_lab_axes `
    --run-root training_runs/v5_strict_all22_binary_meds_one_seed_2026-09-30 `
    --output-dir training_runs/v5_strict_all22_binary_meds_one_seed_2026-09-30/paired_analysis `
    --seeds 2026 *> (Join-Path $runRoot 'analysis.log')
if ($LASTEXITCODE -ne 0) { throw 'Paired analysis failed; inspect analysis.log' }
