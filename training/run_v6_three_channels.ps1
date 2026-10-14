# 本机持续工作流：全量CT/小结抽取 -> 完整事实接口 -> 六臂训练 -> GPU重载。
# 不修改原始底账，不调用外部LLM，不接受失败/半完成抽取作为正式模型输入。
param(
    [Parameter(Mandatory=$true)][int]$OwnedServerPid
)
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\Administrator\Desktop\aecopd'
Set-Location -LiteralPath $taskRoot
$taskPython = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe'
$taskJobs = 'patient_data/processed/v6_bonsai_tasks_2026-10-10_validated'
$taskCt = 'patient_data/processed/v6_bonsai_ct_full_2026-10-10'
$taskDischarge = 'patient_data/processed/v6_bonsai_discharge_full_2026-10-10'
$taskFacts = 'patient_data/processed/v6_clinical_facts_2026-10-10'
$taskRun = 'training_runs/v6_three_channels_seed42_2026-10-10'
$taskStatus = Join-Path $taskRoot 'work/v6_three_channels_2026-10-10/pipeline_status.json'
$taskServer = Get-CimInstance Win32_Process -Filter "ProcessId=$OwnedServerPid"
if (-not $taskServer -or $taskServer.ExecutablePath -ne 'D:\ywj\Bonsai-demo\bin\cuda\llama-server.exe' -or
    -not $taskServer.CommandLine.Contains('Ternary-Bonsai-2-27B-PTQ1_0.gguf')) {
    throw 'Expected owned Bonsai CUDA service is not running; do not stop any unrelated process.'
}
$taskServerCreation = $taskServer.CreationDate
function Write-TaskStage([string]$stage, [string]$detail='') {
    $taskRecord = @{status=$stage; detail=$detail; pid=$PID; updated=(Get-Date).ToString('o');
        jobs=$taskJobs; ct=$taskCt; discharge=$taskDischarge; facts=$taskFacts; experiment=$taskRun}
    $taskRecord | ConvertTo-Json | Set-Content -LiteralPath $taskStatus -Encoding utf8
    Write-Output "STAGE: $stage"
}
function Invoke-TaskPython([string[]]$taskArguments) {
    & $taskPython @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed, exit=$LASTEXITCODE" }
}
function Invoke-TaskGPU([string[]]$taskArguments) {
    & docker run --rm --network none --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v "${taskRoot}:/workspace" -w /workspace aecopd-baselines:v1 python @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "GPU stage failed, exit=$LASTEXITCODE" }
}
try {
    Write-TaskStage 'extracting_all_chest_CT'
    # 正式批处理统一关闭思考（2026-10-10用户决定）；runner仍保留独立研究选项。
    # 这不会自动升级现有合并章节任务；短主题任务须另建、独立验收后再替换入口。
    Invoke-TaskPython @('-m','patient_data.clinical_facts.runner','--tasks',$taskJobs,'--channel','chest_ct','--parallel','4','--output',$taskCt)
    if ((Get-Content -LiteralPath "$taskCt/status.json" -Raw | ConvertFrom-Json).status -ne 'completed') { throw 'CT extraction has failed tasks.' }
    Write-TaskStage 'extracting_all_discharge_sections'
    Invoke-TaskPython @('-m','patient_data.clinical_facts.runner','--tasks',$taskJobs,'--channel','discharge_all','--parallel','4','--output',$taskDischarge)
    if ((Get-Content -LiteralPath "$taskDischarge/status.json" -Raw | ConvertFrom-Json).status -ne 'completed') { throw 'Discharge extraction has failed tasks.' }
    Write-TaskStage 'building_and_checking_complete_fact_interface'
    Invoke-TaskPython @('-m','patient_data.clinical_facts.features','--tasks',$taskJobs,'--ct',$taskCt,'--discharge',$taskDischarge,'--output',$taskFacts)
    # 只停止本工作流接管且身份/创建时点仍一致的服务，给训练释放显存。
    $taskCurrentServer = Get-CimInstance Win32_Process -Filter "ProcessId=$OwnedServerPid"
    if ($taskCurrentServer -and $taskCurrentServer.CreationDate -eq $taskServerCreation -and
        $taskCurrentServer.ExecutablePath -eq $taskServer.ExecutablePath) { Stop-Process -Id $OwnedServerPid }
    Write-TaskStage 'full_six_arm_GPU_training'
    Invoke-TaskGPU @('-m','training.v6.facts_experiment','--facts-dir',$taskFacts,'--output',$taskRun)
    Write-TaskStage 'independent_GPU_reload_and_common_coverage'
    Invoke-TaskGPU @('-m','training.v6.facts_experiment','--output',$taskRun,'--verify')
    Invoke-TaskGPU @('-m','training.v6.facts_coverage','--run',$taskRun)
    Write-TaskStage 'completed_all_three_channels'
} catch {
    Write-TaskStage 'failed_preserved_for_review' $_.Exception.Message
    throw
}
