param(
    [ValidateSet('All','InventoryPhenotype','CohortNote','StructuredMaster','LandmarkAudit','Verify')]
    [string]$Stage = 'All',
    [string]$Database = 'mimiciv',
    [string]$HostName = '127.0.0.1',
    [string]$Username = 'postgres',
    [string]$PsqlPath = 'psql'
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$createdPassword = $false
$executionTimes = [System.Collections.Generic.List[object]]::new()

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'outputs'))) {
    New-Item -ItemType Directory -Path (Join-Path $projectRoot 'outputs') | Out-Null
}

if ($PsqlPath -eq 'psql' -and -not (Get-Command psql -ErrorAction SilentlyContinue)) {
    $detectedPsql = @(
        'C:\Program Files\PostgreSQL\18\bin\psql.exe',
        'C:\Program Files\PostgreSQL\17\bin\psql.exe',
        'C:\Program Files\PostgreSQL\16\bin\psql.exe',
        'C:\Program Files\PostgreSQL\15\bin\psql.exe'
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($detectedPsql) {
        $PsqlPath = $detectedPsql
    }
    else {
        throw '找不到 psql。请安装 PostgreSQL 客户端，或通过 -PsqlPath 指定 psql.exe 的完整路径。'
    }
}

if (-not $env:PGPASSWORD) {
    $securePassword = Read-Host '请输入 PostgreSQL 密码' -AsSecureString
    $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    try {
        $env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
        $createdPassword = $true
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }
}

function Invoke-ProjectSql {
    param([string]$FileName)
    Write-Host "正在执行 $FileName"
    $startedAt = Get-Date
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $status = 'passed'
    try {
        & $PsqlPath -h $HostName -U $Username -d $Database -v ON_ERROR_STOP=1 -P pager=off -f (Join-Path 'sql' $FileName)
        if ($LASTEXITCODE -ne 0) {
            $status = 'failed'
            throw "$FileName 执行失败，退出码 $LASTEXITCODE"
        }
    }
    catch {
        $status = 'failed'
        throw
    }
    finally {
        $stopwatch.Stop()
        $executionTimes.Add([PSCustomObject]@{
            script = $FileName
            started_at = $startedAt.ToString('yyyy-MM-dd HH:mm:ss')
            elapsed_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
            status = $status
        })
    }
}

$groups = @{
    InventoryPhenotype = @('00_inventory.sql','01_icd_phenotype_audit.sql')
    CohortNote = @('02_build_strict_aecopd.sql','03_longitudinal_history.sql','04_note_coverage.sql')
    StructuredMaster = @('05_structured_field_audit.sql','06_master_cohort.sql')
    LandmarkAudit = @('07_two_landmark_availability_audit.sql')
    Verify = @('99_verify.sql')
}

Push-Location $projectRoot
try {
    $selected = switch ($Stage) {
        'All' { $groups.InventoryPhenotype + $groups.CohortNote + $groups.StructuredMaster + $groups.LandmarkAudit + $groups.Verify }
        default { $groups[$Stage] }
    }
    foreach ($script in $selected) {
        Invoke-ProjectSql -FileName $script
    }
    Write-Host '执行完成。请查看 outputs 目录和 docs\EXECUTION_REPORT.md。'
}
finally {
    if ($executionTimes.Count -gt 0) {
        $executionTimes | Export-Csv -LiteralPath (Join-Path $projectRoot 'outputs\execution_times.csv') -NoTypeInformation -Encoding utf8
    }
    Pop-Location
    if ($createdPassword) {
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    }
}


