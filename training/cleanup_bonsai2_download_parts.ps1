param([switch]$Execute)

# 只处理本次 Bonsai 2 PTQ1_0 下载留下的 19 个明确命名的临时文件。
$ErrorActionPreference = 'Stop'
$base = 'D:\ywj\Bonsai-demo\models\bonsai2-gguf\27B'
$model = Join-Path $base 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
$expectedHash = '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'
if (-not (Test-Path -LiteralPath $model)) {
    throw 'Verified Bonsai 2 model is missing; cleanup refused.'
}
$names = @(
    'Ternary-Bonsai-2-27B-PTQ1_0.single-stream.partial',
    'range-test.bin', 'range-50mb-test.bin'
)
foreach ($index in 0..7) {
    $names += ('part{0:D2}.bin' -f $index)
    $names += ('tail{0:D2}.bin' -f $index)
}
$targets = foreach ($name in $names) {
    $path = Join-Path $base $name
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path
        if ($item.PSIsContainer -or $item.DirectoryName -ne $base) {
            throw "Unexpected cleanup target: $path"
        }
        $item
    }
}
$bytes = ($targets | Measure-Object -Property Length -Sum).Sum
Write-Output ("Temporary files: {0}; total: {1:N2} GiB" -f $targets.Count, ($bytes / 1GB))
$targets | Select-Object FullName, Length
if (-not $Execute) {
    Write-Output 'Dry run only. Re-run with -Execute after checking the target list.'
    return
}
if ((Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'Bonsai 2 model hash mismatch; cleanup refused.'
}
foreach ($target in $targets) {
    Remove-Item -LiteralPath $target.FullName -Force
}
Write-Output ("Removed {0} temporary files; verified model preserved." -f $targets.Count)
