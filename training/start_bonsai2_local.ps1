# 本机 Bonsai 2 文本服务。仅监听 loopback，不开启 Web UI 或跨站凭证。
param(
    [ValidateSet('auto', 'off')][string]$ReasoningMode = 'auto',
    [ValidateRange(-1, 16384)][int]$ReasoningBudget = 2048,
    [ValidateRange(8192, 32768)][int]$ContextTokens = 32768,
    [ValidateRange(1, 4)][int]$Slots = 4,
    [ValidateRange(1024, 65535)][int]$Port = 8080
)
$ErrorActionPreference = 'Stop'
# 总上下文由槽位分摊；并发不能以减少每项原文的可用上下文为代价。
if (($ContextTokens % $Slots) -ne 0 -or ($ContextTokens / $Slots) -lt 8192) {
    throw 'Each Bonsai slot needs at least 8192 context tokens; no silent truncation.'
}
$binary = 'D:\ywj\Bonsai-demo\bin\cuda\llama-server.exe'
$model = 'D:\ywj\Bonsai-demo\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PTQ1_0.gguf'
if (-not (Test-Path -LiteralPath $binary) -or -not (Test-Path -LiteralPath $model)) {
    throw 'Bonsai 2 binary or verified GGUF is missing.'
}
$arguments = @(
    '-m', $model,
    '--host', '127.0.0.1', '--port', "$Port",
    '-ngl', '99', '-fa', 'on', '-c', "$ContextTokens", '-np', "$Slots",
    '-cb',
    '--jinja', '--reasoning', $ReasoningMode, '--reasoning-budget', "$ReasoningBudget",
    '--reasoning-format', 'deepseek',
    '--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.05',
    '--repeat-penalty', '1.0',
    '--no-webui', '--cors-origins', 'localhost', '--no-cors-credentials'
)
& $binary @arguments
exit $LASTEXITCODE
