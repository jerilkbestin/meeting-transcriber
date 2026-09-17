param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$CudaDllDirectory
)
$ErrorActionPreference = "Stop"
$taskName = "Jarvis-Whisper"
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw "Task $taskName already exists; inspect it before replacing it."
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$CudaDllDirectory = (Resolve-Path -LiteralPath $CudaDllDirectory).Path
$launcher = Join-Path $PSScriptRoot "start-server.ps1"
$arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Python "{1}" -CudaDllDirectory "{2}"' -f $launcher, $Python, $CudaDllDirectory
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument $arguments -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Description "Local CUDA Whisper on the dedicated Ethernet link"
