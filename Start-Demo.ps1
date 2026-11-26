param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$demoRoot = $PSScriptRoot
$demoPython = Join-Path $demoRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $demoPython)) { throw 'Prepare the project Python environment first.' }
$demoProfile = Get-Content -LiteralPath (Join-Path $demoRoot 'config\demo.json') -Raw | ConvertFrom-Json
$demoPort = [int]$demoProfile.local_port
$demoAddress = 'http://127.0.0.1:' + $demoPort
$demoRuntime = Join-Path $demoRoot 'runtime\gpu-demo'
New-Item -ItemType Directory -Path $demoRuntime -Force | Out-Null
$demoListener = Get-NetTCPConnection -LocalPort $demoPort -State Listen -ErrorAction SilentlyContinue
if (!$demoListener) {
    $demoProcess = Start-Process -FilePath $demoPython -ArgumentList @('-u', (Join-Path $demoRoot 'scripts\start_demo.py'), '--no-browser') -WorkingDirectory $demoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoRuntime 'launcher.stdout.log') -RedirectStandardError (Join-Path $demoRuntime 'launcher.stderr.log') -PassThru
    [System.IO.File]::WriteAllText((Join-Path $demoRuntime 'launcher.pid'), [string]$demoProcess.Id)
}
$demoDeadline = [DateTime]::UtcNow.AddSeconds(45)
$demoReady = $false
while ([DateTime]::UtcNow -lt $demoDeadline) {
    try {
        $demoDeployment = Invoke-RestMethod -Uri ($demoAddress + '/api/v1/deployment') -TimeoutSec 2
        $demoStatus = Invoke-RestMethod -Uri ($demoAddress + '/ready') -TimeoutSec 2
        if ($demoDeployment.location -eq 'remote' -and $demoDeployment.mode -eq 'real' -and $demoStatus.ready -and $demoStatus.mock -eq $false) { $demoReady = $true; break }
    } catch { }
    if ($demoProcess -and $demoProcess.HasExited) { throw 'The connection ended. See runtime/gpu-demo/launcher.stderr.log.' }
    Start-Sleep -Milliseconds 500
}
if (!$demoReady) { throw 'The service is unavailable. Check the connection log and GPU server.' }
Write-Output ('Ready: ' + $demoAddress)
if (!$NoBrowser) { Start-Process -FilePath $demoAddress }
