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
$demoSocket = New-Object System.Net.Sockets.TcpClient
try { $demoListener = $demoSocket.ConnectAsync('127.0.0.1', $demoPort).Wait(500) -and $demoSocket.Connected } catch { $demoListener = $false } finally { $demoSocket.Dispose() }
if (!$demoListener) {
    $demoScript = '"' + (Join-Path $demoRoot 'scripts\start_demo.py') + '"'
    $demoProcess = Start-Process -FilePath $demoPython -ArgumentList @('-u', $demoScript, '--no-browser') -WorkingDirectory $demoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoRuntime 'launcher.stdout.log') -RedirectStandardError (Join-Path $demoRuntime 'launcher.stderr.log') -PassThru
    [System.IO.File]::WriteAllText((Join-Path $demoRuntime 'launcher.pid'), [string]$demoProcess.Id)
}
$demoDeadline = [DateTime]::UtcNow.AddSeconds(45)
$demoReady = $false
while ([DateTime]::UtcNow -lt $demoDeadline) {
    try {
        $demoDeployment = Invoke-RestMethod -Uri ($demoAddress + '/api/v1/deployment') -TimeoutSec 2
        $demoStatus = Invoke-RestMethod -Uri ($demoAddress + '/health') -TimeoutSec 2
        if ($demoDeployment.transport -eq 'ssh' -and $demoDeployment.local_demo -and $demoStatus.status -eq 'ok') { $demoReady = $true; break }
    } catch { }
    if ($demoProcess -and $demoProcess.HasExited) { throw 'The local page could not start. See runtime/gpu-demo/launcher.stderr.log.' }
    Start-Sleep -Milliseconds 500
}
if (!$demoReady) { throw 'The local page is unavailable. Check the local port and runtime/gpu-demo/launcher.stderr.log.' }
if ($demoListener) {
    try { Invoke-RestMethod -Uri ($demoAddress + '/api/v1/deployment') -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 2 | Out-Null } catch { }
}
Write-Output ('Ready: ' + $demoAddress)
if (!$NoBrowser) { Start-Process -FilePath $demoAddress }
