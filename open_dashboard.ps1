param(
    [string]$SshHost = "connect.weste.seetacloud.com",
    [int]$SshPort = 35832,
    [string]$SshUser = "root",
    [string]$RemoteRoot = "/root/autodl-tmp/n2p3-net",
    [int]$LocalPort = 18812,
    [int]$RemotePort = 8812,
    [string]$IdentityFile = (Join-Path $env:USERPROFILE ".ssh\id_ed25519_n2p3_cloud_20260827"),
    [string]$Run,
    [switch]$NoBrowser,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$dashboardUrl = "http://127.0.0.1:$LocalPort/dashboard.html"
$listener = $null
$ssh = $null

function Get-LocalListener {
    Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Get-ProcessCommandLine([int]$ProcessId) {
    try {
        (Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId").CommandLine
    } catch {
        $null
    }
}

function Test-Dashboard {
    try {
        $status = Invoke-RestMethod "http://127.0.0.1:$LocalPort/api/status" -TimeoutSec 4
        if (-not $status.ok) { return $null }
        return $status
    } catch {
        return $null
    }
}

function Ensure-RemoteDashboard {
    $remoteCommand = "cd $RemoteRoot && mkdir -p logs && if ! pgrep -f '[d]ashboard_server.py --port $RemotePort' >/dev/null; then nohup .venv/bin/python experiments/dashboard_server.py --port $RemotePort --bind 127.0.0.1 --directory experiments > logs/dashboard-http.log 2>&1 < /dev/null & fi"
    Write-Host "[monitor] Remote dashboard API is unavailable; starting remote dashboard on port $RemotePort."
    Write-Host "[monitor] SSH may prompt for the password again. Training processes are not restarted."
    & ssh.exe @sshAuthArgs -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p $SshPort "${SshUser}@${SshHost}" $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start the remote dashboard service."
    }
}

function Stop-Tunnel {
    $connection = Get-LocalListener
    if (-not $connection) {
        Write-Host "[monitor] No listener on local port $LocalPort."
        return
    }
    $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
    $commandLine = Get-ProcessCommandLine $connection.OwningProcess
    if ($process.ProcessName -ne "ssh" -or $commandLine -notmatch "-L\s*${LocalPort}:127\.0\.0\.1:${RemotePort}") {
        throw "Local port $LocalPort is occupied by another process: PID $($connection.OwningProcess)"
    }
    Stop-Process -Id $connection.OwningProcess -Force
    Write-Host "[monitor] Local SSH tunnel stopped. Cloud training was not affected."
}

if ($Stop) {
    Stop-Tunnel
    exit 0
}

if (-not (Test-Path -LiteralPath $IdentityFile -PathType Leaf)) {
    throw "SSH identity file does not exist: $IdentityFile"
}

$sshAuthArgs = @(
    "-i", $IdentityFile,
    "-o", "IdentitiesOnly=yes"
)

$listener = Get-LocalListener
if ($listener) {
    $existingStatus = Test-Dashboard
    if (-not $existingStatus) {
        $commandLine = Get-ProcessCommandLine $listener.OwningProcess
        throw "Local port $LocalPort is occupied, but it is not a usable dashboard tunnel: $commandLine"
    }
    Write-Host "[monitor] Reusing existing SSH tunnel (PID $($listener.OwningProcess))."
} else {
    $sshArgs = @(
        "-N"
    ) + $sshAuthArgs + @(
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "StrictHostKeyChecking=no",
        "-L", "${LocalPort}:127.0.0.1:${RemotePort}",
        "-p", $SshPort,
        "${SshUser}@${SshHost}"
    )
    Write-Host "[monitor] Opening SSH tunnel: 127.0.0.1:$LocalPort -> remote 127.0.0.1:$RemotePort"
    Write-Host "[monitor] SSH may prompt for a password; it is not stored in this script."
    $ssh = Start-Process -FilePath "ssh.exe" -ArgumentList $sshArgs -NoNewWindow -PassThru
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        if ($ssh.HasExited) {
            throw "SSH tunnel exited early, ExitCode=$($ssh.ExitCode)"
        }
        $listener = Get-LocalListener
    } while (-not $listener -and (Get-Date) -lt $deadline)
    if (-not $listener) {
        Stop-Process -Id $ssh.Id -Force -ErrorAction SilentlyContinue
        throw "Timed out waiting for local port $LocalPort."
    }
}

$status = $null
$deadline = (Get-Date).AddSeconds(15)
do {
    $status = Test-Dashboard
    if (-not $status) { Start-Sleep -Milliseconds 500 }
} while (-not $status -and (Get-Date) -lt $deadline)
if (-not $status) {
    Ensure-RemoteDashboard
    $deadline = (Get-Date).AddSeconds(15)
    do {
        $status = Test-Dashboard
        if (-not $status) { Start-Sleep -Milliseconds 500 }
    } while (-not $status -and (Get-Date) -lt $deadline)
}
if (-not $status) {
    throw "SSH tunnel is up, but dashboard API is still unavailable on remote port $RemotePort."
}

$targetUrl = $dashboardUrl
if ($Run) {
    $targetUrl += "?run=" + [uri]::EscapeDataString($Run)
}
Write-Host "[monitor] Dashboard port: $($status.port)"
Write-Host "[monitor] Auto run: $($status.latest_run)"
Write-Host "[monitor] URL: $targetUrl"

if (-not $NoBrowser) {
    Start-Process $targetUrl
}

if ($ssh) {
    Write-Host "[monitor] Tunnel is running. Keep this window open; press Ctrl+C to stop it."
    Wait-Process -Id $ssh.Id
    Write-Host "[monitor] SSH tunnel exited."
}
