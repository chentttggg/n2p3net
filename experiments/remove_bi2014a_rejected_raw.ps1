$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rawRoot = (Resolve-Path (Join-Path $workspace 'mne_data/MNE-braininvaders2014a-data/zenodo/3266223')).Path
$ledgerPath = Join-Path $workspace 'experiments/runs/bi61_retirement_20260906/raw_deletion.json'
$targets = foreach ($subject in 62, 63, 64) {
    foreach ($suffix in '', '.zip') {
        $candidate = Join-Path $rawRoot ("subject_{0}{1}" -f $subject, $suffix)
        if (Test-Path -LiteralPath $candidate) {
            $resolved = (Resolve-Path -LiteralPath $candidate).Path
            if ([System.IO.Path]::GetDirectoryName($resolved) -ne $rawRoot) {
                throw "Deletion target escaped the declared BI2014a directory: $resolved"
            }
            $item = Get-Item -LiteralPath $resolved
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                throw "Unexpected link at deletion target: $resolved"
            }
            $item
        }
    }
}
$files = foreach ($target in $targets) {
    $members = if ($target.PSIsContainer) { Get-ChildItem -LiteralPath $target.FullName -Recurse -File } else { $target }
    foreach ($member in $members) {
        if (-not $member.FullName.StartsWith($rawRoot + [System.IO.Path]::DirectorySeparatorChar)) {
            throw "Raw member escaped the verified root"
        }
        [ordered]@{ path = $member.FullName; bytes = $member.Length; sha256 = (Get-FileHash -LiteralPath $member.FullName -Algorithm SHA256).Hash.ToLower() }
    }
}
$ledger = [ordered]@{
    policy = 'bi2014a_exclude_62_64_20260906'
    author_source = 'https://hal.science/hal-02171575/document#page=4'
    status = 'verified_before_deletion'
    files = @($files)
}
$ledger | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ledgerPath -Encoding utf8
foreach ($target in $targets) {
    Remove-Item -LiteralPath $target.FullName -Recurse -Force
    if (Test-Path -LiteralPath $target.FullName) { throw "Deletion failed: $($target.FullName)" }
}
$ledger.status = 'deleted'
$ledger | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ledgerPath -Encoding utf8
Write-Output ("Deleted {0} raw files; ledger: {1}" -f @($files).Count, $ledgerPath)
