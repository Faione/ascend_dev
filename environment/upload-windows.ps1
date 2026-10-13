param(
    [Parameter(Mandatory = $true)]
    [string]$Server,             # user@host
    [Parameter(Mandatory = $true)]
    [string]$RemoteRoot,         # e.g. /home/user/cann-run
    [string]$DownloadDir = ''    # browser download directory
)

$ErrorActionPreference = 'Stop'
if (-not $DownloadDir) {
    # 默认查找当前 Windows 用户的浏览器下载目录。
    $profileDir = [Environment]::GetFolderPath('UserProfile')
    if (-not $profileDir) { throw 'Set -DownloadDir to your browser download directory.' }
    $DownloadDir = Join-Path -Path $profileDir -ChildPath 'Downloads'
}
if ($Server -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$') { throw 'Server must be user@host.' }
if ($RemoteRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteRoot -match '(^|/)\.\.(/|$)') {
    throw 'RemoteRoot must be an absolute Linux path without spaces or .. segments.'
}
if (-not (Test-Path -LiteralPath $DownloadDir -PathType Container)) { throw "Directory not found: $DownloadDir" }
foreach ($command in @('ssh', 'scp')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Missing $command. Enable Windows OpenSSH Client." }
}
$names = @(
    # 一次上传 Toolkit、910/910B/950 算子包及两个 Python wheel。
    'Ascend-cann-toolkit_9.1.0_linux-x86_64.run',
    'Ascend-cann-910-ops_9.1.0_linux-x86_64.run',
    'Ascend-cann-910b-ops_9.1.0_linux-x86_64.run',
    'Ascend-cann-950-ops_9.1.0_linux-x86_64.run',
    'torch-2.10.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl',
    'torch_npu-2.10.0.post4-cp310-cp310-manylinux_2_28_x86_64.whl'
)
foreach ($name in $names) {
    # 上传前先确认文件存在，并打印本地 SHA256 供人工核对。
    $file = Join-Path -Path $DownloadDir -ChildPath $name
    if (-not (Test-Path -LiteralPath $file -PathType Leaf) -or (Get-Item -LiteralPath $file).Length -eq 0) {
        throw "Missing or empty file: $file"
    }
    Write-Host "$name SHA256: $((Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash)"
}
$remoteDir = $RemoteRoot.TrimEnd('/')
& ssh $Server "mkdir -p -- '$remoteDir'"
if ($LASTEXITCODE -ne 0) { throw 'Remote directory creation failed.' }
Push-Location $DownloadDir
try {
    $localFiles = @($names | ForEach-Object { "./$_" })
    & scp @localFiles "${Server}:$remoteDir/"
    if ($LASTEXITCODE -ne 0) { throw 'SCP upload failed.' }
} finally {
    Pop-Location
}
$remotePaths = @($names | ForEach-Object { "'$remoteDir/$_'" })
$remoteHashes = & ssh $Server ('sha256sum ' + ($remotePaths -join ' '))
# 逐文件对比本地与服务器哈希，避免大安装包在传输中损坏。
if ($LASTEXITCODE -ne 0) { throw 'Remote SHA256 check failed.' }
foreach ($name in $names) {
    $expected = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $DownloadDir $name)).Hash.ToLowerInvariant()
    $line = $remoteHashes | Where-Object { $_ -like "* $remoteDir/$name" } | Select-Object -First 1
    if (-not $line -or (($line -split '\s+')[0]).ToLowerInvariant() -ne $expected) {
        throw "SHA256 mismatch after upload: $name"
    }
}
Write-Host "Ready: $remoteDir"
