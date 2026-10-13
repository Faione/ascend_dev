$cannBase = 'https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0'
# 只在浏览器打开下载页；文件仍需由用户保存到同一个本地目录。
$urls = @(
    "$cannBase/Ascend-cann-toolkit_9.1.0_linux-x86_64.run",
    "$cannBase/Ascend-cann-910-ops_9.1.0_linux-x86_64.run",
    "$cannBase/Ascend-cann-910b-ops_9.1.0_linux-x86_64.run",
    "$cannBase/Ascend-cann-950-ops_9.1.0_linux-x86_64.run",
    'https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl',
    'https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp310-cp310-manylinux_2_28_x86_64.whl'
)
foreach ($url in $urls) {
    Write-Host $url
    Start-Process $url
}
