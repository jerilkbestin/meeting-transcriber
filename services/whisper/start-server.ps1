param(
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$BindAddress = "192.168.77.1",
    [string]$CudaDllDirectory = "",
    [ValidateSet("int8", "float16")][string]$ComputeType = "int8"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
if ($CudaDllDirectory) {
    if (-not (Test-Path -LiteralPath (Join-Path $CudaDllDirectory "cublas64_12.dll"))) {
        throw "CUDA 12 cuBLAS DLL not found in $CudaDllDirectory"
    }
    $env:PATH = "$CudaDllDirectory;$env:PATH"
}
$env:WHISPER_MODEL = "large-v3-turbo"
$env:WHISPER_DEVICE = "cuda"
$env:WHISPER_COMPUTE = $ComputeType
# One worker = one resident model. Bind only the dedicated wired address.
& $Python -m uvicorn server:app --app-dir $PSScriptRoot --host $BindAddress --port 8000 --workers 1
exit $LASTEXITCODE
