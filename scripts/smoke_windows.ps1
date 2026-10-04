# Run on the build host; no physical GPU is required.
param(
    [string]$Binary = (Join-Path $PSScriptRoot '..\cppminer.exe'),
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\build\win\cmake')
)
$ErrorActionPreference = 'Stop'
$Binary = (Resolve-Path -LiteralPath $Binary).Path
$BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path

# Let the native process finish before shortening its output. A downstream
# Select-Object -First can terminate the pipeline before LASTEXITCODE is set.
$helpOutput = & $Binary --help
$helpExitCode = $LASTEXITCODE
if ($helpExitCode -ne 0) { throw "--help failed ($helpExitCode)" }
$helpOutput | Select-Object -First 3

& $Binary --mock --simd-test
if ($LASTEXITCODE -ne 0) { throw "--simd-test failed ($LASTEXITCODE)" }
& $Binary --mock --prepack-test
if ($LASTEXITCODE -ne 0) { throw "--prepack-test failed ($LASTEXITCODE)" }
$proofOutput = & $Binary --backend cpu --mock --mock-diff 40 --m 1 --n 1 --max-nonce 4 --verify --threads 1
if ($LASTEXITCODE -ne 0) { throw "CPU proof test failed ($LASTEXITCODE)" }
if (($proofOutput -join "`n") -notmatch '\[plain\] verify OK') { throw 'Real proof verification was not observed' }
$proofOutput
ctest --test-dir $BuildDir -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "protocol tests failed ($LASTEXITCODE)" }
