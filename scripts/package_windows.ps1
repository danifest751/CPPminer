param(
    [Parameter(Mandatory=$true)][string]$CudaRoot,
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [string]$Root = (Join-Path $PSScriptRoot '..'),
    [string]$OutputDir,
    [string]$SourceCommit = '',
    [string]$Version = 'v0.5-fork.5'
)
$ErrorActionPreference='Stop'
$Root=(Resolve-Path -LiteralPath $Root).Path
$cache=@{}
foreach($line in Get-Content -LiteralPath (Join-Path $Root 'build\win\cmake\CMakeCache.txt')) {
    if($line -match '^([^#/:][^:]*):[^=]+=(.*)$') { $cache[$Matches[1]]=$Matches[2] }
}
$backends=@()
foreach($backend in 'CPU','CUDA','OPENCL','ONEDNN','ESIMD') {
    if($cache["CP_ENABLE_$backend"] -ne 'ON') { throw "Release requires CP_ENABLE_$backend=ON" }
    $backends += $backend.ToLowerInvariant()
}
if($cache['CP_PROOF_FFI'] -ne 'ON' -or -not(Test-Path (Join-Path $Root 'rust\cp-proof-ffi\target\release\cp_proof_ffi.lib'))) {
    throw 'Release requires the real Rust proof library'
}
$compilerPackage=Get-ChildItem (Join-Path $RuntimeRoot 'conda-meta') -Filter 'dpcpp_impl_win-64-*.json' | Select-Object -First 1
if(-not $compilerPackage) { throw 'Intel compiler version metadata missing' }
$compilerVersion=(Get-Content -LiteralPath $compilerPackage.FullName -Raw | ConvertFrom-Json).version
if(-not $OutputDir) { $OutputDir=Join-Path $Root 'cppminer-win64-cuda' }
$OutputDir=[IO.Path]::GetFullPath($OutputDir)
if(Test-Path -LiteralPath $OutputDir) { throw "Output directory already exists: $OutputDir" }
New-Item -ItemType Directory $OutputDir | Out-Null
Copy-Item -LiteralPath (Join-Path $Root 'cppminer.exe') -Destination $OutputDir
Copy-Item -LiteralPath (Join-Path $Root 'kernels') -Destination $OutputDir -Recurse
Copy-Item (Join-Path $CudaRoot 'bin\cudart64_*.dll') $OutputDir
$vswhere="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(-not $vs) { throw 'MSVC installation not found' }
$crt=Get-ChildItem "$vs\VC\Redist\MSVC" -Recurse -Directory -Filter Microsoft.VC143.CRT |
    Where-Object { $_.FullName -like '*\x64\*' -and $_.FullName -notmatch 'onecore|debug' } | Select-Object -First 1
$omp=Get-ChildItem "$vs\VC\Redist\MSVC" -Recurse -Directory -Filter Microsoft.VC143.OpenMP |
    Where-Object { $_.FullName -like '*\x64\*' -and $_.FullName -notmatch 'onecore|debug' } | Select-Object -First 1
if(-not $crt -or -not $omp) { throw 'MSVC redistributable folders missing' }
Copy-Item "$($crt.FullName)\*.dll" $OutputDir
Copy-Item "$($omp.FullName)\*.dll" $OutputDir
$dumpbin=Get-ChildItem "$vs\VC\Tools\MSVC" -Recurse -Filter dumpbin.exe |
    Where-Object FullName -like '*Hostx64\x64*' | Select-Object -First 1
if(-not $dumpbin) { throw 'dumpbin missing' }
& (Join-Path $PSScriptRoot 'package_esimd_runtime.ps1') -KernelDir (Join-Path $OutputDir 'kernels') `
    -RuntimeRoot $RuntimeRoot -Dumpbin $dumpbin.FullName -AdditionalSearchDirs $crt.FullName,$omp.FullName
Copy-Item -LiteralPath (Join-Path $RuntimeRoot 'Library\bin\OpenCL.dll') -Destination $OutputDir
foreach($name in 'README.md','CHANGELOG.md','LICENSE') { Copy-Item -LiteralPath (Join-Path $Root $name) -Destination $OutputDir }
Copy-Item -LiteralPath (Join-Path $Root "docs\releases\$Version.md") -Destination (Join-Path $OutputDir 'RELEASE_NOTES.md')
if(Test-Path (Join-Path $CudaRoot 'EULA.txt')) { Copy-Item (Join-Path $CudaRoot 'EULA.txt') (Join-Path $OutputDir 'CUDA-LICENSE.txt') }
$start=@'
@echo off
cd /d "%~dp0"
set "WALLET=YOUR_WALLET"
if not "%~1"=="" set "WALLET=%~1"
if "%WALLET%"=="YOUR_WALLET" (
    echo Set WALLET in this script or pass it as the first argument.
    exit /b 1
)
set "WORKER=%~2"
if "%WORKER%"=="" set "WORKER=cppminer"
:mine
cppminer.exe --backend BACKEND_PLACEHOLDER --devices 0 --verify --pool POOL_PLACEHOLDER --wallet "%WALLET%" --worker "%WORKER%"
timeout /t 5 /nobreak >nul
goto mine
'@
foreach($entry in @(
    @{File='start-herominers.bat';Backend='cuda';Pool='stratum+tcp://ru.pearl.herominers.com:1200'},
    @{File='start-herominers-intel.bat';Backend='onednn';Pool='stratum+tcp://ru.pearl.herominers.com:1200'},
    @{File='start-kryptex.bat';Backend='cuda';Pool='stratum+tcp://prl.kryptex.network:7048'}
)) {
    $start.Replace('BACKEND_PLACEHOLDER',$entry.Backend).Replace('POOL_PLACEHOLDER',$entry.Pool) |
        Set-Content (Join-Path $OutputDir $entry.File) -Encoding ascii
}
$info=@{
    version=$Version;source_commit=$SourceCommit;platform='windows-x64';
    backends=$backends;cuda_architectures=@($cache['CP_CUDA_ARCH'] -split ';');
    oneapi_dpcpp=$compilerVersion;proof_ffi=$true;
    binary_sha256=(Get-FileHash (Join-Path $OutputDir 'cppminer.exe') -Algorithm SHA256).Hash.ToLowerInvariant();
    esimd_sha256=(Get-FileHash (Join-Path $OutputDir 'kernels\cp_esimd.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
}
$info | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $OutputDir 'BUILD_INFO.json') -Encoding utf8
Write-Host "Windows package prepared: $OutputDir"
