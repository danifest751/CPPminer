# Validate the redistributable package with all toolchain search paths removed.
param([Parameter(Mandatory=$true)][string]$PackageDir)
$ErrorActionPreference='Stop'
$PackageDir=(Resolve-Path -LiteralPath $PackageDir).Path
$savedPath=$env:PATH
$savedLocation=Get-Location
try {
    $env:PATH="$env:SystemRoot\System32;$env:SystemRoot"
    Set-Location $env:TEMP
    $binary=Join-Path $PackageDir 'cppminer.exe'
    foreach($arguments in @(
        @('--help'),
        @('--mock','--simd-test'),
        @('--mock','--prepack-test'),
        @('--backend','cpu','--mock','--mock-diff','40','--m','1','--n','1','--max-nonce','4','--verify','--threads','1')
    )) {
        $output=& $binary @arguments
        if($LASTEXITCODE -ne 0) { throw "Portable package test failed: $arguments ($LASTEXITCODE)" }
        if($arguments -contains '--verify' -and ($output -join "`n") -notmatch '\[plain\] verify OK') {
            throw 'Portable package did not verify a real proof'
        }
        Write-Host "Portable test passed: $arguments"
    }
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class EsimdPackageProbe {
    [DllImport("kernel32", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern IntPtr LoadLibraryExW(string path, IntPtr file, uint flags);
    [DllImport("kernel32", CharSet=CharSet.Unicode, SetLastError=true)]
    public static extern bool SetDllDirectoryW(string path);
    [DllImport("kernel32", CharSet=CharSet.Ansi, ExactSpelling=true)]
    public static extern IntPtr GetProcAddress(IntPtr module, string name);
    [DllImport("kernel32")] public static extern bool FreeLibrary(IntPtr module);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate uint AbiVersion();
}
'@
    # Match the miner: UR's proxy performs additional loads by name after the
    # initial import resolution, so it also needs the kernels search directory.
    if(-not [EsimdPackageProbe]::SetDllDirectoryW((Join-Path $PackageDir 'kernels'))) {
        throw 'Cannot register bundled runtime search directory'
    }
    foreach($name in 'cp_esimd.dll','ur_adapter_opencl.dll') {
        $module=[EsimdPackageProbe]::LoadLibraryExW((Join-Path $PackageDir "kernels\$name"),[IntPtr]::Zero,8)
        if($module -eq [IntPtr]::Zero) { throw "Cannot load $name (Win32 $([Runtime.InteropServices.Marshal]::GetLastWin32Error()))" }
        try {
            if($name -eq 'cp_esimd.dll') {
                foreach($symbol in 'cp_esimd_abi_version','cp_esimd_create','cp_esimd_scan_panel','cp_esimd_wait','cp_esimd_destroy') {
                    if([EsimdPackageProbe]::GetProcAddress($module,$symbol) -eq [IntPtr]::Zero) { throw "Missing export $symbol" }
                }
                $address=[EsimdPackageProbe]::GetProcAddress($module,'cp_esimd_abi_version')
                $abi=[Runtime.InteropServices.Marshal]::GetDelegateForFunctionPointer($address,[EsimdPackageProbe+AbiVersion])
                if($abi.Invoke() -ne 2) { throw 'ESIMD ABI does not match the host (expected 2)' }
            }
            Write-Host "Portable DLL test passed: $name"
        } finally { [void][EsimdPackageProbe]::FreeLibrary($module) }
    }
} finally {
    if('EsimdPackageProbe' -as [type]) { [void][EsimdPackageProbe]::SetDllDirectoryW($null) }
    $env:PATH=$savedPath
    Set-Location $savedLocation
}
