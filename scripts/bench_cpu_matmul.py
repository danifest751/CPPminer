#!/usr/bin/env python3
"""Build/run exact CPU multiplication experiments without CMake, Rust or CI.

Requires x86-64 GCC, OpenMP and the already-vendored BLAKE3 C sources.
Production Case33 sources are compiled unchanged, with their usual ISA flags.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=None, help="path to g++")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sample-ms", type=float, default=40)
    parser.add_argument("--cpu", type=int, default=2, help="Windows logical CPU affinity")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    cxx = args.cxx or shutil.which("g++")
    if not cxx and Path("C:/msys64/ucrt64/bin/g++.exe").is_file():
        cxx = "C:/msys64/ucrt64/bin/g++.exe"
    if not cxx:
        parser.error("GCC g++ not found; use --cxx")
    cxx = Path(cxx).resolve()
    cc = cxx.with_name(cxx.name.replace("g++", "gcc"))
    if not cc.is_file():
        parser.error("expected gcc next to g++")
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    env["PATH"] = str(cxx.parent) + os.pathsep + env.get("PATH", "")
    compiler = subprocess.check_output([str(cxx), "--version"], env=env, text=True).splitlines()[0]
    common = ["-O3", "-DNDEBUG", "-fopenmp", "-ffunction-sections", "-fdata-sections",
              "-DCASE33_BUILD_X86=1", "-DCP_ENABLE_CPU=1",
              "-DBLAKE3_NO_SSE2", "-DBLAKE3_NO_SSE41", "-DBLAKE3_NO_AVX2", "-DBLAKE3_NO_AVX512",
              "-I" + str(root / "include"), "-I" + str(root / "src/cpu/gemm"),
              "-I" + str(root / "third_party/blake3")]
    sources = [("tests/cp_cpu_matmul_bench.cpp", [])]
    for name, flags in [
        ("case33_gemm_xor", []), ("case33_cpu_features", []),
        ("case33_gemm_xor_neon", []), ("case33_gemm_xor_dotprod", []), ("case33_gemm_xor_i8mm", []),
        ("case33_gemm_xor_avx2", ["-mavx2"]), ("case33_gemm_xor_ssse3", ["-mssse3"]),
        ("case33_gemm_xor_avxvnni", ["-mavx2", "-mavxvnni"]),
        ("case33_gemm_xor_avx512bw", ["-mavx512f", "-mavx512bw", "-mavx512vl"]),
        ("case33_gemm_xor_avx512vnni", ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512vnni"]),
    ]:
        sources.append(("src/cpu/gemm/" + name + ".cpp", flags))
    # Link real cancellation/pool dependencies; none of their connection entry points are called.
    sources += [(name, []) for name in ["src/common/cp_noise.c", "src/common/cp_job_ctrl.cpp", "src/common/cp_pool.cpp",
                                      "src/common/cp_util.cpp", "src/common/cp_json_frame.cpp", "src/common/cp_tcp.cpp",
                                      "src/common/cp_qpow_pool.cpp", "src/common/cp_state.cpp",
                                      "third_party/blake3/blake3.c", "third_party/blake3/blake3_dispatch.c", "third_party/blake3/blake3_portable.c"]]
    build = root / "build/cpu-matmul-experiment"
    build.mkdir(parents=True, exist_ok=True)
    objects, commands = [], []
    for index, (name, flags) in enumerate(sources):
        obj = build / (str(index) + ".o")
        objects.append(str(obj))
        command = [str(cc if name.endswith(".c") else cxx)] + common
        if name.endswith(".cpp"):
            command += ["-std=c++17"]
        command += flags + ["-c", str(root / name), "-o", str(obj)]
        commands.append(command)
    exe = build / ("cp_cpu_matmul_bench.exe" if os.name == "nt" else "cp_cpu_matmul_bench")
    link = [str(cxx), "-fopenmp", "-pthread", "-Wl,--gc-sections"] + objects
    if os.name == "nt":
        link += ["-static", "-lws2_32", "-lbcrypt"]
    link += ["-o", str(exe)]
    # Rebuild on any relevant source/header/compiler/flag change.
    digest = hashlib.sha256(json.dumps([compiler, commands, link]).encode())
    files = {root / name for name, _ in sources}
    files.update((root / "include").glob("*.h*"))
    files.update((root / "src/cpu/gemm").glob("*.h*"))
    files.update((root / "third_party/blake3").glob("*.h"))
    files.add(root / "tests/pearl_matmul_algorithms.hpp")
    for path in sorted(files):
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    stamp = build / "fingerprint.txt"
    if not exe.is_file() or not stamp.is_file() or stamp.read_text() != fingerprint:
        for command, (name, _) in zip(commands, sources):
            print("Compiling " + name, flush=True)
            subprocess.run(command, env=env, check=True)
        print("Linking unchanged Case33 backend and experiments", flush=True)
        subprocess.run(link, env=env, check=True)
        stamp.write_text(fingerprint)
    run = [str(exe), "--repeats", str(args.repeats), "--sample-ms", str(args.sample_ms), "--cpu", str(args.cpu)]
    if args.quick:
        run.append("--quick")
    if args.verify_only:
        run.append("--verify-only")
    started = time.time()
    rows = []
    with subprocess.Popen(run, env=env, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            row = json.loads(line)
            rows.append(row)
            if row["type"] == "measurement":
                print(f'{row["workload"]} {row["m"]}x{row["n"]}x{row["k"]} {row["pattern"]}: '
                      f'{row["method"]} {row["median_ms"]:.4f} ms', flush=True)
            else:
                print(line.strip(), flush=True)
        if proc.wait() != 0:
            raise RuntimeError("benchmark/correctness check failed; no successful report written")
    if not rows or rows[-1].get("type") != "validation" or not rows[-1].get("passed"):
        raise RuntimeError("missing successful validation record")
    cpu_name = platform.processor()
    if os.name == "nt":
        cpu_name = subprocess.check_output(["powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
    report = {"schema": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
              "elapsed_seconds": time.time() - started, "cpu": cpu_name, "platform": platform.platform(),
              "compiler": compiler, "build_fingerprint": fingerprint,
              "compile_commands": commands, "link_command": link, "run_command": run, "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
