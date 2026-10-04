"""Build the standalone exact CUDA multiplication experiment (Linux/CUDA).

No miner, Rust proof build, pool traffic, or CI is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="sm_75", choices=["sm_75", "sm_80", "sm_86"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if root / "src" == output or root / "src" in output.parents:
        parser.error("build outside src/")
    output.mkdir(parents=True, exist_ok=True)
    includes = ["-I"+str(root/p) for p in ("include", "src/cuda", "src/cuda/cutlass",
        "third_party/cutlass_override", "third_party/cutlass/include", "third_party/cutlass/examples/35_gemm_softmax", "third_party/blake3")]
    definitions = ["-DBLAKE3_NO_SSE2", "-DBLAKE3_NO_SSE41", "-DBLAKE3_NO_AVX2", "-DBLAKE3_NO_AVX512", "-DCP_ENABLE_CPU=0"]
    sources = ["src/common/cp_noise.c", "src/common/cp_job_ctrl.cpp", "src/common/cp_pool.cpp",
               "src/common/cp_util.cpp", "src/common/cp_json_frame.cpp", "src/common/cp_tcp.cpp",
               "src/common/cp_qpow_pool.cpp", "src/common/cp_state.cpp",
               "third_party/blake3/blake3.c", "third_party/blake3/blake3_dispatch.c", "third_party/blake3/blake3_portable.c"]
    commands, objects = [], []
    for i, source in enumerate(sources):
        obj = output / (str(i)+".o")
        command = ["cc" if source.endswith(".c") else "c++", "-O3", "-fopenmp", "-ffunction-sections", "-fdata-sections"]
        if source.endswith(".cpp"):
            command.append("-std=c++17")
        command += definitions + includes + ["-c", str(root/source), "-o", str(obj)]
        subprocess.run(command, check=True)
        commands.append(command)
        objects.append(str(obj))
    nvcc = "/usr/local/cuda/bin/nvcc"
    exe = output / "matmul-bench"
    command = [nvcc, "-std=c++17", "-O3", "-arch="+args.arch, "-lineinfo", "-Xptxas=-v",
               "-Xcompiler=-fopenmp", "-Xlinker=--gc-sections"]+includes+[
                   str(root/"tests/cp_cuda_matmul_bench.cu"), *objects, "-o", str(exe)]
    subprocess.run(command, check=True)
    commands.append(command)
    tracked = ["tests/cp_cuda_matmul_bench.cu", "scripts/build_cuda_matmul_bench.py", "src/common/cp_noise.c"]
    tracked += [str(p.relative_to(root)) for p in (root/"src/cuda/cutlass").glob("*") if p.is_file()]
    manifest = {"architecture": args.arch, "commands": commands,
                "nvcc_version": subprocess.check_output([nvcc, "--version"], text=True),
                "source_sha256": {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in tracked},
                "binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest()}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Built " + str(exe), flush=True)


if __name__ == "__main__":
    main()
