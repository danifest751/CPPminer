"""Build controlled standalone ablations in a temporary directory, not the miner.

Run on Linux with CUDA: python3 scripts/build_cuda_stage_profile.py --output build/stage-profile
"""
import argparse
import hashlib
import json
import pathlib
import subprocess


def replace_once(source, old, new):
    assert source.count(old) == 1, 'profile hook changed; inspect the current kernel'
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    original = root / 'src/cuda/cutlass'
    kernel = (original/'gemm_inline_xor_kernel.h').read_text()
    types = (original/'cp_cutlass_gemm_types.h').read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    hook = '''cp_cutlass_jackpot_try(
            jackpot_words[t], params.jackpot.ptr_a_key8, params.jackpot.bound,
            row_period_eff, col_period_eff, vt, params.jackpot.ptr_found,
            params.jackpot.ptr_out_t_rows, params.jackpot.ptr_out_t_cols);'''
    sink = '''uint32_t diagnostic = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int word = 0; word < CP_CUTLASS_JACKPOT_WORDS; ++word)
          diagnostic ^= jackpot_words[t][word];
        const size_t index = (static_cast<size_t>(cta_r) * tile_cols + cta_c) * kHashTilesPerCta + vt;
        params.jackpot.ptr_out_t_rows[index] = static_cast<int>(diagnostic);'''
    callback = '''[&](int ms_idx, typename Mma::FragmentC const &acc) {
            HashTilePolicy::milestone_xor(
                acc, lane_idx, [&](int t, uint32_t xv) {
                  cp_cutlass_jackpot_fold_step(jackpot_words[t], ms_idx, xv);
                });
          });'''
    manifest = dict(kernel_sha256=hashlib.sha256(kernel.encode()).hexdigest(),
                    types_sha256=hashlib.sha256(types.encode()).hexdigest(), variants={})
    for variant in ('full', 'no-final-hash', 'final-milestone-only'):
        modified = kernel
        if variant != 'full':
            modified = replace_once(modified, hook, sink)
        if variant == 'final-milestone-only':
            modified = replace_once(modified, callback, callback.replace(
                '{\n            HashTilePolicy',
                '{\n            if (ms_idx != K / R_RANK - 1) return;\n            HashTilePolicy', 1))
        destination = args.output / variant
        destination.mkdir(exist_ok=True)
        (destination/'gemm_inline_xor_kernel.h').write_text(modified)
        (destination/'cp_cutlass_gemm_types.h').write_text(types)
        binary = destination/'profile'
        command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '-arch=sm_75', '-lineinfo',
                   '-DCP_PROFILE_VARIANT="'+variant+'"', '-I'+str(destination)]
        for path in ('include', 'src/cuda', 'src/cuda/cutlass', 'third_party/cutlass_override',
                     'third_party/cutlass/include', 'third_party/cutlass/examples/35_gemm_softmax'):
            command.append('-I'+str(root/path))
        command += [str(root/'tests/cp_cuda_stage_profile.cu'), '-o', str(binary)]
        subprocess.run(command, check=True)
        manifest['variants'][variant] = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                                            generated_kernel_sha256=hashlib.sha256(modified.encode()).hexdigest())
        print('Built profile variant ' + variant, flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
