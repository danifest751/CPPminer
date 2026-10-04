"""Diagnostic XOR/fold ablations, never valid mining binaries.

All controls retain final-accumulator output dependencies. Removing fold state
changes register allocation; differences are end-to-end sensitivity, not cycle
accounting. Independent CPU checks cover sampled diagnostic tile outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from build_cuda_stage_profile import replace_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    for protected in ('src', 'third_party'):
        assert out != root/protected and root/protected not in out.parents
    out.mkdir(parents=True, exist_ok=True)
    (out/'manifest.json').unlink(missing_ok=True)
    kernel = (args.control/'gemm_inline_xor_kernel.h').read_text()
    test = (root/'tests/cp_cuda_stage_profile.cu').read_text()
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
    manifest = {'completed': False, 'diagnostic_only': True, 'variants': {}}
    for variant in ('full', 'no-final-hash', 'xor-only', 'final-milestone-only'):
        dest = out/variant
        shutil.copytree(args.control, dest, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('full', 'verify', '*.sass', '*.log', '*.cubin'))
        modified = kernel
        if variant != 'full':
            modified = replace_once(modified, hook, sink)
        if variant == 'xor-only':
            modified = replace_once(modified,
                '                  cp_cutlass_jackpot_fold_step(jackpot_words[t], ms_idx, xv);\n                });',
                '                  jackpot_words[t][0] ^= xv;\n                });')
        if variant == 'final-milestone-only':
            modified = replace_once(modified, callback, callback.replace(
                '{\n            HashTilePolicy',
                '{\n            if (ms_idx != K / R_RANK - 1) return;\n            HashTilePolicy'))
        (dest/'gemm_inline_xor_kernel.h').write_text(modified)
        helper = dest/'cp_cutlass_jackpot.cuh'
        # No verification digest capture in these diagnostic binaries.
        helper.write_text(helper.read_text())
        diagnostic_test = test
        if variant == 'xor-only':
            diagnostic_test = replace_once(diagnostic_test,
                '        word = (word << 13 | word >> 19) ^ folded;',
                '        words[0] ^= folded;')
        source = dest/'ablation.cu'; source.write_text(diagnostic_test)
        command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '-arch=sm_75', '-lineinfo',
            '-Xptxas=-v', '-DCP_FEED_VERIFY=0', '-DCP_PROFILE_WARMUP=40',
            '-DCP_PROFILE_VARIANT="'+variant+'"', '-I'+str(dest)]
        command += ['-I'+str(root/p) for p in ('include', 'src/cuda', 'src/cuda/cutlass',
            'third_party/cutlass_override', 'third_party/cutlass/include',
            'third_party/cutlass/examples/35_gemm_softmax')]
        command += [str(source), '-o', str(dest/'profile')]
        with (dest/'build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        raw = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(dest/'profile')], text=True)
        (dest/'profile.sass').write_text(raw)
        manifest['variants'][variant] = {
            'sha256': hashlib.sha256((dest/'profile').read_bytes()).hexdigest(),
            'kernel_sha256': hashlib.sha256(modified.encode()).hexdigest(),
            'command': command}
        print('BUILT ablation '+variant, flush=True)
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    manifest['completed'] = True
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
