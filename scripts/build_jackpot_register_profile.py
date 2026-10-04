"""Compare constant-index jackpot folds without changing the production miner.

Linux/CUDA: python3 scripts/build_jackpot_register_profile.py --output build/jackpot-profile
The full variants retain every milestone and BLAKE3. Diagnostic variants replace
only the final target/hash with a checksum, as in build_cuda_stage_profile.py.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

from build_cuda_stage_profile import replace_once


def fold_body(style):
    expression = "cp_cutlass_rotl32(jackpot_words[{i}], CP_CUTLASS_JACKPOT_LROT) ^ partial_xor"
    if style == 'switch':
        cases = '\n'.join('    case %d: jackpot_words[%d] = %s; break;' %
                          (i, i, expression.format(i=i)) for i in range(16))
        return '    switch (step % CP_CUTLASS_JACKPOT_WORDS) {\n' + cases + '\n    }'
    if style == 'predicated':
        return '''    const int tid = step % CP_CUTLASS_JACKPOT_WORDS;
    #pragma unroll
    for (int word = 0; word < CP_CUTLASS_JACKPOT_WORDS; ++word) {
        if (word == tid)
            jackpot_words[word] = cp_cutlass_rotl32(jackpot_words[word], CP_CUTLASS_JACKPOT_LROT) ^ partial_xor;
    }'''
    raise ValueError(style)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    original = root/'src/cuda/cutlass'
    if args.output.resolve() == original or original in args.output.resolve().parents:
        parser.error('--output must be outside src/cuda/cutlass')
    sources = {name: (original/name).read_text() for name in (
        'gemm_inline_xor_kernel.h', 'cp_cutlass_gemm_types.h', 'cp_cutlass_jackpot.cuh')}
    dynamic = '''    const int tid = step % CP_CUTLASS_JACKPOT_WORDS;
    jackpot_words[tid] =
        cp_cutlass_rotl32(jackpot_words[tid], CP_CUTLASS_JACKPOT_LROT) ^ partial_xor;'''
    if sources['cp_cutlass_jackpot.cuh'].count(dynamic) != 1:
        parser.error('expected the unmodified dynamic-index jackpot helper')
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
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'manifest.json').unlink(missing_ok=True)
    includes = ['-I'+str(root/path) for path in (
        'include', 'src/cuda', 'src/cuda/cutlass', 'third_party/cutlass_override',
        'third_party/cutlass/include', 'third_party/cutlass/examples/35_gemm_softmax', 'third_party/blake3')]
    base = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '-arch=sm_75', '-lineinfo']
    blake = root/'third_party/blake3'
    objects = []
    for name in ('blake3', 'blake3_dispatch', 'blake3_portable'):
        obj = args.output/(name+'.o')
        subprocess.run(['cc', '-O3', '-DBLAKE3_NO_SSE2', '-DBLAKE3_NO_SSE41',
                        '-DBLAKE3_NO_AVX2', '-DBLAKE3_NO_AVX512', '-I'+str(blake),
                        '-c', str(blake/(name+'.c')), '-o', str(obj)], check=True)
        objects.append(str(obj))
    manifest = dict(source_sha256={name: hashlib.sha256(source.encode()).hexdigest()
                                  for name, source in sources.items()},
                    harness_sha256={name: hashlib.sha256((root/'tests'/name).read_bytes()).hexdigest()
                                    for name in ('cp_cuda_stage_profile.cu', 'cp_cuda_jackpot_fold_test.cu')},
                    nvcc_version=subprocess.check_output([base[0], '--version'], text=True),
                    architecture='sm_75', warmup_launches=80, variants={})
    for style in ('dynamic', 'switch', 'predicated'):
        destination = args.output/style
        # All CUTLASS wrapper headers must resolve the same helper copy. Merely
        # overriding three -I headers mixes original/generated #pragma-once files.
        shutil.copytree(original, destination, dirs_exist_ok=True)
        generated = dict(sources)
        if style != 'dynamic':
            generated['cp_cutlass_jackpot.cuh'] = replace_once(
                generated['cp_cutlass_jackpot.cuh'], dynamic, fold_body(style))
        for name, source in generated.items():
            (destination/name).write_text(source)
        oracle = destination/'fold-test'
        subprocess.run(base+['-I'+str(destination)]+includes+[
            str(root/'tests/cp_cuda_jackpot_fold_test.cu'), *objects, '-o', str(oracle)], check=True)
        full = destination/'full'
        subprocess.run(base+['-DCP_PROFILE_VARIANT="'+style+'"', '-DCP_PROFILE_DIAGNOSTIC=0',
                            '-DCP_PROFILE_WARMUP=80', '-I'+str(destination)]+includes+[
                            str(root/'tests/cp_cuda_stage_profile.cu'), '-o', str(full)], check=True)
        diagnostic = destination/'diagnostic'
        generated['gemm_inline_xor_kernel.h'] = replace_once(generated['gemm_inline_xor_kernel.h'], hook, sink)
        (destination/'gemm_inline_xor_kernel.h').write_text(generated['gemm_inline_xor_kernel.h'])
        subprocess.run(base+['-DCP_PROFILE_VARIANT="'+style+'-diagnostic"', '-DCP_PROFILE_DIAGNOSTIC=1',
                            '-DCP_PROFILE_WARMUP=80', '-I'+str(destination)]+includes+[
                            str(root/'tests/cp_cuda_stage_profile.cu'), '-o', str(diagnostic)], check=True)
        # Preserve exactly the full kernel source next to its binary too.
        (destination/'gemm_inline_xor_kernel.full.h').write_text(sources['gemm_inline_xor_kernel.h'])
        manifest['variants'][style] = dict(
            helper_sha256=hashlib.sha256(generated['cp_cutlass_jackpot.cuh'].encode()).hexdigest(),
            binaries={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (oracle, full, diagnostic)})
        print('Built jackpot profile '+style, flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
