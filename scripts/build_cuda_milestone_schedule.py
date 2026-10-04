"""Build private sm75 callback-placement candidates; production is untouched.

Control uses the already validated .cg loads and constant-index fold. Both
placements preserve each prefix before the next MMA writes its accumulator.
Use run_cuda_research_profile.py for bounded timings and exact restoration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from build_cuda_data_feed_profile import diagnostic, memory
from build_cuda_stage_profile import replace_once
from build_jackpot_register_profile import fold_body


def placement(source, variant):
    if variant in ('baseline', 'ptx_balanced'):
        return source
    if variant == 'ptx_before':
        variant = 'before_loads'
    callback = '''      if (since_ms == kMilestoneIters) {
        cb(ms_idx++, accum);
        since_ms = 0;
      }'''
    source = replace_once(source, callback, '')
    pending = '''        if (warp_mma_k == 0 && since_ms == kMilestoneIters) {
          cb(ms_idx++, accum);
          since_ms = 0;
        }
'''
    if variant == 'after_loads':
        anchor = '        this->warp_mma(accum, warp_frag_A[warp_mma_k % 2],'
    else:
        assert variant == 'before_loads'
        anchor = '        if (warp_mma_k == Base::kWarpGemmIterations - 1) {'
    # The final complete (or partial) milestone is flushed by the original
    # since_ms > 0 callback after the loop. No extra iteration/load is issued.
    return replace_once(source, anchor, pending+'\n'+anchor)


def balanced_xor(source):
    """Eight inputs in four logic instructions, depth two instead of four.

    Do not change the lane mapping, reduce-scatter or transcript order. Plain
    non-volatile register PTX gives ptxas freedom to schedule independent LOP3s.
    """
    start = source.index('          uint32_t x = 0u;')
    end = source.index('          w[h * 8 + a * 4 + b] = x;', start)
    assert source[start:end].count('x ^=') == 1
    replacement = '''          int const e0 = 2 * (b * kRowIters + 4 * h + a);
          int const e1 = 2 * ((b + 4) * kRowIters + 4 * h + a);
          int const e2 = 2 * (b * kRowIters + 4 * h + a + 2);
          int const e3 = 2 * ((b + 4) * kRowIters + 4 * h + a + 2);
          uint32_t x;
          asm("{ .reg .b32 p, q, r;\\n"
              "lop3.b32 p, %1, %2, %3, 0x96;\\n"
              "lop3.b32 q, %4, %5, %6, 0x96;\\n"
              "xor.b32 r, %7, %8;\\n"
              "lop3.b32 %0, p, q, r, 0x96;\\n}"
              : "=r"(x)
              : "r"(uint32_t(accum[e0])), "r"(uint32_t(accum[e0+1])),
                "r"(uint32_t(accum[e1])), "r"(uint32_t(accum[e1+1])),
                "r"(uint32_t(accum[e2])), "r"(uint32_t(accum[e2+1])),
                "r"(uint32_t(accum[e3])), "r"(uint32_t(accum[e3+1])));
'''
    return source[:start]+replacement+source[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['baseline','after_loads','before_loads'],
                        choices=['baseline','after_loads','before_loads','ptx_balanced','ptx_before'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    for protected in ('src', 'third_party'):
        assert output != root/protected and root/protected not in output.parents
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').unlink(missing_ok=True)
    old = json.loads((args.objects/'manifest.json').read_text())
    objects = [args.objects/(str(i)+'.o') for i in range(11)]
    # Verify the sources of all reused common/BLAKE3 objects.
    common = ['src/common/cp_noise.c', 'src/common/cp_job_ctrl.cpp',
              'src/common/cp_pool.cpp', 'src/common/cp_util.cpp',
              'src/common/cp_json_frame.cpp', 'src/common/cp_tcp.cpp',
              'src/common/cp_qpow_pool.cpp', 'src/common/cp_state.cpp',
              'third_party/blake3/blake3.c', 'third_party/blake3/blake3_dispatch.c',
              'third_party/blake3/blake3_portable.c']
    for path in common:
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == old['source_sha256'][path]
    assert all(p.is_file() for p in objects)
    includes = ['-I'+str(root/p) for p in ('include', 'src/cuda', 'src/cuda/cutlass',
        'third_party/cutlass_override', 'third_party/cutlass/include',
        'third_party/cutlass/examples/35_gemm_softmax', 'third_party/blake3')]
    manifest = {'group': 'milestone-schedule', 'completed': False, 'variants': {},
                'commands': [], 'reused_objects': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in objects}}
    nvcc = '/usr/local/cuda/bin/nvcc'
    manifest['nvcc_version'] = subprocess.check_output([nvcc, '--version'], text=True)
    assert 'baseline' in args.variants and len(set(args.variants)) == len(args.variants)
    for variant in args.variants:
        dest = output/variant
        shutil.copytree(root/'src/cuda/cutlass', dest, dirs_exist_ok=True)
        shutil.copytree(root/'third_party/cutlass/include/cutlass/arch', dest/'cutlass/arch', dirs_exist_ok=True)
        mem = dest/'cutlass/arch/memory.h'
        mem.write_text(memory(mem.read_text(), 'cg128'))
        helper = dest/'cp_cutlass_jackpot.cuh'
        dynamic = '''    const int tid = step % CP_CUTLASS_JACKPOT_WORDS;
    jackpot_words[tid] =
        cp_cutlass_rotl32(jackpot_words[tid], CP_CUTLASS_JACKPOT_LROT) ^ partial_xor;'''
        helper.write_text(diagnostic(replace_once(helper.read_text(), dynamic, fold_body('predicated'))))
        loop = dest/'mma_milestone.h'
        loop.write_text(placement(loop.read_text(), variant))
        if variant.startswith('ptx_'):
            policy = dest/'hash_tile_policy.h'
            policy.write_text(balanced_xor(policy.read_text()))
        test = (root/'tests/cp_cuda_data_feed_profile.cu').read_text()
        test = replace_once(test, '        compare(digests.host(), expected.digests, "all keyed BLAKE3 digests");', '''        compare(digests.host(), expected.digests, "all keyed BLAKE3 digests");
        // Also test the null-dump mining branch, which has a separate callback.
        CUDA(cudaMemset(digests.p, 0xa5, digests.count * 4));
        status(op.initialize(in.m, in.n, 4096, in.m, in.n, in.a.p, in.b.p,
                             nullptr, in.n / 128, tiles, &jackpot));
        status(op());
        CUDA(cudaDeviceSynchronize());
        compare(digests.host(), expected.digests, "all mining-branch BLAKE3 digests");''')
        test_source = dest/'profile.cu'
        test_source.write_text(test)
        record = {'compiled': False, 'binaries': {}, 'generated_sha256': {
            str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.rglob('*') if p.is_file() and p.suffix in ('.h', '.cuh', '.cu')}}
        for verify, kind in ((0, 'full'), (1, 'verify')):
            binary = dest/kind
            command = [nvcc, '-std=c++17', '-O3', '-arch=sm_75', '-lineinfo', '-Xptxas=-v',
                '-Xcompiler=-fopenmp', '-Xlinker=--gc-sections',
                '-DCP_FEED_VARIANT="'+variant+'"', '-DCP_FEED_VERIFY='+str(verify),
                '-I'+str(dest), *includes, str(root/'tests/cp_cuda_data_feed_profile.cu'),
                *[str(p) for p in objects], '-o', str(binary)]
            command[command.index(str(root/'tests/cp_cuda_data_feed_profile.cu'))] = str(test_source)
            manifest['commands'].append(command)
            with (dest/(kind+'-build.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            record['binaries'][kind] = {'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
            if not verify:
                sass = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(binary)], text=True)
                (dest/'full.sass').write_text(sass)
                record['sass_sha256'] = hashlib.sha256(sass.encode()).hexdigest()
        record['compiled'] = True
        manifest['variants'][variant] = record
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('BUILT '+variant, flush=True)
    manifest['completed'] = True
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
