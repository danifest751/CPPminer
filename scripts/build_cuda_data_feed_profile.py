"""Build isolated sm_75 data-feed variants; never modify production/vendor headers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from build_cuda_stage_profile import replace_once

VARIANTS = ('baseline', 'early', 'late1', 'cg128', 'cs128', 'prefetch64', 'no_prefetch')


def mainloop(source, variant):
    start = source.index('        if (warp_mma_k == 0) {')
    end = source.index('\n\n        this->warp_mma', start)
    load = source[start:end]
    result = source[:start]+source[end:]
    if variant == 'early':
        anchor = '        if (warp_mma_k == Base::kWarpGemmIterations - 1) {'
        return replace_once(result, anchor, load+'\n\n'+anchor)
    if variant == 'late1':
        mma = '''        this->warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                       warp_frag_B[warp_mma_k % 2], accum);'''
        return replace_once(result, mma, mma+'\n\n'+load.replace('warp_mma_k == 0','warp_mma_k == 1'))
    return source


def memory(source, variant):
    old = 'ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];'
    changes = dict(cg128='ld.global.cg.L2::128B.v4.u32', cs128='ld.global.cs.L2::128B.v4.u32',
                   prefetch64='ld.global.L2::64B.v4.u32', no_prefetch='ld.global.v4.u32')
    if variant in changes:
        return replace_once(source, old, changes[variant]+' {%0, %1, %2, %3}, [%4];')
    return source


def diagnostic(source):
    # Verification-only digest capture, after the real BLAKE3 computation and
    # before the unchanged target check. No extra instructions in timed builds.
    source = replace_once(source, '#define CP_CUTLASS_JACKPOT_WORDS 16', '''#define CP_CUTLASS_JACKPOT_WORDS 16
#if CP_FEED_VERIFY
__device__ uint32_t* cp_feed_digest_sink;
__device__ int cp_feed_digest_columns;
#endif''')
    return replace_once(source, '    b3_compress64(a_key8, msg, digest);', '''    b3_compress64(a_key8, msg, digest);
#if CP_FEED_VERIFY
    const size_t tile = (size_t(row_period) * cp_feed_digest_columns + col_period) * 256 + thread_idx;
    for (int i = 0; i < 8; ++i) cp_feed_digest_sink[tile * 8 + i] = digest[i];
#endif''')


def sass(binary):
    raw = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True)
    binary.with_suffix('.sass').write_text(raw)
    result = {}
    for function, body in re.findall(r'Function : ([^\n]+)\n(.*?)(?=\n\s*Function :|\Z)',raw,re.S):
        if 'FusedKernelEntry' not in function:
            continue
        shape = '256x128' if 'ILi256ELi128ELi64' in function else '128x128'
        # Keep instruction encodings and scheduling bits. No line-info dump used.
        normalized = '\n'.join(line.strip() for line in body.splitlines() if '/*' in line)
        result[shape] = dict(sha256=hashlib.sha256(normalized.encode()).hexdigest(),
            static_counts={op:len(re.findall(r'\b'+op+r'(?:\.|\s)',body)) for op in
                           ('IMMA','LDG','LDS','STS','LDL','STL','LEA','IADD3','BAR','SHFL')})
    assert set(result)=={'128x128','256x128'}, 'missing fused SASS kernels'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    if root/'src'==output or root/'src' in output.parents or root/'third_party'==output or root/'third_party' in output.parents:
        parser.error('build outside src/ and third_party/')
    output.mkdir(parents=True,exist_ok=True)
    # A failed rebuild must not leave an older complete build manifest usable.
    (output/'manifest.json').unlink(missing_ok=True)
    original=root/'src/cuda/cutlass'
    main_source=(original/'mma_milestone.h').read_text()
    memory_source=(root/'third_party/cutlass/include/cutlass/arch/memory.h').read_text()
    includes=['-I'+str(root/p) for p in ('include','src/cuda','src/cuda/cutlass','third_party/cutlass_override',
        'third_party/cutlass/include','third_party/cutlass/examples/35_gemm_softmax','third_party/blake3')]
    definitions=['-DBLAKE3_NO_SSE2','-DBLAKE3_NO_SSE41','-DBLAKE3_NO_AVX2','-DBLAKE3_NO_AVX512','-DCP_ENABLE_CPU=0']
    sources=['src/common/cp_noise.c','src/common/cp_job_ctrl.cpp','src/common/cp_pool.cpp','src/common/cp_util.cpp',
        'src/common/cp_json_frame.cpp','src/common/cp_tcp.cpp','src/common/cp_qpow_pool.cpp','src/common/cp_state.cpp',
        'third_party/blake3/blake3.c','third_party/blake3/blake3_dispatch.c','third_party/blake3/blake3_portable.c']
    commands,objects=[],[]
    for i,path in enumerate(sources):
        obj=output/(str(i)+'.o')
        command=['cc' if path.endswith('.c') else 'c++','-O3','-fopenmp','-ffunction-sections','-fdata-sections']
        if path.endswith('.cpp'): command.append('-std=c++17')
        command+=definitions+includes+['-c',str(root/path),'-o',str(obj)]
        subprocess.run(command,check=True); commands.append(command); objects.append(str(obj))
    nvcc='/usr/local/cuda/bin/nvcc'
    tracked=sources+['tests/cp_cuda_data_feed_profile.cu','scripts/build_cuda_data_feed_profile.py']
    tracked+=[str(p.relative_to(root)) for p in original.glob('*') if p.is_file()]
    tracked+=['third_party/cutlass/include/cutlass/arch/memory.h']
    manifest=dict(architecture='sm_75',nvcc_version=subprocess.check_output([nvcc,'--version'],text=True),
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in tracked},variants={},commands=commands)
    for variant in VARIANTS:
        destination=output/variant
        shutil.copytree(original,destination,dirs_exist_ok=True)
        # memory.h includes architecture helpers by relative path. Keep those
        # helpers in the same private include tree, with their license notices.
        shutil.copytree(root/'third_party/cutlass/include/cutlass/arch',destination/'cutlass/arch',dirs_exist_ok=True)
        (destination/'mma_milestone.h').write_text(mainloop(main_source,variant))
        mem=destination/'cutlass/arch/memory.h'; mem.parent.mkdir(parents=True,exist_ok=True)
        mem.write_text(memory(memory_source,variant))
        helper=destination/'cp_cutlass_jackpot.cuh'
        helper.write_text(diagnostic(helper.read_text()))
        record=dict(generated_sha256={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (destination/'mma_milestone.h',mem,helper)},binaries={})
        for verify,name in ((0,'full'),(1,'verify')):
            binary=destination/name
            command=[nvcc,'-std=c++17','-O3','-arch=sm_75','-lineinfo','-Xptxas=-v',
                '-Xcompiler=-fopenmp','-Xlinker=--gc-sections','-DCP_FEED_VARIANT="'+variant+'"',
                '-DCP_FEED_VERIFY='+str(verify),'-I'+str(destination)]+includes+[
                str(root/'tests/cp_cuda_data_feed_profile.cu'),*objects,'-o',str(binary)]
            with (destination/(name+'-build.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            commands.append(command)
            record['binaries'][name]=dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            if not verify: record['sass']=sass(binary)
        manifest['variants'][variant]=record
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Built data-feed variant '+variant,flush=True)


if __name__=='__main__':
    main()
