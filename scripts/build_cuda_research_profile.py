"""Build isolated exact Pearl research kernels for sm_75.

Uses the data-feed oracle and real common/noise/BLAKE3 objects. No production or
shared dependency header is edited. Run one group after the preceding GPU test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from build_cuda_data_feed_profile import diagnostic, memory
from build_cuda_stage_profile import replace_once
from build_jackpot_register_profile import fold_body


GROUPS = {
    'cache': ('baseline', 'cg_a', 'cg_b', 'cg_ab'),
    'swizzle': ('baseline', 'group1', 'group2', 'group4', 'group16', 'transpose'),
    'pipeline': ('baseline', 'k32_s2', 'k32_s3', 'k32_s4', 'k64_s3_small'),
    'minimal': ('baseline', 'direct', 'lookahead', 'lookahead_cg'),
    'native': ('baseline',),
    'lookahead': ('baseline', 'reg_k64', 'reg_k32'),
    'lookahead_postbar': ('baseline', 'post_b_k64', 'post_b_k32', 'post_ab_k64'),
}


def selective_cache(root, destination, variant):
    """Give operand iterators distinct types; keep address/predicate code intact."""
    vendor = root/'third_party/cutlass/include/cutlass'
    source = (vendor/'arch/memory.h').read_text()
    begin = source.index('template <typename AccessType>\nstruct global_load<AccessType,\n                   16')
    end = source.index('template <typename AccessType>', begin+len('template <typename AccessType>'))
    specialization = source[begin:end].replace('global_load', 'cp_research_load_cg')
    specialization = specialization.replace('ld.global.L2::128B', 'ld.global.cg.L2::128B')
    license_text=source[:source.index('/*!')]
    (destination/'cp_research_cache.h').write_text(license_text+'''#pragma once
#include "cutlass/arch/memory.h"
namespace cutlass { namespace arch {
template<class T, int N> struct cp_research_load_cg : global_load<T,N> {
  CUTLASS_DEVICE cp_research_load_cg(T& value, void const* ptr, bool guard)
      : global_load<T,N>(value,ptr,guard) {}
};
'''+specialization+'\n}}\n')
    iterator = (vendor/'transform/threadblock/predicated_tile_iterator.h').read_text()
    iterator = iterator.replace('PredicatedTileIterator', 'CpResearchCgTileIterator')
    iterator = iterator.replace('cutlass::arch::global_load<', 'cutlass::arch::cp_research_load_cg<')
    iterator = iterator.replace('#include "cutlass/arch/memory.h"', '#include "cp_research_cache.h"')
    (destination/'cp_research_cg_iterator.h').write_text(iterator)
    mma = (vendor/'gemm/threadblock/default_mma.h').read_text()
    mma = mma.replace('#include "cutlass/transform/threadblock/predicated_tile_iterator.h"',
                      '#include "cutlass/transform/threadblock/predicated_tile_iterator.h"\n#include "cp_research_cg_iterator.h"')
    for operand in ('A','B'):
        if variant in ('cg_ab','cg_'+operand.lower()):
            pattern = r'(using Iterator'+operand+r'\s*=\s*cutlass::transform::threadblock::)PredicatedTileIterator<'
            mma, count = re.subn(pattern, r'\1CpResearchCgTileIterator<', mma)
            assert count > 0, 'operand iterator hook missing'
    file = destination/'cutlass/gemm/threadblock/default_mma.h'
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(mma)


def traversal(source, variant):
    if variant.startswith('group'):
        return replace_once(source, 'GemmIdentityThreadblockSwizzle<8>',
                            'GemmIdentityThreadblockSwizzle<'+variant[5:]+'>')
    if variant == 'transpose':
        declaration = '''struct CpResearchTransposeSwizzle {
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord get_tiled_shape(cutlass::gemm::GemmCoord problem,
      cutlass::gemm::GemmCoord tile, int split_k_slices) const {
    return {(problem.m()+tile.m()-1)/tile.m(),
            (problem.n()+tile.n()-1)/tile.n(), split_k_slices};
  }
  CUTLASS_HOST_DEVICE int get_log_tile(cutlass::gemm::GemmCoord) const { return 0; }
  CUTLASS_HOST_DEVICE dim3 get_grid_shape(cutlass::gemm::GemmCoord tiled) const {
    return dim3(tiled.n(),tiled.m(),tiled.k());
  }
  CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int) const {
    return {int(blockIdx.y),int(blockIdx.x),int(blockIdx.z)};
  }
};
'''
        source = replace_once(source, 'namespace cp_cutlass {', 'namespace cp_cutlass {\n'+declaration)
        return replace_once(source, 'cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>',
                            'CpResearchTransposeSwizzle')
    return source


def pipeline(source, variant):
    if variant == 'baseline':
        return source
    # Sm80-tag multistage with the sm75 MMA atom falls back to ordinary
    # vector global/shared copies and CTA barriers on sm75. No cp.async used.
    stages = int(variant.split('_s')[1].split('_')[0])
    k = 32 if variant.startswith('k32') else 64
    source = replace_once(source, 'using TensorOpWarpShape = cutlass::gemm::GemmShape<64, 64, 64>;',
                          'using TensorOpWarpShape = cutlass::gemm::GemmShape<64, 64, '+str(k)+'>;')
    source = replace_once(source, 'using Shape256x128x64 = cutlass::gemm::GemmShape<256, 128, 64>;',
                          'using Shape256x128x64 = cutlass::gemm::GemmShape<'+('128' if variant.endswith('small') else '256')+', 128, '+str(k)+'>;')
    for name,shape in (('Gemm128x128TensorOp','cutlass::gemm::GemmShape<128, 128, 64>'),
                       ('Gemm256x128TensorOp','Shape256x128x64')):
        start = source.index('using '+name+' = GemmTypesCase10<')
        end = source.index(';',start)+1
        block = source[start:end]
        if stages > 2:
            block = block.replace('cutlass::arch::Sm75','cutlass::arch::Sm80')
        block = block.replace('TensorOpInstructionShape, 2, 16', 'TensorOpInstructionShape, '+str(stages)+', 16')
        if k == 32: block = block.replace('cutlass::gemm::GemmShape<128, 128, 64>',
                                        'cutlass::gemm::GemmShape<128, 128, 32>')
        source = source[:start]+block+source[end:]
    return source


def register_lookahead(source):
    """Two future global fragments in registers; unchanged two-stage shared ring.

    Phase-unroll makes fragment indices constant. Refilling occurs only after
    the old fragment's shared store, so a pending global load can overlap two
    compute tiles without overwriting a future tile or adding shared stages.
    """
    start=source.index('        if (warp_mma_k == 0) {')
    end=source.index('\n\n        this->warp_mma',start)
    source=source[:start]+source[end:]
    source=replace_once(source,'''    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    int gemm_k_iterations = total_iters;
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);''','''    FragmentA tb_frag_A[2];
    FragmentB tb_frag_B[2];
    iterator_A.clear_mask(total_iters <= 1);
    iterator_B.clear_mask(total_iters <= 1);
    CUTLASS_PRAGMA_UNROLL
    for (int phase = 0; phase < 2; ++phase) {
      tb_frag_A[phase].clear(); iterator_A.load(tb_frag_A[phase]); ++iterator_A;
      tb_frag_B[phase].clear(); iterator_B.load(tb_frag_B[phase]); ++iterator_B;
      iterator_A.clear_mask(total_iters <= phase + 2);
      iterator_B.clear_mask(total_iters <= phase + 2);
    }
    int gemm_k_iterations = total_iters;''')
    source=replace_once(source,'for (; gemm_k_iterations > 0; --gemm_k_iterations) {',
                          '''for (; gemm_k_iterations > 0;) {
      CUTLASS_PRAGMA_UNROLL
      for (int phase = 0; phase < 2; ++phase) {
        if (gemm_k_iterations <= 0) break;''')
    source=replace_once(source,'''          this->smem_iterator_A_.store(this->transform_A_(tb_frag_A));
          this->smem_iterator_B_.store(this->transform_B_(tb_frag_B));''','''          this->smem_iterator_A_.store(this->transform_A_(tb_frag_A[phase]));
          this->smem_iterator_B_.store(this->transform_B_(tb_frag_B[phase]));
          tb_frag_A[phase].clear(); iterator_A.load(tb_frag_A[phase]); ++iterator_A;
          tb_frag_B[phase].clear(); iterator_B.load(tb_frag_B[phase]); ++iterator_B;
          iterator_A.clear_mask(gemm_k_iterations <= 4);
          iterator_B.clear_mask(gemm_k_iterations <= 4);''')
    source=replace_once(source,'''    }

    if (since_ms > 0)''','''        --gemm_k_iterations;
      }
    }

    if (since_ms > 0)''')
    return source


def postbar_lookahead(source,only_b):
    source=register_lookahead(source)
    refill='''          tb_frag_A[phase].clear(); iterator_A.load(tb_frag_A[phase]); ++iterator_A;
          tb_frag_B[phase].clear(); iterator_B.load(tb_frag_B[phase]); ++iterator_B;
          iterator_A.clear_mask(gemm_k_iterations <= 4);
          iterator_B.clear_mask(gemm_k_iterations <= 4);'''
    source=replace_once(source,refill,'')
    mma='''        this->warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                       warp_frag_B[warp_mma_k % 2], accum);'''
    if only_b:
        source=replace_once(source,'FragmentA tb_frag_A[2];','FragmentA tb_frag_A;')
        source=replace_once(source,'      tb_frag_A[phase].clear(); iterator_A.load(tb_frag_A[phase]); ++iterator_A;\n','')
        source=replace_once(source,'      iterator_A.clear_mask(total_iters <= phase + 2);\n','')
        source=replace_once(source,'this->transform_A_(tb_frag_A[phase])','this->transform_A_(tb_frag_A)')
        source=replace_once(source,mma,'''        if (warp_mma_k == 0) {
          tb_frag_A.clear(); iterator_A.load(tb_frag_A); ++iterator_A;
          iterator_A.clear_mask(gemm_k_iterations <= 2);
        }
'''+mma)
        refill='''          tb_frag_B[phase].clear(); iterator_B.load(tb_frag_B[phase]); ++iterator_B;
          iterator_B.clear_mask(gemm_k_iterations <= 4);'''
    # Prefetch after the CTA barrier, so a freshly issued future LDG does not
    # immediately meet that barrier. Only the now-free register slot is reused.
    return replace_once(source,mma,mma+'''
        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
'''+refill+'''
        }''')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=GROUPS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--objects',type=Path,required=True,
                        help='existing data-feed object directory with its manifest')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    if root/'src' == output or root/'src' in output.parents or root/'third_party' == output or root/'third_party' in output.parents:
        parser.error('output must be outside source/dependency trees')
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').unlink(missing_ok=True)
    previous=json.loads((args.objects/'manifest.json').read_text())
    objects=[args.objects/(str(i)+'.o') for i in range(11)]
    common_sources=['src/common/cp_noise.c','src/common/cp_job_ctrl.cpp','src/common/cp_pool.cpp','src/common/cp_util.cpp',
        'src/common/cp_json_frame.cpp','src/common/cp_tcp.cpp','src/common/cp_qpow_pool.cpp','src/common/cp_state.cpp',
        'third_party/blake3/blake3.c','third_party/blake3/blake3_dispatch.c','third_party/blake3/blake3_portable.c']
    for file in common_sources:
        assert hashlib.sha256((root/file).read_bytes()).hexdigest()==previous['source_sha256'][file], 'stale reused object: '+file
    assert all(file.is_file() for file in objects)
    includes=['-I'+str(root/p) for p in ('include','src/cuda','src/cuda/cutlass',
        'third_party/cutlass_override','third_party/cutlass/include','third_party/cutlass/examples/35_gemm_softmax','third_party/blake3')]
    nvcc='/usr/local/cuda/bin/nvcc';original=root/'src/cuda/cutlass'
    manifest={'group':args.group,'nvcc_version':subprocess.check_output([nvcc,'--version'],text=True),
              'reused_object_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
              'variants':{},'commands':[]}
    for variant in GROUPS[args.group]:
        dest=output/variant;shutil.copytree(original,dest,dirs_exist_ok=True)
        helper=dest/'cp_cutlass_jackpot.cuh'
        helper_source=helper.read_text()
        if args.group=='native':
            dynamic='''    const int tid = step % CP_CUTLASS_JACKPOT_WORDS;
    jackpot_words[tid] =
        cp_cutlass_rotl32(jackpot_words[tid], CP_CUTLASS_JACKPOT_LROT) ^ partial_xor;'''
            helper_source=replace_once(helper_source,dynamic,fold_body('predicated'))
            shutil.copytree(root/'third_party/cutlass/include/cutlass/arch',dest/'cutlass/arch',dirs_exist_ok=True)
            mem=dest/'cutlass/arch/memory.h';mem.write_text(memory(mem.read_text(),'cg128'))
        helper.write_text(diagnostic(helper_source))
        types=dest/'cp_cutlass_gemm_types.h';source=types.read_text()
        if args.group=='cache' and variant!='baseline':selective_cache(root,dest,variant)
        if args.group=='swizzle':source=traversal(source,variant)
        if args.group=='pipeline':source=pipeline(source,variant)
        if args.group=='lookahead' and variant!='baseline':
            if variant=='reg_k32':source=pipeline(source,'k32_s2')
            mainloop=dest/'mma_milestone.h';mainloop.write_text(register_lookahead(mainloop.read_text()))
        if args.group=='lookahead_postbar' and variant!='baseline':
            if variant.endswith('k32'):source=pipeline(source,'k32_s2')
            mainloop=dest/'mma_milestone.h';mainloop.write_text(postbar_lookahead(mainloop.read_text(),variant.startswith('post_b_')))
        types.write_text(source)
        test_source=root/'tests/cp_cuda_data_feed_profile.cu'
        if args.group in ('minimal','native'):
            test=(root/'tests/cp_cuda_data_feed_profile.cu').read_text()
            # Equal reduced warmup for this deliberately high-traffic prototype
            # and its control. Keep the complete INT64/hash oracle unchanged.
            if args.group=='minimal':test=replace_once(test,'i < 40;', 'i < 4;')
            if args.group=='minimal' and variant!='baseline':
                shutil.copy2(root/'tests/cp_cuda_minimal_mma.cuh',dest/'cp_cuda_minimal_mma.cuh')
                test=replace_once(test,'#include "cp_cutlass_gemm_types.h"',
                                  '#include "cp_cutlass_gemm_types.h"\n#include "cp_cuda_minimal_mma.cuh"')
                test=replace_once(test,'cp_cutlass::FusedMilestoneGemmOp<T> op;', 'cp_research::MinimalOp op;')
                assert test.count('cp_cutlass::FusedKernelEntry<T>')==2
                test=test.replace('cp_cutlass::FusedKernelEntry<T>','cp_research::minimal_kernel')
                test=replace_once(test,'sizeof(typename T::GemmKernel::SharedStorage)', '0')
                test=replace_once(test,'T::GemmKernel::kThreadCount', '256')
            if args.group=='native':
                shutil.copy2(root/'tests/cp_cuda_native_schedule.cuh',dest/'cp_cuda_native_schedule.cuh')
                test=replace_once(test,'#include "cp_cutlass_gemm_types.h"',
                                  '#include "cp_cutlass_gemm_types.h"\n#include "cp_cuda_native_schedule.cuh"')
                test=replace_once(test,'cp_cutlass::FusedMilestoneGemmOp<T> op;', 'cp_research::NativeOp<T> op;')
                test=replace_once(test,'CUDA(cudaMemcpyToSymbol(cp_feed_digest_sink, &digests.p, sizeof(digests.p)));',
                                  'cp_research::native_symbol("cp_feed_digest_sink", &digests.p, sizeof(digests.p));')
                test=replace_once(test,'CUDA(cudaMemcpyToSymbol(cp_feed_digest_columns, &tile_cols, sizeof(tile_cols)));',
                                  'cp_research::native_symbol("cp_feed_digest_columns", &tile_cols, sizeof(tile_cols));')
            test_source=dest/'profile.cu';test_source.write_text(test)
        record={'generated_sha256':{str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in dest.rglob('*') if p.is_file() and p.suffix in ('.h','.cuh','.cu')},'binaries':{}}
        try:
            for verify,name in ((0,'full'),(1,'verify')):
                binary=dest/name
                command=[nvcc,'-std=c++17','-O3','-arch=sm_75','-lineinfo','-Xptxas=-v','-Xcompiler=-fopenmp',
                    '-Xlinker=--gc-sections','-DCP_FEED_VARIANT="'+variant+'"','-DCP_FEED_VERIFY='+str(verify),
                    '-DCP_MINIMAL_PREFETCH='+('1' if variant.startswith('lookahead') else '0'),
                    '-DCP_MINIMAL_CG='+('1' if variant.endswith('_cg') else '0'),
                    '-I'+str(dest)]+includes+[str(test_source),*[str(p) for p in objects],'-o',str(binary)]
                manifest['commands'].append(command)
                if args.group=='native':command+=['-L/usr/local/cuda/lib64/stubs','-lcuda']
                with (dest/(name+'-build.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                record['binaries'][name]={'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
                if args.group=='native':
                    cubin=dest/(name+'.cubin')
                    cmd=[nvcc,'--cubin','-std=c++17','-O3','-arch=sm_75','-lineinfo','-Xptxas=-v',
                         '-DCP_FEED_VARIANT="'+variant+'"','-DCP_FEED_VERIFY='+str(verify),'-I'+str(dest)]+includes+[str(test_source),'-o',str(cubin)]
                    manifest['commands'].append(cmd)
                    with (dest/(name+'-cubin.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                    disassembly=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(cubin)],text=True)
                    functions={}
                    for function in re.findall(r'Function : ([^\n]+)',disassembly):
                        if 'FusedKernelEntry' in function:
                            functions['256' if 'ILi256ELi128ELi64' in function else '128']=function
                    assert set(functions)=={'128','256'}
                    record.setdefault('cubins',{})[name]={'path':str(cubin),'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest(),'functions':functions}
                if not verify:
                    raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True)
                    (dest/'full.sass').write_text(raw)
                    record['sass_sha256']=hashlib.sha256(raw.encode()).hexdigest()
            record['compiled']=True
        except subprocess.CalledProcessError as error:
            record['compiled']=False;record['error']=str(error)
        manifest['variants'][variant]=record
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('BUILT '+variant+' '+str(record['compiled']),flush=True)
    manifest['completed']=True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
