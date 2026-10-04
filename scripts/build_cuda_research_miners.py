"""Build the R1 cache/jackpot factorial in an explicitly private checkout.

Dependencies, including CUTLASS, must be populated beforehand. Reject symlinked
CUTLASS/wrapper directories: this script temporarily changes private headers,
restores them in finally, and never modifies the installed release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from build_jackpot_register_profile import fold_body
from build_cuda_stage_profile import replace_once


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    mem=root/'third_party/cutlass/include/cutlass/arch/memory.h'
    helper=root/'src/cuda/cutlass/cp_cutlass_jackpot.cuh'
    for file in (mem,helper):
        assert file.is_file() and file.resolve().is_relative_to(root), 'use private copied headers'
        assert not any(p.is_symlink() for p in (file,*file.parents) if p!=root.parent), 'use private copied headers'
    original_mem=mem.read_text();original_helper=helper.read_text()
    old='ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];'
    dynamic='''    const int tid = step % CP_CUTLASS_JACKPOT_WORDS;
    jackpot_words[tid] =
        cp_cutlass_rotl32(jackpot_words[tid], CP_CUTLASS_JACKPOT_LROT) ^ partial_xor;'''
    pred=replace_once(original_helper,dynamic,fold_body('predicated'))
    assert original_mem.count(old)==1
    cmake=root/'build/miner';out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').unlink(missing_ok=True)
    configure=['cmake','-S',str(root),'-B',str(cmake),'-DCMAKE_BUILD_TYPE=Release',
               '-DCP_ENABLE_CUDA=ON','-DCP_CUDA_ARCH=75','-DCP_ENABLE_OPENCL=OFF',
               '-DCP_ENABLE_CUBLAS=OFF','-DCP_ENABLE_WGPU=OFF']
    subprocess.run(configure,check=True)
    report={'architecture':'sm_75','configure':configure,'variants':{}}
    try:
        for name,cg,predicated in [('baseline',False,False),('cg',True,False),('predicated',False,True),('combined',True,True)]:
            mem.write_text(replace_once(original_mem,old,old.replace('ld.global.','ld.global.cg.')) if cg else original_mem)
            helper.write_text(pred if predicated else original_helper)
            start=time.monotonic()
            subprocess.run(['cmake','--build',str(cmake),'-j','4'],check=True)
            dest=out/name;dest.mkdir(exist_ok=True)
            shutil.copy2(cmake/'cppminer',dest/'cppminer')
            shutil.copy2(mem,dest/'memory.h');shutil.copy2(helper,dest/helper.name)
            report['variants'][name]={'cg':cg,'predicated':predicated,'seconds':time.monotonic()-start,
                'binary_sha256':hashlib.sha256((dest/'cppminer').read_bytes()).hexdigest(),
                'memory_sha256':hashlib.sha256(mem.read_bytes()).hexdigest(),
                'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest()}
            (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print('BUILT '+name,flush=True)
        report['completed']=True
    finally:
        mem.write_text(original_mem);helper.write_text(original_helper)
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
