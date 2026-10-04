"""Bounded sm75 SASS schedules from existing cubins, with preserved metadata.

Requires a private CuAssembler checkout plus sympy and pyelftools. Attempts an
assembler round-trip first. Original instruction words can be moved directly if
CUDA-toolchain metadata prevents reassembly: control-word split/merge must still
round-trip every text section exactly. No branch, barrier, LDSM or dependency
chain is moved. Exact GPU oracles remain required before timing each candidate.
"""
import argparse
import copy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from sass_milestone_stats import parse as parse_sass, loops_of


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--assembler',type=Path,required=True)
    parser.add_argument('--packages',type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(args.assembler))
    if args.packages:sys.path.insert(0,str(args.packages))
    os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
    from CuAsm.CubinFile import CubinFile
    from CuAsm.CuAsmParser import CuAsmParser
    from CuAsm.CuControlCode import CuControlCode
    from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
    from CuAsm.CuInsFeeder import CuInsFeeder
    from CuAsm.CuSMVersion import CuSMVersion
    from elftools.elf.elffile import ELFFile
    build=args.build.resolve();manifest_path=build/'manifest.json'
    manifest=json.loads(manifest_path.read_text());assert manifest.get('completed')
    assert list(manifest['variants'])==['baseline'], 'start from a fresh native baseline'
    baseline=manifest['variants']['baseline'];assert baseline['compiled']
    manifest['completed']=False
    report={'assembler_commit':subprocess.check_output(['git','-C',str(args.assembler),'rev-parse','HEAD'],text=True).strip(),
            'roundtrip':{},'candidates':{}}
    manifest['native_schedule']=report
    def save():manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    def sections(path):
        elf=ELFFile(BytesIO(path.read_bytes()))
        return {s.name:{'header':{k:s.header[k] for k in ('sh_type','sh_flags','sh_size','sh_info','sh_link','sh_entsize')},'data':s.data()}
                for s in elf.iter_sections()}
    def equal_sections(before,after,text=True):
        if set(before)!=set(after):return False
        return all(before[name]==after[name] for name in before if text or not name.startswith('.text.'))
    repos=CuInsAssemblerRepos.getDefaultRepos(75)
    asm_files={};originals={};section_maps={};tensor_ranges={}
    for kind in ('full','verify'):
        original=Path(baseline['cubins'][kind]['path']);originals[kind]=original
        assert hashlib.sha256(original.read_bytes()).hexdigest()==baseline['cubins'][kind]['sha256']
        sass=build/('original-'+kind+'.sass')
        sass.write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(original)],text=True))
        tensor_ranges[kind]={}
        for function,instructions in parse_sass(sass).items():
            if 'FusedKernelEntry' in function:
                tensor_ranges[kind]['.text.'+function]=[
                    (instructions[first][0],instructions[last][0],
                     'dump-loop' if any(op.startswith('STG') for _,op,_ in instructions[first:last+1]) else 'mine-loop')
                    for first,last,_ in loops_of(instructions)]
        repos.update(CuInsFeeder(str(sass),archfilter=75))
        asm=build/('original-'+kind+'.cuasm');CubinFile(str(original)).saveAsCuAsm(str(asm));asm_files[kind]=asm
        section_maps[kind]=sections(original)
        for name,sec in section_maps[kind].items():
            if name.startswith('.text.'):
                controls,instructions=CuSMVersion.splitCtrlCodeFromBytes_7x_8x(sec['data'])
                assert CuSMVersion.mergeCtrlCodes_7x_8x(instructions,controls)==sec['data'], 'control split/merge failed'
    repo_file=build/'private-instruction-repos.txt';repos.save2file(str(repo_file))
    roundtrip_ok=True
    for kind in ('full','verify'):
        target=build/('roundtrip-'+kind+'.cubin')
        try:
            assembler=CuAsmParser();assembler.setInsAsmRepos(str(repo_file),75)
            assembler.parse(str(asm_files[kind]));assembler.saveAsCubin(str(target))
            same=equal_sections(section_maps[kind],sections(target))
            report['roundtrip'][kind]={'assembled':True,'all_section_data_and_resource_headers_equal':same,
                'whole_file_identical':target.read_bytes()==originals[kind].read_bytes(),
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
            roundtrip_ok &= same
        except Exception as error:
            roundtrip_ok=False;report['roundtrip'][kind]={'assembled':False,'error':str(error)}
        save()
    if roundtrip_ok:
        name='roundtrip';dest=build/name;dest.mkdir(exist_ok=True);record=copy.deepcopy(baseline)
        for kind in ('full','verify'):
            shutil.copy2(build/'baseline'/kind,dest/kind)
            shutil.copy2(build/('roundtrip-'+kind+'.cubin'),dest/(kind+'.cubin'))
            record['cubins'][kind].update(path=str(dest/(kind+'.cubin')),sha256=hashlib.sha256((dest/(kind+'.cubin')).read_bytes()).hexdigest())
        manifest['variants'][name]=record

    # Conservative superset of operand registers: four registers per mentioned
    # GPR/UR token includes vector destinations, 64-bit addresses and IMMA pairs.
    def registers(asm):
        result=set()
        scalar=bool(re.match(r'(?:@!?P\d+\s+)?(?:SHF|IADD3|MOV|LOP3)\b',asm))
        for prefix,number in re.findall(r'\b(UR|R|UP|P)(\d+)\b',asm):
            value=int(number)
            for offset in range(4 if not scalar and prefix in ('R','UR') else 1):result.add(prefix+str(value+offset))
        return result
    pattern=re.compile(r'\[([^]]+)\]\s*/\*([0-9a-f]+)\*/\s*(.*?;)')
    def parse(line):
        match=pattern.search(line)
        if not match:return None
        ctrl,address,asm=match.groups()
        return {'address':int(address,16),'control':CuControlCode.encode(ctrl),'asm':asm,'registers':registers(asm)}
    schedules={}
    for kind,asm_file in asm_files.items():
        lines=asm_file.read_text().splitlines();eligible=[];section=None;history=[]
        for index,line in enumerate(lines):
            if '.section ' in line or '.section\t' in line:
                match=re.search(r'\.section\s+([^,\s]+)',line)
                section=match.group(1).strip('"') if match else None;history=[]
            current=parse(line)
            if not current:
                if line.strip() and not line.lstrip().startswith('//'):history=[]
                continue
            if section and 'FusedKernelEntry' in section and current['asm'].lstrip().startswith(('LDG.','@')) and ' LDG.' in ' '+current['asm']:
                previous=parse(lines[index-1]) if index else None
                loop=next((item for item in tensor_ranges[kind].get(section,[]) if previous and item[0]<=previous['address'] and current['address']<=item[1]),None)
                if loop and previous and re.match(r'(?:@!?P\d+\s+)?(?:IMMA|IADD3|LEA|IMAD|MOV|LOP3|SHF)\b',previous['asm']):
                    pc=CuControlCode(previous['control']);lc=CuControlCode(current['control'])
                    if (not (previous['registers'] & current['registers'])
                        and pc.Read==7 and pc.Barrier==0 and pc.Stall+lc.Stall<=15
                        and (pc.Write==7 or (lc.Read!=pc.Write and lc.Write!=pc.Write and not lc.Barrier & (1<<pc.Write)))
                        and current['address']==previous['address']+16
                        ):
                        # Moving the LDG earlier needs its address/predicate inputs
                        # ready already. Require 16 explicit scheduled cycles since
                        # any touch, conservatively counting reads as dependencies.
                        bracket=re.search(r'\[([^]]+)\]',current['asm'])
                        inputs=set()
                        for prefix,number in re.findall(r'\b(UR|R)(\d+)\b',bracket.group(1) if bracket else ''):
                            inputs|={prefix+str(int(number)+offset) for offset in range(2)}
                        inputs|={p for p in registers(current['asm'].split('LDG.')[0]) if p.startswith(('P','UP'))}
                        gap=0;ready=False
                        for old in reversed(history[:-1]):
                            gap+=CuControlCode(old['control']).Stall
                            if gap>=16:ready=True;break
                            if old['registers'] & inputs:break
                        if ready:
                            eligible.append({'section':section,'previous':previous,'load':current,'input_gap_cycles':gap,
                                             'loop_start':loop[0],'loop_end':loop[1],'loop_kind':loop[2]})
            history.append(current)
        schedules[kind]=eligible
        report['candidates'][kind]={'eligible_independent_pairs':len(eligible)}
        print('ELIGIBLE '+kind+' '+str(len(eligible)),flush=True);save()

    for name,limit,scoreboard in (('load_1',1,False),('load_4',4,False),('load_8',8,False),
                                  ('load_1_scoreboard',1,True),('load_4_scoreboard',4,True)):
        dest=build/name;dest.mkdir(exist_ok=True);record=copy.deepcopy(baseline);changes={}
        for kind in ('full','verify'):
            raw=bytearray(originals[kind].read_bytes());elf=ELFFile(BytesIO(raw))
            used=set();selected=[];per_loop={}
            for item in schedules[kind]:
                key=(item['section'],item['previous']['address'])
                category=(item['section'],item['loop_start'])
                if per_loop.get(category,0)>=limit:continue
                if key in used or (key[0],key[1]+16) in used or (key[0],key[1]-16) in used:continue
                section=elf.get_section_by_name(item['section']);base=section.header['sh_offset'];offset=base+key[1]
                assert 0<=key[1] and key[1]+32<=section.header['sh_size'], 'instruction outside text section'
                a=bytes(raw[offset:offset+16]);b=bytes(raw[offset+16:offset+32])
                controls,codes=CuSMVersion.splitCtrlCodeFromBytes_7x_8x(a+b)
                assert controls==[item['previous']['control'],item['load']['control']]
                pcontrol=CuControlCode(controls[0]);lcontrol=CuControlCode(controls[1])
                # Keep original scoreboard/yield fields with their instructions;
                # preserve the moved ALU's downstream latency by adding LDG stall.
                # Also test the original stall for dynamically scoreboarded
                # IMMA results. Treat this as a candidate, not a proved hazard
                # model: all exact GPU checks are mandatory before timing it.
                stall=pcontrol.Stall if scoreboard and pcontrol.Write!=7 else pcontrol.Stall+lcontrol.Stall
                moved_control=(controls[0]&~15)|stall
                swapped=CuSMVersion.mergeCtrlCodes_7x_8x([codes[1],codes[0]],[controls[1],moved_control])
                raw[offset:offset+32]=swapped;used.add(key);selected.append(item)
                per_loop[category]=per_loop.get(category,0)+1
            if not selected:record['compiled']=False;record['error']='no eligible independent pair in '+kind;break
            target=dest/(kind+'.cubin');target.write_bytes(raw)
            after=sections(target)
            assert equal_sections(section_maps[kind],after,text=False), 'metadata modified'
            assert all(section_maps[kind][n]['header']==after[n]['header'] for n in after), 'resource header modified'
            assert any(section_maps[kind][n]!=after[n] for n in after if n.startswith('.text.')), 'unchanged native schedule'
            shutil.copy2(build/'baseline'/kind,dest/kind)
            record['cubins'][kind].update(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            (dest/(kind+'.sass')).write_text(subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(target)],text=True))
            changes[kind]=[{k:v for k,v in item.items() if k not in ('previous','load')} |
                           {'before':[item['previous']['asm'],item['load']['asm']],
                            'previous_address':item['previous']['address'],'load_address':item['load']['address']}
                          for item in selected]
        record['native_changes']=changes;record['scoreboard_stall_candidate']=scoreboard
        if record['compiled']:
            for other,other_record in manifest['variants'].items():
                if other_record.get('compiled') and all(record['cubins'][k]['sha256']==other_record['cubins'][k]['sha256'] for k in ('full','verify')):
                    record.update(compiled=False,duplicate_of=other,error='duplicate cubin schedule');break
        manifest['variants'][name]=record;save()
        print('BUILT native '+name+' '+str(record['compiled']),flush=True)
    report['control_split_merge_all_text_exact']=True
    report['method']='original 128-bit instruction words; only order and explicit stall fields changed'
    manifest['completed']=True;save()


if __name__=='__main__':main()
