"""Intel hardware test; no external pool, wallet, rental or CI execution."""
from pathlib import Path
import sys,os,time,re,json,subprocess,statistics
import argparse
parser=argparse.ArgumentParser(description="Validate Intel DPAS and measure complete zero-target scans on a loopback pool.")
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
options=parser.parse_args()
base=options.output.resolve();base.mkdir(parents=True,exist_ok=True)
binary=options.binary.resolve()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from cp_pool_session_integration import PoolProbe,job
result={'cases':[],'dimensions':32768,'attempts_per_case':5}
for label,policy,extra in [('khr','khr',{}),('dpas-4x1','dpas',{}),('dpas-2x1','dpas',{'CP_OCL_DPAS_TM':'2','CP_OCL_DPAS_TN':'1'}),('dpas-1x1','dpas',{'CP_OCL_DPAS_TM':'1','CP_OCL_DPAS_TN':'1'})]:
    env=dict(os.environ,OMP_NUM_THREADS='2',**extra)
    # Each tile layout candidate must pass the complete prefix-GEMM oracle and proof verifier.
    for name,args in [('align',['--align-test']),('mock',['--mock','--mock-diff','40','--verify','--max-nonce','8'])]:
        with (base/(label+'-'+name+'.log')).open('w') as log:
            p=subprocess.run([str(binary),'--backend','opencl','--ocl-dot',policy,'--m','8','--n','8',*args],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        assert p.returncode==0,label+' '+name
    with PoolProbe(binary,('--backend','opencl','--ocl-dot',policy,'--m','32','--n','32','--max-nonce','5'),env=env) as pool:
        pool.send({'id':pool.auth_id,'result':True});pool.send(job('a380-'+label))
        deadline=time.monotonic()+180
        while '[plain] job stopped (max_nonce)' not in pool.output():
            assert pool.process.poll() is None and time.monotonic()<deadline,label+' timeout/exited'
            time.sleep(.1)
        text=pool.output()
    (base/(label+'-full-scan.log')).write_text(text)
    times=[(float(a),float(b)) for a,b in re.findall(r'\[ocl\] attempt timing: prep=([0-9.]+)s scan=([0-9.]+)s',text)]
    assert len(times)==5 and 'plain_proof SHARE' not in text
    assert not any(marker in text for marker in ('MISMATCH','verify FAIL','CL_OUT_OF_RESOURCES'))
    macs=32768**2*4096
    measured=times[1:]
    result['cases'].append({'label':label,'policy':policy,'environment':extra,'timings_seconds':times,'full_attempt_tmac_s':macs/statistics.mean(a+b for a,b in measured)/1e12,'scan_tmac_s':macs/statistics.mean(b for a,b in measured)/1e12,'spill_bytes_per_work_item':re.findall(r'register spill: (\d+) B/WI',text),'alignment_and_proof_passed':True})
    (base/'benchmark-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS '+label+' '+str(round(result['cases'][-1]['full_attempt_tmac_s'],3))+' TMAC/s',flush=True)
result['completed']=True
(base/'benchmark-result.json').write_text(json.dumps(result,indent=2)+'\n')
