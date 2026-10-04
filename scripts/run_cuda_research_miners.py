"""Guarded R1 factorial: verified mock proofs and fixed-work complete attempts.

Server-local restore snapshots and wallet-bearing production logs remain private.
Requires the isolated /w/experiment-research-suite and installed v0.5-fork.4.
"""
import argparse,hashlib,json,os,pathlib,re,signal,statistics,subprocess,sys,time
ROOT=pathlib.Path('/w/experiment-research-suite');OUT=ROOT/'results'
RELEASE=pathlib.Path('/w/releases/v0.5-fork.4');REFERENCE=RELEASE/'cppminer-linux-x64-cuda/cppminer'
BUILD=ROOT/'build/research-miners';VARIANTS=('baseline','cg','predicated','combined');CONTROL=BUILD/'baseline/cppminer';BINARY=BUILD/'combined/cppminer'
child=None;stopped=False;result={}

def alive(pid):
    try:return (pathlib.Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def stop():
    global child
    if child and child.poll() is None:
        child.terminate()
        try:child.wait(timeout=12)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    child=None

def gpu():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used','--format=csv,noheader,nounits'],text=True)
    return [float(v.strip()) for v in s.strip().split(',')]

def save():
    tmp=OUT/'miner.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(OUT/'miner.json')

def run(name,args,environment,timeout=240):
    global child
    path=OUT/(name+'.log')
    with path.open('w') as file:
        child=subprocess.Popen(args,env=environment,stdout=file,stderr=subprocess.STDOUT,cwd=ROOT)
        try:code=child.wait(timeout=timeout)
        finally:stop()
    content=path.read_text(errors='replace')
    assert code==0,name+' failed: '+content[-1800:]
    print('PASS '+name,flush=True)
    return content

def fixed(label,binary,mode,count,warmup,environment):
    samples=[]
    environment=dict(environment,CP_CUDA_A_MODE=mode,CP_CUDA_A_CHECK='0',CP_CUDA_TB='256x128',CP_CUDA_OVERLAP='1',CP_CUDA_A_UPDATES='4096')
    start=time.monotonic()
    with PoolProbe(binary,('--backend','cuda','--no-fee','--m','128','--n','128','--max-nonce',str(count)),env=environment) as pool:
        pool.send({'id':pool.auth_id,'result':True});pool.send(job('jackpot-fixed-work'))
        until=time.monotonic()+180;next_report=time.monotonic()+30
        while '[plain] job stopped (max_nonce)' not in pool.output():
            assert pool.process.poll() is None and time.monotonic()<until,'fixed-work scan stopped early/timed out'
            samples.append(dict(elapsed_seconds=time.monotonic()-start,values=gpu()))
            if time.monotonic()>next_report:
                print('FIXED progress '+label+': '+str(pool.output().count('[gpu] attempt timing:'))+' attempts',flush=True);next_report+=30
            time.sleep(.5)
        content=pool.output()
    (OUT/(label+'.log')).write_text(content)
    times=[(float(a),float(b),float(c)) for a,b,c in re.findall(r'\[gpu\] attempt timing: prep=([0-9.]+)s scan=([0-9.]+)s ([0-9.]+) TMAC/s',content)]
    assert len(times)==count and 'plain_proof SHARE' not in content,label+' wrong attempt count or early share'
    assert not any(s in content for s in ('MISMATCH','CUDA error','verify FAILED'))
    good=times[warmup:]
    row=dict(label=label,variant=binary.parent.name,signal_a_mode=mode,attempts=count,warmup_discarded=warmup,effective_tmac=len(good)*131072*131072*4096/sum(a+b for a,b,c in good)/1e12,prep_ms=1000*statistics.mean(a for a,b,c in good),scan_tmac=statistics.mean(c for a,b,c in good),timings=times,gpu_samples=samples)
    print('FIXED '+json.dumps({k:v for k,v in row.items() if k not in ('timings','gpu_samples')}),flush=True)
    return row

def interrupt(signum,frame):raise KeyboardInterrupt('bounded miner experiment interrupted')

def main():
    global OUT,VARIANTS,stopped,result,PoolProbe,job
    options=argparse.ArgumentParser(description=__doc__)
    options.add_argument('--confirmation',action='store_true',help='Longer baseline/combined ABBA confirmation')
    options.add_argument('--output',type=pathlib.Path)
    options=options.parse_args()
    OUT=options.output or ROOT/'results';OUT.mkdir(parents=True,exist_ok=True)
    VARIANTS=('baseline','combined') if options.confirmation else ('baseline','cg','predicated','combined')
    sys.path.insert(0,str(ROOT/'tests'))
    from cp_pool_session_integration import PoolProbe,job
    manifest=json.loads((BUILD/'manifest.json').read_text());assert manifest.get('completed') and set(VARIANTS)<=set(manifest['variants'])
    for name in VARIANTS:assert hashlib.sha256((BUILD/name/'cppminer').read_bytes()).hexdigest()==manifest['variants'][name]['binary_sha256']
    pid=int((RELEASE/'miner.pid').read_text());proc=pathlib.Path('/proc')/str(pid)
    assert alive(pid) and pathlib.Path(os.readlink(proc/'exe')).resolve()==REFERENCE.resolve()
    args=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0');assert args[args.index('--worker')+1]=='cmp50hx'
    env=dict(v.split('=',1) for v in (proc/'environ').read_bytes().decode().strip('\0').split('\0'))
    restore=dict(argv=args,environment=env,cwd=os.readlink(proc/'cwd'),stdout=os.readlink(proc/'fd/1'),stderr=os.readlink(proc/'fd/2'))
    fd=os.open(OUT/'miner-restore.json',os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as file:json.dump(restore,file)
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()=='3e5a589350558afc34c5d15297612ef16aa3273d0999c3b5bc3d1e9737fd300e'

    result=dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),confirmation=options.confirmation,reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),experiment=manifest,fixed_blocks=[],gpu_columns=['temperature_c','sm_clock_mhz','memory_clock_mhz','power_w','memory_mib'])
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupt)
    signal.alarm(2400)
    try:
        os.kill(pid,signal.SIGTERM)
        until=time.monotonic()+15
        while alive(pid):
            assert time.monotonic()<until;time.sleep(.1)
        stopped=True
        print('STOPPED release; guarded whole-miner jackpot experiment started',flush=True)
        test_env=dict(env,LD_LIBRARY_PATH=str(REFERENCE.parent)+':'+env.get('LD_LIBRARY_PATH',''),CP_CUDA_A_UPDATES='4096',CP_CUDA_OVERLAP='1')
        run('ctest-jackpot',['ctest','--test-dir',str(ROOT/'build/miner'),'--output-on-failure'],test_env)
        content=run('alignment-jackpot',[str(BINARY),'--backend','cuda','--align-test-prod','--m','8','--n','8'],dict(test_env,CP_CUDA_A_MODE='dense'),timeout=360)
        assert '[align-test-prod] GPU pipeline OK' in content
        result['alignment_passed']=True
        for variant in VARIANTS:
            for tile in ('256x128','128x128'):
                for mode in ('dense','incremental'):
                    content=run('mock-'+variant+'-'+tile+'-'+mode,[str(BUILD/variant/'cppminer'),'--backend','cuda','--no-fee','--mock','--mock-diff','40','--m','128','--n','128','--verify','--max-nonce','4'],dict(test_env,CP_CUDA_TB=tile,CP_CUDA_A_MODE=mode,CP_CUDA_A_CHECK='1' if mode=='incremental' else '0'))
                    assert '[mock] PASS' in content
                    result.setdefault('verified_mock_cases',[]).append(dict(variant=variant,tile=tile,mode=mode,passed=True));save()
        result['warmup']=fixed('warmup-miner',CONTROL,'dense',40,4,test_env);save()
        for mode in ('dense','incremental'):
            order=list(VARIANTS)+list(reversed(VARIANTS))
            for block,variant in enumerate(order,1):
                row=fixed(mode+'-'+str(block)+'-'+variant,BUILD/variant/'cppminer',mode,65 if options.confirmation else 36,5 if options.confirmation else 4,test_env)
                row['block']=block;result['fixed_blocks'].append(row);save()
        result['completed']=True
    except BaseException as error:
        result['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        signal.alarm(0);stop()
        if stopped or not alive(pid):
            with open(restore['stdout'],'ab',buffering=0) as out,open(restore['stderr'],'ab',buffering=0) as errors:
                production=subprocess.Popen(restore['argv'],cwd=restore['cwd'],env=restore['environment'],stdout=out,stderr=errors,start_new_session=True)
            time.sleep(3)
            assert production.poll() is None,'release restoration failed'
            current=pathlib.Path('/proc')/str(production.pid)
            assert pathlib.Path(os.readlink(current/'exe')).resolve()==REFERENCE.resolve()
            assert (current/'cmdline').read_bytes().decode().strip('\0').split('\0')==restore['argv']
            assert dict(v.split('=',1) for v in (current/'environ').read_bytes().decode().strip('\0').split('\0'))==restore['environment']
            assert os.readlink(current/'cwd')==restore['cwd']
            (RELEASE/'miner.pid').write_text(str(production.pid)+'\n')
            deployment=RELEASE/'deployment-result.json';report=json.loads(deployment.read_text());report.update(pid=production.pid,last_restart_reason='restored after R1 whole-miner cache/jackpot experiment',last_restart_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));deployment.write_text(json.dumps(report,indent=2)+'\n')
            result['restored_pid']=production.pid
            print('RESTORED original v0.5-fork.4 wallet/pool/worker/environment',flush=True)
        save()


if __name__=='__main__':main()
