"""Bounded sm75 experiment runner with exact installed-miner restoration.

Only sanitized numerical results leave the server. restore.json contains private
process state, is mode 0600, and must remain server-local. This runner does not
change GPU power/clock settings or install an experimental production binary.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--release',type=Path,default=Path('/w/releases/v0.5-fork.4'))
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--repeats',type=int,default=60)
    parser.add_argument('--tile',choices=('256x128','128x128'),default='256x128')
    args=parser.parse_args()
    assert 60 <= args.seconds <= 2400
    assert 3 <= args.repeats <= 1000
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    build=args.build;release=args.release
    binary=release/'cppminer-linux-x64-cuda/cppminer'
    reference_sha='3e5a589350558afc34c5d15297612ef16aa3273d0999c3b5bc3d1e9737fd300e'
    child=None;stopped=False;done=threading.Event();result={}
    def alive(pid):
        try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
        except FileNotFoundError:return False
    def save():
        temporary=out/'result.json.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out/'result.json')
    def stop_child():
        nonlocal child
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        child=None
    def interrupt(signum,frame):raise KeyboardInterrupt('bounded research experiment interrupted')
    def sampler():
        while not done.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
                result['gpu_samples'].append({'seconds':time.monotonic()-started,'values':[float(x.strip()) for x in raw.strip().split(',')]})
            except Exception:pass
            done.wait(1)
    def run(name,variant,kind,arguments,timeout,**labels):
        nonlocal child
        start=time.monotonic()-started
        with (out/(name+'.log')).open('w') as log:
            environment=dict(os.environ,OMP_NUM_THREADS='4')
            cubin=manifest['variants'][variant].get('cubins',{}).get(kind)
            if cubin:
                environment.update(CP_RESEARCH_CUBIN=cubin['path'],
                    CP_RESEARCH_FUNCTION_128=cubin['functions']['128'],CP_RESEARCH_FUNCTION_256=cubin['functions']['256'])
            child=subprocess.Popen([str(build/variant/kind),*arguments],stdout=log,stderr=subprocess.STDOUT,env=environment)
            try:code=child.wait(timeout=timeout)
            finally:stop_child()
        records=[json.loads(line) for line in (out/(name+'.log')).read_text().splitlines() if line.startswith('{')]
        passed=code==0 and bool(records) and records[-1].get('type')=='complete' and records[-1].get('passed')
        row={'name':name,'variant':variant,'binary':kind,'args':arguments,'started_seconds':start,'ended_seconds':time.monotonic()-started,'records':records,'passed':bool(passed),**labels}
        result['runs'].append(row);save()
        measurements=[r for r in records if r['type']=='measurement']
        print(('PASS ' if passed else 'FAIL ')+name+(' median='+str(statistics.median(measurements[0]['gpu_ms']))+' ms' if measurements else ''),flush=True)
        return passed
    manifest=json.loads((build/'manifest.json').read_text());assert manifest.get('completed')
    variants=[name for name,row in manifest['variants'].items() if row.get('compiled',False)]
    assert 'baseline' in variants
    for name in variants:
        for kind in ('full','verify'):
            assert hashlib.sha256((build/name/kind).read_bytes()).hexdigest()==manifest['variants'][name]['binaries'][kind]['sha256']
            cubin=manifest['variants'][name].get('cubins',{}).get(kind)
            if cubin:assert hashlib.sha256(Path(cubin['path']).read_bytes()).hexdigest()==cubin['sha256']
    pid=int((release/'miner.pid').read_text());proc=Path('/proc')/str(pid)
    assert alive(pid) and Path(os.readlink(proc/'exe')).resolve()==binary.resolve()
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==reference_sha
    argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
    assert argv[argv.index('--worker')+1]=='cmp50hx'
    environment=dict(v.split('=',1) for v in (proc/'environ').read_bytes().decode().strip('\0').split('\0'))
    snapshot={'argv':argv,'environment':environment,'cwd':os.readlink(proc/'cwd'),'stdout':os.readlink(proc/'fd/1'),'stderr':os.readlink(proc/'fd/2')}
    assert all(Path(snapshot[key]).is_file() or snapshot[key]=='/dev/null' for key in ('stdout','stderr'))
    with os.fdopen(os.open(out/'restore.json',os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600),'w') as file:json.dump(snapshot,file)
    result={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'reference_sha256':reference_sha,
            'manifest':manifest,'repeats_per_block':args.repeats,'runs':[],'gpu_samples':[],'gpu_columns':['temperature_c','sm_clock_mhz','memory_clock_mhz','power_w','memory_mib','utilization_percent']}
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,interrupt)
    signal.alarm(args.seconds);started=time.monotonic();thread=threading.Thread(target=sampler,daemon=True);thread.start()
    try:
        os.kill(pid,signal.SIGTERM);deadline=time.monotonic()+15
        while alive(pid):
            assert time.monotonic()<deadline,'original miner did not stop';time.sleep(.1)
        stopped=True;print('STOPPED original release; protected snapshot saved',flush=True);save()
        valid=[]
        for variant in variants:
            if run('verify-'+variant,variant,'verify',[],180,phase='correctness'):valid.append(variant)
        assert 'baseline' in valid, 'baseline oracle failed'
        timing_args=[str(args.repeats)]+(['large'] if args.tile=='256x128' else [])
        assert run('conditioning','baseline','full',['300']+timing_args[1:],120,phase='conditioning')
        # Four balanced orders; observations are block medians, not thousands
        # of correlated kernel launches treated as independent trials.
        rotated=valid[len(valid)//2:]+valid[:len(valid)//2]
        orders=[valid,list(reversed(valid)),rotated,list(reversed(rotated))]
        result['orders']=orders;save()
        for round_id,order in enumerate(orders):
            for position,variant in enumerate(order):
                assert run('r'+str(round_id)+'-p'+str(position)+'-'+variant,variant,'full',timing_args,180,
                           phase='measurement',round=round_id,position=position,tile=args.tile)
        result['completed']=True
    except BaseException as error:
        result.update(completed=False,error_type=type(error).__name__,error=str(error));print('FAILED '+str(error),flush=True)
    finally:
        signal.alarm(0);stop_child();done.set();thread.join(timeout=6)
        if stopped or not alive(pid):
            with open(snapshot['stdout'],'ab',buffering=0) as stdout,open(snapshot['stderr'],'ab',buffering=0) as stderr:
                original=subprocess.Popen(snapshot['argv'],cwd=snapshot['cwd'],env=snapshot['environment'],stdout=stdout,stderr=stderr,start_new_session=True)
            time.sleep(3);assert original.poll() is None,'original restoration failed'
            current=Path('/proc')/str(original.pid)
            assert Path(os.readlink(current/'exe')).resolve()==binary.resolve()
            assert (current/'cmdline').read_bytes().decode().strip('\0').split('\0')==snapshot['argv']
            assert dict(v.split('=',1) for v in (current/'environ').read_bytes().decode().strip('\0').split('\0'))==snapshot['environment']
            assert os.readlink(current/'cwd')==snapshot['cwd']
            (release/'miner.pid').write_text(str(original.pid)+'\n')
            deployment=release/'deployment-result.json'
            if deployment.exists():
                report=json.loads(deployment.read_text());report.update(pid=original.pid,last_restart_reason='restored after CUDA '+manifest['group']+' research experiment',last_restart_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));deployment.write_text(json.dumps(report,indent=2)+'\n')
            result['restoration']={'pid':original.pid,'arguments_match':True,'environment_match':True,'cwd_match':True,'original_executable':True}
            print('RESTORED original v0.5-fork.4 exactly',flush=True)
        result['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':main()
