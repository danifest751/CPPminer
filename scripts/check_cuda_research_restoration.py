"""Audit the restored release independently; never print private process state.

The snapshot stays on the test server. Only boolean checks, binary SHA and a
fresh completed-attempt count are written to the optional public result file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--release',type=Path,default=Path('/w/releases/v0.5-fork.4'))
    parser.add_argument('--sha256',default='3e5a589350558afc34c5d15297612ef16aa3273d0999c3b5bc3d1e9737fd300e')
    parser.add_argument('--seconds',type=int,default=12)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();assert 10<=args.seconds<=60
    snapshot=json.loads(args.snapshot.read_text())
    pid=int((args.release/'miner.pid').read_text());proc=Path('/proc')/str(pid)
    binary=args.release/'cppminer-linux-x64-cuda/cppminer'
    assert Path(os.readlink(proc/'exe')).resolve()==binary.resolve(),'wrong executable'
    digest=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert digest==args.sha256,'release binary changed'
    argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
    environment=dict(v.split('=',1) for v in (proc/'environ').read_bytes().decode().strip('\0').split('\0'))
    assert argv==snapshot['argv'],'arguments differ'
    assert environment==snapshot['environment'],'environment differs'
    assert os.readlink(proc/'cwd')==snapshot['cwd'],'working directory differs'
    assert all(os.readlink(proc/('fd/'+str(fd)))==snapshot[key] for fd,key in ((1,'stdout'),(2,'stderr'))),'log destinations differ'
    worker=argv[argv.index('--worker')+1]
    workers=[]
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:
            command=(item/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if '--worker' in command and command[command.index('--worker')+1].startswith(worker):
                workers.append(int(item.name))
        except (OSError,UnicodeError,IndexError):pass
    assert sorted(workers)==[pid],'additional mining worker found'
    log=Path(snapshot['stdout'])
    before=log.read_text(errors='replace').count('[gpu] attempt timing:')
    time.sleep(args.seconds)
    after=log.read_text(errors='replace').count('[gpu] attempt timing:')
    assert after>before and (proc/'stat').read_text().split()[2]!='Z','mining has not resumed'
    result={'pid':pid,'original_binary':True,'binary_sha256':digest,'argv_match':True,
            'environment_match':True,'cwd_match':True,'log_destinations_match':True,
            'unique_worker':True,'observation_seconds':args.seconds,'fresh_attempts':after-before}
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
