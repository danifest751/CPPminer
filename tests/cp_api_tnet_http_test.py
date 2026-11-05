"""Actual CPU stats listener contract used by Worknet; never starts mining."""
import json
import socket
import subprocess
import sys
import time
import urllib.request
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
p = subprocess.Popen([sys.argv[1], str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    deadline = time.monotonic() + 3
    while True:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/summary', timeout=1) as r:
                data = json.load(r)
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.02)
    assert data['miner'] == 'cppminer' and data['algo'] == 'tnet' and data['backend'] == 'cuda'
    assert data['worker'] == 'resource-fixture' and data['pool'] == 'local-test'
    assert data['devices'][0]['uuid'] == 'GPU-11111111-2222-3333-4444-555555555555'
    assert data['devices'][0]['pci'] == '0000:01:00.0'
    assert data['hashrate']['windows'] == [10, 60, 900]
    time.sleep(.65)
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/summary', timeout=1) as r:
        data = json.load(r)
    assert data['hashrate']['total'][0] > 0
    print('TNet summary device identity, worker, rate windows and work counters passed; CPU fixture only')
finally:
    p.terminate()
    try:
        p.wait(timeout=2)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
