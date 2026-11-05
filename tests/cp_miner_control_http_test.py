"""Exercise the real host HTTP implementation with a simulated idle callback."""
import http.client
import json
import os
import socket
import subprocess
import sys
import time

key = 'fixture-miner-owner-key-never-use-live'
with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
proc = subprocess.Popen([sys.argv[1], '--serve', str(port)], env={**os.environ, 'TEST_MINER_CONTROL_KEY': key})
headers = {'Authorization': 'Bearer '+key, 'X-ReQuant-Miner-Owner': '2'*64, 'X-ReQuant-Miner-Instance': '1'*64}

def request(action, lease=None, extra=None):
    h = {**headers, **(extra or {})}
    if lease:
        h['X-ReQuant-Miner-Lease'] = lease
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
    c.request('GET' if action == 'status' else 'POST', '/v1/control/'+action, headers=h)
    r = c.getresponse()
    code, data = r.status, json.loads(r.read())
    c.close()
    return code, data

try:
    for _ in range(100):
        try:
            code, data = request('status', extra={'X-ReQuant-Miner-Instance': ''})
            if data.get('state') == 'awaiting_operator':
                break
        except OSError:
            pass
        time.sleep(.01)
    assert code == 200 and data['state'] == 'awaiting_operator'
    assert len(data['instance']) == 64 and data['lease'] == data['instance']
    headers['X-ReQuant-Miner-Instance'] = data['instance']
    assert request('resume', '3'*64)[0] == 409
    assert request('resume', data['lease'])[0] == 200
    for _ in range(100):
        _, data = request('status')
        if data['state'] == 'mining':
            break
        time.sleep(.01)
    assert data['state'] == 'mining'
    assert request('status', extra={'Authorization': 'Bearer wrong'})[0] == 401
    assert request('status', extra={'X-ReQuant-Miner-Owner': '9'*64})[0] == 409
    assert request('status', extra={'X-ReQuant-Miner-Instance': '9'*64})[0] == 409
    assert request('quiesce', '3'*64)[0] == 200
    for _ in range(100):
        _, data = request('status')
        if data['state'] == 'quiescent':
            break
        time.sleep(.01)
    assert data['gpu_quiescent'] is True and data['lease'] == '3'*64
    assert request('resume', '4'*64)[0] == 409
    assert request('resume', '3'*64)[1]['state'] == 'mining'
    with socket.create_connection(('127.0.0.1', port), timeout=3) as c:
        raw = ('GET /v1/control/status HTTP/1.1\r\nAuthorization: Bearer '+key+
               '\r\nAuthorization: Bearer '+key+'\r\n\r\n').encode()
        c.sendall(raw)
        assert c.recv(4096).startswith(b'HTTP/1.1 400 ')
    assert request('status', extra={'Transfer-Encoding': 'chunked'})[0] == 400
    print('host HTTP authentication, ownership, lease and quiescence tests passed; no CUDA validation claimed')
finally:
    proc.terminate()
    proc.wait(timeout=5)

# Restart with the same owner/key must produce a new unarmed process instance.
old_instance = headers['X-ReQuant-Miner-Instance']
with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
proc = subprocess.Popen([sys.argv[1], '--serve', str(port)], env={**os.environ, 'TEST_MINER_CONTROL_KEY': key})
try:
    for _ in range(100):
        try:
            code, data = request('status', extra={'X-ReQuant-Miner-Instance': ''})
            if code == 200:
                break
        except OSError:
            pass
        time.sleep(.01)
    assert data['state'] == 'awaiting_operator' and data['instance'] != old_instance
    assert request('status')[0] == 409
    assert request('resume', old_instance)[0] == 409
    print('host restart generated a fresh instance and rejected stale control before operator reconciliation')
finally:
    proc.terminate()
    proc.wait(timeout=5)
