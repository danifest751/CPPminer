"""Loopback experiment, not a multi-pool implementation in CPPminer.

Measure the current miner's reconnect and a tiny two-session protocol model.
Run: python tests/cp_pool_failover_experiment.py path/to/cppminer --output result.json
All credentials and work in this script are synthetic; no real pool is contacted.
"""
import argparse
import json
import pathlib
import socket
import statistics
import threading
import time

from cp_pool_session_integration import PoolProbe, job, wait_for


def send(connection, message):
    connection.sendall((json.dumps(message) + '\n').encode())


def recv_line(connection):
    data = bytearray()
    while not data.endswith(b'\n'):
        chunk = connection.recv(1)
        if not chunk:
            raise EOFError('closed session')
        data += chunk
        if len(data) > 65536:
            raise ValueError('oversized message')
    return json.loads(data)


class MockEndpoint:
    """One connection, delayed authorization, then two jobs with the same ID."""
    def __init__(self, delay_ms, marker):
        self.delay = delay_ms / 1000
        self.marker = marker
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.listener.settimeout(5)
        self.address = self.listener.getsockname()
        self.connection = None
        self.done = threading.Event()
        self.errors = []
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        try:
            self.connection, _ = self.listener.accept()
            self.connection.settimeout(5)
            auth = recv_line(self.connection)
            assert auth['method'] == 'mining.authorize'
            time.sleep(self.delay)
            send(self.connection, {'id': auth['id'], 'result': True})
            for revision in (1, 2):
                work = job('same-id-on-both-pools')
                work['params']['header'] = (f'{self.marker + revision:02x}' * 76)
                send(self.connection, work)
            self.done.wait(5)
        except BaseException as error:
            self.errors.append(repr(error))

    def close(self):
        self.done.set()
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        self.listener.close()
        self.thread.join(timeout=6)
        assert not self.thread.is_alive(), 'mock pool thread leaked'
        assert not self.errors, self.errors


class Session:
    def __init__(self, endpoint, generation):
        self.socket = socket.create_connection(endpoint.address, timeout=5)
        self.socket.settimeout(5)
        self.owner = (endpoint.address, generation)
        self.latest = None
        send(self.socket, {'id': 1, 'method': 'mining.authorize',
                           'params': {'wallet': 'loopback-test'}})
        response = recv_line(self.socket)
        assert response['id'] == 1 and response['result'] is True
        self.authorized = True
        for _ in range(2):
            notification = recv_line(self.socket)
            assert notification['method'] == 'mining.notify'
            self.latest = notification['params']
        self.work_owner = self.token_for(self.latest)

    def token_for(self, work):
        # Target/certificate changes with an unchanged ID/header also change work.
        return (self.owner, json.dumps(work, sort_keys=True, separators=(',', ':')))

    def can_route(self, proof_owner):
        return self.authorized and proof_owner == self.work_owner

    def close(self):
        self.socket.close()


def protocol_model(delay_ms, warm, blackhole=False):
    primary = MockEndpoint(0, 10)
    reserve = MockEndpoint(delay_ms, 20)
    a = b = None
    try:
        a = Session(primary, 1)
        if warm:
            b = Session(reserve, 1)
        started = time.perf_counter()
        if blackhole:
            # Explicit 200 ms policy in this model; production has no such idle timer.
            a.socket.settimeout(.2)
            try:
                a.socket.recv(1)
                raise AssertionError('expected silent primary')
            except socket.timeout:
                pass
        else:
            primary.connection.shutdown(socket.SHUT_RDWR)
            assert a.socket.recv(1) == b''
        detected = time.perf_counter()
        if b is None:
            b = Session(reserve, 1)
        promoted = time.perf_counter()
        assert a.latest['job_id'] == b.latest['job_id']
        assert a.latest['header'] != b.latest['header']
        assert b.latest['header'] == '16' * 76, 'latest reserve update was lost'
        assert not b.can_route(a.work_owner), 'primary proof crossed pool/session'
        assert b.can_route(b.work_owner)
        assert not b.can_route(b.token_for(dict(b.latest, header='15' * 76))), 'old reserve job was accepted'
        assert not b.can_route(b.token_for(dict(b.latest, target='ff' * 32))), 'changed target was accepted'
        assert not b.can_route(b.token_for(dict(b.latest, cert_version=2))), 'changed certificate was accepted'
        assert not b.can_route(((reserve.address, 0), b.work_owner[1])), 'old session was accepted'
        return dict(detection_ms=(detected-started)*1000,
                    after_detection_ms=(promoted-detected)*1000,
                    total_ms=(promoted-started)*1000)
    finally:
        if a:
            a.close()
        if b:
            b.close()
        primary.close()
        reserve.close()


def current_miner_eof(binary):
    with PoolProbe(binary) as pool:
        pool.send({'id': pool.auth_id, 'result': True})
        pool.send(job('before-eof'))
        wait_for(pool, '[job] notify id=before-eof ')
        started = time.perf_counter()
        pool.connection.shutdown(socket.SHUT_RDWR)
        pool.connection.close()
        pool.listener.settimeout(10)
        pool.connection, _ = pool.listener.accept()
        pool.connection.settimeout(5)
        auth = recv_line(pool.connection)
        pool.send({'id': auth['id'], 'result': True})
        pool.send(job('after-eof'))
        wait_for(pool, '[job] notify id=after-eof ')
        return (time.perf_counter() - started) * 1000


def current_miner_silence(binary, seconds=35):
    with PoolProbe(binary) as pool:
        pool.send({'id': pool.auth_id, 'result': True})
        pool.send(job('silent-after-valid-job'))
        wait_for(pool, '[job] notify id=silent-after-valid-job ')
        pool.listener.settimeout(seconds)
        try:
            retry, _ = pool.listener.accept()
            retry.close()
            reconnected = True
        except socket.timeout:
            reconnected = False
        assert pool.process.poll() is None, pool.output()
        assert 'plain_proof SHARE' not in pool.output(), 'zero target unexpectedly found a share'
        return dict(observed_seconds=seconds, reconnected=reconnected,
                    pending_submits=0, note='Valid job received; peer holds TCP open and sends no further data.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = dict(scope='Current miner baseline plus separate protocol model; no production failover added.',
                  current_miner_eof_ms=[current_miner_eof(args.binary.resolve()) for _ in range(3)], model=[])
    for delay in (0, 50, 200):
        for warm in (False, True):
            samples = [protocol_model(delay, warm) for _ in range(10)]
            report['model'].append(dict(authorize_delay_ms=delay, warm=warm, samples=samples,
                                        median_total_ms=statistics.median(x['total_ms'] for x in samples)))
    report['blackhole_model'] = [dict(warm=warm, **protocol_model(200, warm, True)) for warm in (False, True)]
    report['ownership_checks'] = 'PASS latest reserve job; colliding job IDs; primary proof; old reserve header/target/certificate; old session'
    print('PASS two-session model and current-miner EOF; measuring 35 seconds of silence', flush=True)
    report['current_miner_silence'] = current_miner_silence(args.binary.resolve())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'model'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
