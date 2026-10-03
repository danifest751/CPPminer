"""Exercise the real miner against loopback pools, without a wallet or GPU.

Run: python tests/cp_pool_session_integration.py path/to/cppminer
Handshake timeouts take 30 seconds, share timeouts 60; cases run concurrently.
"""

import base64
import concurrent.futures
import gzip
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time


class PoolProbe:
    def __init__(self, binary, extra_args=()):
        self.binary_dir = tempfile.TemporaryDirectory(prefix="cppminer-pool-test-")
        test_binary = pathlib.Path(self.binary_dir.name) / binary.name
        shutil.copy2(binary, test_binary)
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(2)
        self.listener.settimeout(5)
        self.log = tempfile.TemporaryFile(mode="w+")
        self.process = subprocess.Popen([
            str(test_binary), "--backend", "cpu", "--m", "1", "--n", "1",
            "--threads", "2", "--no-fee", "--max-nonce", "1",
            "--wallet", "loopback-test",
            "--pool", f"stratum+tcp://127.0.0.1:{self.listener.getsockname()[1]}",
        ] + list(extra_args), stdout=self.log, stderr=self.log)
        self.connection = None

    def __enter__(self):
        try:
            self.connection, _ = self.listener.accept()
            self.connection.settimeout(5)
            data = b""
            while b"\n" not in data:
                chunk = self.connection.recv(4096)
                if not chunk:
                    raise AssertionError("miner disconnected before authorize")
                data += chunk
            self.auth_id = json.loads(data.split(b"\n", 1)[0])["id"]
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def send(self, message):
        self.connection.sendall((json.dumps(message) + "\n").encode())

    def output(self):
        self.log.seek(0)
        return self.log.read()

    def wait_reconnect(self, timeout, noise=None):
        self.listener.settimeout(0.1)
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if noise:
                try:
                    self.send(noise)
                except OSError:
                    noise = None
            try:
                retry, _ = self.listener.accept()
                retry.close()
                return
            except socket.timeout:
                if self.process.poll() is not None:
                    raise AssertionError(self.output())
        raise AssertionError("no reconnect within deadline:\n" + self.output())

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.connection:
            self.connection.close()
        self.listener.close()
        self.log.close()
        self.binary_dir.cleanup()


def rejected_authorize(binary):
    with PoolProbe(binary) as pool:
        pool.send({"id": pool.auth_id, "result": False,
                   "error": {"code": -1, "message": "test rejection"}})
        pool.wait_reconnect(5)
        assert "pool rejected authorization" in pool.output()
    return "rejected authorize reconnects"


def quantus_job_buffer(binary):
    # Exercise both writes into main's current-job buffer, including queued jobs.
    with PoolProbe(binary, ("--algo", "quantus")) as pool:
        job = {"job_id": "q" * 127, "mining_hash": "00" * 32,
               "target": "00" * 64, "extranonce": "", "difficulty": 1}
        pool.send({"id": pool.auth_id, "result": {
            "id": "loopback-session", "status": "OK", "job": job}})
        for job_id in (job["job_id"], "next-quantus-job"):
            if job_id != job["job_id"]:
                job = dict(job, job_id=job_id)
                pool.send({"method": "job", "params": job})
            until = time.monotonic() + 10
            while time.monotonic() < until:
                output = pool.output()
                assert pool.process.poll() is None, output
                if f"[qpow] mine job={job_id} " in output:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(pool.output())
    return "Quantus long job id and queued job fit the current-job buffer"


def missing_authorize(binary):
    with PoolProbe(binary) as pool:
        pool.wait_reconnect(35, {"id": pool.auth_id + 999, "result": True})
        assert "no authorize response for 30 s" in pool.output()
    return "wrong-id traffic cannot extend authorize deadline"


def missing_first_job(binary):
    with PoolProbe(binary) as pool:
        pool.send({"id": pool.auth_id, "result": True, "type": "v2"})
        pool.wait_reconnect(35, {"method": "mining.notify",
                                "params": {"job_id": "bad", "header": "zz"}})
        assert "no valid first job for 30 s" in pool.output()
    return "invalid notifications cannot extend first-job deadline"


def job_before_authorize(binary):
    with PoolProbe(binary) as pool:
        pool.send({"method": "mining.notify", "params": {
            "job_id": "early-job", "header": "00" * 76,
            "target": "00" * 32, "cert_version": 3}})
        time.sleep(0.3)
        assert "attempt timing:" not in pool.output(), "mined before authorize ACK"
        pool.send({"id": pool.auth_id, "result": {"type": "v2"}, "error": None})
        until = time.monotonic() + 5
        while time.monotonic() < until:
            output = pool.output()
            if "stopped after max_nonce=1" in output:
                assert "pool answered type v2" in output
                return "early job is preserved and mined after accepted authorize"
            time.sleep(0.05)
        raise AssertionError(pool.output())


def changed_job_identity(binary):
    with PoolProbe(binary) as pool:
        pool.send({"id": pool.auth_id, "result": True})
        job = {"job_id": "same-id", "header": "00" * 76,
               "target": "00" * 32, "cert_version": 3}
        for expected in range(1, 5):
            pool.send({"method": "mining.notify", "params": job})
            until = time.monotonic() + 5
            while time.monotonic() < until:
                if pool.output().count("stopped after max_nonce=1") == expected:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(pool.output())
            if expected == 1:
                # Equivalent duplicate must not restart work (also catches sizeof(pointer)).
                pool.send({"method": "mining.notify", "params": job})
                until = time.monotonic() + 2
                while "duplicate notify ignored" not in pool.output() and time.monotonic() < until:
                    time.sleep(0.05)
                assert "duplicate notify ignored" in pool.output(), pool.output()
                job = dict(job, header="00" * 75 + "01")
            elif expected == 2:
                job = dict(job, target="00" * 31 + "01")
            elif expected == 3:
                job = dict(job, cert_version=2)
    return "duplicate jobs are ignored; full header, target and certificate changes restart work"


def receive_share(pool):
    pool.send({"id": pool.auth_id, "result": True, "type": "v2"})
    pool.send({"method": "mining.notify", "params": {
        "job_id": "submit-test", "header": "00" * 76,
        "target": "0000004000000000" + "00" * 24, "cert_version": 3}})
    data = b""
    while b"\n" not in data:
        chunk = pool.connection.recv(65536)
        assert chunk, pool.output()
        data += chunk
    share = json.loads(data.split(b"\n", 1)[0])
    assert share["method"] == "mining.submit", share
    compressed = base64.b64decode(share["params"]["plain_proof"])
    proof = gzip.decompress(compressed)
    assert len(proof) > 1000 and len(compressed) < len(proof)
    return share["id"]


def accepted_share(binary):
    with PoolProbe(binary) as pool:
        submit_id = receive_share(pool)
        pool.send({"id": submit_id, "result": True, "error": None})
        pool.listener.settimeout(62)
        try:
            retry, _ = pool.listener.accept()
            retry.close()
            raise AssertionError("acknowledged share caused reconnect:\n" + pool.output())
        except socket.timeout:
            output = pool.output()
            assert pool.process.poll() is None, output
            assert "[pool] submit response:" in output, output
            assert "no pool reply to a submit" not in output, output
    return "gzip share ACK stays connected past the 60-second deadline"


def unacknowledged_share(binary):
    with PoolProbe(binary) as pool:
        submit_id = receive_share(pool)
        pool.wait_reconnect(65, {"id": submit_id + 999, "result": True})
        assert "no pool reply to a submit for 60 s" in pool.output()
    return "unrelated reply traffic cannot acknowledge a share or prevent reconnect"


if __name__ == "__main__":
    binary = pathlib.Path(sys.argv[1]).resolve()
    # Each miner has its own executable directory and proof files.
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(case, binary) for case in (
            changed_job_identity, quantus_job_buffer, rejected_authorize, missing_authorize, missing_first_job, job_before_authorize,
            accepted_share, unacknowledged_share)]
        for future in futures:
            print("PASS:", future.result(), flush=True)
