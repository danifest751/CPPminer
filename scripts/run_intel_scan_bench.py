"""Intel GPU scan benchmark on a loopback pool; no external pool, wallet or CI.

Each case runs a mock share through the real proof verifier, then times complete
zero-target scans (no early hit) and reports full-attempt and scan TMAC/s.
"""
from pathlib import Path
import argparse, json, os, re, statistics, subprocess, sys, time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--size', type=int, default=32, help='--m/--n in units of 1024')
parser.add_argument('--attempts', type=int, default=5, help='first attempt is discarded')
parser.add_argument('--skip-mock', action='store_true',
                    help='diagnostic runs only (e.g. CASE5_XOR_NOP): no proof check')
parser.add_argument('--case', action='append', default=[],
                    help='label=backend args, e.g. "onednn=--backend onednn" '
                         '(env vars as KEY=VAL tokens before the args)')
options = parser.parse_args()
base = options.output.resolve()
base.mkdir(parents=True, exist_ok=True)
binary = options.binary.resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from cp_pool_session_integration import PoolProbe, job

cases = options.case or ['onednn=--backend onednn', 'dpas=--backend opencl --ocl-dot dpas']
dim = options.size * 1024
result = {'dimensions': dim, 'attempts_per_case': options.attempts, 'cases': []}
timing = re.compile(r'\[(?:ocl|onednn)\] attempt timing: prep=([0-9.]+)s scan=([0-9.]+)s')

def run_case(spec):
    label, _, rest = spec.partition('=')
    tokens = rest.split()
    extra = dict(t.split('=', 1) for t in tokens if '=' in t and not t.startswith('-'))
    args = [t for t in tokens if not ('=' in t and not t.startswith('-'))]
    env = dict(os.environ, OMP_NUM_THREADS='2', **extra)

    if not options.skip_mock:
        with (base / (label + '-mock.log')).open('w') as log:
            p = subprocess.run([str(binary), *args, '--m', '8', '--n', '8', '--mock', '--mock-diff', '40',
                                '--verify', '--max-nonce', '8', '--no-fee'],
                               env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        mock = (base / (label + '-mock.log')).read_text()
        assert p.returncode == 0 and '[mock] PASS' in mock, label + ' mock proof failed'

    with PoolProbe(binary, (*args, '--m', str(options.size), '--n', str(options.size),
                            '--max-nonce', str(options.attempts)), env=env) as pool:
        pool.send({'id': pool.auth_id, 'result': True})
        pool.send(job('intel-' + label))
        deadline = time.monotonic() + 1800
        while '[plain] job stopped (max_nonce)' not in pool.output():
            assert pool.process.poll() is None and time.monotonic() < deadline, label + ' timeout/exited'
            time.sleep(.2)
        text = pool.output()
    (base / (label + '-full-scan.log')).write_text(text)
    times = [(float(a), float(b)) for a, b in timing.findall(text)]
    assert len(times) == options.attempts and 'plain_proof SHARE' not in text, label + ' bad scan log'
    assert not any(m in text for m in ('MISMATCH', 'verify FAIL', 'CL_OUT_OF_RESOURCES')), label
    macs = dim * dim * 4096
    measured = times[1:]
    case = {'label': label, 'args': args, 'environment': extra, 'timings_seconds': times,
            'full_attempt_tmac_s': macs / statistics.mean(a + b for a, b in measured) / 1e12,
            'scan_tmac_s': macs / statistics.mean(b for a, b in measured) / 1e12,
            'mock_proof_passed': not options.skip_mock}
    result['cases'].append(case)
    (base / 'benchmark-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS %s full=%.3f scan=%.3f TMAC/s' % (label, case['full_attempt_tmac_s'], case['scan_tmac_s']),
          flush=True)


failed = []
for spec in cases:
    try:
        run_case(spec)
    except (AssertionError, subprocess.TimeoutExpired) as e:
        failed.append({'case': spec, 'error': str(e)})
        result['failed'] = failed
        print('FAIL %s' % e, flush=True)
result['completed'] = True
(base / 'benchmark-result.json').write_text(json.dumps(result, indent=2) + '\n')
