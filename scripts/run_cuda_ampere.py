"""Bounded sm86 sweep: exact oracles, balanced panel runs, real miner ABBA.

Run inside the unpacked bundle. Requires no compiler, wallet or remote pool.
Only child test processes are stopped; clocks/power limits are never changed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

from ampere_variants import TILES
# In source checkouts the loopback helper lives under tests; bundles carry it
# beside this runner, so they do not need the checkout or its dependencies.
if (Path(__file__).resolve().parents[1] / 'tests').is_dir():
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'tests'))
from cp_pool_session_integration import PoolProbe, job

ATTEMPT = re.compile(r'\[gpu\] attempt timing: prep=([0-9.]+)s scan=([0-9.]+)s ([0-9.]+) TMAC/s')
WORK = 131072 * 131072 * 4096 / 1e12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bundle(bundle, manifest):
    if manifest.get('schema') != 1 or not manifest.get('completed') or manifest.get('architecture') != 'sm_86':
        raise RuntimeError('bundle is incomplete or incompatible')
    files = dict(manifest['runtime_files'], **manifest['support_files'])
    for record in manifest['variants'].values():
        if not record.get('compiled'):
            raise RuntimeError('uncompiled variant in manifest')
        files.update({name: info['sha256'] for name, info in record['files'].items()})
    for name, expected in files.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or digest(path) != expected:
            raise RuntimeError('bundle SHA256 mismatch: ' + name)


def profile_records(text, variant, tile=None):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    completed = [r for r in rows if r.get('type') == 'complete']
    metadata = [r for r in rows if r.get('type') == 'metadata']
    if len(completed) != 1 or completed[0].get('passed') is not True or len(metadata) != 1:
        raise RuntimeError('missing/duplicate successful profile completion')
    if metadata[0].get('variant') != variant:
        raise RuntimeError('wrong profile variant')
    if tile is None:
        expected = {(t, p) for t in TILES for p in range(4)}
        correct = [r for r in rows if r.get('type') == 'correctness']
        if (len(correct) != len(expected) or {(r['tile'], r['pattern']) for r in correct} != expected
                or not all(r.get('passed') is True for r in correct)
                or completed[0].get('checked_values') != 589824):
            raise RuntimeError('incomplete prefix/digest/mining-path oracle coverage')
        return completed[0]
    measured = [r for r in rows if r.get('type') == 'measurement']
    pipeline = [r for r in rows if r.get('type') == 'pipeline']
    if len(measured) != 1 or len(pipeline) != 1 or measured[0]['tile'] != tile or pipeline[0]['tile'] != tile:
        raise RuntimeError('wrong or missing profile shape/pipeline')
    row = measured[0]
    values = row['gpu_ms']
    if len(values) < 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values):
        raise RuntimeError('invalid GPU timings')
    if row.get('variant') != variant or (row['m'], row['n'], row['k']) != (4096, 131072, 4096):
        raise RuntimeError('wrong timed workload')
    return dict(row, pipeline=pipeline[0], median_ms=statistics.median(values))


def attempt_result(text, count, discard):
    values = [tuple(map(float, match)) for match in ATTEMPT.findall(text)]
    if len(values) != count or not 0 <= discard < count:
        raise RuntimeError('incomplete full-miner attempt block')
    if 'plain_proof SHARE' in text or any(s in text for s in ('MISMATCH', 'CUDA error', 'verify FAILED')):
        raise RuntimeError('invalid fixed-work miner block')
    kept = values[discard:]
    if not all(math.isfinite(p) and math.isfinite(s) and p >= 0 and s > 0 for p, s, _ in kept):
        raise RuntimeError('invalid full-miner timings')
    return {'effective_tmac': len(kept) * WORK / sum(p + s for p, s, _ in kept),
            'prep_ms': 1000 * statistics.mean(p for p, _, _ in kept),
            'attempts': values, 'discarded': discard}


class Sweep:
    def __init__(self, bundle, output, device, seconds):
        self.bundle, self.output, self.device = bundle, output, device
        self.deadline = time.monotonic() + seconds
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('CP_')}
        self.env.update(CUDA_VISIBLE_DEVICES=device, CP_CUDA_OVERLAP='1', CP_CUDA_A_UPDATES='4096',
                        LD_LIBRARY_PATH=str(bundle / 'lib') + ':' + self.env.get('LD_LIBRARY_PATH', ''))
        self.report = {'completed': False, 'verified': {}, 'rejected': {}, 'panels': [],
                       'proofs': [], 'miner_blocks': [], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

    def remaining(self, maximum):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('experiment budget exhausted; partial results preserved')
        return min(maximum, remaining)

    def save(self):
        path = self.output / 'results.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report, indent=2) + '\n')
        temporary.replace(path)

    def telemetry(self):
        command = ['nvidia-smi', '-i', self.device, '--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
                   '--format=csv,noheader,nounits']
        result = subprocess.run(command, capture_output=True, text=True, timeout=self.remaining(10))
        return {'elapsed_remaining': self.deadline - time.monotonic(), 'values': result.stdout.strip()}

    def run(self, label, command, env=None, maximum=180):
        self.remaining(1)
        print('RUN ' + label, flush=True)
        path = self.output / (label + '.log')
        with path.open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env or self.env, start_new_session=True)
            try:
                code = child.wait(timeout=self.remaining(maximum))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        content = path.read_text(errors='replace')
        if code != 0:
            raise RuntimeError(label + ' failed; see ' + path.name)
        return content

    def fixed(self, variant, tile, mode, label, count, discard):
        env = dict(self.env, CP_CUDA_TB=tile, CP_CUDA_A_MODE=mode, CP_CUDA_A_CHECK='0')
        samples = [self.telemetry()]
        with PoolProbe(self.bundle / variant / 'cppminer',
                       ('--backend', 'cuda', '--cuda-mma', 'tensorop80', '--devices', '0',
                        '--m', '128', '--n', '128', '--max-nonce', str(count)), env=env) as pool:
            pool.send({'id': pool.auth_id, 'result': True})
            pool.send(job('ampere-fixed-work'))
            until = time.monotonic() + self.remaining(180)
            while '[plain] job stopped (max_nonce)' not in pool.output():
                if pool.process.poll() is not None or time.monotonic() >= until:
                    raise RuntimeError('fixed-work miner exited early or timed out: ' + label)
                samples.append(self.telemetry())
                time.sleep(.5)
            content = pool.output()
        (self.output / (label + '.log')).write_text(content)
        result = dict(attempt_result(content, count, discard), variant=variant, tile=tile,
                      mode=mode, label=label, telemetry=samples)
        self.report['miner_blocks'].append(result)
        self.save()
        print('BLOCK ' + label + ' %.3f effective TMAC/s' % result['effective_tmac'], flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--device', default='0', help='physical GPU index from nvidia-smi')
    parser.add_argument('--budget-seconds', type=int, default=2700)
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--finalists', type=int, default=3)
    parser.add_argument('--attempts', type=int, default=24)
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--ncu', action='store_true', help='capture one optional baseline Nsight Compute report')
    args = parser.parse_args()
    if (not args.device.isdigit() or not 60 <= args.budget_seconds <= 3300
            or not 3 <= args.repeats <= 200 or not 1 <= args.finalists <= 5 or not 8 <= args.attempts <= 120):
        parser.error('invalid device/budget/repeats/finalists/attempts')
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / 'manifest.json').read_text())
    validate_bundle(bundle, manifest)
    variants = args.variants or list(manifest['variants'])
    if 'baseline' not in variants or len(set(variants)) != len(variants) or any(v not in manifest['variants'] for v in variants):
        parser.error('select unique compiled variants including baseline')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('use a new output directory to preserve previous results')
    out.mkdir(parents=True, exist_ok=True)
    sweep = Sweep(bundle, out, args.device, args.budget_seconds)
    def interrupt(*_):
        raise TimeoutError('interrupted; only test children are stopped')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupt)
    signal.alarm(args.budget_seconds)
    try:
        info = sweep.run('gpu', ['nvidia-smi', '-i', args.device,
            '--query-gpu=name,uuid,compute_cap,driver_version,power.limit,memory.total', '--format=csv,noheader'])
        fields = next(csv.reader(info.splitlines(), skipinitialspace=True))
        if fields[2].strip() != '8.6':
            raise RuntimeError('this precompiled bundle requires an sm86 GPU')
        # CUDA's default device ordering can differ from NVML/nvidia-smi on a
        # mixed-GPU host. Pin the physical GPU by UUID, then use logical device 0.
        sweep.env['CUDA_VISIBLE_DEVICES'] = fields[1].strip()
        busy = sweep.run('gpu-processes', ['nvidia-smi', '-i', args.device,
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
        if any(line.strip().isdigit() for line in busy.splitlines()):
            raise RuntimeError('selected GPU already has compute processes; use an idle rented GPU')
        sweep.report.update(gpu=fields, bundle_sha256=digest(bundle / 'manifest.json'))
        for variant in variants:
            try:
                text = sweep.run('verify-' + variant, [str(bundle / variant / 'verify')])
                sweep.report['verified'][variant] = profile_records(text, variant)
            except RuntimeError as error:
                if variant == 'baseline':
                    raise
                sweep.report['rejected'][variant] = str(error)
            sweep.save()
        valid = [v for v in variants if v in sweep.report['verified']]
        combinations = [(v, t) for v in valid for t in TILES]
        # Two forward/reverse pairs expose thermal/clock drift. Each invocation
        # checks CPU prefix samples before timing and performs forty warmups.
        for round_id in range(4):
            order = combinations if round_id % 2 == 0 else list(reversed(combinations))
            for variant, tile in order:
                label = 'panel-%d-%s-%s' % (round_id, variant, tile)
                before = sweep.telemetry()
                text = sweep.run(label, [str(bundle / variant / 'full'), str(args.repeats), tile])
                measured = dict(profile_records(text, variant, tile), round=round_id,
                                telemetry=[before, sweep.telemetry()])
                sweep.report['panels'].append(measured)
                sweep.save()
        ranking = []
        for variant, tile in combinations:
            blocks = [r for r in sweep.report['panels'] if r['variant'] == variant and r['tile'] == tile]
            ranking.append({'variant': variant, 'tile': tile,
                            'median_ms': statistics.mean(r['median_ms'] for r in blocks),
                            'block_medians_ms': [r['median_ms'] for r in blocks]})
        ranking.sort(key=lambda r: r['median_ms'])
        sweep.report['panel_ranking'] = ranking
        sweep.save()
        finalists = [(r['variant'], r['tile']) for r in ranking
                     if (r['variant'], r['tile']) != ('baseline', '128x128')][:args.finalists]
        for variant, tile in [('baseline', '128x128'), *finalists]:
            binary = str(bundle / variant / 'cppminer')
            env = dict(sweep.env, CP_CUDA_TB=tile, CP_CUDA_A_MODE='dense')
            text = sweep.run('align-' + variant + '-' + tile, [binary, '--backend', 'cuda',
                '--cuda-mma', 'tensorop80', '--align-test-prod', '--m', '8', '--n', '8'], env)
            if 'GPU pipeline OK' not in text or 'tile-xor OK' not in text or 'mismatch' in text.lower():
                raise RuntimeError('production alignment failed: ' + variant + '/' + tile)
            for mode in ('dense', 'incremental'):
                env = dict(env, CP_CUDA_A_MODE=mode, CP_CUDA_A_CHECK='1' if mode == 'incremental' else '0')
                text = sweep.run('proof-' + variant + '-' + tile + '-' + mode,
                    [binary, '--backend', 'cuda', '--cuda-mma', 'tensorop80', '--no-fee',
                     '--mock', '--mock-diff', '40', '--m', '8', '--n', '8', '--verify', '--max-nonce', '16'], env)
                if '[mock] PASS' not in text:
                    raise RuntimeError('real Rust proof verification failed: ' + variant + '/' + mode)
                sweep.report['proofs'].append({'variant': variant, 'tile': tile, 'mode': mode, 'passed': True})
                sweep.save()
        confirmations = []
        for mode in ('dense', 'incremental'):
            sweep.fixed('baseline', '128x128', mode, 'conditioning-' + mode, 20, 4)
            for number, (variant, tile) in enumerate(finalists):
                blocks = []
                for index, (v, t) in enumerate([('baseline', '128x128'), (variant, tile),
                                               (variant, tile), ('baseline', '128x128')]):
                    label = 'abba-%s-%d-%d-%s-%s' % (mode, number, index, v, t)
                    blocks.append(sweep.fixed(v, t, mode, label, args.attempts, 4))
                def pooled(indices):
                    attempts = [x for i in indices for x in blocks[i]['attempts'][4:]]
                    return len(attempts) * WORK / sum(p + s for p, s, _ in attempts)
                control, candidate = pooled((0, 3)), pooled((1, 2))
                confirmations.append({'variant': variant, 'tile': tile, 'mode': mode,
                    'baseline_tmac': control, 'candidate_tmac': candidate,
                    'gain_percent': (candidate / control - 1) * 100})
                sweep.report['confirmations'] = confirmations
                sweep.save()
        if args.ncu:
            try:
                if not shutil.which('ncu'):
                    raise RuntimeError('Nsight Compute is unavailable')
                sweep.run('ncu-baseline', ['ncu', '--set', 'full', '--kernel-name', 'regex:FusedKernelEntry',
                    '--launch-skip', '41', '--launch-count', '1', '--export', str(out / 'baseline'),
                    str(bundle / 'baseline/full'), '3', '128x128'], maximum=120)
                sweep.report['ncu'] = 'captured'
            except RuntimeError as error:
                sweep.report['ncu'] = str(error)
        sweep.report['completed'] = True
        summary = ['# Ampere experiment results', '',
            '| Variant | Tile | A mode | Baseline TMAC/s | Candidate TMAC/s | Change |',
            '|---|---|---|---:|---:|---:|']
        for row in confirmations:
            summary.append('| {variant} | {tile} | {mode} | {baseline_tmac:.3f} | {candidate_tmac:.3f} | {gain_percent:+.2f}% |'.format(**row))
        summary += ['', 'These are paired complete-attempt measurements, not pool-income estimates.',
                    'Inspect repeated blocks and telemetry before choosing a production default.']
        (out / 'SUMMARY.md').write_text('\n'.join(summary) + '\n')
        print('COMPLETE ' + str(out / 'SUMMARY.md'), flush=True)
    except BaseException as error:
        sweep.report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        signal.alarm(0)
        sweep.save()


if __name__ == '__main__':
    main()
