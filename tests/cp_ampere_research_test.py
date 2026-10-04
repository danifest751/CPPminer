"""CPU tests for distributed XOR semantics and experiment acceptance gates."""
import json
from pathlib import Path
import random
import sys
import tempfile
import hashlib
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cuda_ampere import attempt_result, profile_records, validate_bundle
from ampere_variants import TILES, VARIANTS, headers, profile_source


class AmpereResearchTest(unittest.TestCase):
    def test_subgroup_reduction_preserves_every_distributed_output(self):
        rng = random.Random(719)
        for halves in (1, 2):
            for _ in range(100):
                partials = [[rng.getrandbits(32) for _ in range(8 * halves)] for _ in range(32)]
                # Existing three butterfly steps with each lane keeping its half.
                scattered = partials
                count = 8 * halves
                for mask in (8, 4, 1):
                    size = count // 2
                    next_step = []
                    for lane in range(32):
                        hi = bool(lane & mask)
                        peer = lane ^ mask
                        peer_hi = bool(peer & mask)
                        keep = scattered[lane][size:] if hi else scattered[lane][:size]
                        send = scattered[peer][:size] if peer_hi else scattered[peer][size:]
                        next_step.append([a ^ b for a, b in zip(keep, send)])
                    scattered, count = next_step, size
                # Independent native reduction of one fixed partial per subgroup.
                for lane in range(32):
                    members = 0x3333 << (lane & 18)
                    contributors = [p for p in range(32) if members & (1 << p)]
                    self.assertEqual(len(contributors), 8)
                    self.assertIn(lane, contributors)
                    owner = (((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + (lane & 1)) * halves
                    for t in range(halves):
                        expected = 0
                        for peer in contributors:
                            expected ^= partials[peer][owner + t]
                        self.assertEqual(scattered[lane][t], expected)

    def test_partial_oracle_output_is_rejected(self):
        rows = [{'type': 'metadata', 'variant': 'baseline'},
                {'type': 'complete', 'passed': True, 'checked_values': 589824}]
        rows += [{'type': 'correctness', 'tile': t, 'pattern': p, 'passed': True}
                 for t in TILES for p in range(4)]
        text = '\n'.join(map(json.dumps, rows))
        self.assertEqual(profile_records(text, 'baseline')['checked_values'], 589824)
        with self.assertRaises(RuntimeError):
            profile_records('\n'.join(map(json.dumps, rows[:-1])), 'baseline')
        rows[1]['checked_values'] -= 1
        with self.assertRaises(RuntimeError):
            profile_records('\n'.join(map(json.dumps, rows)), 'baseline')

    def test_full_attempt_rate_uses_prep_plus_scan(self):
        log = '[gpu] attempt timing: prep=0.100s scan=1.000s 70.000 TMAC/s\n' * 8
        result = attempt_result(log, 8, 4)
        self.assertAlmostEqual(result['effective_tmac'], 131072 * 131072 * 4096 / 1e12 / 1.1)
        with self.assertRaises(RuntimeError):
            attempt_result(log, 9, 4)
        with self.assertRaises(RuntimeError):
            attempt_result(log + 'plain_proof SHARE', 8, 4)

    def test_profile_rejects_nonfinite_timings_and_wrong_shapes(self):
        rows = [{'type': 'metadata', 'variant': 'baseline'},
                {'type': 'complete', 'passed': True},
                {'type': 'pipeline', 'tile': '128x128'},
                {'type': 'measurement', 'variant': 'baseline', 'tile': '128x128',
                 'm': 4096, 'n': 131072, 'k': 4096, 'gpu_ms': [1, 2, 3]}]
        self.assertEqual(profile_records('\n'.join(map(json.dumps, rows)), 'baseline', '128x128')['median_ms'], 2)
        rows[-1]['gpu_ms'][0] = float('nan')
        with self.assertRaises(RuntimeError):
            profile_records('\n'.join(map(json.dumps, rows)), 'baseline', '128x128')
        rows[-1]['gpu_ms'][0] = 1
        with self.assertRaises(RuntimeError):
            profile_records('\n'.join(map(json.dumps, rows)), 'baseline', '256x128')

    def test_bundle_rejects_modified_files_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            binary = root / 'binary'
            binary.write_bytes(b'compiled-test-fixture')
            checksum = hashlib.sha256(binary.read_bytes()).hexdigest()
            manifest = {'schema': 1, 'completed': True, 'architecture': 'sm_86',
                        'runtime_files': {}, 'support_files': {}, 'variants': {
                            'baseline': {'compiled': True, 'files': {'binary': {'sha256': checksum}}}}}
            validate_bundle(root, manifest)
            binary.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                validate_bundle(root, manifest)
            manifest['variants']['baseline']['files'] = {'../binary': {'sha256': checksum}}
            with self.assertRaises(RuntimeError):
                validate_bundle(root, manifest)

    def test_all_private_hooks_match_current_source(self):
        root = Path(__file__).resolve().parents[1]
        original = {p.name: p.read_text() for p in (root / 'src/cuda/cutlass').glob('*') if p.is_file()}
        self.assertEqual(headers(original, VARIANTS['baseline']), original)
        for options in VARIANTS.values():
            headers(original, options)
        source = profile_source((root / 'tests/cp_cuda_data_feed_profile.cu').read_text())
        self.assertIn('all mining-branch BLAKE3 digests', source)


if __name__ == '__main__':
    unittest.main()
