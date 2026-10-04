"""Build a self-contained sm86 experiment bundle before renting a GPU (Linux)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import re

from ampere_variants import VARIANTS, TILES, headers, profile_source
from build_cuda_data_feed_profile import diagnostic

COMMON = ('src/common/cp_noise.c', 'src/common/cp_job_ctrl.cpp', 'src/common/cp_pool.cpp',
          'src/common/cp_util.cpp', 'src/common/cp_json_frame.cpp', 'src/common/cp_tcp.cpp',
          'src/common/cp_qpow_pool.cpp', 'src/common/cp_state.cpp', 'third_party/blake3/blake3.c',
          'third_party/blake3/blake3_dispatch.c', 'third_party/blake3/blake3_portable.c')
WRAPPERS = ('cp_cutlass_gemm_types.h', 'mma_milestone_multistage.h',
            'cp_cutlass_jackpot.cuh', 'hash_tile_policy.h')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def execute(command, log, commands):
    commands.append(command)
    with log.open('w') as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--refresh-tools', action='store_true',
                        help='regenerate after Python-only edits; reuse only identical compiler inputs/commands')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if not out.is_relative_to(root / 'build') or out == root / 'build':
        parser.error('use a dedicated directory underneath this checkout/build/')
    if args.jobs < 1 or args.jobs > 16 or len(set(args.variants)) != len(args.variants):
        parser.error('invalid jobs or duplicate variants')
    if 'baseline' not in args.variants:
        parser.error('baseline is mandatory')
    if out.exists() and any(out.iterdir()) and not (args.resume or args.refresh_tools):
        parser.error('output must be empty; use --resume for an interrupted identical build')
    out.mkdir(parents=True, exist_ok=True)
    original = {p.name: p.read_text() for p in (root / 'src/cuda/cutlass').glob('*')
                if p.is_file() and p.suffix in ('.h', '.cuh')}
    dependencies = ('third_party/cutlass/include/cutlass/cutlass.h',
                    'third_party/blake3/blake3.h',
                    'rust/cp-proof-ffi/target/release/libcp_proof_ffi.a')
    for name in dependencies:
        if not (root / name).is_file():
            parser.error('populate dependency first: ' + name)
    # Hash all compiler inputs including vendored CUDA headers and Rust proof.
    inputs = [p for directory in ('src', 'include', 'tests', 'scripts',
                                  'third_party/cutlass/include', 'third_party/cutlass_override',
                                  'third_party/cutlass/examples/35_gemm_softmax', 'third_party/blake3')
              for p in (root / directory).rglob('*')
              if p.is_file() and p.suffix in ('.c', '.cu', '.cpp', '.h', '.hpp', '.cuh', '.py')]
    inputs += [root / 'CMakeLists.txt', root / dependencies[-1]]
    source_hashes = {str(p.relative_to(root)): sha(p) for p in inputs}
    manifest_path = out / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous['source_sha256'] != source_hashes:
        changed = {name for name in source_hashes.keys() | previous['source_sha256'].keys()
                   if source_hashes.get(name) != previous['source_sha256'].get(name)}
        if not args.refresh_tools or any(not name.endswith('.py') for name in changed):
            parser.error('compiler source/dependency changed; use a fresh output directory')
    nvcc = shutil.which('nvcc')
    if not nvcc:
        parser.error('CUDA nvcc is required for preparation')
    manifest = previous or {'schema': 1, 'architecture': 'sm_86', 'completed': False,
        'source_sha256': source_hashes, 'variants': {}, 'commands': [],
        'nvcc_version': subprocess.check_output([nvcc, '--version'], text=True),
        'runtime': {'os': 'Linux x86_64', 'glibc': '2.39 or newer (Ubuntu 24.04)',
                    'cuda_runtime': 'bundled CUDA 12', 'python': '3.10 or newer'}}
    manifest['completed'] = False
    manifest['source_sha256'] = source_hashes
    previous_commands = list(manifest['commands'])
    old_toolchain = manifest['nvcc_version']
    if old_toolchain != subprocess.check_output([nvcc, '--version'], text=True):
        parser.error('CUDA toolchain changed; use a fresh output directory')
    manifest['commands'] = []
    save(manifest_path, manifest)
    include_paths = ('include', 'src/cuda', 'src/cuda/cutlass', 'third_party/cutlass_override',
                     'third_party/cutlass/include', 'third_party/cutlass/examples/35_gemm_softmax',
                     'third_party/blake3')
    includes = ['-I' + str(root / name) for name in include_paths]
    objects = []
    for i, source in enumerate(COMMON):
        obj = out / (str(i) + '.o')
        command = ['cc' if source.endswith('.c') else 'c++', '-O3', '-fopenmp',
                   '-ffunction-sections', '-fdata-sections']
        if source.endswith('.cpp'):
            command += ['-std=c++17']
        command += ['-DBLAKE3_NO_SSE2', '-DBLAKE3_NO_SSE41', '-DBLAKE3_NO_AVX2',
                    '-DBLAKE3_NO_AVX512', '-DCP_ENABLE_CPU=0', *includes,
                    '-c', str(root / source), '-o', str(obj)]
        if previous_commands and command != previous_commands[i]:
            parser.error('common compiler flags changed; use a fresh output directory')
        execute(command, out / (str(i) + '-build.log'), manifest['commands'])
        objects.append(str(obj))
    # Build the full miner in an owned scratch tree. Only its four wrapper
    # headers change; external dependencies are read-only, and the real Rust
    # proof archive is linked. Restoring these headers cannot touch production.
    scratch = out / 'miner-source'
    scratch.mkdir(exist_ok=True)
    for name in ('src', 'include'):
        shutil.copytree(root / name, scratch / name, dirs_exist_ok=True)
    shutil.copy2(root / 'CMakeLists.txt', scratch / 'CMakeLists.txt')
    for name in ('third_party', 'rust', 'tests'):
        link = scratch / name
        if not link.exists():
            link.symlink_to(root / name, target_is_directory=True)
    cmake = out / 'miner-build'
    configure = ['cmake', '-S', str(scratch), '-B', str(cmake), '-DCMAKE_BUILD_TYPE=Release',
                 '-DCP_ENABLE_CUDA=ON', '-DCP_CUDA_ARCH=86', '-DCP_ENABLE_CPU=ON',
                 '-DCP_ENABLE_OPENCL=OFF', '-DCP_ENABLE_ONEDNN=OFF', '-DCP_ENABLE_ESIMD=OFF',
                 '-DCP_ENABLE_CUBLAS=OFF', '-DCP_ENABLE_WGPU=OFF', '-DCP_PROOF_FFI=ON',
                 '-DCARGO_EXECUTABLE:FILEPATH=', '-DBUILD_TESTING=OFF',
                 '-DCP_STATIC_LIBCXX=ON']
    if previous_commands and configure != previous_commands[len(COMMON)]:
        parser.error('miner configuration changed; use a fresh output directory')
    execute(configure, out / 'configure.log', manifest['commands'])
    if 'linking prebuilt' not in (out / 'configure.log').read_text():
        raise RuntimeError('real prebuilt proof FFI must be linked; stub is forbidden')
    profile = profile_source((root / 'tests/cp_cuda_data_feed_profile.cu').read_text())
    package = out / 'bundle'
    package.mkdir(exist_ok=True)
    for name in ('run_cuda_ampere.py', 'ampere_variants.py', 'build_cuda_stage_profile.py'):
        shutil.copy2(root / 'scripts' / name, package / name)
    shutil.copy2(root / 'tests/cp_pool_session_integration.py', package / 'cp_pool_session_integration.py')
    shutil.copy2(root / 'scripts/ampere-runtime.Dockerfile', package / 'Dockerfile')
    shutil.copy2(root / 'docs/cuda_ampere_research.md', package / 'README.md')
    try:
        for variant in args.variants:
            old = manifest['variants'].get(variant, {})
            generated = headers(original, VARIANTS[variant])
            expected_headers = dict(generated)
            expected_headers['cp_cutlass_jackpot.cuh'] = diagnostic(generated['cp_cutlass_jackpot.cuh'])
            header_hashes = {name: hashlib.sha256(source.encode()).hexdigest()
                             for name, source in expected_headers.items()}
            dest = out / variant
            profile_hash = hashlib.sha256(profile.encode()).hexdigest()
            expected_commands = []
            for verify, kind in ((0, 'full'), (1, 'verify')):
                expected_commands.append([nvcc, '-std=c++17', '-O3', '-arch=sm_86', '-lineinfo', '-Xptxas=-v',
                    '-Xcompiler=-fopenmp', '-Xlinker=--gc-sections', '-DCP_FEED_VARIANT="' + variant + '"',
                    '-DCP_FEED_VERIFY=' + str(verify), '-I' + str(dest), *includes,
                    str(dest / 'profile.cu'), *objects, '-o', str(package / variant / kind)])
            old_commands = [c for c in previous_commands if c and c[-1] in
                            (str(package / variant / 'full'), str(package / variant / 'verify'))]
            profile_unchanged = (dest / 'profile.cu').is_file() and sha(dest / 'profile.cu') == profile_hash
            if (old.get('compiled') and old.get('generated_sha256') == header_hashes
                    and profile_unchanged and old_commands == expected_commands
                    and all((package / path).is_file() and sha(package / path) == info['sha256']
                            for path, info in old['files'].items())):
                manifest['commands'].extend(expected_commands)
                sass_file = dest / 'full.sass'
                if sass_file.exists():
                    if sha(sass_file) != old['sass_sha256']:
                        raise RuntimeError('cached SASS hash mismatch: ' + variant)
                    shutil.copy2(sass_file, package / variant / 'full.sass')
                    old['files'][variant + '/full.sass'] = {'sha256': sha(sass_file)}
                print('REUSE ' + variant, flush=True)
                continue
            started = time.monotonic()
            dest.mkdir(exist_ok=True)
            for name, source in generated.items():
                (dest / name).write_text(source)
            for name in WRAPPERS:
                (scratch / 'src/cuda/cutlass' / name).write_text(generated[name])
            (dest / 'cp_cutlass_jackpot.cuh').write_text(diagnostic(generated['cp_cutlass_jackpot.cuh']))
            (dest / 'profile.cu').write_text(profile)
            binaries = package / variant
            binaries.mkdir(exist_ok=True)
            record = {'options': VARIANTS[variant], 'compiled': False, 'files': {},
                      'generated_sha256': {name: sha(dest / name) for name in generated}}
            for verify, kind in ((0, 'full'), (1, 'verify')):
                binary = binaries / kind
                command = [nvcc, '-std=c++17', '-O3', '-arch=sm_86', '-lineinfo', '-Xptxas=-v',
                           '-Xcompiler=-fopenmp', '-Xlinker=--gc-sections',
                           '-DCP_FEED_VARIANT="' + variant + '"', '-DCP_FEED_VERIFY=' + str(verify),
                           '-I' + str(dest), *includes, str(dest / 'profile.cu'), *objects, '-o', str(binary)]
                execute(command, dest / (kind + '-build.log'), manifest['commands'])
                if not verify:
                    dump = subprocess.check_output([str(Path(nvcc).parent / 'cuobjdump'),
                        '--dump-sass', '--gpu-architecture', 'sm_86', str(binary)], text=True)
                    (dest / 'full.sass').write_text(dump)
                    (binaries / 'full.sass').write_text(dump)
                    record['sass_sha256'] = hashlib.sha256(dump.encode()).hexdigest()
                    record['sass_contains_redux'] = 'REDUX' in dump
                    record['static_instructions'] = {name: {
                        op: len(re.findall(r'\b' + op + r'(?:\.|\s)', body))
                        for op in ('IMMA', 'LDGSTS', 'LDSM', 'LDL', 'STL', 'BAR', 'SHFL', 'REDUX')}
                        for name, body in re.findall(r'Function : ([^\n]+)\n(.*?)(?=\n\s*Function :|\Z)', dump, re.S)
                        if 'FusedKernelEntry' in name}
            execute(['cmake', '--build', str(cmake), '--target', 'cppminer', '-j', str(args.jobs)],
                    dest / 'miner-build.log', manifest['commands'])
            shutil.copy2(cmake / 'cppminer', binaries / 'cppminer')
            for binary in binaries.iterdir():
                record['files'][str(binary.relative_to(package))] = {'sha256': sha(binary)}
            record.update(compiled=True, seconds=time.monotonic() - started)
            manifest['variants'][variant] = record
            save(manifest_path, manifest)
            print('BUILT ' + variant, flush=True)
    finally:
        for name in WRAPPERS:
            (scratch / 'src/cuda/cutlass' / name).write_text(original[name])
    runtime = Path(nvcc).resolve().parents[1] / 'targets/x86_64-linux/lib/libcudart.so.12'
    (package / 'lib').mkdir(exist_ok=True)
    shutil.copy2(runtime.resolve(), package / 'lib/libcudart.so.12')
    manifest['runtime_files'] = {'lib/libcudart.so.12': sha(package / 'lib/libcudart.so.12')}
    manifest['support_files'] = {name: sha(package / name) for name in (
        'run_cuda_ampere.py', 'ampere_variants.py', 'build_cuda_stage_profile.py', 'cp_pool_session_integration.py',
        'Dockerfile', 'README.md')}
    manifest['completed'] = all(manifest['variants'].get(v, {}).get('compiled') for v in args.variants)
    save(manifest_path, manifest)
    save(package / 'manifest.json', manifest)
    shutil.make_archive(str(out / 'ampere-sm86'), 'gztar', root_dir=out, base_dir='bundle')
    print('READY ' + str(out / 'ampere-sm86.tar.gz'), flush=True)


if __name__ == '__main__':
    main()
