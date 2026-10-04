"""Private Ampere research headers. Production and vendor headers are never edited."""
from build_cuda_stage_profile import replace_once


VARIANTS = {
    'baseline': {},
    'before_loads': {'schedule': 'before_loads'},
    'async_shadow': {'schedule': 'async_shadow'},
    'predicated': {'predicated': True},
    'before_predicated': {'schedule': 'before_loads', 'predicated': True},
    'shadow_predicated': {'schedule': 'async_shadow', 'predicated': True},
    'redux': {'redux': True},
    'redux_predicated': {'redux': True, 'predicated': True},
    'shadow_redux_predicated': {'schedule': 'async_shadow', 'redux': True, 'predicated': True},
    'stages2': {'stages': 2},
    'stages4': {'stages': 4},
    'warp32': {'warp_m': 32},
    'cache_a_ca': {'cache_a': 'Always'},
    'cache_b_ca': {'cache_b': 'Always'},
    'cache_ab_ca': {'cache_a': 'Always', 'cache_b': 'Always'},
    'group4': {'swizzle': 4},
    'group16': {'swizzle': 16},
}
TILES = ('128x128', '256x128', '128x256')


def schedule(source, placement):
    """Fold a completed prefix before the next MMA can change its accumulator.

    async_shadow issues the next copy group first, then folds, then loads shared
    fragments. The moved copy is removed from its former position, preserving
    the number/order of copies, fences, waits and K iterations. Flush the final
    prefix exactly once without issuing another K iteration.
    """
    source = replace_once(source, '  CUTLASS_DEVICE\n  void mac_loop_iter(',
                          '  template <typename Callback>\n  CUTLASS_DEVICE\n  void mac_loop_iter(')
    source = replace_once(source, '                     int &gemm_k_iterations) {',
                          '                     int &gemm_k_iterations, Callback &&cb,\n'
                          '                     bool pending, int previous_ms) {')
    anchor = '''      this->warp_tile_iterator_A_.set_kgroup_index(
          (warp_mma_k + 1) % Base::kWarpGemmIterations);'''
    early_copy = '''        copy_group<false>(iterator_A, iterator_B, 0, 0,
                          Detail::kAccessesPerGroupA,
                          Detail::kAccessesPerGroupB);
'''
    if placement == 'async_shadow':
        source = replace_once(source, '      if (warp_mma_k < Base::kWarpGemmIterations - 1) {',
                              '      if (warp_mma_k > 0 && warp_mma_k < Base::kWarpGemmIterations - 1) {')
    else:
        early_copy = ''
    pending = '      if (warp_mma_k == 0) {\n' + early_copy + '''        if (pending)
          cb(previous_ms, accum);
      }

'''
    source = replace_once(source, anchor, pending + anchor)
    old = '''        mac_loop_iter(pipe_state, accum, iterator_A, iterator_B,
                      gemm_k_iterations);
      cb(ms_idx, accum);'''
    new = '''        mac_loop_iter(pipe_state, accum, iterator_A, iterator_B,
                      gemm_k_iterations, cb, ms_idx > 0 && i == 0, ms_idx - 1);'''
    source = replace_once(source, old, new)
    return replace_once(source, '    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {',
                        '''    if (num_ms > 0)
      cb(num_ms - 1, accum);

    if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {''')


def redux(source):
    old = '''    /* Reduce-scatter across the 8 lanes sharing (b4, b1). */
    reduce_scatter_step<kPartials, 8>(w, lane);
    reduce_scatter_step<kPartials / 2, 4>(w, lane);
    reduce_scatter_step<kPartials / 4, 1>(w, lane);

    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < kHalves; ++t)
      emit(t, w[t]);'''
    new = '''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Four disjoint groups. Within each, bits 3, 2 and 0 enumerate the eight
    // contributing lanes. Every member executes EVERY reduction for the same
    // partial index; only selection of its owned output is lane-dependent.
    unsigned const members = 0x00003333u << (lane & 18);
    int const owner = (((lane >> 3) & 1) * 4 +
                       ((lane >> 2) & 1) * 2 + (lane & 1)) * kHalves;
    uint32_t owned[kHalves] = {};
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kPartials; ++j) {
      uint32_t const x = __reduce_xor_sync(members, w[j]);
      CUTLASS_PRAGMA_UNROLL
      for (int t = 0; t < kHalves; ++t)
        if (j == owner + t)
          owned[t] = x;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < kHalves; ++t)
      emit(t, owned[t]);
#else
''' + old + '\n#endif'
    return replace_once(source, old, new)


def headers(original, options):
    result = dict(original)
    if options.get('schedule'):
        key = 'mma_milestone_multistage.h'
        result[key] = schedule(result[key], options['schedule'])
    if options.get('predicated'):
        key = 'cp_cutlass_jackpot.cuh'
        result[key] = replace_once(result[key],
            '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750',
            '#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750 || __CUDA_ARCH__ >= 800)')
    if options.get('redux'):
        result['hash_tile_policy.h'] = redux(result['hash_tile_policy.h'])
    key = 'cp_cutlass_gemm_types.h'
    if options.get('stages'):
        for macro in ('CP_SM80_STAGES', 'CP_SM80_STAGES_BIG'):
            result[key] = replace_once(result[key], '#define ' + macro + ' 3',
                                       '#define ' + macro + ' ' + str(options['stages']))
    if options.get('warp_m'):
        result[key] = replace_once(result[key],
            'using TensorOp80WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;',
            'using TensorOp80WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;')
        # The larger shapes contain 512 threads. Let ptxas enforce their
        # per-block register budget instead of generating an unlaunchable CTA.
        result[key] = replace_once(result[key], '__global__ void\nFusedKernelEntry(',
            '__global__ __launch_bounds__(GemmTypesT::GemmKernel::kThreadCount) void\nFusedKernelEntry(')
    if options.get('swizzle'):
        result[key] = replace_once(result[key], 'GemmIdentityThreadblockSwizzle<8>',
                                  'GemmIdentityThreadblockSwizzle<' + str(options['swizzle']) + '>')
    # Change only this private loop's async copies, without patching CUTLASS.
    key = 'mma_milestone_multistage.h'
    for operand in ('a', 'b'):
        if options.get('cache_' + operand):
            for function in ('cp_async_zfill', 'cp_async'):
                target = function + '<kSrcBytes, CacheOp' + operand.upper() + '>'
                result[key] = replace_once(result[key], target,
                    function + '<kSrcBytes, arch::CacheOperation::' + options['cache_' + operand] + '>')
    return result


def profile_source(source):
    """Reuse the independent INT64/CPU BLAKE3 oracle with Ampere types."""
    source = replace_once(source, 'gpu.major * 10 + gpu.minor != 75',
                          'gpu.major * 10 + gpu.minor != 86')
    source = replace_once(source, 'this experiment targets sm_75 only',
                          'this precompiled experiment targets sm_86 (RTX 30 series) only')
    for tile in ('128x128', '256x128'):
        source = source.replace('Gemm' + tile + 'TensorOp>', 'Gemm' + tile + 'TensorOp80>')
    source = replace_once(source,
        '                run<cp_cutlass::Gemm256x128TensorOp80>(in, pattern, 0, "256x128");',
        '''                run<cp_cutlass::Gemm256x128TensorOp80>(in, pattern, 0, "256x128");
                run<cp_cutlass::Gemm128x256TensorOp80>(in, pattern, 0, "128x256");''')
    source = replace_once(source, '            if (argc > 2)',
                          '            if (argc > 2 && std::string(argv[2]) == "128x256")\n'
                          '                run<cp_cutlass::Gemm128x256TensorOp80>(in, 3, repeats, "128x256");\n'
                          '            else if (argc > 2 && std::string(argv[2]) == "256x128")')
    source = replace_once(source, '        compare(digests.host(), expected.digests, "all keyed BLAKE3 digests");',
        '''        compare(digests.host(), expected.digests, "all keyed BLAKE3 digests");
        CUDA(cudaMemset(digests.p, 0xa5, digests.count * 4));
        status(op.initialize(in.m, in.n, 4096, in.m, in.n, in.a.p, in.b.p,
                             nullptr, in.n / 128, tiles, &jackpot));
        status(op());
        CUDA(cudaDeviceSynchronize());
        compare(digests.host(), expected.digests, "all mining-branch BLAKE3 digests");''')
    # Print actual pipeline type separately: CUTLASS may choose a synchronous
    # specialization for two stages. Never infer cp.async from the label alone.
    source = replace_once(source, '        cudaFuncAttributes attr;',
                          '''        std::printf("{\\\"type\\\":\\\"pipeline\\\",\\\"tile\\\":\\\"%s\\\","
                    "\\\"multistage\\\":%s,\\\"stages\\\":%d,\\\"threads\\\":%d}\\n",
                    shape, T::kMultistage ? "true" : "false", T::kStages,
                    T::GemmKernel::kThreadCount);
        cudaFuncAttributes attr;''')
    return source
