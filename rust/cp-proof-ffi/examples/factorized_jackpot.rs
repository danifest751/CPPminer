//! Exact CPU experiment for zero signal-B. This does not change the miner.
//! cargo run --release --manifest-path rust/cp-proof-ffi/Cargo.toml --example factorized_jackpot
use pearl_blake3::blake3_digest;
use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams,
    SeedDerivation,
};
use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
use zk_pow::circuit::chip::compute_jackpot;
use zk_pow::circuit::pearl_noise::{
    compute_noise, generate_permutation_matrix, generate_uniform_random_matrix,
};

const K: usize = 4096;
const R: usize = 128;
const M: usize = 128;
const ROWS: [u32; 8] = [0, 1, 2, 3, 16, 17, 18, 19];
const COLS: [u32; 8] = [0, 1, 2, 3, 32, 33, 34, 35];

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

struct Case {
    derivation: String,
    signal: String,
    seed: u64,
    anchor: u32,
    milestones: usize,
    maximum_projection_abs: i32,
    projection_entries: usize,
    projection_outside_int8: usize,
    two_signed_int8_limbs_sufficient_for_this_case: bool,
    hash: String,
}

fn run(derivation: SeedDerivation, mode: &str, seed: u64, anchor: u32) -> Case {
    let header = IncompleteBlockHeader {
        version: 1,
        prev_block: [seed as u8; 32],
        merkle_root: [(seed >> 8) as u8; 32],
        timestamp: 1_790_000_000 + seed as u32,
        nbits: 0x207fffff,
    };
    let config = MiningConfiguration {
        common_dim: K as u32,
        rank: R as u16,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&ROWS).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&COLS).unwrap(),
        moe: None,
    };
    let mut state = seed;
    let mut a = vec![0u8; M * K];
    match mode {
        "dense" => {
            for value in &mut a {
                *value = (((splitmix(&mut state) >> 32) & 127) as i8 - 64) as u8;
            }
        }
        "sparse" => {
            for col in 0..K {
                let random = splitmix(&mut state);
                let row = random as usize % M;
                a[row * K + col] = (((random >> 32) & 127) as i8 - 64) as u8;
            }
        }
        "zero" => {}
        _ => unreachable!(),
    }
    let b = vec![0u8; M * K];
    let mut params = PublicProofParams::new(
        header, derivation, config, [0; 32], [0; 32], [0; 32], M as u32, M as u32, anchor, anchor,
    );
    let job_key = params.job_key();
    params.hash_a = blake3_digest(&a, Some(job_key));
    params.hash_b = blake3_digest(&b, Some(job_key));
    let compiled = CompiledPublicParams::from(&params);
    let noise = compute_noise(&compiled);
    let secret_a: Vec<Vec<i8>> = compiled
        .a_rows_indices
        .iter()
        .map(|&row| a[row * K..(row + 1) * K].iter().map(|&x| x as i8).collect())
        .collect();
    let secret_b = vec![vec![0i8; K]; compiled.w];
    let noisy_a: Vec<Vec<i32>> = secret_a
        .iter()
        .zip(&noise.a)
        .map(|(a, n)| {
            a.iter()
                .zip(n)
                .map(|(&a, &n)| a as i32 + n as i32)
                .collect()
        })
        .collect();
    let mut label = [0u8; 32];
    label[..8].copy_from_slice(b"B_tensor");
    let pairs = generate_permutation_matrix(&label, &compiled.commitment_hash.0, K, R);
    let e_b = generate_uniform_random_matrix(
        &label,
        &compiled.commitment_hash.0,
        &compiled.b_cols_indices,
        R,
    );
    // Check the independently regenerated compact B against the reference noise.
    for v in 0..compiled.w {
        for l in 0..K {
            assert_eq!(
                noise.b[v][l] as i32,
                e_b[v][pairs[l][0] as usize] as i32 - e_b[v][pairs[l][1] as usize] as i32
            );
        }
    }
    let mut direct = vec![vec![0i32; compiled.w]; compiled.h];
    let mut factorized_tile = vec![vec![0i32; compiled.w]; compiled.h];
    let mut projection = vec![vec![0i32; R]; compiled.h];
    let mut jackpot = [0u32; 16];
    let mut max_abs = 0;
    let mut outside = 0;
    let mut entries = 0;
    let mut fits_two = true;
    for end in (R..=K).step_by(R) {
        for u in 0..compiled.h {
            for l in end - R..end {
                let [plus, minus] = pairs[l];
                projection[u][plus as usize] += noisy_a[u][l];
                projection[u][minus as usize] -= noisy_a[u][l];
            }
            for &d in &projection[u] {
                max_abs = max_abs.max(d.abs());
                entries += 1;
                outside += usize::from(!(-128..=127).contains(&d));
                let low = d as i8 as i32;
                let high = (d - low) / 256;
                fits_two &= (-128..=127).contains(&high);
                assert_eq!(d, low + high * 256);
            }
            for v in 0..compiled.w {
                for l in end - R..end {
                    direct[u][v] += noisy_a[u][l] * noise.b[v][l] as i32;
                }
                let factorized: i32 = projection[u]
                    .iter()
                    .zip(&e_b[v])
                    .map(|(&d, &e)| d * e as i32)
                    .sum();
                assert_eq!(direct[u][v], factorized, "prefix {end}, tile {u},{v}");
                factorized_tile[u][v] = factorized;
                // An exact two-pass INT8 product where the observed coefficients fit.
                if fits_two {
                    let low: i32 = projection[u]
                        .iter()
                        .zip(&e_b[v])
                        .map(|(&d, &e)| d as i8 as i32 * e as i32)
                        .sum();
                    let high: i32 = projection[u]
                        .iter()
                        .zip(&e_b[v])
                        .map(|(&d, &e)| ((d - d as i8 as i32) / 256) * e as i32)
                        .sum();
                    assert_eq!(direct[u][v], low + 256 * high);
                }
            }
        }
        let xor = factorized_tile
            .iter()
            .flatten()
            .fold(0u32, |acc, &c| acc ^ c as u32);
        let slot = (end / R - 1) % 16;
        jackpot[slot] = jackpot[slot].rotate_left(13) ^ xor;
    }
    let reference = compute_jackpot(&compiled, &secret_a, &secret_b, &noise);
    assert_eq!(jackpot, reference);
    let hash = compute_jackpot_hash(&jackpot, compiled.commitment_hash.1);
    assert_eq!(
        hash,
        compute_jackpot_hash(&reference, compiled.commitment_hash.1)
    );
    Case {
        derivation: format!("{derivation:?}"),
        signal: mode.into(),
        seed,
        anchor,
        milestones: K / R,
        maximum_projection_abs: max_abs,
        projection_entries: entries,
        projection_outside_int8: outside,
        two_signed_int8_limbs_sufficient_for_this_case: fits_two,
        hash: hash.iter().map(|x| format!("{x:02x}")).collect(),
    }
}

fn main() {
    let mut cases = Vec::new();
    for derivation in [SeedDerivation::Legacy, SeedDerivation::Salted] {
        for mode in ["zero", "sparse", "dense"] {
            for seed in [1, 19, 257, 65537] {
                for anchor in [0, 64] {
                    cases.push(run(derivation, mode, seed, anchor));
                }
            }
        }
    }
    println!("{}", json_report(&cases));
}

fn json_report(cases: &[Case]) -> String {
    // Keep this executable's dependency set identical to the proof FFI.
    let mut report = String::from("{\n  \"k\":4096, \"rank\":128, \"tile_h\":8, \"tile_w\":8,\n");
    report.push_str("  \"direct_mac_per_tile\":262144, \"all_prefix_factorized_mac_per_tile\":262144,\n  \"projection_additions_per_tile\":65536,\n  \"cases\":[\n");
    for (i, c) in cases.iter().enumerate() {
        report.push_str(&format!("    {{\"derivation\":\"{}\",\"signal\":\"{}\",\"seed\":{},\"anchor\":{},\"milestones\":{},\"maximum_projection_abs\":{},\"projection_entries\":{},\"projection_outside_int8\":{},\"two_signed_int8_limbs_sufficient_for_this_case\":{},\"hash\":\"{}\"}}{}\n", c.derivation, c.signal, c.seed, c.anchor, c.milestones, c.maximum_projection_abs, c.projection_entries, c.projection_outside_int8, c.two_signed_int8_limbs_sufficient_for_this_case, c.hash, if i+1 == cases.len() { "" } else { "," }));
    }
    report.push_str("  ]\n}");
    report
}
