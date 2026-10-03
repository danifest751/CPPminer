#include "cp_fee.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

/*
 * Developer fee wallets — light XOR obfuscation (defeats casual `strings`).
 * Not real secrecy: recoverable at runtime from authorize traffic or reversing.
 *
 * To change an address:
 *   python internal/encode_fee_wallet.py 'prl1...'   # or qzpp...
 * and paste the printed bytes into the matching k_*_dev_wallet_enc below.
 *
 * Quantus fee wallet is empty until a qzpp address is encoded; fee stays off.
 */

static const unsigned char k_dev_wallet_key[] = {
    'c', 'p', 0x9e, 'm', 'i', 'n', 'A', 0x11, 'z'
};

static const unsigned char k_pearl_dev_wallet_enc[] = {
    0x13, 0x02, 0xf2, 0x5c, 0x19, 0x56, 0x73, 0x7a, 0x17, 0x12, 0x05, 0xad,
    0x09, 0x13, 0x0a, 0x73, 0x76, 0x4d, 0x5a, 0x15, 0xec, 0x15, 0x1e, 0x5c,
    0x3b, 0x24, 0x12, 0x1b, 0x1c, 0xe4, 0x06, 0x5a, 0x57, 0x38, 0x7f, 0x1e,
    0x10, 0x13, 0xf8, 0x0c, 0x04, 0x04, 0x20, 0x76, 0x0a, 0x16, 0x1e, 0xf2,
    0x55, 0x0e, 0x09, 0x79, 0x64, 0x1c, 0x12, 0x1b, 0xef, 0x0b, 0x5c, 0x57,
    0x38, 0x27, 0x4f,
};

/* Placeholder: no Quantus fee address encoded yet (len 0 → fee disabled). */
static const unsigned char k_quantus_dev_wallet_enc[1] = {0};
static const size_t k_quantus_dev_wallet_enc_len = 0;

/*
 * Fork builds (danifest751/CPPminer releases): the 1% fee goes to the fork
 * maintainer instead. Pearl fee work is mined on the fork's fee pool
 * (HeroMiners) under the maintainer's Pearl address, whatever pool the user
 * picked; the miner reconnects there for the fee cycle and back afterwards.
 * If the fee pool cannot be reached CP_FEE_POOL_MAX_FAILS times in a row, the
 * fee falls back to the user's pool: a Kryptex account worker on Kryptex, the
 * Pearl address elsewhere. All of this is plain on purpose and stated in the
 * release notes.
 */
static const char k_fork_fee_pool_host[] = "ru.pearl.herominers.com";
static const int k_fork_fee_pool_port = 1200;
static const char k_fork_fee_pool_worker[] = "devfee";
static const char k_fork_kryptex_fee_wallet[] = "krxX8QJ872.devfee";
static const char k_fork_pearl_fee_wallet[] =
    "prl1pp9k3spr6l0c0s00mlmcnpktm5yfp0up9lu92deuvj2hnp38s8x0svyseuq";
#define CP_FEE_POOL_MAX_FAILS 3
static char g_pool_host[256];
static int g_fee_pool_fails = 0;

static char g_user_wallet[256];
static char g_dev_wallet[256];
static int g_enabled = 0;
static int g_auth_is_dev = 0;
static int g_fee_active = 0;
static uint64_t g_debt = 0;
static uint64_t g_tiles_per_matrix = 0; /* T: tiles (Pearl) or hash quantum (Quantus) */
static CpAlgoId g_algo = CP_ALGO_PEARL;

static void load_dev_wallet(CpAlgoId algo)
{
    const unsigned char* enc = k_pearl_dev_wallet_enc;
    size_t n = sizeof(k_pearl_dev_wallet_enc);
    if(algo == CP_ALGO_QUANTUS){
        enc = k_quantus_dev_wallet_enc;
        n = k_quantus_dev_wallet_enc_len;
    }
    g_dev_wallet[0] = 0;
    if(algo == CP_ALGO_PEARL){
        int kryptex = 0;
        for(const char* p = g_pool_host; *p && !kryptex; p++){
            const char* q = "kryptex";
            const char* s = p;
            while(*q && *s && tolower((unsigned char)*s) == *q){ s++; q++; }
            kryptex = (*q == 0);
        }
        const char* w = kryptex ? k_fork_kryptex_fee_wallet : k_fork_pearl_fee_wallet;
        strncpy(g_dev_wallet, w, sizeof(g_dev_wallet) - 1);
        g_dev_wallet[sizeof(g_dev_wallet) - 1] = 0;
        return;
    }
    if(n == 0) return;
    const size_t klen = sizeof(k_dev_wallet_key);
    if(n >= sizeof(g_dev_wallet)) return;
    for(size_t i = 0; i < n; i++)
        g_dev_wallet[i] = (char)(enc[i] ^ k_dev_wallet_key[i % klen]);
    g_dev_wallet[n] = 0;
}

static uint64_t threshold_tiles(void)
{
    if(g_tiles_per_matrix == 0) return 0;
    return (uint64_t)CP_FEE_PERIOD * g_tiles_per_matrix;
}

void cp_fee_set_pool_host(const char* host)
{
    strncpy(g_pool_host, host ? host : "", sizeof(g_pool_host) - 1);
    g_pool_host[sizeof(g_pool_host) - 1] = 0;
}

void cp_fee_init(const char* user_wallet, int enable, CpAlgoId algo)
{
    g_algo = algo;
    g_user_wallet[0] = 0;
    if(user_wallet){
        strncpy(g_user_wallet, user_wallet, sizeof(g_user_wallet) - 1);
        g_user_wallet[sizeof(g_user_wallet) - 1] = 0;
    }
    load_dev_wallet(algo);
    fflush(stdout);

    /* Compare the account part only ("acct.worker" -> "acct"): mining to the
     * fee account itself must not switch back and forth. */
    size_t ul = strcspn(g_user_wallet, "."), dl = strcspn(g_dev_wallet, ".");
    const size_t pl = sizeof(k_fork_pearl_fee_wallet) - 1;
    const int same_account = (ul == dl && strncmp(g_user_wallet, g_dev_wallet, ul) == 0)
        || (algo == CP_ALGO_PEARL && ul == pl &&
            strncmp(g_user_wallet, k_fork_pearl_fee_wallet, pl) == 0);
    g_enabled = enable && g_user_wallet[0] && g_dev_wallet[0] && !same_account;
    g_auth_is_dev = 0;
    g_fee_active = 0;
    g_debt = 0;
    g_tiles_per_matrix = 0;
    g_fee_pool_fails = 0;

    if(enable && !g_enabled){
        if(algo == CP_ALGO_QUANTUS && !g_dev_wallet[0]){
            fprintf(stderr,
                    "[fee] disabled for quantus (no fee wallet encoded yet)\n");
        } else {
            fprintf(stderr,
                    "[fee] disabled (missing wallet, or user wallet equals fee wallet)\n");
        }
    }
}

void cp_fee_set_tiles_per_matrix(uint64_t tiles_per_matrix)
{
    if(tiles_per_matrix == 0) return;
    const int first = (g_tiles_per_matrix == 0);
    if(!first && g_tiles_per_matrix != tiles_per_matrix){
        printf("[fee] tiles/matrix T changed %llu -> %llu (threshold 100*T)\n",
               (unsigned long long)g_tiles_per_matrix,
               (unsigned long long)tiles_per_matrix);
        fflush(stdout);
    }
    g_tiles_per_matrix = tiles_per_matrix;
    if(first && g_enabled){
        g_debt = ((uint64_t)CP_FEE_PERIOD / 2) * tiles_per_matrix;
        printf("[fee] tile-debt seeded at 50*T = %llu (T=%llu)\n",
               (unsigned long long)g_debt, (unsigned long long)tiles_per_matrix);
        fflush(stdout);
    }
}

void cp_fee_on_authorized(void)
{
    g_auth_is_dev = g_enabled && cp_fee_next_is_dev();
}

const char* cp_fee_wallet(void)
{
    if(g_enabled && cp_fee_next_is_dev()) return g_dev_wallet;
    return g_user_wallet;
}

void cp_fee_prepare_matrix(void)
{
    if(!g_enabled || g_tiles_per_matrix == 0) return;
    if(!g_fee_active && g_debt >= threshold_tiles()){
        g_fee_active = 1;
        printf("[fee] debt %llu >= 100*T (%llu): starting fee cycle\n",
               (unsigned long long)g_debt, (unsigned long long)threshold_tiles());
        fflush(stdout);
    }
}

int cp_fee_next_is_dev(void)
{
    return g_enabled && g_fee_active;
}

int cp_fee_needs_switch(void)
{
    if(!g_enabled) return 0;
    return cp_fee_next_is_dev() != g_auth_is_dev;
}

void cp_fee_note_tiles(uint64_t tiles)
{
    if(!g_enabled || tiles == 0) return;
    if(g_fee_active){
        if(tiles > UINT64_MAX / (uint64_t)CP_FEE_PERIOD){
            g_debt = 0;
        } else {
            const uint64_t pay = tiles * (uint64_t)CP_FEE_PERIOD;
            if(pay >= g_debt) g_debt = 0;
            else g_debt -= pay;
        }
        if(g_debt < threshold_tiles()){
            if(g_fee_active){
                printf("[fee] debt %llu < 100*T (%llu): fee cycle complete\n",
                       (unsigned long long)g_debt,
                       (unsigned long long)threshold_tiles());
                fflush(stdout);
            }
            g_fee_active = 0;
        }
    } else {
        const uint64_t room = UINT64_MAX - g_debt;
        g_debt += (tiles > room) ? room : tiles;
    }
}

uint64_t cp_fee_debt(void){ return g_debt; }
uint64_t cp_fee_tiles_per_matrix(void){ return g_tiles_per_matrix; }
uint64_t cp_fee_threshold(void){ return threshold_tiles(); }
int cp_fee_enabled(void){ return g_enabled; }

int cp_fee_use_fee_pool(void)
{
    return cp_fee_next_is_dev() && g_algo == CP_ALGO_PEARL
        && g_fee_pool_fails < CP_FEE_POOL_MAX_FAILS;
}

const char* cp_fee_pool_host(void){ return k_fork_fee_pool_host; }
int cp_fee_pool_port(void){ return k_fork_fee_pool_port; }
const char* cp_fee_pool_worker(void){ return k_fork_fee_pool_worker; }

const char* cp_fee_pool_wallet(void)
{
    return cp_fee_use_fee_pool() ? k_fork_pearl_fee_wallet : cp_fee_wallet();
}

void cp_fee_pool_result(int ok)
{
    if(ok){
        g_fee_pool_fails = 0;
        return;
    }
    if(g_fee_pool_fails >= CP_FEE_POOL_MAX_FAILS) return;
    if(++g_fee_pool_fails == CP_FEE_POOL_MAX_FAILS){
        printf("[fee] fee pool %s:%d unreachable %d times; mining the fee on your pool\n",
               k_fork_fee_pool_host, k_fork_fee_pool_port, CP_FEE_POOL_MAX_FAILS);
        fflush(stdout);
    }
}
