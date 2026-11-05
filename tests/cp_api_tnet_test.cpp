// CPU-only harness for TNet's public telemetry contract; no CUDA or network mining.
#include "cp_api.h"
#include "cp_tnet.h"
#include "cp_miner_control.h"
#include <chrono>
#include <cstdlib>
#include <thread>
extern "C" {
char worker_global[64] = "legacy";
char agent_global[64] = "cppminer/test";
double cp_now_sec(void) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
extern "C" int cp_tnet_cuda_selftest(int) { return 0; }
extern "C" int cp_tnet_cuda_bench(int, int, double) { return 0; }
extern "C" int cp_tnet_cuda_solo(const char*, const char*, const char* worker, int, int, double, long long) {
    cp_api_set_worker(worker);
    cp_api_set_pool("local-test", 0, 1);
    cp_api_add_device_identity("synthetic device", "0000:01:00.0", "GPU-11111111-2222-3333-4444-555555555555");
    for (;;) {
        if (cp_miner_control_checkpoint([] { return true; }) < 0) return 1;
        cp_api_add_work(1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
int main(int argc, char** argv) {
    if (argc > 2) return cp_tnet_main(argc, argv);
    if (argc != 2) return 2;
    cp_api_set_algo("tnet", "cuda");
    cp_api_set_worker("resource-fixture");
    cp_api_set_pool("local-test", 0, 1);
    cp_api_add_device_identity("synthetic device", "0000:01:00.0", "GPU-11111111-2222-3333-4444-555555555555");
    if (cp_api_start("127.0.0.1", std::atoi(argv[1])) != 0) return 1;
    for (;;) { cp_api_add_work(1000); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
}
