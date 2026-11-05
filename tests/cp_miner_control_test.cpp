#include "cp_miner_control.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <thread>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--serve") {
        assert(cp_miner_control_start(std::atoi(argv[2]), std::string(64, '2').c_str(), "TEST_MINER_CONTROL_KEY", 0) == 0);
        for (;;) {
            assert(cp_miner_control_checkpoint([] { return true; }) >= 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    CpMinerControl startup;
    startup.bootstrap(std::string(64, 'a'));
    auto ready = std::async(std::launch::async, [&] { return startup.wait_start(); });
    assert(ready.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    assert(startup.command("resume", std::string(64, 'b')) == 409);
    assert(startup.command("resume", std::string(64, 'a')) == 200);
    assert(ready.get() == 0);
    CpMinerControl c;
    const std::string lease(64, '3'), other(64, '4');
    assert(c.command("quiesce", lease) == 503);
    assert(c.checkpoint([] { assert(false); return true; }) == 0);
    assert(c.command("quiesce", lease) == 200);
    std::promise<void> entered, gpu_done;
    auto finished = gpu_done.get_future();
    auto worker = std::async(std::launch::async, [&] {
        return c.checkpoint([&] { entered.set_value(); finished.wait(); return true; });
    });
    entered.get_future().wait();
    std::string state, current;
    c.snapshot(state, current);
    assert(state == "quiescing" && current == lease);
    assert(c.command("resume", other) == 409);
    gpu_done.set_value();
    for (int i = 0; i < 100; ++i) {
        c.snapshot(state, current);
        if (state == "quiescent") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    assert(state == "quiescent" && current == lease);
    assert(worker.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    assert(c.command("quiesce", other) == 409);
    assert(c.command("resume", lease) == 200);
    assert(worker.get() == 1);
    c.snapshot(state, current);
    assert(state == "mining" && current.empty());
    assert(c.command("quiesce", lease) == 200);
    assert(c.checkpoint([] { return false; }) == -1);
    assert(c.command("resume", lease) == 503);
    c.snapshot(state, current);
    assert(state == "fault");
    std::puts("host control state tests passed; no CUDA validation claimed");
}
