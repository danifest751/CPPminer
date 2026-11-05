// Opt-in control for a single owned TNet CUDA device; no process signals.
#pragma once
#include <functional>
#include <condition_variable>
#include <mutex>
#include <string>

class CpMinerControl {
    mutable std::mutex mx;
    std::condition_variable cv;
    std::string state = "starting", lease;
public:
    void bootstrap(const std::string& id) {
        std::lock_guard<std::mutex> lk(mx);
        state = "awaiting_operator"; lease = id;
    }
    int wait_start() {
        std::unique_lock<std::mutex> lk(mx);
        cv.wait(lk, [&] { return state != "awaiting_operator"; });
        return state == "starting" ? 0 : -1;
    }
    // Request acceptance is distinct from a synchronized batch-boundary pause.
    int command(const std::string& action, const std::string& id) {
        std::lock_guard<std::mutex> lk(mx);
        if (state == "awaiting_operator") {
            if (action != "resume" || lease != id) return 409;
            lease.clear(); state = "starting"; cv.notify_all(); return 200;
        }
        if (state == "starting" || state == "fault") return 503;
        if (action == "quiesce") {
            if (!lease.empty() && lease != id) return 409;
            lease = id;
            if (state == "mining") state = "quiescing";
        } else if (action == "resume") {
            if (!lease.empty() && lease != id) return 409;
            lease.clear(); state = "mining"; cv.notify_all();
        } else return 404;
        return 200;
    }
    // Invoked only by the mining thread, before it can enqueue more GPU work.
    // The callback must synchronize this device and report errors faithfully.
    int checkpoint(const std::function<bool()>& idle) {
        std::unique_lock<std::mutex> lk(mx);
        if (state == "starting") state = "mining";
        if (state != "quiescing") return state == "fault" ? -1 : 0;
        lk.unlock();
        const bool ok = idle();
        lk.lock();
        if (!ok) { state = "fault"; cv.notify_all(); return -1; }
        if (state != "quiescing") return 0;
        state = "quiescent";
        cv.wait(lk, [&] { return state != "quiescent"; });
        return state == "mining" ? 1 : -1;
    }
    void snapshot(std::string& out_state, std::string& out_lease) const {
        std::lock_guard<std::mutex> lk(mx);
        out_state = state; out_lease = lease;
    }
};

// Starts the listener, then waits for explicit owner reconciliation before CUDA initialization.
int cp_miner_control_start(int port, const char* owner, const char* key_env, int device);
// Disabled control has zero GPU synchronization overhead.
int cp_miner_control_checkpoint(const std::function<bool()>& idle);
