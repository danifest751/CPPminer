// Device-wide Intel GPU hardware counters for a few seconds, averaged.
//
// Opens a Level Zero time-based metric streamer on GPU 0 (any process's work
// is counted, so it can watch an OpenCL miner running next to it), reads the
// raw OA reports and prints the mean of every metric over reports in which
// the GPU was busy.
//
//   g++ -O2 -o l0_metric_sample l0_metric_sample.cpp -lze_loader
//   ZET_ENABLE_METRICS=1 ./l0_metric_sample [seconds=10] [group=ComputeBasic]
//
// Needs `sysctl dev.i915.perf_stream_paranoid=0` (or root) and the Intel
// metrics-discovery library.
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x)                                                              \
    do {                                                                      \
        ze_result_t r_ = (x);                                                 \
        if (r_ != ZE_RESULT_SUCCESS) {                                        \
            std::fprintf(stderr, "%s failed: 0x%x\n", #x, (unsigned)r_);      \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static double as_double(const zet_typed_value_t &v) {
    switch (v.type) {
    case ZET_VALUE_TYPE_UINT32: return v.value.ui32;
    case ZET_VALUE_TYPE_UINT64: return (double)v.value.ui64;
    case ZET_VALUE_TYPE_FLOAT32: return v.value.fp32;
    case ZET_VALUE_TYPE_FLOAT64: return v.value.fp64;
    case ZET_VALUE_TYPE_BOOL8: return v.value.b8;
    default: return 0.0;
    }
}

int main(int argc, char **argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 10;
    const std::string want = argc > 2 ? argv[2] : "ComputeBasic";

    CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t n = 1;
    ze_driver_handle_t driver;
    CHECK(zeDriverGet(&n, &driver));
    n = 1;
    ze_device_handle_t device;
    CHECK(zeDeviceGet(driver, &n, &device));

    uint32_t ngroups = 0;
    CHECK(zetMetricGroupGet(device, &ngroups, nullptr));
    std::vector<zet_metric_group_handle_t> groups(ngroups);
    CHECK(zetMetricGroupGet(device, &ngroups, groups.data()));
    zet_metric_group_handle_t group = nullptr;
    for (auto g : groups) {
        zet_metric_group_properties_t p{ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES};
        CHECK(zetMetricGroupGetProperties(g, &p));
        if (want == p.name && (p.samplingType & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED)) {
            group = g;
        }
    }
    if (!group) {
        std::fprintf(stderr, "time-based metric group %s not found\n", want.c_str());
        return 1;
    }
    uint32_t nmetrics = 0;
    CHECK(zetMetricGet(group, &nmetrics, nullptr));
    std::vector<zet_metric_handle_t> metrics(nmetrics);
    CHECK(zetMetricGet(group, &nmetrics, metrics.data()));
    std::vector<std::string> names(nmetrics);
    int busy_idx = -1;
    for (uint32_t i = 0; i < nmetrics; ++i) {
        zet_metric_properties_t p{ZET_STRUCTURE_TYPE_METRIC_PROPERTIES};
        CHECK(zetMetricGetProperties(metrics[i], &p));
        names[i] = std::string(p.name) + (p.resultUnits[0] ? std::string("[") + p.resultUnits + "]" : "");
        if (!std::strcmp(p.name, "GpuBusy")) busy_idx = (int)i;
    }

    ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx;
    CHECK(zeContextCreate(driver, &cd, &ctx));
    CHECK(zetContextActivateMetricGroups(ctx, device, 1, &group));
    zet_metric_streamer_desc_t sd{ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC};
    sd.notifyEveryNReports = 32768;
    sd.samplingPeriod = 10'000'000; // 10 ms
    zet_metric_streamer_handle_t streamer;
    CHECK(zetMetricStreamerOpen(ctx, device, group, &sd, nullptr, &streamer));

    std::vector<uint8_t> raw;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (std::chrono::steady_clock::now() < end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        size_t size = 0;
        CHECK(zetMetricStreamerReadData(streamer, UINT32_MAX, &size, nullptr));
        if (!size) continue;
        const size_t off = raw.size();
        raw.resize(off + size);
        CHECK(zetMetricStreamerReadData(streamer, UINT32_MAX, &size, raw.data() + off));
        raw.resize(off + size);
    }
    CHECK(zetMetricStreamerClose(streamer));

    uint32_t nvalues = 0;
    CHECK(zetMetricGroupCalculateMetricValues(group, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
                                              raw.size(), raw.data(), &nvalues, nullptr));
    std::vector<zet_typed_value_t> values(nvalues);
    CHECK(zetMetricGroupCalculateMetricValues(group, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
                                              raw.size(), raw.data(), &nvalues, values.data()));
    const uint32_t reports = nmetrics ? nvalues / nmetrics : 0;
    std::vector<double> sum(nmetrics, 0.0);
    uint32_t used = 0;
    for (uint32_t r = 0; r < reports; ++r) {
        const zet_typed_value_t *row = &values[(size_t)r * nmetrics];
        if (busy_idx >= 0 && as_double(row[busy_idx]) < 90.0) continue; // idle gaps
        for (uint32_t i = 0; i < nmetrics; ++i) sum[i] += as_double(row[i]);
        ++used;
    }
    std::printf("group=%s reports=%u busy_reports=%u\n", want.c_str(), reports, used);
    for (uint32_t i = 0; i < nmetrics && used; ++i)
        std::printf("%-34s %14.3f\n", names[i].c_str(), sum[i] / used);
    zetContextActivateMetricGroups(ctx, device, 0, nullptr);
    zeContextDestroy(ctx);
    return 0;
}
