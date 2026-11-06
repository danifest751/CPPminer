// TNet on the CPU against Requant's vectors (tnet-v1-small.jsonl by default; give tnet-v1-frozen.jsonl to check the real
// parameters, which needs about 600 MiB): every ticket of a row, and the row itself when all its tickets fit in a scan.
#include "../src/tnet/cp_tnet_cpu.hpp"

#include <algorithm>
#include <fstream>

static std::string field(const std::string& line, const std::string& key) {
    const std::string k = "\"" + key + "\": ";
    size_t at = line.find(k);
    if (at == std::string::npos) return "";
    at += k.size();
    if (line[at] == '"') {
        const size_t end = line.find('"', at + 1);
        return line.substr(at + 1, end - at - 1);
    }
    size_t end = at;
    while (end < line.size() && line[end] != ',' && line[end] != '}') ++end;
    return line.substr(at, end - at);
}

static std::vector<std::string> hex_list(const std::string& line, const std::string& key) {
    std::vector<std::string> out;
    size_t at = line.find("\"" + key + "\": [");
    if (at == std::string::npos) return out;
    const size_t end = line.find(']', at);
    at = line.find('"', line.find('[', at));
    while (at != std::string::npos && at < end) {
        const size_t q = line.find('"', at + 1);
        out.push_back(line.substr(at + 1, q - at - 1));
        at = line.find('"', q + 1);
    }
    return out;
}

static std::vector<uint8_t> from_hex(const std::string& s) {
    std::vector<uint8_t> out(s.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<uint8_t>(strtoul(s.substr(2 * i, 2).c_str(), nullptr, 16));
    return out;
}

static int check_line(const std::string& line, bool portable) {
    RqWork wk;
    wk.n = atoi(field(line, "n").c_str());
    wk.b = atoi(field(line, "b").c_str());
    wk.L = atoi(field(line, "L").c_str());
    wk.w = atoi(field(line, "w").c_str());
    wk.mult = atoi(field(line, "mult").c_str());
    wk.seed = from_hex(field(line, "epoch"));
    wk.digest = from_hex(field(line, "hd"));
    wk.target.assign(32, 0xff);
    const uint64_t nonce = strtoull(field(line, "nonce").c_str(), nullptr, 10);
    const uint32_t row = static_cast<uint32_t>(atoi(field(line, "i").c_str()));
    const std::vector<std::string> want = hex_list(line, "tickets");

    std::unique_ptr<RqDevice> dev = rq_cpu_device(0, portable);
    if (!dev->init(wk, 1)) return 1;
    dev->set_epoch(wk.seed);
    std::vector<RqHit> hits;
    dev->scan(wk, nonce, row, 1, hits);
    const size_t expect = std::min<size_t>(want.size(), RQ_MAX_HITS);
    if (hits.size() != expect) {
        printf("row %u: %zu tickets, expected %zu\n", row, hits.size(), expect);
        return 1;
    }
    std::vector<uint8_t> whole;
    for (const RqHit& h : hits) {
        uint8_t got[32];
        rq_cpu_ticket(reinterpret_cast<const int8_t*>(h.piece.data()), wk.w, wk.digest.data(), nonce, h.row, h.c, got);
        if (h.row != row || to_hex(got, 32) != want[h.c]) {
            printf("row %u: ticket %u differs\n", row, h.c);
            return 1;
        }
        whole.insert(whole.end(), h.piece.begin(), h.piece.end());
    }
    if (hits.size() == want.size()) {
        uint8_t got[32];
        host_sha256(whole, got);
        if (to_hex(got, 32) != field(line, "row_sha256")) {
            printf("row %u: row_sha256 differs\n", row);
            return 1;
        }
    }
    printf("row %u (%s): %zu tickets ok\n", row, portable ? "portable" : dev->kernel().c_str(), hits.size());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: cp_tnet_cpu_test VECTORS.jsonl\n");
        return 2;
    }
    if (rq_cpu_selftest(0) != 0) return 1;
    std::ifstream in(argv[1]);
    std::string line;
    int lines = 0, bad = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        ++lines;
        bad += check_line(line, false);
        // The portable kernel at the real parameters would take minutes and adds nothing to the small vectors.
        if (line.find("\"n\": 256,") != std::string::npos) bad += check_line(line, true);
    }
    if (!lines) {
        fprintf(stderr, "no vectors in %s\n", argv[1]);
        return 2;
    }
    printf("%d vectors, %d failed\n", lines, bad);
    return bad ? 1 : 0;
}
