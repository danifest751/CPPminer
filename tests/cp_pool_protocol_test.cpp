#include "cp_json_frame.h"
#include "cp_json_text.hpp"
#include "cp_util.h"
#include "cp_pool_session.hpp"
#include "cp_fee.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <thread>

static void test_response_ids()
{
    int id = -1, accepted = 0;
    assert(cp_json_rpc_response("{\"id\":7,\"result\":true,\"error\":null}", &id, &accepted));
    assert(id == 7 && accepted);
    assert(cp_json_rpc_response("{\"result\":{\"id\":99},\"id\":7}", &id, &accepted));
    assert(id == 7 && accepted);
    assert(cp_json_rpc_response("{\"id\":0,\"result\":false}", &id, &accepted));
    assert(id == 0 && !accepted);
    assert(cp_json_rpc_response("{\"id\":7,\"error\":{\"code\":-1}}", &id, &accepted));
    assert(!accepted);
    assert(!cp_json_rpc_response("{\"result\":{\"id\":7}}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":null,\"result\":true}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":\"7\",\"result\":true}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":7.5,\"result\":true}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":2147483648,\"result\":true}", &id, &accepted));
    assert(!cp_json_rpc_response("{\"id\":7,\"method\":\"job\",\"result\":true}", &id, &accepted));
}

static void test_submit_deadlines()
{
    CpPoolSession session;
    assert(session.begin_submit(2, 100));
    assert(!session.finish_submit(999)); // Unrelated responses must not clear the request.
    assert(!session.begin_submit(2, 110)); // Nor may duplicate ids move its deadline.
    assert(session.begin_submit(3, 150));
    assert(session.expired(159.9) == CpPoolSession::Timeout::None);
    assert(session.expired(160) == CpPoolSession::Timeout::Submit);
    assert(session.finish_submit(3)); // An out-of-order ACK leaves id 2 outstanding.
    assert(session.expired(165) == CpPoolSession::Timeout::Submit);
    assert(session.finish_submit(2));
    assert(!session.has_pending_submits());
    assert(session.expired(200) == CpPoolSession::Timeout::None);

    // A fast ACK can arrive on the reader thread before send returns.
    assert(session.begin_submit(4, 200));
    std::thread reader([&] { assert(session.finish_submit(4)); });
    reader.join();
    assert(!session.has_pending_submits());
    assert(session.expired(300) == CpPoolSession::Timeout::None);

    assert(session.begin_submit(5, 300));
    assert(session.finish_submit(5)); // Send failure removes only that request.
    assert(!session.has_pending_submits());
    assert(session.begin_submit(6, 300));
    session.reset(); // Reconnect discards old ids/deadlines.
    assert(!session.finish_submit(6));
    assert(session.expired(400) == CpPoolSession::Timeout::None);
}

static void test_handshake_deadlines()
{
    CpPoolSession session;
    session.begin_authorize(1, 100);
    assert(!session.authorize_response(999, true, true, 110));
    assert(session.expired(129.9) == CpPoolSession::Timeout::None);
    assert(session.expired(130) == CpPoolSession::Timeout::Authorize);

    session.reset();
    session.begin_authorize(2, 200);
    assert(session.authorize_response(2, false, true, 201));
    assert(!session.authorized() && !session.proof_gzip());

    session.reset();
    session.begin_authorize(3, 300);
    assert(session.authorize_response(3, true, true, 305));
    assert(session.authorized() && session.proof_gzip());
    assert(session.expired(334.9) == CpPoolSession::Timeout::None);
    assert(session.expired(335) == CpPoolSession::Timeout::FirstJob);
    session.received_job();
    assert(session.expired(400) == CpPoolSession::Timeout::None);

    // Some pools notify before the authorize ACK; keep the job but still wait for ACK.
    session.reset();
    session.begin_authorize(4, 400);
    session.received_job();
    assert(!session.authorized());
    assert(session.authorize_response(4, true, false, 410));
    assert(session.authorized() && !session.proof_gzip());
    assert(session.expired(500) == CpPoolSession::Timeout::None);

    session.reset();
    session.begin_authorize(5, 500);
    session.received_job();
    assert(session.expired(530) == CpPoolSession::Timeout::Authorize);
    session.reset();
    assert(!session.authorized() && !session.proof_gzip());
}

static void test_fee_pool_fallback()
{
    cp_fee_set_pool_host("prl.kryptex.network");
    cp_fee_init("test-account", 1, CP_ALGO_PEARL);
    cp_fee_set_tiles_per_matrix(1);
    cp_fee_note_tiles(CP_FEE_PERIOD);
    cp_fee_prepare_matrix();
    assert(cp_fee_use_fee_pool());
    cp_fee_pool_result(0);
    cp_fee_pool_result(0);
    assert(cp_fee_use_fee_pool());
    cp_fee_pool_result(1); // A valid first job resets consecutive failures.
    cp_fee_pool_result(0);
    cp_fee_pool_result(0);
    assert(cp_fee_use_fee_pool());
    cp_fee_pool_result(0);
    assert(!cp_fee_use_fee_pool());
    assert(cp_fee_next_is_dev());
    assert(!strcmp(cp_fee_pool_wallet(), cp_fee_wallet()));
    cp_fee_init("test-account", 0, CP_ALGO_PEARL);
}

int main()
{
    test_response_ids();
    test_submit_deadlines();
    test_handshake_deadlines();
    test_fee_pool_fallback();
    size_t len = 0;
    const char* combined = "{\"job_id\":\"a}b\\\"{c\",\"target\":[1,2]}{\"id\":2}";
    assert(cp_json_object_length(combined, strlen(combined), &len) == 1);
    assert(len == strlen("{\"job_id\":\"a}b\\\"{c\",\"target\":[1,2]}"));
    const char* partial = "{\"id\":\"unfinished";
    assert(cp_json_object_length(partial, strlen(partial), &len) == 0);
    assert(cp_json_object_length("{\"id\":[1}}", strlen("{\"id\":[1}}"), &len) == -1);

    char value[16];
    assert(cp_json_str("{\"note\":\"fake \\\"id\\\":\\\"x\\\"\",\"id\":\"a\\\"b\"}",
                       "id", value, sizeof(value)) == 1);
    assert(strcmp(value, "a\"b") == 0);
    assert(cp_json_str("{\"id\":\"12345678901234567\"}", "id", value, sizeof(value)) == 0);
    assert(value[0] == 0);
    assert(cp_json_str("{\"id\":\"unterminated}", "id", value, sizeof(value)) == 0);
    assert(cp_json_num("{\"note\":\"fake \\\"seq\\\":99\",\"seq\":7}", "seq") == 7);

    uint8_t bytes[2] = {};
    assert(cp_hex_to_bytes("0aFf", bytes, 2) == 2 && bytes[0] == 10 && bytes[1] == 255);
    assert(cp_hex_to_bytes("0g", bytes, 2) == 0);
    assert(cp_hex_to_bytes("g0", bytes, 2) == 0);
    assert(cp_json_escape("a\"\\\nb") == "a\\\"\\\\\\u000ab");
    return 0;
}
