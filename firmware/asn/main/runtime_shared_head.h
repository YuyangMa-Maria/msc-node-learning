// Runtime API for evaluating and replacing the ASN shared risk head.

#pragma once

#include <cstddef>
#include <cstdint>

#include "model_exchange_protocol.h"

struct RuntimeHeadSnapshot {
    uint16_t version;
    model_exchange::SharedHeadFormat format;
    float threshold;
    uint32_t generation;
};

struct RuntimeHeadTransition {
    RuntimeHeadSnapshot before;
    RuntimeHeadSnapshot after;
    bool had_live_representation;
    float live_logit_before;
    float live_logit_after;
    float golden_output;
    float golden_error;
    int64_t elapsed_us;
};

bool runtime_shared_head_init(
    const model_exchange::SharedHeadPackage &initial_package,
    float local_threshold);
bool runtime_shared_head_activate(
    const uint8_t *payload,
    size_t payload_bytes,
    model_exchange::SharedHeadFormat format,
    uint16_t version,
    float local_threshold,
    float expected_golden_output,
    RuntimeHeadTransition &transition);
bool runtime_shared_head_rollback(RuntimeHeadTransition &transition);
bool runtime_shared_head_evaluate(
    const float *representation,
    size_t values,
    float &logit,
    RuntimeHeadSnapshot &snapshot);
RuntimeHeadSnapshot runtime_shared_head_snapshot();
