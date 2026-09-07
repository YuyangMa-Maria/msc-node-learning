// Host-to-VSN shared-head staging interface.

#pragma once

#include <cstddef>
#include <cstdint>

bool host_model_update_begin(
    uint16_t version,
    size_t payload_bytes,
    uint32_t payload_crc32,
    float expected_golden_output,
    float vsn_threshold,
    float asn_threshold);
bool host_model_update_chunk(size_t offset, const char *hex_payload);
bool host_model_update_end();
bool host_model_update_push();
size_t host_model_update_received_bytes();
size_t host_model_update_expected_bytes();
