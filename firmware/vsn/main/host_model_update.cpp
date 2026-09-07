// Stages and validates a MARH model update received over USB serial.
// Receipt, VSN activation and BLE relay are separate phases: an incomplete host
// upload cannot become active locally or be forwarded to the ASN.

#include "host_model_update.h"

#include <cmath>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "model_exchange_protocol.h"
#include "runtime_shared_head.h"
#include "shared_head_payloads.h"
#include "vsn_model_exchange.h"

namespace {

constexpr char kTag[] = "host_model_update";

uint8_t *g_payload = nullptr;
size_t g_capacity = 0;
size_t g_expected_bytes = 0;
size_t g_received_bytes = 0;
uint32_t g_expected_crc32 = 0;
uint16_t g_version = 0;
float g_expected_golden_output = 0.0f;
float g_vsn_threshold = 0.5f;
float g_asn_threshold = 0.5f;
bool g_receiving = false;
bool g_validated = false;

int hex_value(char value)
{
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

bool ensure_capacity(size_t bytes)
{
    if (g_payload != nullptr && g_capacity >= bytes) {
        return true;
    }
    if (g_payload != nullptr) {
        heap_caps_free(g_payload);
        g_payload = nullptr;
        g_capacity = 0;
    }
    g_payload = static_cast<uint8_t *>(heap_caps_malloc(
        model_exchange::kSharedHeadFp32Bytes,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (g_payload != nullptr) {
        g_capacity = model_exchange::kSharedHeadFp32Bytes;
    }
    return g_payload != nullptr && g_capacity >= bytes;
}

}  // namespace

bool host_model_update_begin(
    uint16_t version,
    size_t payload_bytes,
    uint32_t payload_crc32,
    float expected_golden_output,
    float vsn_threshold,
    float asn_threshold)
{
    const auto format = model_exchange::SharedHeadFormat::kInt8Symmetric;
    if (version < 0x0200 ||
        !model_exchange::valid_package_contract(format, payload_bytes) ||
        !std::isfinite(expected_golden_output) ||
        !std::isfinite(vsn_threshold) || !std::isfinite(asn_threshold) ||
        vsn_threshold < 0.0f || vsn_threshold > 1.0f ||
        asn_threshold < 0.0f || asn_threshold > 1.0f ||
        !ensure_capacity(payload_bytes)) {
        return false;
    }
    g_expected_bytes = payload_bytes;
    g_received_bytes = 0;
    g_expected_crc32 = payload_crc32;
    g_version = version;
    g_expected_golden_output = expected_golden_output;
    g_vsn_threshold = vsn_threshold;
    g_asn_threshold = asn_threshold;
    g_receiving = true;
    g_validated = false;
    ESP_LOGI(
        kTag,
        "HOST_MODEL_BEGIN version=0x%04x bytes=%u crc32=0x%08x vsn_threshold=%.3f asn_threshold=%.3f",
        version,
        static_cast<unsigned>(payload_bytes),
        payload_crc32,
        vsn_threshold,
        asn_threshold);
    return true;
}

bool host_model_update_chunk(size_t offset, const char *hex_payload)
{
    if (!g_receiving || hex_payload == nullptr || offset != g_received_bytes) {
        ESP_LOGW(
            kTag,
            "HOST_MODEL_CHUNK_STATE_FAIL receiving=%s offset=%u expected_offset=%u payload=%s",
            g_receiving ? "yes" : "no",
            static_cast<unsigned>(offset),
            static_cast<unsigned>(g_received_bytes),
            hex_payload == nullptr ? "null" : "present");
        return false;
    }
    const size_t hex_chars = std::strlen(hex_payload);
    if (hex_chars == 0 || (hex_chars % 2) != 0 ||
        g_received_bytes + hex_chars / 2 > g_expected_bytes) {
        ESP_LOGW(
            kTag,
            "HOST_MODEL_CHUNK_LENGTH_FAIL hex_chars=%u received=%u expected=%u",
            static_cast<unsigned>(hex_chars),
            static_cast<unsigned>(g_received_bytes),
            static_cast<unsigned>(g_expected_bytes));
        return false;
    }
    for (size_t i = 0; i < hex_chars; i += 2) {
        const int high = hex_value(hex_payload[i]);
        const int low = hex_value(hex_payload[i + 1]);
        if (high < 0 || low < 0) {
            ESP_LOGW(kTag, "HOST_MODEL_CHUNK_HEX_FAIL index=%u", static_cast<unsigned>(i));
            return false;
        }
        g_payload[g_received_bytes++] = static_cast<uint8_t>((high << 4) | low);
    }
    return true;
}

bool host_model_update_end()
{
    if (!g_receiving || g_received_bytes != g_expected_bytes) {
        return false;
    }
    const uint32_t computed_crc32 = model_exchange::crc32_ieee(g_payload, g_received_bytes);
    if (computed_crc32 != g_expected_crc32) {
        ESP_LOGE(
            kTag,
            "HOST_MODEL_CRC_FAIL expected=0x%08x computed=0x%08x",
            g_expected_crc32,
            computed_crc32);
        g_receiving = false;
        return false;
    }
    // CRC proves transport integrity; runtime activation adds an execution-level
    // golden check before changing the active slot.
    RuntimeHeadTransition transition{};
    const bool activated = runtime_shared_head_activate(
        g_payload,
        g_received_bytes,
        model_exchange::SharedHeadFormat::kInt8Symmetric,
        g_version,
        g_vsn_threshold,
        g_expected_golden_output,
        transition);
    g_receiving = false;
    g_validated = activated;
    ESP_LOGI(
        kTag,
        "HOST_MODEL_VALIDATE result=%s version=0x%04x bytes=%u golden_error=%.9f active_generation=%u",
        activated ? "PASS" : "FAIL",
        g_version,
        static_cast<unsigned>(g_received_bytes),
        transition.golden_error,
        static_cast<unsigned>(transition.after.generation));
    return activated;
}

bool host_model_update_push()
{
    // Relay only the candidate already accepted by the VSN. The ASN still runs
    // its own protocol and golden-output checks before installation.
    if (!g_validated) {
        return false;
    }
    const model_exchange::SharedHeadPackage package{
        g_payload,
        g_received_bytes,
        g_expected_crc32,
        g_version,
        model_exchange::SharedHeadFormat::kInt8Symmetric,
        g_expected_golden_output,
    };
    return vsn_model_exchange_push_package(package, g_asn_threshold);
}

size_t host_model_update_received_bytes()
{
    return g_received_bytes;
}

size_t host_model_update_expected_bytes()
{
    return g_expected_bytes;
}
