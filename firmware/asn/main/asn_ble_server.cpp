// NimBLE peripheral for ASN summaries, sampling commands and shared-head transfer.
// Decision notifications and model updates use separate characteristics because
// their frequency, payload size and failure handling are intentionally different.

#include "asn_ble_server.h"
#include "asn_sampling_control.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "esp_err.h"
#include "esp_bt.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "host/ble_att.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "model_exchange_protocol.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "node_summary_protocol.h"
#include "nvs_flash.h"
#include "os/os_mbuf.h"
#include "runtime_shared_head.h"
#include "sampling_control_protocol.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

namespace {

constexpr char kTag[] = "asn_ble";
constexpr char kDeviceName[] = "NL-ASN-01";
constexpr uint16_t kNoConnection = BLE_HS_CONN_HANDLE_NONE;
constexpr int64_t kModelRxInactivityTimeoutUs = 5000000;

const ble_uuid128_t kServiceUuid = BLE_UUID128_INIT(
    0x01, 0xd3, 0x7f, 0x3f, 0x8b, 0x8b, 0x31, 0xa6,
    0x2f, 0x4f, 0x2f, 0x4d, 0x01, 0x00, 0x1a, 0xc5);
const ble_uuid128_t kSummaryUuid = BLE_UUID128_INIT(
    0x01, 0xd3, 0x7f, 0x3f, 0x8b, 0x8b, 0x31, 0xa6,
    0x2f, 0x4f, 0x2f, 0x4d, 0x02, 0x00, 0x1a, 0xc5);
const ble_uuid128_t kAckUuid = BLE_UUID128_INIT(
    0x01, 0xd3, 0x7f, 0x3f, 0x8b, 0x8b, 0x31, 0xa6,
    0x2f, 0x4f, 0x2f, 0x4d, 0x03, 0x00, 0x1a, 0xc5);
const ble_uuid128_t kModelControlUuid = BLE_UUID128_INIT(
    0x01, 0xd3, 0x7f, 0x3f, 0x8b, 0x8b, 0x31, 0xa6,
    0x2f, 0x4f, 0x2f, 0x4d, 0x04, 0x00, 0x1a, 0xc5);
const ble_uuid128_t kModelDataRxUuid = BLE_UUID128_INIT(
    0x01, 0xd3, 0x7f, 0x3f, 0x8b, 0x8b, 0x31, 0xa6,
    0x2f, 0x4f, 0x2f, 0x4d, 0x05, 0x00, 0x1a, 0xc5);
const ble_uuid128_t kModelStatusUuid = BLE_UUID128_INIT(
    0x01, 0xd3, 0x7f, 0x3f, 0x8b, 0x8b, 0x31, 0xa6,
    0x2f, 0x4f, 0x2f, 0x4d, 0x06, 0x00, 0x1a, 0xc5);
const ble_uuid128_t kModelDataTxUuid = BLE_UUID128_INIT(
    0x01, 0xd3, 0x7f, 0x3f, 0x8b, 0x8b, 0x31, 0xa6,
    0x2f, 0x4f, 0x2f, 0x4d, 0x07, 0x00, 0x1a, 0xc5);
const ble_uuid128_t kSamplingControlUuid = BLE_UUID128_INIT(
    0x01, 0xd3, 0x7f, 0x3f, 0x8b, 0x8b, 0x31, 0xa6,
    0x2f, 0x4f, 0x2f, 0x4d, 0x08, 0x00, 0x1a, 0xc5);

enum CharacteristicId : uintptr_t {
    kSummaryCharacteristic = 1,
    kAckCharacteristic = 2,
    kModelControlCharacteristic = 3,
    kModelDataRxCharacteristic = 4,
    kModelStatusCharacteristic = 5,
    kModelDataTxCharacteristic = 6,
    kSamplingControlCharacteristic = 7,
};

// Transfer state is kept until commit or timeout so a partial update never
// replaces the active head. The inference task continues using the last
// validated slot throughout the transfer.
uint8_t g_own_addr_type = 0;
uint16_t g_conn_handle = kNoConnection;
uint16_t g_summary_handle = 0;
bool g_subscribed = false;
uint16_t g_sequence = 0;
uint16_t g_last_tx_sequence = 0;
int64_t g_last_tx_us = 0;
node_learning::NodeSummaryV1 g_last_summary{};
uint8_t *g_model_rx_buffer = nullptr;
size_t g_model_rx_capacity = 0;
size_t g_model_rx_bytes = 0;
size_t g_model_rx_chunk_payload_bytes = 0;
uint16_t g_model_rx_next_chunk = 0;
uint16_t g_model_rx_transfer_id = 0;
uint16_t g_model_rx_model_version = 0;
size_t g_model_rx_expected_bytes = 0;
uint32_t g_model_rx_expected_crc32 = 0;
float g_model_rx_expected_golden_output = 0.0f;
float g_model_rx_target_threshold = 0.5f;
bool g_model_metadata_ready = false;
model_exchange::SharedHeadFormat g_model_rx_format =
    model_exchange::SharedHeadFormat::kFp32;
model_exchange::TransferDirection g_model_rx_direction =
    model_exchange::TransferDirection::kVsnToAsn;
int64_t g_model_rx_last_activity_us = 0;
uint16_t g_model_pull_chunk = 0;
uint16_t g_model_pull_transfer_id = 0;
int64_t g_model_transfer_started_us = 0;
bool g_model_rx_active = false;
bool g_have_completed_push = false;
uint16_t g_last_completed_push_transfer_id = 0;
const model_exchange::SharedHeadPackage *g_model_pull_package = nullptr;
model_exchange::ModelStatusV1 g_model_status{};
bool g_runtime_rollback_tested = false;

void log_runtime_transition(
    const char *action,
    const RuntimeHeadTransition &transition,
    bool pass)
{
    ESP_LOGI(
        kTag,
        "RUNTIME_HEAD_%s result=%s before_version=0x%04x after_version=0x%04x "
        "before_generation=%u after_generation=%u before_format=%s after_format=%s "
        "before_threshold=%.3f after_threshold=%.3f live_ready=%s "
        "live_logit_before=%.9f live_logit_after=%.9f golden_error=%.9f elapsed_us=%lld",
        action,
        pass ? "PASS" : "FAIL",
        transition.before.version,
        transition.after.version,
        static_cast<unsigned>(transition.before.generation),
        static_cast<unsigned>(transition.after.generation),
        model_exchange::format_name(transition.before.format),
        model_exchange::format_name(transition.after.format),
        transition.before.threshold,
        transition.after.threshold,
        transition.had_live_representation ? "yes" : "no",
        transition.live_logit_before,
        transition.live_logit_after,
        transition.golden_error,
        static_cast<long long>(transition.elapsed_us));
}

const model_exchange::SharedHeadPackage *asn_source_package(
    model_exchange::SharedHeadFormat format)
{
    if (format == model_exchange::SharedHeadFormat::kFp32) {
        return &model_exchange::kAsnFederatedFP32;
    }
    if (format == model_exchange::SharedHeadFormat::kInt8Symmetric) {
        return &model_exchange::kAsnFederatedINT8;
    }
    return nullptr;
}

void update_model_status(
    model_exchange::TransferStatus status,
    uint16_t transfer_id,
    uint16_t model_version,
    model_exchange::SharedHeadFormat format,
    model_exchange::TransferDirection direction,
    uint16_t chunks,
    uint16_t payload_bytes,
    uint32_t computed_crc32)
{
    g_model_status = {};
    g_model_status.status = static_cast<uint8_t>(status);
    g_model_status.transfer_id = transfer_id;
    g_model_status.model_version = model_version;
    g_model_status.format = static_cast<uint8_t>(format);
    g_model_status.direction = static_cast<uint8_t>(direction);
    g_model_status.chunks_received = chunks;
    g_model_status.payload_bytes = payload_bytes;
    g_model_status.computed_crc32 = computed_crc32;
    model_exchange::seal(g_model_status);
}

void clear_model_rx_state()
{
    g_model_rx_active = false;
    g_model_rx_bytes = 0;
    g_model_rx_next_chunk = 0;
    g_model_rx_chunk_payload_bytes = 0;
    g_model_rx_expected_bytes = 0;
    g_model_rx_expected_crc32 = 0;
    g_model_rx_expected_golden_output = 0.0f;
    g_model_rx_target_threshold = 0.5f;
    g_model_metadata_ready = false;
    g_model_rx_last_activity_us = 0;
}

void reject_model_rx(
    model_exchange::TransferStatus status,
    const char *reason,
    uint16_t transfer_id,
    uint16_t model_version,
    model_exchange::SharedHeadFormat format,
    model_exchange::TransferDirection direction,
    uint32_t computed_crc32 = 0)
{
    const RuntimeHeadSnapshot active = runtime_shared_head_snapshot();
    update_model_status(
        status,
        transfer_id,
        model_version,
        format,
        direction,
        g_model_rx_next_chunk,
        static_cast<uint16_t>(g_model_rx_bytes),
        computed_crc32);
    ESP_LOGW(
        kTag,
        "MODEL_RX_REJECT reason=%s status=%s transfer=%u model_version=0x%04x "
        "chunks=%u bytes=%u active_version=0x%04x active_generation=%u preserved=yes",
        reason,
        model_exchange::status_name(status),
        transfer_id,
        model_version,
        g_model_rx_next_chunk,
        static_cast<unsigned>(g_model_rx_bytes),
        active.version,
        static_cast<unsigned>(active.generation));
    clear_model_rx_state();
}

bool expire_model_rx_if_needed()
{
    if (!g_model_rx_active || g_model_rx_last_activity_us == 0) {
        return false;
    }
    const int64_t idle_us = esp_timer_get_time() - g_model_rx_last_activity_us;
    if (idle_us <= kModelRxInactivityTimeoutUs) {
        return false;
    }
    ESP_LOGW(
        kTag,
        "MODEL_RX_TIMEOUT transfer=%u idle_ms=%.3f limit_ms=%.3f",
        g_model_rx_transfer_id,
        idle_us / 1000.0,
        kModelRxInactivityTimeoutUs / 1000.0);
    reject_model_rx(
        model_exchange::TransferStatus::kTimeout,
        "inactivity_timeout",
        g_model_rx_transfer_id,
        g_model_rx_model_version,
        g_model_rx_format,
        g_model_rx_direction);
    return true;
}

int copy_mbuf(os_mbuf *source, void *destination, size_t capacity, uint16_t &copied)
{
    copied = 0;
    return ble_hs_mbuf_to_flat(source, destination, capacity, &copied);
}

int handle_model_control(uint16_t conn_handle, ble_gatt_access_ctxt *context)
{
    model_exchange::ModelControlV1 control{};
    uint16_t copied = 0;
    if (copy_mbuf(context->om, &control, sizeof(control), copied) != 0 ||
        copied != sizeof(control) || !model_exchange::validate(control)) {
        ESP_LOGE(kTag, "MODEL_CONTROL_INVALID");
        return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    }

    const auto format = static_cast<model_exchange::SharedHeadFormat>(control.format);
    const auto direction = static_cast<model_exchange::TransferDirection>(control.direction);
    const auto opcode = static_cast<model_exchange::ControlOpcode>(control.opcode);
    const size_t chunk_payload_bytes = model_exchange::chunk_payload_capacity(
        ble_att_mtu(conn_handle));

    expire_model_rx_if_needed();

    if (opcode == model_exchange::ControlOpcode::kPreparePushMetadata) {
        if (direction != model_exchange::TransferDirection::kVsnToAsn ||
            control.model_version < 0x0200 ||
            !model_exchange::valid_package_contract(format, control.payload_bytes)) {
            reject_model_rx(
                control.model_version < 0x0200
                    ? model_exchange::TransferStatus::kVersionFailure
                    : model_exchange::TransferStatus::kProtocolFailure,
                "metadata_contract",
                control.transfer_id,
                control.model_version,
                format,
                direction);
            return BLE_ATT_ERR_UNLIKELY;
        }
        if (g_model_rx_active) {
            reject_model_rx(
                model_exchange::TransferStatus::kAborted,
                "superseded_by_metadata",
                g_model_rx_transfer_id,
                g_model_rx_model_version,
                g_model_rx_format,
                g_model_rx_direction);
        }
        float expected_golden_output = 0.0f;
        std::memcpy(
            &expected_golden_output,
            &control.payload_crc32,
            sizeof(expected_golden_output));
        const float target_threshold = control.chunk_index / 65535.0f;
        if (!std::isfinite(expected_golden_output) || !std::isfinite(target_threshold)) {
            reject_model_rx(
                model_exchange::TransferStatus::kProtocolFailure,
                "metadata_values",
                control.transfer_id,
                control.model_version,
                format,
                direction);
            return BLE_ATT_ERR_UNLIKELY;
        }
        g_model_rx_transfer_id = control.transfer_id;
        g_model_rx_model_version = control.model_version;
        g_model_rx_format = format;
        g_model_rx_direction = direction;
        g_model_rx_expected_bytes = control.payload_bytes;
        g_model_rx_expected_golden_output = expected_golden_output;
        g_model_rx_target_threshold = target_threshold;
        g_model_metadata_ready = true;
        update_model_status(
            model_exchange::TransferStatus::kReady,
            control.transfer_id,
            control.model_version,
            format,
            direction,
            0,
            0,
            0);
        ESP_LOGI(
            kTag,
            "MODEL_RX_METADATA transfer=%u version=0x%04x bytes=%u expected=%.9f threshold=%.6f",
            control.transfer_id,
            control.model_version,
            control.payload_bytes,
            expected_golden_output,
            target_threshold);
        return 0;
    }

    if (opcode == model_exchange::ControlOpcode::kBeginPush) {
        if (control.model_version < 0x0200) {
            reject_model_rx(
                model_exchange::TransferStatus::kVersionFailure,
                "model_version_mismatch",
                control.transfer_id,
                control.model_version,
                format,
                direction);
            return BLE_ATT_ERR_UNLIKELY;
        }
        if (direction != model_exchange::TransferDirection::kVsnToAsn ||
            !model_exchange::valid_package_contract(format, control.payload_bytes) ||
            chunk_payload_bytes == 0 || !g_model_metadata_ready ||
            control.transfer_id != g_model_rx_transfer_id ||
            control.model_version != g_model_rx_model_version ||
            format != g_model_rx_format || control.payload_bytes != g_model_rx_expected_bytes) {
            reject_model_rx(
                model_exchange::TransferStatus::kProtocolFailure,
                "begin_contract",
                control.transfer_id,
                control.model_version,
                format,
                direction);
            return BLE_ATT_ERR_UNLIKELY;
        }
        if (g_have_completed_push &&
            control.transfer_id <= g_last_completed_push_transfer_id) {
            reject_model_rx(
                model_exchange::TransferStatus::kReplayFailure,
                "transfer_id_replay",
                control.transfer_id,
                control.model_version,
                format,
                direction);
            return BLE_ATT_ERR_UNLIKELY;
        }
        if (g_model_rx_active) {
            reject_model_rx(
                model_exchange::TransferStatus::kAborted,
                "superseded_by_new_begin",
                g_model_rx_transfer_id,
                g_model_rx_model_version,
                g_model_rx_format,
                g_model_rx_direction);
        }
        if (g_model_rx_buffer == nullptr || g_model_rx_capacity < control.payload_bytes) {
            if (g_model_rx_buffer != nullptr) {
                heap_caps_free(g_model_rx_buffer);
            }
            g_model_rx_buffer = static_cast<uint8_t *>(heap_caps_malloc(
                model_exchange::kSharedHeadFp32Bytes,
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
            g_model_rx_capacity = g_model_rx_buffer == nullptr
                ? 0
                : model_exchange::kSharedHeadFp32Bytes;
        }
        if (g_model_rx_buffer == nullptr) {
            ESP_LOGE(kTag, "MODEL_RX_ALLOCATION_FAIL bytes=%u", control.payload_bytes);
            return BLE_ATT_ERR_INSUFFICIENT_RES;
        }
        g_model_rx_bytes = 0;
        g_model_rx_next_chunk = 0;
        g_model_rx_transfer_id = control.transfer_id;
        g_model_rx_model_version = control.model_version;
        g_model_rx_expected_bytes = control.payload_bytes;
        g_model_rx_expected_crc32 = control.payload_crc32;
        g_model_rx_format = format;
        g_model_rx_direction = direction;
        g_model_rx_chunk_payload_bytes = chunk_payload_bytes;
        g_model_transfer_started_us = esp_timer_get_time();
        g_model_rx_last_activity_us = g_model_transfer_started_us;
        g_model_rx_active = true;
        update_model_status(
            model_exchange::TransferStatus::kReceiving,
            control.transfer_id,
            control.model_version,
            format,
            direction,
            0,
            0,
            0);
        ESP_LOGI(
            kTag,
            "MODEL_RX_BEGIN transfer=%u format=%s version=0x%04x bytes=%u "
            "crc32=0x%08x expected=%.9f threshold=%.3f mtu=%u chunk_payload=%u",
            control.transfer_id,
            model_exchange::format_name(format),
            control.model_version,
            control.payload_bytes,
            control.payload_crc32,
            g_model_rx_expected_golden_output,
            g_model_rx_target_threshold,
            ble_att_mtu(conn_handle),
            static_cast<unsigned>(chunk_payload_bytes));
        return 0;
    }

    if (opcode == model_exchange::ControlOpcode::kCommitPush) {
        if (!g_model_rx_active ||
            direction != model_exchange::TransferDirection::kVsnToAsn ||
            control.transfer_id != g_model_rx_transfer_id ||
            control.model_version != g_model_rx_model_version ||
            format != g_model_rx_format ||
            control.payload_bytes != g_model_rx_expected_bytes ||
            control.payload_crc32 != g_model_rx_expected_crc32 ||
            g_model_rx_bytes != g_model_rx_expected_bytes) {
            reject_model_rx(
                model_exchange::TransferStatus::kProtocolFailure,
                "incomplete_commit",
                control.transfer_id,
                control.model_version,
                format,
                direction);
            return BLE_ATT_ERR_UNLIKELY;
        }
        const uint32_t computed_crc = model_exchange::crc32_ieee(
            g_model_rx_buffer,
            g_model_rx_bytes);
        float output = 0.0f;
        const bool crc_pass = computed_crc == g_model_rx_expected_crc32 &&
            computed_crc == control.payload_crc32;
        const bool execute_pass = crc_pass && model_exchange::evaluate_shared_head(
            g_model_rx_buffer,
            g_model_rx_bytes,
            format,
            output);
        const float absolute_error = execute_pass
            ? std::fabs(output - g_model_rx_expected_golden_output)
            : INFINITY;
        const bool parity_pass = execute_pass && absolute_error < 1.0e-4f;
        RuntimeHeadTransition activation{};
        bool activation_pass = parity_pass && runtime_shared_head_activate(
            g_model_rx_buffer,
            g_model_rx_bytes,
            format,
            control.model_version,
            g_model_rx_target_threshold,
            g_model_rx_expected_golden_output,
            activation);
        log_runtime_transition("ACTIVATE", activation, activation_pass);
        bool rollback_pass = true;
        bool reactivate_pass = true;
        if (activation_pass && !g_runtime_rollback_tested &&
            format == model_exchange::SharedHeadFormat::kInt8Symmetric) {
            RuntimeHeadTransition rollback{};
            rollback_pass = runtime_shared_head_rollback(rollback);
            log_runtime_transition("ROLLBACK", rollback, rollback_pass);
            RuntimeHeadTransition reactivate{};
            reactivate_pass = rollback_pass && runtime_shared_head_activate(
                g_model_rx_buffer,
                g_model_rx_bytes,
                format,
                control.model_version,
                g_model_rx_target_threshold,
                g_model_rx_expected_golden_output,
                reactivate);
            log_runtime_transition("REACTIVATE", reactivate, reactivate_pass);
            g_runtime_rollback_tested = rollback_pass && reactivate_pass;
        }
        const bool install_pass = activation_pass && rollback_pass && reactivate_pass;
        const auto final_status = !crc_pass
            ? model_exchange::TransferStatus::kCrcFailure
            : parity_pass && install_pass
                ? model_exchange::TransferStatus::kComplete
                : model_exchange::TransferStatus::kExecutionFailure;
        update_model_status(
            final_status,
            control.transfer_id,
            control.model_version,
            format,
            direction,
            g_model_rx_next_chunk,
            static_cast<uint16_t>(g_model_rx_bytes),
            computed_crc);
        const double elapsed_ms = (esp_timer_get_time() - g_model_transfer_started_us) / 1000.0;
        const double throughput_bps = elapsed_ms > 0.0
            ? static_cast<double>(g_model_rx_bytes) * 1000.0 / elapsed_ms
            : 0.0;
        ESP_LOGI(
            kTag,
            "MODEL_RX_COMPLETE transfer=%u direction=VSN_TO_ASN format=%s bytes=%u "
            "chunks=%u elapsed_ms=%.3f throughput_Bps=%.3f crc=%s execute=%s "
            "output=%.9f expected=%.9f abs_error=%.9f staged=yes hot_install=%s active_version=0x%04x",
            control.transfer_id,
            model_exchange::format_name(format),
            static_cast<unsigned>(g_model_rx_bytes),
            g_model_rx_next_chunk,
            elapsed_ms,
            throughput_bps,
            crc_pass ? "PASS" : "FAIL",
            parity_pass ? "PASS" : "FAIL",
            output,
            g_model_rx_expected_golden_output,
            absolute_error,
            install_pass ? "PASS" : "FAIL",
            runtime_shared_head_snapshot().version);
        if (parity_pass && install_pass) {
            g_have_completed_push = true;
            g_last_completed_push_transfer_id = control.transfer_id;
            clear_model_rx_state();
        } else {
            reject_model_rx(
                !crc_pass
                    ? model_exchange::TransferStatus::kCrcFailure
                    : model_exchange::TransferStatus::kExecutionFailure,
                !crc_pass ? "payload_crc32" : "execution_or_activation",
                control.transfer_id,
                control.model_version,
                format,
                direction,
                computed_crc);
        }
        return parity_pass && install_pass ? 0 : BLE_ATT_ERR_UNLIKELY;
    }

    if (opcode == model_exchange::ControlOpcode::kPreparePull) {
        g_model_pull_package = asn_source_package(format);
        if (direction != model_exchange::TransferDirection::kAsnToVsn ||
            g_model_pull_package == nullptr ||
            control.payload_bytes != g_model_pull_package->size ||
            control.payload_crc32 != g_model_pull_package->crc32 ||
            chunk_payload_bytes == 0) {
            ESP_LOGE(kTag, "MODEL_TX_PREPARE_CONTRACT_FAIL transfer=%u", control.transfer_id);
            g_model_pull_package = nullptr;
            return BLE_ATT_ERR_UNLIKELY;
        }
        g_model_pull_transfer_id = control.transfer_id;
        g_model_pull_chunk = 0;
        g_model_transfer_started_us = esp_timer_get_time();
        update_model_status(
            model_exchange::TransferStatus::kReady,
            control.transfer_id,
            g_model_pull_package->model_version,
            format,
            direction,
            0,
            static_cast<uint16_t>(g_model_pull_package->size),
            g_model_pull_package->crc32);
        ESP_LOGI(
            kTag,
            "MODEL_TX_READY transfer=%u direction=ASN_TO_VSN format=%s version=0x%04x "
            "bytes=%u crc32=0x%08x mtu=%u chunk_payload=%u",
            control.transfer_id,
            model_exchange::format_name(format),
            g_model_pull_package->model_version,
            static_cast<unsigned>(g_model_pull_package->size),
            g_model_pull_package->crc32,
            ble_att_mtu(conn_handle),
            static_cast<unsigned>(chunk_payload_bytes));
        return 0;
    }

    if (opcode == model_exchange::ControlOpcode::kSelectPullChunk) {
        if (direction != model_exchange::TransferDirection::kAsnToVsn ||
            g_model_pull_package == nullptr ||
            control.transfer_id != g_model_pull_transfer_id ||
            control.chunk_index >= model_exchange::chunk_count(
                g_model_pull_package->size,
                chunk_payload_bytes)) {
            ESP_LOGE(kTag, "MODEL_TX_CHUNK_SELECT_FAIL transfer=%u chunk=%u",
                control.transfer_id, control.chunk_index);
            return BLE_ATT_ERR_UNLIKELY;
        }
        g_model_pull_chunk = control.chunk_index;
        return 0;
    }

    return BLE_ATT_ERR_UNLIKELY;
}

int handle_model_chunk_write(uint16_t conn_handle, ble_gatt_access_ctxt *context)
{
    expire_model_rx_if_needed();
    uint8_t packet[model_exchange::kPreferredMtu - 3]{};
    uint16_t copied = 0;
    if (!g_model_rx_active ||
        copy_mbuf(context->om, packet, sizeof(packet), copied) != 0) {
        if (g_model_rx_active) {
            reject_model_rx(
                model_exchange::TransferStatus::kProtocolFailure,
                "chunk_copy",
                g_model_rx_transfer_id,
                g_model_rx_model_version,
                g_model_rx_format,
                g_model_rx_direction);
        }
        return BLE_ATT_ERR_UNLIKELY;
    }
    model_exchange::ModelChunkHeaderV1 header{};
    const uint8_t *payload = nullptr;
    if (!model_exchange::parse_chunk(packet, copied, header, payload)) {
        ESP_LOGE(kTag, "MODEL_RX_CHUNK_VALIDATION_FAIL transfer=%u expected_chunk=%u",
            g_model_rx_transfer_id, g_model_rx_next_chunk);
        reject_model_rx(
            model_exchange::TransferStatus::kProtocolFailure,
            "chunk_crc_or_header",
            g_model_rx_transfer_id,
            g_model_rx_model_version,
            g_model_rx_format,
            g_model_rx_direction);
        return BLE_ATT_ERR_UNLIKELY;
    }
    if (header.transfer_id != g_model_rx_transfer_id ||
        header.chunk_index != g_model_rx_next_chunk ||
        header.payload_bytes > g_model_rx_chunk_payload_bytes) {
        ESP_LOGE(kTag, "MODEL_RX_CHUNK_FAIL transfer=%u expected_chunk=%u",
            g_model_rx_transfer_id, g_model_rx_next_chunk);
        reject_model_rx(
            model_exchange::TransferStatus::kProtocolFailure,
            "chunk_order_or_transfer",
            g_model_rx_transfer_id,
            g_model_rx_model_version,
            g_model_rx_format,
            g_model_rx_direction);
        return BLE_ATT_ERR_UNLIKELY;
    }
    const size_t offset = static_cast<size_t>(header.chunk_index) *
        g_model_rx_chunk_payload_bytes;
    if (offset != g_model_rx_bytes ||
        offset + header.payload_bytes > g_model_rx_capacity) {
        ESP_LOGE(kTag, "MODEL_RX_CHUNK_RANGE_FAIL chunk=%u offset=%u bytes=%u",
            header.chunk_index, static_cast<unsigned>(offset), header.payload_bytes);
        reject_model_rx(
            model_exchange::TransferStatus::kProtocolFailure,
            "chunk_range",
            g_model_rx_transfer_id,
            g_model_rx_model_version,
            g_model_rx_format,
            g_model_rx_direction);
        return BLE_ATT_ERR_UNLIKELY;
    }
    std::memcpy(g_model_rx_buffer + offset, payload, header.payload_bytes);
    g_model_rx_bytes += header.payload_bytes;
    ++g_model_rx_next_chunk;
    g_model_rx_last_activity_us = esp_timer_get_time();
    if (g_model_rx_next_chunk % 10 == 0 || header.payload_bytes < g_model_rx_chunk_payload_bytes) {
        ESP_LOGI(kTag, "MODEL_RX_PROGRESS transfer=%u chunks=%u bytes=%u",
            g_model_rx_transfer_id,
            g_model_rx_next_chunk,
            static_cast<unsigned>(g_model_rx_bytes));
    }
    return 0;
}

int handle_model_chunk_read(uint16_t conn_handle, ble_gatt_access_ctxt *context)
{
    if (g_model_pull_package == nullptr) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    const size_t chunk_payload_bytes = model_exchange::chunk_payload_capacity(
        ble_att_mtu(conn_handle));
    const size_t offset = static_cast<size_t>(g_model_pull_chunk) * chunk_payload_bytes;
    if (chunk_payload_bytes == 0 || offset >= g_model_pull_package->size) {
        return BLE_ATT_ERR_INVALID_OFFSET;
    }
    const size_t bytes = std::min(
        chunk_payload_bytes,
        g_model_pull_package->size - offset);
    uint8_t packet[model_exchange::kPreferredMtu - 3]{};
    const size_t packet_bytes = model_exchange::build_chunk(
        packet,
        sizeof(packet),
        g_model_pull_transfer_id,
        g_model_pull_chunk,
        g_model_pull_package->data + offset,
        bytes);
    if (packet_bytes == 0 || os_mbuf_append(context->om, packet, packet_bytes) != 0) {
        return BLE_ATT_ERR_INSUFFICIENT_RES;
    }
    const size_t total_chunks = model_exchange::chunk_count(
        g_model_pull_package->size,
        chunk_payload_bytes);
    if (static_cast<size_t>(g_model_pull_chunk) + 1 == total_chunks) {
        const double elapsed_ms = (esp_timer_get_time() - g_model_transfer_started_us) / 1000.0;
        const double throughput_bps = elapsed_ms > 0.0
            ? static_cast<double>(g_model_pull_package->size) * 1000.0 / elapsed_ms
            : 0.0;
        ESP_LOGI(
            kTag,
            "MODEL_TX_COMPLETE transfer=%u direction=ASN_TO_VSN format=%s bytes=%u "
            "chunks=%u elapsed_ms=%.3f throughput_Bps=%.3f",
            g_model_pull_transfer_id,
            model_exchange::format_name(g_model_pull_package->format),
            static_cast<unsigned>(g_model_pull_package->size),
            static_cast<unsigned>(total_chunks),
            elapsed_ms,
            throughput_bps);
    }
    return 0;
}

int gatt_access(
    uint16_t conn_handle,
    uint16_t attr_handle,
    ble_gatt_access_ctxt *context,
    void *argument)
{
    const uintptr_t characteristic = reinterpret_cast<uintptr_t>(argument);
    if (characteristic == kSummaryCharacteristic && context->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        return os_mbuf_append(
            context->om,
            &g_last_summary,
            sizeof(g_last_summary)) == 0
            ? 0
            : BLE_ATT_ERR_INSUFFICIENT_RES;
    }

    if (characteristic == kAckCharacteristic && context->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        uint16_t acknowledged_sequence = 0;
        uint16_t copied = 0;
        const int result = ble_hs_mbuf_to_flat(
            context->om,
            &acknowledged_sequence,
            sizeof(acknowledged_sequence),
            &copied);
        if (result != 0 || copied != sizeof(acknowledged_sequence)) {
            return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
        }
        const double rtt_ms = acknowledged_sequence == g_last_tx_sequence
            ? (esp_timer_get_time() - g_last_tx_us) / 1000.0
            : -1.0;
        ESP_LOGI(
            kTag,
            "BLE_ACK sequence=%u expected=%u rtt_ms=%.3f matched=%s",
            acknowledged_sequence,
            g_last_tx_sequence,
            rtt_ms,
            acknowledged_sequence == g_last_tx_sequence ? "yes" : "no");
        return 0;
    }

    if (characteristic == kModelControlCharacteristic &&
        context->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        return handle_model_control(conn_handle, context);
    }
    if (characteristic == kModelDataRxCharacteristic &&
        context->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        return handle_model_chunk_write(conn_handle, context);
    }
    if (characteristic == kModelStatusCharacteristic &&
        context->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        expire_model_rx_if_needed();
        return os_mbuf_append(context->om, &g_model_status, sizeof(g_model_status)) == 0
            ? 0
            : BLE_ATT_ERR_INSUFFICIENT_RES;
    }
    if (characteristic == kModelDataTxCharacteristic &&
        context->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        return handle_model_chunk_read(conn_handle, context);
    }
    if (characteristic == kSamplingControlCharacteristic &&
        context->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        sampling_control::MessageV1 message{};
        uint16_t copied = 0;
        const int result = ble_hs_mbuf_to_flat(
            context->om,
            &message,
            sizeof(message),
            &copied);
        if (result != 0 || copied != sizeof(message) ||
            !sampling_control::validate(message)) {
            ESP_LOGW(kTag, "SAMPLING_CONTROL_REJECT reason=invalid_payload bytes=%u", copied);
            return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
        }
        const auto command = static_cast<sampling_control::Command>(message.command);
        const bool applied = asn_sampling_apply(command, message.request_id);
        ESP_LOGI(
            kTag,
            "SAMPLING_CONTROL_RX command=%s request=%u result=%s",
            sampling_control::command_name(command),
            message.request_id,
            applied ? "PASS" : "FAIL");
        return applied ? 0 : BLE_ATT_ERR_UNLIKELY;
    }

    ESP_LOGW(
        kTag,
        "Unsupported GATT access op=%d attr_handle=%u conn_handle=%u",
        context->op,
        attr_handle,
        conn_handle);
    return BLE_ATT_ERR_UNLIKELY;
}

const ble_gatt_chr_def kCharacteristics[] = {
    {
        .uuid = &kSummaryUuid.u,
        .access_cb = gatt_access,
        .arg = reinterpret_cast<void *>(kSummaryCharacteristic),
        .descriptors = nullptr,
        .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
        .min_key_size = 0,
        .val_handle = &g_summary_handle,
        .cpfd = nullptr,
    },
    {
        .uuid = &kAckUuid.u,
        .access_cb = gatt_access,
        .arg = reinterpret_cast<void *>(kAckCharacteristic),
        .descriptors = nullptr,
        .flags = BLE_GATT_CHR_F_WRITE,
        .min_key_size = 0,
        .val_handle = nullptr,
        .cpfd = nullptr,
    },
    {
        .uuid = &kModelControlUuid.u,
        .access_cb = gatt_access,
        .arg = reinterpret_cast<void *>(kModelControlCharacteristic),
        .descriptors = nullptr,
        .flags = BLE_GATT_CHR_F_WRITE,
        .min_key_size = 0,
        .val_handle = nullptr,
        .cpfd = nullptr,
    },
    {
        .uuid = &kModelDataRxUuid.u,
        .access_cb = gatt_access,
        .arg = reinterpret_cast<void *>(kModelDataRxCharacteristic),
        .descriptors = nullptr,
        .flags = BLE_GATT_CHR_F_WRITE,
        .min_key_size = 0,
        .val_handle = nullptr,
        .cpfd = nullptr,
    },
    {
        .uuid = &kModelStatusUuid.u,
        .access_cb = gatt_access,
        .arg = reinterpret_cast<void *>(kModelStatusCharacteristic),
        .descriptors = nullptr,
        .flags = BLE_GATT_CHR_F_READ,
        .min_key_size = 0,
        .val_handle = nullptr,
        .cpfd = nullptr,
    },
    {
        .uuid = &kModelDataTxUuid.u,
        .access_cb = gatt_access,
        .arg = reinterpret_cast<void *>(kModelDataTxCharacteristic),
        .descriptors = nullptr,
        .flags = BLE_GATT_CHR_F_READ,
        .min_key_size = 0,
        .val_handle = nullptr,
        .cpfd = nullptr,
    },
    {
        .uuid = &kSamplingControlUuid.u,
        .access_cb = gatt_access,
        .arg = reinterpret_cast<void *>(kSamplingControlCharacteristic),
        .descriptors = nullptr,
        .flags = BLE_GATT_CHR_F_WRITE,
        .min_key_size = 0,
        .val_handle = nullptr,
        .cpfd = nullptr,
    },
    {},
};

const ble_gatt_svc_def kServices[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &kServiceUuid.u,
        .includes = nullptr,
        .characteristics = kCharacteristics,
    },
    {},
};

void advertise();

int gap_event(ble_gap_event *event, void *)
{
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            g_conn_handle = event->connect.conn_handle;
            ESP_LOGI(kTag, "BLE_CONNECTED conn_handle=%u", g_conn_handle);
        } else {
            ESP_LOGW(kTag, "BLE_CONNECT_FAILED status=%d", event->connect.status);
            advertise();
        }
        return 0;

    case BLE_GAP_EVENT_DISCONNECT:
        ESP_LOGW(kTag, "BLE_DISCONNECTED reason=%d", event->disconnect.reason);
        if (g_model_rx_active) {
            reject_model_rx(
                model_exchange::TransferStatus::kAborted,
                "link_disconnect",
                g_model_rx_transfer_id,
                g_model_rx_model_version,
                g_model_rx_format,
                g_model_rx_direction);
        }
        g_conn_handle = kNoConnection;
        g_subscribed = false;
        clear_model_rx_state();
        g_model_pull_package = nullptr;
        advertise();
        return 0;

    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == g_summary_handle) {
            g_subscribed = event->subscribe.cur_notify != 0;
            ESP_LOGI(
                kTag,
                "BLE_SUBSCRIPTION notify=%s conn_handle=%u",
                g_subscribed ? "enabled" : "disabled",
                event->subscribe.conn_handle);
        }
        return 0;

    case BLE_GAP_EVENT_ADV_COMPLETE:
        advertise();
        return 0;

    case BLE_GAP_EVENT_MTU:
        ESP_LOGI(kTag, "BLE_MTU value=%u", event->mtu.value);
        return 0;

    default:
        return 0;
    }
}

void advertise()
{
    ble_hs_adv_fields fields{};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.uuids128 = const_cast<ble_uuid128_t *>(&kServiceUuid);
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;
    int result = ble_gap_adv_set_fields(&fields);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_ADV_FIELDS_FAIL rc=%d", result);
        return;
    }

    ble_hs_adv_fields response{};
    response.name = reinterpret_cast<uint8_t *>(const_cast<char *>(kDeviceName));
    response.name_len = std::strlen(kDeviceName);
    response.name_is_complete = 1;
    result = ble_gap_adv_rsp_set_fields(&response);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_SCAN_RESPONSE_FAIL rc=%d", result);
        return;
    }

    ble_gap_adv_params parameters{};
    parameters.conn_mode = BLE_GAP_CONN_MODE_UND;
    parameters.disc_mode = BLE_GAP_DISC_MODE_GEN;
    result = ble_gap_adv_start(
        g_own_addr_type,
        nullptr,
        BLE_HS_FOREVER,
        &parameters,
        gap_event,
        nullptr);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_ADV_START_FAIL rc=%d", result);
        return;
    }
    ESP_LOGI(kTag, "BLE_ADVERTISING name=%s", kDeviceName);
}

void on_reset(int reason)
{
    ESP_LOGE(kTag, "BLE_HOST_RESET reason=%d", reason);
}

void on_sync()
{
    int result = ble_hs_util_ensure_addr(0);
    if (result == 0) {
        result = ble_hs_id_infer_auto(0, &g_own_addr_type);
    }
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_ADDRESS_FAIL rc=%d", result);
        return;
    }
    advertise();
}

void host_task(void *)
{
    ESP_LOGI(kTag, "BLE_HOST_TASK_STARTED");
    nimble_port_run();
    nimble_port_freertos_deinit();
}

}  // namespace

bool asn_ble_init()
{
    esp_err_t result = nvs_flash_init();
    if (result == ESP_ERR_NVS_NO_FREE_PAGES ||
        result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        result = nvs_flash_init();
    }
    if (result != ESP_OK) {
        ESP_LOGE(kTag, "BLE_NVS_INIT_FAIL result=%s", esp_err_to_name(result));
        return false;
    }

    result = nimble_port_init();
    if (result != ESP_OK) {
        ESP_LOGE(kTag, "BLE_INIT_FAIL result=%s", esp_err_to_name(result));
        return false;
    }
    const esp_err_t default_power = esp_ble_tx_power_set(
        ESP_BLE_PWR_TYPE_DEFAULT,
        ESP_PWR_LVL_P9);
    const esp_err_t advertising_power = esp_ble_tx_power_set(
        ESP_BLE_PWR_TYPE_ADV,
        ESP_PWR_LVL_P9);
    ESP_LOGI(
        kTag,
        "BLE_TX_POWER default=%s advertising=%s level=+9dBm",
        esp_err_to_name(default_power),
        esp_err_to_name(advertising_power));

    ble_hs_cfg.reset_cb = on_reset;
    ble_hs_cfg.sync_cb = on_sync;
    if (ble_att_set_preferred_mtu(model_exchange::kPreferredMtu) != 0) {
        ESP_LOGE(kTag, "BLE_MTU_CONFIG_FAIL value=%u", model_exchange::kPreferredMtu);
        return false;
    }
    ble_svc_gap_init();
    ble_svc_gatt_init();
    if (ble_gatts_count_cfg(kServices) != 0 || ble_gatts_add_svcs(kServices) != 0) {
        ESP_LOGE(kTag, "BLE_GATT_INIT_FAIL");
        return false;
    }
    if (ble_svc_gap_device_name_set(kDeviceName) != 0) {
        ESP_LOGE(kTag, "BLE_DEVICE_NAME_FAIL");
        return false;
    }

    nimble_port_freertos_init(host_task);
    update_model_status(
        model_exchange::TransferStatus::kIdle,
        0,
        0,
        model_exchange::SharedHeadFormat::kFp32,
        model_exchange::TransferDirection::kVsnToAsn,
        0,
        0,
        0);
    ESP_LOGI(
        kTag,
        "BLE_INIT_PASS summary_bytes=%u shared_head_parameters=%u fp32_bytes=%u int8_bytes=%u",
        sizeof(g_last_summary),
        static_cast<unsigned>(model_exchange::kSharedHeadParameterCount),
        static_cast<unsigned>(model_exchange::kSharedHeadFp32Bytes),
        static_cast<unsigned>(model_exchange::kSharedHeadInt8Bytes));
    return true;
}

bool asn_ble_publish(float risk_score, float confidence, uint8_t status)
{
    node_learning::NodeSummaryV1 summary{};
    summary.node_id = static_cast<uint8_t>(node_learning::NodeId::kAsn);
    summary.modality = static_cast<uint8_t>(node_learning::Modality::kAcoustic);
    summary.status = status;
    summary.sequence = ++g_sequence;
    summary.uptime_ms = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    summary.risk_q15 = node_learning::encode_q15(risk_score);
    summary.confidence_q15 = node_learning::encode_q15(confidence);
    summary.model_version = runtime_shared_head_snapshot().version;
    node_learning::seal(summary);
    g_last_summary = summary;

    if (g_conn_handle == kNoConnection || !g_subscribed) {
        ESP_LOGW(
            kTag,
            "BLE_TX_SKIPPED sequence=%u connected=%s subscribed=%s",
            summary.sequence,
            g_conn_handle != kNoConnection ? "yes" : "no",
            g_subscribed ? "yes" : "no");
        return false;
    }

    os_mbuf *packet = ble_hs_mbuf_from_flat(&summary, sizeof(summary));
    if (packet == nullptr) {
        ESP_LOGE(kTag, "BLE_TX_ALLOCATION_FAIL sequence=%u", summary.sequence);
        return false;
    }
    g_last_tx_sequence = summary.sequence;
    g_last_tx_us = esp_timer_get_time();
    const int result = ble_gatts_notify_custom(
        g_conn_handle,
        g_summary_handle,
        packet);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_TX_FAIL sequence=%u rc=%d", summary.sequence, result);
        return false;
    }

    ESP_LOGI(
        kTag,
        "BLE_TX sequence=%u bytes=%u risk=%.6f confidence=%.6f status=%u crc=0x%04x",
        summary.sequence,
        sizeof(summary),
        risk_score,
        confidence,
        status,
        summary.crc16);
    return true;
}
