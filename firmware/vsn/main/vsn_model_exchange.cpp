// Reliable BLE transfer and fault tests for compatible shared risk heads.
// GATT writes are synchronised and acknowledged chunk by chunk. The nine fault
// cases exercise rejection paths while the last validated runtime head remains
// available to the sensing task.

#include "vsn_model_exchange.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "host/ble_att.h"
#include "host/ble_gap.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "model_exchange_protocol.h"
#include "nvs.h"
#include "os/os_mbuf.h"
#include "runtime_shared_head.h"

namespace {

constexpr char kTag[] = "vsn_model_ble";
constexpr EventBits_t kReadyBit = BIT0;
constexpr TickType_t kGattTimeout = pdMS_TO_TICKS(5000);
constexpr TickType_t kReconnectTimeout = pdMS_TO_TICKS(20000);
constexpr int kFaultCaseCount = 9;
constexpr char kTransferNvsNamespace[] = "model_xfer";
constexpr char kNextTransferIdKey[] = "next_id";
constexpr uint16_t kInitialCommandTransferId = 1000;

EventGroupHandle_t g_events = nullptr;
SemaphoreHandle_t g_gatt_complete = nullptr;
SemaphoreHandle_t g_command_lock = nullptr;
portMUX_TYPE g_handle_lock = portMUX_INITIALIZER_UNLOCKED;
VsnModelExchangeHandles g_handles{};
int g_gatt_status = -1;
uint8_t g_read_buffer[model_exchange::kPreferredMtu - 3]{};
size_t g_read_bytes = 0;
uint8_t *g_pull_buffer = nullptr;
size_t g_pull_capacity = 0;
bool g_experiment_complete = false;
uint16_t g_next_command_transfer_id = kInitialCommandTransferId;

bool initialise_transfer_counter()
{
    nvs_handle_t handle{};
    esp_err_t result = nvs_open(kTransferNvsNamespace, NVS_READWRITE, &handle);
    if (result != ESP_OK) {
        ESP_LOGE(kTag, "TRANSFER_ID_NVS_OPEN_FAIL result=%s", esp_err_to_name(result));
        return false;
    }

    uint16_t stored_id = kInitialCommandTransferId;
    result = nvs_get_u16(handle, kNextTransferIdKey, &stored_id);
    if (result == ESP_ERR_NVS_NOT_FOUND || stored_id < kInitialCommandTransferId) {
        stored_id = kInitialCommandTransferId;
        result = nvs_set_u16(handle, kNextTransferIdKey, stored_id);
        if (result == ESP_OK) {
            result = nvs_commit(handle);
        }
    }
    nvs_close(handle);
    if (result != ESP_OK) {
        ESP_LOGE(kTag, "TRANSFER_ID_NVS_INIT_FAIL result=%s", esp_err_to_name(result));
        return false;
    }

    g_next_command_transfer_id = stored_id;
    ESP_LOGI(kTag, "TRANSFER_ID_COUNTER_READY next=%u", stored_id);
    return true;
}

bool allocate_command_transfer_id(uint16_t &transfer_id)
{
    if (g_next_command_transfer_id == UINT16_MAX) {
        ESP_LOGE(kTag, "TRANSFER_ID_COUNTER_EXHAUSTED");
        return false;
    }

    const uint16_t next_id = static_cast<uint16_t>(g_next_command_transfer_id + 1);
    nvs_handle_t handle{};
    esp_err_t result = nvs_open(kTransferNvsNamespace, NVS_READWRITE, &handle);
    if (result == ESP_OK) {
        result = nvs_set_u16(handle, kNextTransferIdKey, next_id);
    }
    if (result == ESP_OK) {
        result = nvs_commit(handle);
    }
    if (handle != 0) {
        nvs_close(handle);
    }
    if (result != ESP_OK) {
        ESP_LOGE(kTag, "TRANSFER_ID_NVS_COMMIT_FAIL result=%s", esp_err_to_name(result));
        return false;
    }

    transfer_id = g_next_command_transfer_id;
    g_next_command_transfer_id = next_id;
    return true;
}

float local_threshold_for_version(uint16_t version)
{
    return version == model_exchange::kAsnFederatedINT8.model_version
        ? 0.185f
        : 0.165f;
}

void log_transition(const char *action, const RuntimeHeadTransition &transition, bool pass)
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

bool snapshot_handles(VsnModelExchangeHandles &handles)
{
    // Copy all handles under one critical section so a reconnect cannot produce
    // a mixed set from two different BLE sessions.
    portENTER_CRITICAL(&g_handle_lock);
    handles = g_handles;
    portEXIT_CRITICAL(&g_handle_lock);
    return handles.conn_handle != BLE_HS_CONN_HANDLE_NONE &&
        handles.control_handle != 0 && handles.data_rx_handle != 0 &&
        handles.status_handle != 0 && handles.data_tx_handle != 0;
}

void drain_semaphore()
{
    while (xSemaphoreTake(g_gatt_complete, 0) == pdTRUE) {
    }
}

int gatt_write_complete(
    uint16_t,
    const ble_gatt_error *error,
    ble_gatt_attr *,
    void *)
{
    g_gatt_status = error->status;
    xSemaphoreGive(g_gatt_complete);
    return 0;
}

int gatt_read_complete(
    uint16_t,
    const ble_gatt_error *error,
    ble_gatt_attr *attribute,
    void *)
{
    g_gatt_status = error->status;
    g_read_bytes = 0;
    if (error->status == 0 && attribute != nullptr && attribute->om != nullptr) {
        uint16_t copied = 0;
        if (ble_hs_mbuf_to_flat(
                attribute->om,
                g_read_buffer,
                sizeof(g_read_buffer),
                &copied) == 0) {
            g_read_bytes = copied;
        } else {
            g_gatt_status = BLE_HS_EMSGSIZE;
        }
    }
    xSemaphoreGive(g_gatt_complete);
    return 0;
}

int write_sync_status(uint16_t handle, const void *data, size_t bytes)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles) || bytes > UINT16_MAX) {
        return -1001;
    }
    drain_semaphore();
    g_gatt_status = -1;
    const int result = ble_gattc_write_flat(
        handles.conn_handle,
        handle,
        data,
        static_cast<uint16_t>(bytes),
        gatt_write_complete,
        nullptr);
    if (result != 0) {
        ESP_LOGE(kTag, "MODEL_GATT_WRITE_START_FAIL handle=%u rc=%d", handle, result);
        return result;
    }
    if (xSemaphoreTake(g_gatt_complete, kGattTimeout) != pdTRUE) {
        ESP_LOGE(kTag, "MODEL_GATT_WRITE_TIMEOUT handle=%u", handle);
        return -1002;
    }
    if (g_gatt_status != 0) {
        ESP_LOGE(kTag, "MODEL_GATT_WRITE_FAIL handle=%u status=%d", handle, g_gatt_status);
        return g_gatt_status;
    }
    return 0;
}

bool write_sync(uint16_t handle, const void *data, size_t bytes)
{
    return write_sync_status(handle, data, bytes) == 0;
}

bool read_sync(uint16_t handle, uint8_t *destination, size_t capacity, size_t &bytes)
{
    VsnModelExchangeHandles handles{};
    bytes = 0;
    if (!snapshot_handles(handles)) {
        return false;
    }
    drain_semaphore();
    g_gatt_status = -1;
    const int result = ble_gattc_read(
        handles.conn_handle,
        handle,
        gatt_read_complete,
        nullptr);
    if (result != 0) {
        ESP_LOGE(kTag, "MODEL_GATT_READ_START_FAIL handle=%u rc=%d", handle, result);
        return false;
    }
    if (xSemaphoreTake(g_gatt_complete, kGattTimeout) != pdTRUE) {
        ESP_LOGE(kTag, "MODEL_GATT_READ_TIMEOUT handle=%u", handle);
        return false;
    }
    if (g_gatt_status != 0 || g_read_bytes > capacity) {
        ESP_LOGE(kTag, "MODEL_GATT_READ_FAIL handle=%u status=%d bytes=%u",
            handle, g_gatt_status, static_cast<unsigned>(g_read_bytes));
        return false;
    }
    std::memcpy(destination, g_read_buffer, g_read_bytes);
    bytes = g_read_bytes;
    return true;
}

model_exchange::ModelControlV1 make_control(
    model_exchange::ControlOpcode opcode,
    uint16_t transfer_id,
    const model_exchange::SharedHeadPackage &package,
    model_exchange::TransferDirection direction,
    uint16_t chunk_index = 0)
{
    model_exchange::ModelControlV1 control{};
    control.opcode = static_cast<uint8_t>(opcode);
    control.transfer_id = transfer_id;
    control.model_version = package.model_version;
    control.format = static_cast<uint8_t>(package.format);
    control.direction = static_cast<uint8_t>(direction);
    control.payload_bytes = static_cast<uint16_t>(package.size);
    control.chunk_index = chunk_index;
    control.payload_crc32 = package.crc32;
    model_exchange::seal(control);
    return control;
}

model_exchange::ModelControlV1 make_push_metadata(
    uint16_t transfer_id,
    const model_exchange::SharedHeadPackage &package,
    float target_threshold)
{
    auto control = make_control(
        model_exchange::ControlOpcode::kPreparePushMetadata,
        transfer_id,
        package,
        model_exchange::TransferDirection::kVsnToAsn);
    control.chunk_index = static_cast<uint16_t>(
        std::lround(std::clamp(target_threshold, 0.0f, 1.0f) * 65535.0f));
    static_assert(sizeof(control.payload_crc32) == sizeof(package.expected_golden_output));
    std::memcpy(
        &control.payload_crc32,
        &package.expected_golden_output,
        sizeof(control.payload_crc32));
    model_exchange::seal(control);
    return control;
}

// A transfer is accepted only after per-chunk checks, full-payload CRC and a
// golden-vector test on the receiving node.
bool run_push(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id,
    int repetition,
    float remote_threshold);

bool read_remote_status(
    uint16_t expected_transfer_id,
    model_exchange::TransferStatus expected_status,
    const char *case_name)
{
    VsnModelExchangeHandles handles{};
    uint8_t status_bytes[sizeof(model_exchange::ModelStatusV1)]{};
    size_t status_length = 0;
    if (!snapshot_handles(handles) ||
        !read_sync(handles.status_handle, status_bytes, sizeof(status_bytes), status_length) ||
        status_length != sizeof(model_exchange::ModelStatusV1)) {
        ESP_LOGE(kTag, "FAULT_STATUS_READ_FAIL case=%s transfer=%u",
            case_name, expected_transfer_id);
        return false;
    }
    model_exchange::ModelStatusV1 status{};
    std::memcpy(&status, status_bytes, sizeof(status));
    const auto observed_status = static_cast<model_exchange::TransferStatus>(status.status);
    const bool pass = model_exchange::validate(status) &&
        status.transfer_id == expected_transfer_id && observed_status == expected_status;
    ESP_LOGI(
        kTag,
        "FAULT_STATUS case=%s transfer=%u expected=%s observed=%s chunks=%u bytes=%u "
        "crc32=0x%08x result=%s",
        case_name,
        expected_transfer_id,
        model_exchange::status_name(expected_status),
        model_exchange::status_name(observed_status),
        status.chunks_received,
        status.payload_bytes,
        status.computed_crc32,
        pass ? "PASS" : "FAIL");
    return pass;
}

bool begin_fault_push(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id,
    uint16_t model_version)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles)) {
        return false;
    }
    const auto metadata = make_push_metadata(transfer_id, package, 0.84f);
    if (!write_sync(handles.control_handle, &metadata, sizeof(metadata))) {
        return false;
    }
    auto control = make_control(
        model_exchange::ControlOpcode::kBeginPush,
        transfer_id,
        package,
        model_exchange::TransferDirection::kVsnToAsn);
    control.model_version = model_version;
    model_exchange::seal(control);
    return write_sync(handles.control_handle, &control, sizeof(control));
}

int send_fault_chunk(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id,
    uint16_t chunk_index,
    bool corrupt_chunk_crc,
    bool corrupt_payload)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles)) {
        return -1001;
    }
    const size_t chunk_payload_bytes = model_exchange::chunk_payload_capacity(
        ble_att_mtu(handles.conn_handle));
    const size_t offset = static_cast<size_t>(chunk_index) * chunk_payload_bytes;
    if (chunk_payload_bytes == 0 || offset >= package.size) {
        return -1003;
    }
    const size_t bytes = std::min(chunk_payload_bytes, package.size - offset);
    uint8_t payload[model_exchange::kMaximumChunkPayloadBytes]{};
    std::memcpy(payload, package.data + offset, bytes);
    if (corrupt_payload && bytes > 0) {
        payload[0] ^= 0x5au;
    }
    uint8_t packet[model_exchange::kPreferredMtu - 3]{};
    const size_t packet_bytes = model_exchange::build_chunk(
        packet,
        sizeof(packet),
        transfer_id,
        chunk_index,
        payload,
        bytes);
    if (packet_bytes == 0) {
        return -1004;
    }
    if (corrupt_chunk_crc) {
        packet[packet_bytes - 1] ^= 0xa5u;
    }
    return write_sync_status(handles.data_rx_handle, packet, packet_bytes);
}

int commit_fault_push(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles)) {
        return -1001;
    }
    const auto control = make_control(
        model_exchange::ControlOpcode::kCommitPush,
        transfer_id,
        package,
        model_exchange::TransferDirection::kVsnToAsn);
    return write_sync_status(handles.control_handle, &control, sizeof(control));
}

bool fault_invalid_model_version(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles)) {
        return false;
    }
    auto control = make_control(
        model_exchange::ControlOpcode::kBeginPush,
        transfer_id,
        package,
        model_exchange::TransferDirection::kVsnToAsn);
    control.model_version = 0x0100;
    model_exchange::seal(control);
    const int status = write_sync_status(handles.control_handle, &control, sizeof(control));
    return status != 0 && read_remote_status(
        transfer_id,
        model_exchange::TransferStatus::kVersionFailure,
        "invalid_model_version");
}

bool fault_corrupt_chunk_crc(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    if (!begin_fault_push(package, transfer_id, package.model_version)) {
        return false;
    }
    const int status = send_fault_chunk(package, transfer_id, 0, true, false);
    return status != 0 && read_remote_status(
        transfer_id,
        model_exchange::TransferStatus::kProtocolFailure,
        "corrupt_chunk_crc16");
}

bool fault_out_of_order_chunk(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    if (!begin_fault_push(package, transfer_id, package.model_version)) {
        return false;
    }
    const int status = send_fault_chunk(package, transfer_id, 1, false, false);
    return status != 0 && read_remote_status(
        transfer_id,
        model_exchange::TransferStatus::kProtocolFailure,
        "out_of_order_chunk");
}

bool fault_duplicate_chunk(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    if (!begin_fault_push(package, transfer_id, package.model_version) ||
        send_fault_chunk(package, transfer_id, 0, false, false) != 0) {
        return false;
    }
    const int status = send_fault_chunk(package, transfer_id, 0, false, false);
    return status != 0 && read_remote_status(
        transfer_id,
        model_exchange::TransferStatus::kProtocolFailure,
        "duplicate_chunk");
}

bool fault_incomplete_commit(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    if (!begin_fault_push(package, transfer_id, package.model_version) ||
        send_fault_chunk(package, transfer_id, 0, false, false) != 0 ||
        send_fault_chunk(package, transfer_id, 1, false, false) != 0) {
        return false;
    }
    const int status = commit_fault_push(package, transfer_id);
    return status != 0 && read_remote_status(
        transfer_id,
        model_exchange::TransferStatus::kProtocolFailure,
        "incomplete_commit");
}

bool fault_payload_crc32(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles) ||
        !begin_fault_push(package, transfer_id, package.model_version)) {
        return false;
    }
    const size_t chunk_payload_bytes = model_exchange::chunk_payload_capacity(
        ble_att_mtu(handles.conn_handle));
    const size_t chunks = model_exchange::chunk_count(package.size, chunk_payload_bytes);
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        if (send_fault_chunk(
                package,
                transfer_id,
                static_cast<uint16_t>(chunk),
                false,
                chunk == 2) != 0) {
            return false;
        }
    }
    const int status = commit_fault_push(package, transfer_id);
    return status != 0 && read_remote_status(
        transfer_id,
        model_exchange::TransferStatus::kCrcFailure,
        "payload_crc32_mismatch");
}

bool fault_receive_timeout(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    if (!begin_fault_push(package, transfer_id, package.model_version) ||
        send_fault_chunk(package, transfer_id, 0, false, false) != 0) {
        return false;
    }
    ESP_LOGI(kTag, "FAULT_TIMEOUT_WAIT transfer=%u wait_ms=6000", transfer_id);
    vTaskDelay(pdMS_TO_TICKS(6000));
    return read_remote_status(
        transfer_id,
        model_exchange::TransferStatus::kTimeout,
        "receive_timeout");
}

bool fault_disconnect_mid_transfer(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles) ||
        !begin_fault_push(package, transfer_id, package.model_version) ||
        send_fault_chunk(package, transfer_id, 0, false, false) != 0) {
        return false;
    }
    const int64_t started_us = esp_timer_get_time();
    ESP_LOGI(kTag, "FAULT_DISCONNECT_INJECT transfer=%u conn_handle=%u",
        transfer_id, handles.conn_handle);
    const int terminate_status = ble_gap_terminate(
        handles.conn_handle,
        BLE_ERR_REM_USER_CONN_TERM);
    if (terminate_status != 0) {
        ESP_LOGE(kTag, "FAULT_DISCONNECT_START_FAIL transfer=%u rc=%d",
            transfer_id, terminate_status);
        return false;
    }
    vTaskDelay(pdMS_TO_TICKS(500));
    const EventBits_t ready = xEventGroupWaitBits(
        g_events,
        kReadyBit,
        pdFALSE,
        pdTRUE,
        kReconnectTimeout);
    if ((ready & kReadyBit) == 0) {
        ESP_LOGE(kTag, "FAULT_RECONNECT_TIMEOUT transfer=%u", transfer_id);
        return false;
    }
    const double recovery_ms = (esp_timer_get_time() - started_us) / 1000.0;
    const bool aborted = read_remote_status(
        transfer_id,
        model_exchange::TransferStatus::kAborted,
        "disconnect_mid_transfer");
    ESP_LOGI(kTag, "FAULT_LINK_RECOVERY transfer=%u reconnect_ms=%.3f status=%s",
        transfer_id, recovery_ms, aborted ? "PASS" : "FAIL");
    return aborted;
}

bool fault_replay_transfer(
    const model_exchange::SharedHeadPackage &package,
    uint16_t replayed_transfer_id)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles)) {
        return false;
    }
    const auto control = make_control(
        model_exchange::ControlOpcode::kBeginPush,
        replayed_transfer_id,
        package,
        model_exchange::TransferDirection::kVsnToAsn);
    const int status = write_sync_status(handles.control_handle, &control, sizeof(control));
    return status != 0 && read_remote_status(
        replayed_transfer_id,
        model_exchange::TransferStatus::kReplayFailure,
        "transfer_id_replay");
}

bool run_fault_case_recovery(
    int case_index,
    const char *case_name,
    uint16_t fault_transfer_id,
    uint16_t recovery_transfer_id,
    bool detected)
{
    const bool recovered = run_push(
        model_exchange::kVsnBaseINT8,
        recovery_transfer_id,
        case_index,
        0.84f);
    ESP_LOGI(
        kTag,
        "FAULT_CASE_RESULT case_index=%d case=%s fault_transfer=%u recovery_transfer=%u "
        "detected=%s active_preserved=receiver_verified recovered=%s result=%s",
        case_index,
        case_name,
        fault_transfer_id,
        recovery_transfer_id,
        detected ? "yes" : "no",
        recovered ? "yes" : "no",
        detected && recovered ? "PASS" : "FAIL");
    vTaskDelay(pdMS_TO_TICKS(250));
    return detected && recovered;
}

bool run_push(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id,
    int repetition,
    float remote_threshold)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles)) {
        return false;
    }
    const uint16_t mtu = ble_att_mtu(handles.conn_handle);
    const size_t chunk_payload_bytes = model_exchange::chunk_payload_capacity(mtu);
    const size_t chunks = model_exchange::chunk_count(package.size, chunk_payload_bytes);
    if (chunk_payload_bytes == 0 || chunks > UINT16_MAX) {
        return false;
    }

    const int64_t started_us = esp_timer_get_time();
    const auto metadata = make_push_metadata(transfer_id, package, remote_threshold);
    if (!write_sync(handles.control_handle, &metadata, sizeof(metadata))) {
        return false;
    }
    auto control = make_control(
        model_exchange::ControlOpcode::kBeginPush,
        transfer_id,
        package,
        model_exchange::TransferDirection::kVsnToAsn);
    ESP_LOGI(
        kTag,
        "MODEL_PUSH_BEGIN repetition=%d transfer=%u direction=VSN_TO_ASN format=%s "
        "bytes=%u chunks=%u mtu=%u",
        repetition,
        transfer_id,
        model_exchange::format_name(package.format),
        static_cast<unsigned>(package.size),
        static_cast<unsigned>(chunks),
        mtu);
    if (!write_sync(handles.control_handle, &control, sizeof(control))) {
        return false;
    }

    uint8_t packet[model_exchange::kPreferredMtu - 3]{};
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const size_t offset = chunk * chunk_payload_bytes;
        const size_t bytes = std::min(chunk_payload_bytes, package.size - offset);
        const size_t packet_bytes = model_exchange::build_chunk(
            packet,
            sizeof(packet),
            transfer_id,
            static_cast<uint16_t>(chunk),
            package.data + offset,
            bytes);
        if (packet_bytes == 0 ||
            !write_sync(handles.data_rx_handle, packet, packet_bytes)) {
            ESP_LOGE(kTag, "MODEL_PUSH_CHUNK_FAIL transfer=%u chunk=%u",
                transfer_id, static_cast<unsigned>(chunk));
            return false;
        }
    }

    control = make_control(
        model_exchange::ControlOpcode::kCommitPush,
        transfer_id,
        package,
        model_exchange::TransferDirection::kVsnToAsn);
    if (!write_sync(handles.control_handle, &control, sizeof(control))) {
        return false;
    }
    uint8_t status_bytes[sizeof(model_exchange::ModelStatusV1)]{};
    size_t status_length = 0;
    if (!read_sync(handles.status_handle, status_bytes, sizeof(status_bytes), status_length) ||
        status_length != sizeof(model_exchange::ModelStatusV1)) {
        return false;
    }
    model_exchange::ModelStatusV1 status{};
    std::memcpy(&status, status_bytes, sizeof(status));
    const bool pass = model_exchange::validate(status) &&
        status.status == static_cast<uint8_t>(model_exchange::TransferStatus::kComplete) &&
        status.transfer_id == transfer_id && status.payload_bytes == package.size &&
        status.computed_crc32 == package.crc32;
    const double elapsed_ms = (esp_timer_get_time() - started_us) / 1000.0;
    ESP_LOGI(
        kTag,
        "MODEL_PUSH_RESULT repetition=%d transfer=%u direction=VSN_TO_ASN format=%s "
        "bytes=%u chunks=%u elapsed_ms=%.3f throughput_Bps=%.3f integrity=%s "
        "remote_execute=%s",
        repetition,
        transfer_id,
        model_exchange::format_name(package.format),
        static_cast<unsigned>(package.size),
        static_cast<unsigned>(chunks),
        elapsed_ms,
        elapsed_ms > 0.0 ? package.size * 1000.0 / elapsed_ms : 0.0,
        pass ? "PASS" : "FAIL",
        pass ? "PASS" : "FAIL");
    return pass;
}

bool ensure_pull_buffer()
{
    if (g_pull_buffer != nullptr &&
        g_pull_capacity >= model_exchange::kSharedHeadFp32Bytes) {
        return true;
    }
    if (g_pull_buffer != nullptr) {
        heap_caps_free(g_pull_buffer);
    }
    g_pull_buffer = static_cast<uint8_t *>(heap_caps_malloc(
        model_exchange::kSharedHeadFp32Bytes,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    g_pull_capacity = g_pull_buffer == nullptr
        ? 0
        : model_exchange::kSharedHeadFp32Bytes;
    return g_pull_buffer != nullptr;
}

bool run_pull(
    const model_exchange::SharedHeadPackage &package,
    uint16_t transfer_id,
    int repetition)
{
    VsnModelExchangeHandles handles{};
    if (!snapshot_handles(handles) || !ensure_pull_buffer()) {
        return false;
    }
    const uint16_t mtu = ble_att_mtu(handles.conn_handle);
    const size_t chunk_payload_bytes = model_exchange::chunk_payload_capacity(mtu);
    const size_t chunks = model_exchange::chunk_count(package.size, chunk_payload_bytes);
    if (chunk_payload_bytes == 0 || chunks > UINT16_MAX) {
        return false;
    }
    const int64_t started_us = esp_timer_get_time();
    auto control = make_control(
        model_exchange::ControlOpcode::kPreparePull,
        transfer_id,
        package,
        model_exchange::TransferDirection::kAsnToVsn);
    ESP_LOGI(
        kTag,
        "MODEL_PULL_BEGIN repetition=%d transfer=%u direction=ASN_TO_VSN format=%s "
        "bytes=%u chunks=%u mtu=%u",
        repetition,
        transfer_id,
        model_exchange::format_name(package.format),
        static_cast<unsigned>(package.size),
        static_cast<unsigned>(chunks),
        mtu);
    if (!write_sync(handles.control_handle, &control, sizeof(control))) {
        return false;
    }

    size_t received = 0;
    uint8_t packet[model_exchange::kPreferredMtu - 3]{};
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        control = make_control(
            model_exchange::ControlOpcode::kSelectPullChunk,
            transfer_id,
            package,
            model_exchange::TransferDirection::kAsnToVsn,
            static_cast<uint16_t>(chunk));
        if (!write_sync(handles.control_handle, &control, sizeof(control))) {
            return false;
        }
        size_t packet_bytes = 0;
        if (!read_sync(handles.data_tx_handle, packet, sizeof(packet), packet_bytes)) {
            return false;
        }
        model_exchange::ModelChunkHeaderV1 header{};
        const uint8_t *payload = nullptr;
        if (!model_exchange::parse_chunk(packet, packet_bytes, header, payload) ||
            header.transfer_id != transfer_id || header.chunk_index != chunk ||
            received + header.payload_bytes > g_pull_capacity) {
            ESP_LOGE(kTag, "MODEL_PULL_CHUNK_FAIL transfer=%u chunk=%u",
                transfer_id, static_cast<unsigned>(chunk));
            return false;
        }
        std::memcpy(g_pull_buffer + received, payload, header.payload_bytes);
        received += header.payload_bytes;
    }

    const uint32_t computed_crc = model_exchange::crc32_ieee(g_pull_buffer, received);
    float output = 0.0f;
    const bool crc_pass = received == package.size && computed_crc == package.crc32;
    const bool executed = crc_pass && model_exchange::evaluate_shared_head(
        g_pull_buffer,
        received,
        package.format,
        output);
    const float absolute_error = executed
        ? std::fabs(output - package.expected_golden_output)
        : INFINITY;
    const bool parity_pass = executed && absolute_error < 1.0e-4f;
    RuntimeHeadTransition activation{};
    bool activation_pass = parity_pass && runtime_shared_head_activate(
        g_pull_buffer,
        received,
        package.format,
        package.model_version,
        local_threshold_for_version(package.model_version),
        package.expected_golden_output,
        activation);
    log_transition("ACTIVATE", activation, activation_pass);

    bool rollback_pass = true;
    bool reactivate_pass = true;
    if (activation_pass && repetition == 1 &&
        package.format == model_exchange::SharedHeadFormat::kInt8Symmetric) {
        RuntimeHeadTransition rollback{};
        rollback_pass = runtime_shared_head_rollback(rollback);
        log_transition("ROLLBACK", rollback, rollback_pass);
        RuntimeHeadTransition reactivate{};
        reactivate_pass = rollback_pass && runtime_shared_head_activate(
            g_pull_buffer,
            received,
            package.format,
            package.model_version,
            local_threshold_for_version(package.model_version),
            package.expected_golden_output,
            reactivate);
        log_transition("REACTIVATE", reactivate, reactivate_pass);
    }
    const bool pass = parity_pass && activation_pass && rollback_pass && reactivate_pass;
    const double elapsed_ms = (esp_timer_get_time() - started_us) / 1000.0;
    ESP_LOGI(
        kTag,
        "MODEL_PULL_RESULT repetition=%d transfer=%u direction=ASN_TO_VSN format=%s "
        "bytes=%u chunks=%u elapsed_ms=%.3f throughput_Bps=%.3f crc=%s execute=%s "
        "output=%.9f expected=%.9f abs_error=%.9f staged=yes hot_install=%s active_version=0x%04x",
        repetition,
        transfer_id,
        model_exchange::format_name(package.format),
        static_cast<unsigned>(received),
        static_cast<unsigned>(chunks),
        elapsed_ms,
        elapsed_ms > 0.0 ? received * 1000.0 / elapsed_ms : 0.0,
        crc_pass ? "PASS" : "FAIL",
        pass ? "PASS" : "FAIL",
        output,
        package.expected_golden_output,
        absolute_error,
        activation_pass ? "PASS" : "FAIL",
        runtime_shared_head_snapshot().version);
    return pass;
}

[[maybe_unused]] void experiment_task(void *)
{
    while (true) {
        xEventGroupWaitBits(g_events, kReadyBit, pdFALSE, pdTRUE, portMAX_DELAY);
        if (g_experiment_complete) {
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        ESP_LOGI(kTag, "BLE_FAULT_RECOVERY_ARM_DELAY wait_ms=20000");
        vTaskDelay(pdMS_TO_TICKS(20000));
        if ((xEventGroupGetBits(g_events) & kReadyBit) == 0) {
            continue;
        }
        ESP_LOGI(
            kTag,
            "BLE_FAULT_RECOVERY_EXPERIMENT_BEGIN cases=%d parameters=%u "
            "fp32_bytes=%u int8_bytes=%u internal_free=%u psram_free=%u",
            kFaultCaseCount,
            static_cast<unsigned>(model_exchange::kSharedHeadParameterCount),
            static_cast<unsigned>(model_exchange::kSharedHeadFp32Bytes),
            static_cast<unsigned>(model_exchange::kSharedHeadInt8Bytes),
            static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
            static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));

        bool all_pass = true;
        int passed_cases = 0;
        bool detected = fault_invalid_model_version(model_exchange::kVsnBaseINT8, 300);
        bool case_pass = run_fault_case_recovery(
            1, "invalid_model_version", 300, 301, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        detected = fault_corrupt_chunk_crc(model_exchange::kVsnBaseINT8, 302);
        case_pass = run_fault_case_recovery(
            2, "corrupt_chunk_crc16", 302, 303, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        detected = fault_out_of_order_chunk(model_exchange::kVsnBaseINT8, 304);
        case_pass = run_fault_case_recovery(
            3, "out_of_order_chunk", 304, 305, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        detected = fault_duplicate_chunk(model_exchange::kVsnBaseINT8, 306);
        case_pass = run_fault_case_recovery(
            4, "duplicate_chunk", 306, 307, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        detected = fault_incomplete_commit(model_exchange::kVsnBaseINT8, 308);
        case_pass = run_fault_case_recovery(
            5, "incomplete_commit", 308, 309, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        detected = fault_payload_crc32(model_exchange::kVsnBaseINT8, 310);
        case_pass = run_fault_case_recovery(
            6, "payload_crc32_mismatch", 310, 311, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        detected = fault_receive_timeout(model_exchange::kVsnBaseINT8, 312);
        case_pass = run_fault_case_recovery(
            7, "receive_timeout", 312, 313, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        detected = fault_disconnect_mid_transfer(model_exchange::kVsnBaseINT8, 314);
        case_pass = run_fault_case_recovery(
            8, "disconnect_mid_transfer", 314, 315, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        detected = fault_replay_transfer(model_exchange::kVsnBaseINT8, 315);
        case_pass = run_fault_case_recovery(
            9, "transfer_id_replay", 315, 316, detected);
        all_pass = all_pass && case_pass;
        passed_cases += case_pass;

        const bool reverse_pull_pass = run_pull(
            model_exchange::kAsnFederatedINT8,
            400,
            1);
        all_pass = all_pass && reverse_pull_pass;
        ESP_LOGI(
            kTag,
            "BLE_FAULT_RECOVERY_EXPERIMENT_COMPLETE cases=%d passed=%d "
            "reverse_pull=%s result=%s internal_free=%u psram_free=%u "
            "active_version=0x%04x online_summary_path=retained",
            kFaultCaseCount,
            passed_cases,
            reverse_pull_pass ? "PASS" : "FAIL",
            all_pass ? "PASS" : "FAIL",
            static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
            static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
            runtime_shared_head_snapshot().version);
        g_experiment_complete = true;
    }
}

}  // namespace

bool vsn_model_exchange_init()
{
    g_events = xEventGroupCreate();
    g_gatt_complete = xSemaphoreCreateBinary();
    g_command_lock = xSemaphoreCreateMutex();
    g_handles.conn_handle = BLE_HS_CONN_HANDLE_NONE;
    if (g_events == nullptr || g_gatt_complete == nullptr || g_command_lock == nullptr) {
        ESP_LOGE(kTag, "MODEL_EXCHANGE_RTOS_INIT_FAIL");
        return false;
    }
    if (!initialise_transfer_counter()) {
        return false;
    }
    ESP_LOGI(kTag, "MODEL_EXCHANGE_READY mode=host_controlled automatic_faults=disabled");
    return true;
}

void vsn_model_exchange_start(const VsnModelExchangeHandles &handles)
{
    portENTER_CRITICAL(&g_handle_lock);
    g_handles = handles;
    portEXIT_CRITICAL(&g_handle_lock);
    xEventGroupSetBits(g_events, kReadyBit);
}

void vsn_model_exchange_disconnect()
{
    xEventGroupClearBits(g_events, kReadyBit);
    portENTER_CRITICAL(&g_handle_lock);
    g_handles = {};
    g_handles.conn_handle = BLE_HS_CONN_HANDLE_NONE;
    portEXIT_CRITICAL(&g_handle_lock);
}

bool vsn_model_exchange_ready()
{
    VsnModelExchangeHandles handles{};
    return snapshot_handles(handles);
}

bool vsn_model_exchange_push_vsn_head()
{
    if (g_command_lock == nullptr ||
        xSemaphoreTake(g_command_lock, pdMS_TO_TICKS(1000)) != pdTRUE) {
        return false;
    }
    uint16_t transfer_id = 0;
    if (!allocate_command_transfer_id(transfer_id)) {
        xSemaphoreGive(g_command_lock);
        return false;
    }
    const bool result = run_push(model_exchange::kVsnBaseINT8, transfer_id, 0, 0.84f);
    xSemaphoreGive(g_command_lock);
    return result;
}

bool vsn_model_exchange_push_package(
    const model_exchange::SharedHeadPackage &package,
    float remote_threshold)
{
    if (g_command_lock == nullptr || remote_threshold < 0.0f || remote_threshold > 1.0f ||
        xSemaphoreTake(g_command_lock, pdMS_TO_TICKS(1000)) != pdTRUE) {
        return false;
    }
    uint16_t transfer_id = 0;
    if (!allocate_command_transfer_id(transfer_id)) {
        xSemaphoreGive(g_command_lock);
        return false;
    }
    const bool result = run_push(package, transfer_id, 0, remote_threshold);
    xSemaphoreGive(g_command_lock);
    return result;
}

bool vsn_model_exchange_pull_asn_head()
{
    if (g_command_lock == nullptr ||
        xSemaphoreTake(g_command_lock, pdMS_TO_TICKS(1000)) != pdTRUE) {
        return false;
    }
    uint16_t transfer_id = 0;
    if (!allocate_command_transfer_id(transfer_id)) {
        xSemaphoreGive(g_command_lock);
        return false;
    }
    const bool result = run_pull(model_exchange::kAsnFederatedINT8, transfer_id, 0);
    xSemaphoreGive(g_command_lock);
    return result;
}

bool vsn_model_exchange_rollback_local()
{
    RuntimeHeadTransition transition{};
    const bool result = runtime_shared_head_rollback(transition);
    log_transition("HOST_ROLLBACK", transition, result);
    return result;
}
