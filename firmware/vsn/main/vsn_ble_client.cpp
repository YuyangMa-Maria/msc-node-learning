// NimBLE central that discovers the ASN and receives its risk summaries.
// The latest validated summary is copied into a small thread-safe snapshot;
// inference code never reads mutable NimBLE packet buffers directly.

#include "vsn_ble_client.h"

#include <cstdint>

#include "esp_err.h"
#include "esp_bt.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "host/ble_att.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "esp_central.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "model_exchange_protocol.h"
#include "node_summary_protocol.h"
#include "sampling_control_protocol.h"
#include "nvs_flash.h"
#include "os/os_mbuf.h"
#include "services/gap/ble_svc_gap.h"
#include "system_host_protocol.h"
#include "vsn_sampling_control.h"
#include "vsn_model_exchange.h"

namespace {

constexpr char kTag[] = "vsn_ble";
constexpr char kDeviceName[] = "NL-VSN-01";

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
const ble_uuid16_t kCccdUuid = BLE_UUID16_INIT(BLE_GATT_DSC_CLT_CFG_UUID16);

uint16_t g_summary_handle = 0;
uint16_t g_ack_handle = 0;
uint16_t g_model_control_handle = 0;
uint16_t g_model_data_rx_handle = 0;
uint16_t g_model_status_handle = 0;
uint16_t g_model_data_tx_handle = 0;
uint16_t g_sampling_control_handle = 0;
uint16_t g_conn_handle = BLE_HS_CONN_HANDLE_NONE;
uint16_t g_last_sequence = 0;
uint32_t g_received = 0;
uint32_t g_crc_errors = 0;
uint32_t g_sequence_gaps = 0;
uint32_t g_duplicates = 0;
uint32_t g_connections = 0;
uint32_t g_advertisements_seen = 0;
bool g_have_sequence = false;
SemaphoreHandle_t g_sampling_write_done = nullptr;
int g_sampling_write_status = -1;
uint16_t g_sampling_request_id = 0;
portMUX_TYPE g_snapshot_lock = portMUX_INITIALIZER_UNLOCKED;
vsn_ble_summary_snapshot_t g_latest_summary{};

// Clearing validity on disconnect prevents stale ASN evidence from surviving a
// link outage. Sequence and CRC checks are applied before a new snapshot is set.
void set_connection_state(bool connected, bool clear_summary)
{
    portENTER_CRITICAL(&g_snapshot_lock);
    g_latest_summary.connected = connected;
    if (clear_summary) {
        g_latest_summary.valid = false;
    }
    portEXIT_CRITICAL(&g_snapshot_lock);
}

int gap_event(ble_gap_event *event, void *argument);

void clear_handles()
{
    g_summary_handle = 0;
    g_ack_handle = 0;
    g_model_control_handle = 0;
    g_model_data_rx_handle = 0;
    g_model_status_handle = 0;
    g_model_data_tx_handle = 0;
    g_sampling_control_handle = 0;
}

void scan()
{
    uint8_t own_addr_type = 0;
    int result = ble_hs_id_infer_auto(0, &own_addr_type);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_ADDRESS_FAIL rc=%d", result);
        return;
    }

    ble_gap_disc_params parameters{};
    parameters.passive = 0;
    parameters.filter_duplicates = 1;
    result = ble_gap_disc(
        own_addr_type,
        BLE_HS_FOREVER,
        &parameters,
        gap_event,
        nullptr);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_SCAN_START_FAIL rc=%d", result);
        return;
    }
    ESP_LOGI(kTag, "BLE_SCANNING target=NL-ASN-01");
}

bool advertises_node_service(const ble_gap_disc_desc &discovery)
{
    ble_hs_adv_fields fields{};
    if (ble_hs_adv_parse_fields(
            &fields,
            discovery.data,
            discovery.length_data) != 0) {
        return false;
    }
    for (int index = 0; index < fields.num_uuids128; ++index) {
        if (ble_uuid_cmp(&fields.uuids128[index].u, &kServiceUuid.u) == 0) {
            return true;
        }
    }
    return false;
}

int subscribe_complete(
    uint16_t conn_handle,
    const ble_gatt_error *error,
    ble_gatt_attr *,
    void *)
{
    if (error->status != 0) {
        ESP_LOGE(kTag, "BLE_SUBSCRIBE_FAIL status=%d", error->status);
        return ble_gap_terminate(conn_handle, BLE_ERR_REM_USER_CONN_TERM);
    }
    ESP_LOGI(
        kTag,
        "BLE_SUBSCRIBED conn_handle=%u summary_handle=%u ack_handle=%u "
        "model_control_handle=%u model_data_rx_handle=%u model_status_handle=%u "
        "model_data_tx_handle=%u sampling_control_handle=%u",
        conn_handle,
        g_summary_handle,
        g_ack_handle,
        g_model_control_handle,
        g_model_data_rx_handle,
        g_model_status_handle,
        g_model_data_tx_handle,
        g_sampling_control_handle);
    vsn_model_exchange_start({
        conn_handle,
        g_model_control_handle,
        g_model_data_rx_handle,
        g_model_status_handle,
        g_model_data_tx_handle,
    });
    return 0;
}

void discovery_complete(const peer *remote, int status, void *)
{
    if (status != 0) {
        ESP_LOGE(kTag, "BLE_DISCOVERY_FAIL status=%d", status);
        ble_gap_terminate(remote->conn_handle, BLE_ERR_REM_USER_CONN_TERM);
        return;
    }

    const peer_chr *summary = peer_chr_find_uuid(
        remote,
        &kServiceUuid.u,
        &kSummaryUuid.u);
    const peer_chr *ack = peer_chr_find_uuid(
        remote,
        &kServiceUuid.u,
        &kAckUuid.u);
    const peer_chr *model_control = peer_chr_find_uuid(
        remote,
        &kServiceUuid.u,
        &kModelControlUuid.u);
    const peer_chr *model_data_rx = peer_chr_find_uuid(
        remote,
        &kServiceUuid.u,
        &kModelDataRxUuid.u);
    const peer_chr *model_status = peer_chr_find_uuid(
        remote,
        &kServiceUuid.u,
        &kModelStatusUuid.u);
    const peer_chr *model_data_tx = peer_chr_find_uuid(
        remote,
        &kServiceUuid.u,
        &kModelDataTxUuid.u);
    const peer_chr *sampling_control = peer_chr_find_uuid(
        remote,
        &kServiceUuid.u,
        &kSamplingControlUuid.u);
    const peer_dsc *cccd = peer_dsc_find_uuid(
        remote,
        &kServiceUuid.u,
        &kSummaryUuid.u,
        &kCccdUuid.u);
    if (summary == nullptr || ack == nullptr || model_control == nullptr ||
        model_data_rx == nullptr || model_status == nullptr ||
        model_data_tx == nullptr || sampling_control == nullptr || cccd == nullptr) {
        ESP_LOGE(
            kTag,
            "BLE_CONTRACT_MISMATCH summary=%s ack=%s control=%s rx=%s status=%s tx=%s sampling=%s cccd=%s",
            summary != nullptr ? "yes" : "no",
            ack != nullptr ? "yes" : "no",
            model_control != nullptr ? "yes" : "no",
            model_data_rx != nullptr ? "yes" : "no",
            model_status != nullptr ? "yes" : "no",
            model_data_tx != nullptr ? "yes" : "no",
            sampling_control != nullptr ? "yes" : "no",
            cccd != nullptr ? "yes" : "no");
        ble_gap_terminate(remote->conn_handle, BLE_ERR_REM_USER_CONN_TERM);
        return;
    }

    g_summary_handle = summary->chr.val_handle;
    g_ack_handle = ack->chr.val_handle;
    g_model_control_handle = model_control->chr.val_handle;
    g_model_data_rx_handle = model_data_rx->chr.val_handle;
    g_model_status_handle = model_status->chr.val_handle;
    g_model_data_tx_handle = model_data_tx->chr.val_handle;
    g_sampling_control_handle = sampling_control->chr.val_handle;
    const uint8_t subscription[] = {1, 0};
    const int result = ble_gattc_write_flat(
        remote->conn_handle,
        cccd->dsc.handle,
        subscription,
        sizeof(subscription),
        subscribe_complete,
        nullptr);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_SUBSCRIBE_START_FAIL rc=%d", result);
        ble_gap_terminate(remote->conn_handle, BLE_ERR_REM_USER_CONN_TERM);
    }
}

void start_discovery(uint16_t conn_handle)
{
    if (peer_disc_all(conn_handle, discovery_complete, nullptr) != 0) {
        ESP_LOGE(kTag, "BLE_DISCOVERY_START_FAIL");
        ble_gap_terminate(conn_handle, BLE_ERR_REM_USER_CONN_TERM);
    }
}

int mtu_exchange_complete(
    uint16_t conn_handle,
    const ble_gatt_error *error,
    uint16_t mtu,
    void *)
{
    if (error->status != 0) {
        ESP_LOGW(kTag, "BLE_MTU_EXCHANGE_FAIL status=%d fallback_mtu=%u",
            error->status, ble_att_mtu(conn_handle));
    } else {
        ESP_LOGI(kTag, "BLE_MTU_EXCHANGE_PASS mtu=%u", mtu);
    }
    start_discovery(conn_handle);
    return 0;
}

void record_sequence(uint16_t sequence)
{
    if (!g_have_sequence) {
        g_last_sequence = sequence;
        g_have_sequence = true;
        return;
    }
    const uint16_t expected = static_cast<uint16_t>(g_last_sequence + 1);
    if (sequence == g_last_sequence) {
        ++g_duplicates;
    } else if (sequence != expected) {
        g_sequence_gaps += static_cast<uint16_t>(sequence - expected);
    }
    g_last_sequence = sequence;
}

void acknowledge(uint16_t conn_handle, uint16_t sequence)
{
    if (g_ack_handle == 0) {
        return;
    }
    const int result = ble_gattc_write_flat(
        conn_handle,
        g_ack_handle,
        &sequence,
        sizeof(sequence),
        nullptr,
        nullptr);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_ACK_WRITE_FAIL sequence=%u rc=%d", sequence, result);
    }
}

void handle_summary(uint16_t conn_handle, os_mbuf *payload)
{
    const uint16_t length = OS_MBUF_PKTLEN(payload);
    if (length != sizeof(node_learning::NodeSummaryV1)) {
        ESP_LOGE(
            kTag,
            "BLE_RX_LENGTH_FAIL expected=%u actual=%u",
            sizeof(node_learning::NodeSummaryV1),
            length);
        return;
    }

    node_learning::NodeSummaryV1 summary{};
    if (os_mbuf_copydata(payload, 0, sizeof(summary), &summary) != 0 ||
        !node_learning::validate(summary)) {
        ++g_crc_errors;
        ESP_LOGE(kTag, "BLE_RX_CRC_FAIL count=%u", g_crc_errors);
        return;
    }
    if (summary.node_id != static_cast<uint8_t>(node_learning::NodeId::kAsn) ||
        summary.modality != static_cast<uint8_t>(node_learning::Modality::kAcoustic)) {
        ESP_LOGE(
            kTag,
            "BLE_RX_IDENTITY_FAIL node=%u modality=%u",
            summary.node_id,
            summary.modality);
        return;
    }

    ++g_received;
    record_sequence(summary.sequence);
    const float risk_score = node_learning::decode_q15(summary.risk_q15);
    const float confidence = node_learning::decode_q15(summary.confidence_q15);
    const uint32_t received_ms = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    portENTER_CRITICAL(&g_snapshot_lock);
    g_latest_summary.valid = true;
    g_latest_summary.connected = true;
    g_latest_summary.sequence = summary.sequence;
    g_latest_summary.source_uptime_ms = summary.uptime_ms;
    g_latest_summary.received_local_uptime_ms = received_ms;
    g_latest_summary.risk_score = risk_score;
    g_latest_summary.confidence = confidence;
    g_latest_summary.status = summary.status;
    portEXIT_CRITICAL(&g_snapshot_lock);
    ESP_LOGI(
        kTag,
        "BLE_RX sequence=%u bytes=%u risk=%.6f confidence=%.6f status=%u "
        "source_uptime_ms=%u crc=PASS",
        summary.sequence,
        length,
        risk_score,
        confidence,
        summary.status,
        summary.uptime_ms);
    system_host_report_node_output(
        "ASN",
        "microphone",
        "ble_summary",
        summary.sequence,
        summary.uptime_ms,
        received_ms,
        risk_score,
        confidence,
        summary.status,
        -1.0f,
        -1.0f,
        -1.0f,
        -1.0f,
        -1.0f);
    vsn_sampling_notify_asn_ready();
    acknowledge(conn_handle, summary.sequence);

    if (g_received % 20 == 0) {
        ESP_LOGI(
            kTag,
            "BLE_LINK_SUMMARY received=%u sequence_gaps=%u duplicates=%u "
            "crc_errors=%u connections=%u",
            g_received,
            g_sequence_gaps,
            g_duplicates,
            g_crc_errors,
            g_connections);
    }
}

int gap_event(ble_gap_event *event, void *)
{
    switch (event->type) {
    case BLE_GAP_EVENT_DISC: {
        ++g_advertisements_seen;
        const bool target_service = advertises_node_service(event->disc);
        if (target_service || g_advertisements_seen <= 20 ||
            g_advertisements_seen % 100 == 0) {
            ESP_LOGI(
                kTag,
                "BLE_DISCOVERY_OBSERVED count=%u rssi=%d bytes=%u target_service=%s",
                g_advertisements_seen,
                event->disc.rssi,
                event->disc.length_data,
                target_service ? "yes" : "no");
        }
        if (!target_service) {
            return 0;
        }
        if (ble_gap_disc_cancel() != 0) {
            return 0;
        }
        {
            uint8_t own_addr_type = 0;
            if (ble_hs_id_infer_auto(0, &own_addr_type) != 0) {
                scan();
                return 0;
            }
            ESP_LOGI(kTag, "BLE_TARGET_FOUND rssi=%d", event->disc.rssi);
            const int result = ble_gap_connect(
                own_addr_type,
                &event->disc.addr,
                10000,
                nullptr,
                gap_event,
                nullptr);
            if (result != 0) {
                ESP_LOGE(kTag, "BLE_CONNECT_START_FAIL rc=%d", result);
                scan();
            }
        }
        return 0;
    }

    case BLE_GAP_EVENT_CONNECT: {
        if (event->connect.status != 0) {
            ESP_LOGW(kTag, "BLE_CONNECT_FAIL status=%d", event->connect.status);
            system_host_report_ble(false, g_connections, event->connect.status);
            scan();
            return 0;
        }
        ++g_connections;
        clear_handles();
        g_conn_handle = event->connect.conn_handle;
        g_have_sequence = false;
        set_connection_state(true, true);
        ESP_LOGI(
            kTag,
            "BLE_CONNECTED conn_handle=%u connection_count=%u",
            event->connect.conn_handle,
            g_connections);
        system_host_report_ble(true, g_connections, 0);
        if (peer_add(event->connect.conn_handle) != 0) {
            ESP_LOGE(kTag, "BLE_PEER_ADD_FAIL");
            ble_gap_terminate(event->connect.conn_handle, BLE_ERR_REM_USER_CONN_TERM);
            return 0;
        }
        const int mtu_result = ble_gattc_exchange_mtu(
            event->connect.conn_handle,
            mtu_exchange_complete,
            nullptr);
        if (mtu_result != 0) {
            ESP_LOGW(kTag, "BLE_MTU_EXCHANGE_START_FAIL rc=%d", mtu_result);
            start_discovery(event->connect.conn_handle);
        }
        return 0;
    }

    case BLE_GAP_EVENT_DISCONNECT:
        ESP_LOGW(kTag, "BLE_DISCONNECTED reason=%d", event->disconnect.reason);
        system_host_report_ble(false, g_connections, event->disconnect.reason);
        vsn_model_exchange_disconnect();
        peer_delete(event->disconnect.conn.conn_handle);
        clear_handles();
        g_conn_handle = BLE_HS_CONN_HANDLE_NONE;
        g_have_sequence = false;
        set_connection_state(false, true);
        scan();
        return 0;

    case BLE_GAP_EVENT_NOTIFY_RX:
        if (event->notify_rx.attr_handle == g_summary_handle) {
            handle_summary(event->notify_rx.conn_handle, event->notify_rx.om);
        }
        return 0;

    case BLE_GAP_EVENT_DISC_COMPLETE:
        if (!ble_gap_disc_active()) {
            scan();
        }
        return 0;

    case BLE_GAP_EVENT_MTU:
        ESP_LOGI(kTag, "BLE_MTU value=%u", event->mtu.value);
        return 0;

    default:
        return 0;
    }
}

void on_reset(int reason)
{
    ESP_LOGE(kTag, "BLE_HOST_RESET reason=%d", reason);
    system_host_report_ble(false, g_connections, reason);
}

void on_sync()
{
    const int result = ble_hs_util_ensure_addr(0);
    if (result != 0) {
        ESP_LOGE(kTag, "BLE_IDENTITY_FAIL rc=%d", result);
        return;
    }
    scan();
}

int sampling_write_complete(
    uint16_t,
    const ble_gatt_error *error,
    ble_gatt_attr *,
    void *)
{
    g_sampling_write_status = error == nullptr ? BLE_HS_EUNKNOWN : error->status;
    if (g_sampling_write_done != nullptr) {
        xSemaphoreGive(g_sampling_write_done);
    }
    return 0;
}

bool send_sampling_control(sampling_control::Command command)
{
    if (g_conn_handle == BLE_HS_CONN_HANDLE_NONE ||
        g_sampling_control_handle == 0 || g_sampling_write_done == nullptr) {
        return false;
    }
    while (xSemaphoreTake(g_sampling_write_done, 0) == pdTRUE) {
    }
    sampling_control::MessageV1 message{};
    message.command = static_cast<uint8_t>(command);
    message.request_id = ++g_sampling_request_id;
    sampling_control::seal(message);
    g_sampling_write_status = -1;
    const int result = ble_gattc_write_flat(
        g_conn_handle,
        g_sampling_control_handle,
        &message,
        sizeof(message),
        sampling_write_complete,
        nullptr);
    if (result != 0 ||
        xSemaphoreTake(g_sampling_write_done, pdMS_TO_TICKS(2000)) != pdTRUE) {
        ESP_LOGE(
            kTag,
            "SAMPLING_CONTROL_TX_FAIL command=%s request=%u start_rc=%d",
            sampling_control::command_name(command),
            message.request_id,
            result);
        return false;
    }
    const bool pass = g_sampling_write_status == 0;
    ESP_LOGI(
        kTag,
        "SAMPLING_CONTROL_TX command=%s request=%u result=%s gatt_status=%d bytes=%u",
        sampling_control::command_name(command),
        message.request_id,
        pass ? "PASS" : "FAIL",
        g_sampling_write_status,
        static_cast<unsigned>(sizeof(message)));
    return pass;
}

void host_task(void *)
{
    ESP_LOGI(kTag, "BLE_HOST_TASK_STARTED");
    nimble_port_run();
    nimble_port_freertos_deinit();
}

}  // namespace

extern "C" bool vsn_ble_init(void)
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
    ESP_LOGI(
        kTag,
        "BLE_TX_POWER default=%s level=+9dBm",
        esp_err_to_name(default_power));

    ble_hs_cfg.reset_cb = on_reset;
    ble_hs_cfg.sync_cb = on_sync;
    if (ble_att_set_preferred_mtu(model_exchange::kPreferredMtu) != 0) {
        ESP_LOGE(kTag, "BLE_MTU_CONFIG_FAIL value=%u", model_exchange::kPreferredMtu);
        return false;
    }
#if MYNEWT_VAL(BLE_INCL_SVC_DISCOVERY) || MYNEWT_VAL(BLE_GATT_CACHING_INCLUDE_SERVICES)
    const int peer_result = peer_init(1, 8, 4, 24, 16);
#else
    const int peer_result = peer_init(1, 8, 24, 16);
#endif
    if (peer_result != 0 || ble_svc_gap_device_name_set(kDeviceName) != 0) {
        ESP_LOGE(kTag, "BLE_CENTRAL_STATE_INIT_FAIL rc=%d", peer_result);
        return false;
    }
    if (!vsn_model_exchange_init()) {
        return false;
    }
    if (g_sampling_write_done == nullptr) {
        g_sampling_write_done = xSemaphoreCreateBinary();
    }
    if (g_sampling_write_done == nullptr) {
        ESP_LOGE(kTag, "SAMPLING_CONTROL_SEMAPHORE_FAIL");
        return false;
    }

    nimble_port_freertos_init(host_task);
    ESP_LOGI(kTag, "BLE_INIT_PASS expected_payload_bytes=%u", sizeof(node_learning::NodeSummaryV1));
    return true;
}

extern "C" bool vsn_ble_rescan(void)
{
    if (g_conn_handle != BLE_HS_CONN_HANDLE_NONE) {
        ESP_LOGI(kTag, "BLE_RESCAN_SKIPPED reason=already_connected");
        return true;
    }
    g_advertisements_seen = 0;
    if (ble_gap_disc_active()) {
        const int result = ble_gap_disc_cancel();
        ESP_LOGI(kTag, "BLE_RESCAN_REQUEST action=cancel_active_scan rc=%d", result);
        return result == 0;
    }
    ESP_LOGI(kTag, "BLE_RESCAN_REQUEST action=start_scan");
    scan();
    return ble_gap_disc_active();
}

extern "C" bool vsn_ble_get_latest_summary(
    vsn_ble_summary_snapshot_t *snapshot)
{
    if (snapshot == nullptr) {
        return false;
    }
    portENTER_CRITICAL(&g_snapshot_lock);
    *snapshot = g_latest_summary;
    portEXIT_CRITICAL(&g_snapshot_lock);
    return snapshot->valid;
}

extern "C" bool vsn_ble_set_asn_sampling_automatic(bool automatic)
{
    return send_sampling_control(
        automatic
            ? sampling_control::Command::kAutomatic
            : sampling_control::Command::kManual);
}

extern "C" bool vsn_ble_trigger_asn_sample(void)
{
    return send_sampling_control(sampling_control::Command::kSampleOnce);
}
