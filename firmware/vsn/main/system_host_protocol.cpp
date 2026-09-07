// Versioned USB host protocol and rule-based warning generation.
// The VSN emits complete decisions as NLJSON records. The PC receiver records
// and displays these fields; it does not recalculate fusion or alert severity.

#include "system_host_protocol.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "host_model_update.h"
#include "model_exchange_protocol.h"
#include "runtime_shared_head.h"
#include "vsn_model_exchange.h"
#include "vsn_ble_client.h"
#include "vsn_sampling_control.h"

namespace {

constexpr char kTag[] = "system_host";
constexpr size_t kJsonBufferBytes = 2048;
constexpr size_t kCommandBufferBytes = 192;

struct HostSnapshot {
    // Retain one coherent fusion decision so STATUS can replay it without
    // mixing fields from separate sampling windows.
    bool valid = false;
    uint32_t fusion_index = 0;
    uint32_t uptime_ms = 0;
    uint32_t visual_sequence = 0;
    float visual_risk = 0.0f;
    float visual_confidence = 0.0f;
    uint8_t visual_status = 0;
    bool acoustic_available = false;
    uint16_t acoustic_sequence = 0;
    uint32_t acoustic_age_ms = 0;
    float acoustic_risk = 0.0f;
    float acoustic_confidence = 0.0f;
    uint8_t acoustic_status = 4;
    float fused_risk = 0.0f;
    float fused_confidence = 0.0f;
    int risk_level = 1;
    char risk_trend[12] = "unknown";
    int active_nodes = 0;
    uint8_t triggered_mask = 0;
    char alert_action[32] = "continue_monitoring";
    bool marh_recommended = false;
    char marh_reason[32] = "none";
};

SemaphoreHandle_t g_output_lock = nullptr;
SemaphoreHandle_t g_snapshot_lock = nullptr;
HostSnapshot g_snapshot{};
std::atomic_bool g_marh_active{false};

const char *status_name(uint8_t status)
{
    switch (status) {
    case 0: return "normal";
    case 1: return "warning";
    case 2: return "critical";
    case 3: return "degraded";
    case 4: return "invalid";
    case 5: return "low_power";
    default: return "invalid";
    }
}

bool status_is_usable(uint8_t status)
{
    return status != 4 && status != 5;
}

const char *alert_evidence(
    uint8_t triggered_mask,
    bool acoustic_available,
    uint8_t visual_status,
    uint8_t acoustic_status)
{
    // Prefer nodes that actually triggered. If none did, report the usable
    // coverage rather than implying that unavailable sensors supplied evidence.
    switch (triggered_mask & 0x03u) {
    case 0x01u: return "VSN";
    case 0x02u: return "ASN";
    case 0x03u: return "VSN+ASN";
    default: break;
    }

    const bool visual_usable = status_is_usable(visual_status);
    const bool acoustic_usable =
        acoustic_available && status_is_usable(acoustic_status);
    if (visual_usable && acoustic_usable) return "VSN+ASN";
    if (visual_usable) return "VSN";
    if (acoustic_usable) return "ASN";
    return "none";
}

bool alert_attention_required(int level, int active_nodes, bool marh_recommended)
{
    return active_nodes == 0 || level >= 3 || marh_recommended;
}

const char *alert_severity(
    int level,
    int active_nodes,
    bool marh_recommended)
{
    if (active_nodes == 0) return "unavailable";
    if (level >= 5) return "critical";
    if (level == 4) return "high";
    if (level == 3) return "caution";
    return marh_recommended ? "advisory" : "info";
}

const char *alert_headline(
    int level,
    const char *trend,
    int active_nodes,
    uint8_t triggered_mask,
    const char *marh_reason)
{
    if (active_nodes == 0) return "Insufficient sensor evidence";
    if (level >= 5 && (triggered_mask & 0x03u) == 0x03u) {
        return "Critical corroborated risk";
    }
    if (level >= 5) return "Critical local risk";
    if (level == 4) return "High local risk";
    if (level == 3 && std::strcmp(trend, "rising") == 0) {
        return "Risk proxy is rising";
    }
    if (level == 3) return "Moderate local risk";
    if (std::strcmp(marh_reason, "node_loss") == 0) {
        return "Reduced sensor coverage";
    }
    if (std::strcmp(marh_reason, "node_disagreement") == 0) {
        return "Conflicting node evidence";
    }
    return "Low local risk";
}

void build_alert_message(
    char *output,
    size_t output_size,
    int level,
    const char *trend,
    int active_nodes,
    uint8_t triggered_mask,
    bool acoustic_available,
    uint8_t visual_status,
    uint8_t acoustic_status,
    const char *marh_reason)
{
    // Fixed templates keep the safety wording deterministic and auditable on a
    // constrained node; no language model is involved in alert generation.
    if (active_nodes == 0) {
        std::snprintf(
            output,
            output_size,
            "No valid sensor evidence is available. Do not interpret this as low risk; repeat sampling or request external assessment.");
        return;
    }

    const char *evidence = alert_evidence(
        triggered_mask,
        acoustic_available,
        visual_status,
        acoustic_status);
    char base[256]{};
    if (level <= 2 && std::strcmp(marh_reason, "node_loss") == 0) {
        std::snprintf(
            base,
            sizeof(base),
            "Available evidence indicates a low current risk proxy, but sensor coverage is limited. Continue monitoring and repeat sampling.");
    } else if (level <= 2 && std::strcmp(marh_reason, "node_disagreement") == 0) {
        std::snprintf(
            base,
            sizeof(base),
            "The nodes report conflicting evidence. Repeat sampling and request additional assessment before relying on the local decision.");
    } else if (level <= 2) {
        std::snprintf(
            base,
            sizeof(base),
            "Current local risk proxy is low. Continue monitoring.");
    } else if (level == 3 && std::strcmp(trend, "rising") == 0) {
        std::snprintf(
            base,
            sizeof(base),
            "Moderate and rising risk proxy from %s evidence. Keep distance, repeat sampling, and request further assessment.",
            evidence);
    } else if (level == 3) {
        std::snprintf(
            base,
            sizeof(base),
            "Moderate risk proxy from %s evidence. Maintain caution and continue monitoring.",
            evidence);
    } else if (level == 4) {
        std::snprintf(
            base,
            sizeof(base),
            "High risk proxy from %s evidence. Avoid entering the monitored area and request further assessment.",
            evidence);
    } else if ((triggered_mask & 0x03u) == 0x03u) {
        std::snprintf(
            base,
            sizeof(base),
            "Critical corroborated risk proxy from VSN and ASN. Move away from the monitored area and await professional assessment.");
    } else {
        std::snprintf(
            base,
            sizeof(base),
            "Critical risk proxy from %s evidence. Move away from the monitored area and await professional assessment.",
            evidence);
    }

    const bool degraded_input = visual_status == 3u || visual_status == 4u ||
        visual_status == 5u || (acoustic_available &&
        (acoustic_status == 3u || acoustic_status == 4u || acoustic_status == 5u));
    const char *health_note = "";
    if (!acoustic_available && std::strcmp(marh_reason, "node_loss") != 0) {
        health_note = " Acoustic evidence is unavailable.";
    } else if (degraded_input) {
        health_note = " At least one node input is degraded.";
    }
    std::snprintf(output, output_size, "%s%s", base, health_note);
}

void emit_line(const char *json)
{
    if (g_output_lock != nullptr) {
        xSemaphoreTake(g_output_lock, portMAX_DELAY);
    }
    std::printf("NLJSON %s\n", json);
    std::fflush(stdout);
    if (g_output_lock != nullptr) {
        xSemaphoreGive(g_output_lock);
    }
}

void emit_command_result(const char *command, bool success, const char *detail)
{
    const RuntimeHeadSnapshot head = runtime_shared_head_snapshot();
    char json[kJsonBufferBytes]{};
    std::snprintf(
        json,
        sizeof(json),
        "{\"schema\":1,\"event\":\"command_result\",\"uptime_ms\":%u,"
        "\"command\":\"%s\",\"success\":%s,\"detail\":\"%s\","
        "\"model\":{\"version\":%u,\"generation\":%u,\"format\":\"%s\"}}",
        static_cast<unsigned>(esp_timer_get_time() / 1000),
        command,
        success ? "true" : "false",
        detail,
        head.version,
        static_cast<unsigned>(head.generation),
        model_exchange::format_name(head.format));
    emit_line(json);
}

void emit_marh_state(const char *source)
{
    char json[384]{};
    std::snprintf(
        json,
        sizeof(json),
        "{\"schema\":1,\"event\":\"marh_state\",\"uptime_ms\":%u,"
        "\"active\":%s,\"source\":\"%s\"}",
        static_cast<unsigned>(esp_timer_get_time() / 1000),
        g_marh_active.load() ? "true" : "false",
        source);
    emit_line(json);
}

void emit_node_state()
{
    HostSnapshot snapshot{};
    if (g_snapshot_lock != nullptr) {
        xSemaphoreTake(g_snapshot_lock, portMAX_DELAY);
        snapshot = g_snapshot;
        xSemaphoreGive(g_snapshot_lock);
    }
    const bool asn_online = vsn_model_exchange_ready();
    char json[512]{};
    std::snprintf(
        json,
        sizeof(json),
        "{\"schema\":1,\"event\":\"node_state\",\"uptime_ms\":%u,"
        "\"nodes\":{\"VSN\":{\"online\":true,\"transport\":\"usb_serial\"},"
        "\"ASN\":{\"online\":%s,\"transport\":\"ble\","
        "\"last_summary_available\":%s,\"last_summary_age_ms\":%u}}}",
        static_cast<unsigned>(esp_timer_get_time() / 1000),
        asn_online ? "true" : "false",
        snapshot.valid && snapshot.acoustic_available ? "true" : "false",
        snapshot.valid ? static_cast<unsigned>(snapshot.acoustic_age_ms) : 0u);
    emit_line(json);
}

void emit_snapshot()
{
    HostSnapshot snapshot{};
    if (g_snapshot_lock != nullptr) {
        xSemaphoreTake(g_snapshot_lock, portMAX_DELAY);
        snapshot = g_snapshot;
        xSemaphoreGive(g_snapshot_lock);
    }
    if (!snapshot.valid) {
        emit_command_result("STATUS", false, "fusion_not_available");
        return;
    }
    system_host_report_fusion(
        snapshot.fusion_index,
        snapshot.uptime_ms,
        snapshot.visual_sequence,
        snapshot.visual_risk,
        snapshot.visual_confidence,
        snapshot.visual_status,
        snapshot.acoustic_available,
        snapshot.acoustic_sequence,
        snapshot.acoustic_age_ms,
        snapshot.acoustic_risk,
        snapshot.acoustic_confidence,
        snapshot.acoustic_status,
        snapshot.fused_risk,
        snapshot.fused_confidence,
        snapshot.risk_level,
        snapshot.risk_trend,
        snapshot.active_nodes,
        snapshot.triggered_mask,
        snapshot.alert_action,
        snapshot.marh_recommended,
        snapshot.marh_reason);
}

std::string normalise_command(char *raw)
{
    std::string command(raw == nullptr ? "" : raw);
    while (!command.empty() && std::isspace(static_cast<unsigned char>(command.back()))) {
        command.pop_back();
    }
    size_t first = 0;
    while (first < command.size() &&
           std::isspace(static_cast<unsigned char>(command[first]))) {
        ++first;
    }
    command.erase(0, first);
    std::transform(command.begin(), command.end(), command.begin(), [](unsigned char value) {
        return static_cast<char>(std::toupper(value));
    });
    return command;
}

void command_task(void *)
{
    char raw[kCommandBufferBytes]{};
    while (true) {
        if (std::fgets(raw, sizeof(raw), stdin) == nullptr) {
            clearerr(stdin);
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        const std::string command = normalise_command(raw);
        if (command.empty()) {
            continue;
        }
        if (command == "PING") {
            emit_command_result("PING", true, "pong");
        } else if (command == "STATUS") {
            emit_snapshot();
        } else if (command == "NODE STATUS" || command == "NODE_STATUS") {
            emit_node_state();
        } else if (command == "MARH JOIN" || command == "MARH_JOIN") {
            g_marh_active.store(true);
            emit_marh_state("manual_join");
        } else if (command == "MARH LEAVE" || command == "MARH_LEAVE") {
            g_marh_active.store(false);
            emit_marh_state("manual_leave");
        } else if (command == "BLE RESCAN" || command == "BLE_RESCAN") {
            const bool success = vsn_ble_rescan();
            emit_command_result(
                "BLE_RESCAN",
                success,
                success ? "scan_restart_requested" : "scan_restart_failed");
        } else if (command == "SYSTEM RESTART" || command == "SYSTEM_RESTART") {
            emit_command_result("SYSTEM_RESTART", true, "representative_node_restarting");
            vTaskDelay(pdMS_TO_TICKS(250));
            esp_restart();
        } else if (command == "MODE AUTO" || command == "MODE_AUTO") {
            vsn_sampling_set_automatic(true);
            const bool asn_synchronised = vsn_ble_set_asn_sampling_automatic(true);
            system_host_report_sampling_state(true, asn_synchronised, "host_command");
            emit_command_result(
                "MODE_AUTO",
                asn_synchronised,
                asn_synchronised ? "both_nodes_automatic" : "vsn_automatic_asn_unavailable");
        } else if (command == "MODE MANUAL" || command == "MODE_MANUAL") {
            vsn_sampling_set_automatic(false);
            const bool asn_synchronised = vsn_ble_set_asn_sampling_automatic(false);
            system_host_report_sampling_state(false, asn_synchronised, "host_command");
            emit_command_result(
                "MODE_MANUAL",
                asn_synchronised,
                asn_synchronised ? "both_nodes_manual" : "vsn_manual_asn_unavailable");
        } else if (command == "SAMPLE VSN" || command == "SAMPLE_VSN") {
            const bool manual = !vsn_sampling_is_automatic();
            const bool success = manual && vsn_sampling_trigger_once();
            emit_command_result(
                "SAMPLE_VSN",
                success,
                manual ? (success ? "visual_capture_triggered" : "trigger_failed")
                       : "switch_to_manual_mode");
        } else if (command == "SAMPLE ASN" || command == "SAMPLE_ASN") {
            const bool manual = !vsn_sampling_is_automatic();
            const bool success = manual && vsn_ble_trigger_asn_sample();
            emit_command_result(
                "SAMPLE_ASN",
                success,
                manual ? (success ? "acoustic_capture_triggered" : "asn_not_connected")
                       : "switch_to_manual_mode");
        } else if (command == "SAMPLE BOTH" || command == "SAMPLE_BOTH") {
            const bool manual = !vsn_sampling_is_automatic();
            const bool scheduled = manual && vsn_sampling_schedule_after_asn();
            const bool asn_success = scheduled && vsn_ble_trigger_asn_sample();
            if (scheduled && !asn_success) {
                vsn_sampling_cancel_after_asn();
            }
            const bool success = scheduled && asn_success;
            emit_command_result(
                "SAMPLE_BOTH",
                success,
                !manual ? "switch_to_manual_mode"
                        : (success ? "acoustic_capture_then_visual_capture_scheduled"
                                   : "one_or_more_triggers_failed"));
        } else if (command.rfind("HOST MODEL BEGIN ", 0) == 0) {
            unsigned version = 0;
            unsigned bytes = 0;
            unsigned crc32 = 0;
            float expected = 0.0f;
            float vsn_threshold = 0.0f;
            float asn_threshold = 0.0f;
            const bool parsed = std::sscanf(
                command.c_str(),
                "HOST MODEL BEGIN %x %u %x %f %f %f",
                &version,
                &bytes,
                &crc32,
                &expected,
                &vsn_threshold,
                &asn_threshold) == 6;
            const bool success = parsed && host_model_update_begin(
                static_cast<uint16_t>(version),
                bytes,
                crc32,
                expected,
                vsn_threshold,
                asn_threshold);
            emit_command_result(
                "HOST_MODEL_BEGIN",
                success,
                success ? "int8_staging_ready" : "invalid_update_contract");
        } else if (command.rfind("HOST MODEL CHUNK ", 0) == 0) {
            unsigned offset = 0;
            char hex_payload[129]{};
            const bool parsed = std::sscanf(
                command.c_str(),
                "HOST MODEL CHUNK %u %128s",
                &offset,
                hex_payload) == 2;
            const bool success = parsed && host_model_update_chunk(offset, hex_payload);
            char detail[96]{};
            std::snprintf(
                detail,
                sizeof(detail),
                success ? "accepted_through_%u_bytes"
                        : (parsed ? "chunk_rejected_at_%u" : "chunk_parse_failed_at_%u"),
                success ? static_cast<unsigned>(host_model_update_received_bytes()) : offset);
            emit_command_result("HOST_MODEL_CHUNK", success, detail);
        } else if (command == "HOST MODEL END" || command == "HOST_MODEL_END") {
            const bool success = host_model_update_end();
            char detail[96]{};
            std::snprintf(
                detail,
                sizeof(detail),
                success ? "validated_and_activated_%u_bytes" : "validation_failed_%u_of_%u_bytes",
                static_cast<unsigned>(host_model_update_received_bytes()),
                static_cast<unsigned>(host_model_update_expected_bytes()));
            emit_command_result("HOST_MODEL_END", success, detail);
        } else if (command == "HOST MODEL PUSH" || command == "HOST_MODEL_PUSH") {
            const bool ready = vsn_model_exchange_ready();
            const bool success = ready && host_model_update_push();
            emit_command_result(
                "HOST_MODEL_PUSH",
                success,
                ready ? (success ? "host_candidate_sent_to_asn" : "transfer_failed")
                      : "asn_not_connected");
        } else if (command == "MODEL PUSH" || command == "MODEL_PUSH") {
            const bool ready = vsn_model_exchange_ready();
            const bool success = ready && vsn_model_exchange_push_vsn_head();
            emit_command_result("MODEL_PUSH", success,
                ready ? (success ? "vsn_head_sent_to_asn" : "transfer_failed")
                      : "asn_not_connected");
        } else if (command == "MODEL PULL" || command == "MODEL_PULL") {
            const bool ready = vsn_model_exchange_ready();
            const bool success = ready && vsn_model_exchange_pull_asn_head();
            emit_command_result("MODEL_PULL", success,
                ready ? (success ? "asn_head_activated_on_vsn" : "transfer_failed")
                      : "asn_not_connected");
        } else if (command == "MODEL ROLLBACK" || command == "MODEL_ROLLBACK") {
            const bool success = vsn_model_exchange_rollback_local();
            emit_command_result("MODEL_ROLLBACK", success,
                success ? "previous_local_head_restored" : "no_rollback_available");
        } else if (command == "HELP") {
            emit_command_result(
                "HELP",
                true,
                "PING|STATUS|NODE STATUS|BLE RESCAN|SYSTEM RESTART|MODE AUTO|MODE MANUAL|SAMPLE VSN|SAMPLE ASN|SAMPLE BOTH|MARH JOIN|MARH LEAVE|HOST MODEL BEGIN/CHUNK/END/PUSH|MODEL PUSH|MODEL PULL|MODEL ROLLBACK");
        } else {
            emit_command_result("UNKNOWN", false, "send_HELP_for_supported_commands");
        }
    }
}

}  // namespace

extern "C" bool system_host_protocol_init(void)
{
    g_output_lock = xSemaphoreCreateMutex();
    g_snapshot_lock = xSemaphoreCreateMutex();
    if (g_output_lock == nullptr || g_snapshot_lock == nullptr) {
        ESP_LOGE(kTag, "HOST_PROTOCOL_MUTEX_FAIL");
        return false;
    }
    if (!vsn_sampling_control_init()) {
        ESP_LOGE(kTag, "HOST_PROTOCOL_SAMPLING_INIT_FAIL");
        return false;
    }
    if (xTaskCreate(command_task, "host_commands", 6144, nullptr, 3, nullptr) != pdPASS) {
        ESP_LOGE(kTag, "HOST_PROTOCOL_TASK_FAIL");
        return false;
    }
    ESP_LOGI(kTag, "HOST_PROTOCOL_READY schema=1 transport=usb_serial role=external_receiver_marh_interface");
    return true;
}

extern "C" void system_host_report_boot(
    bool camera_ready,
    bool model_ready,
    bool ble_ready)
{
    char json[512]{};
    std::snprintf(
        json,
        sizeof(json),
        "{\"schema\":1,\"event\":\"boot\",\"uptime_ms\":%u,"
        "\"node\":\"VSN\",\"camera_ready\":%s,\"model_ready\":%s,"
        "\"ble_ready\":%s,\"decision_owner\":\"VSN\","
        "\"receiver_transport\":\"usb_serial\"}",
        static_cast<unsigned>(esp_timer_get_time() / 1000),
        camera_ready ? "true" : "false",
        model_ready ? "true" : "false",
        ble_ready ? "true" : "false");
    emit_line(json);
}

extern "C" void system_host_report_ble(
    bool connected,
    uint32_t connection_count,
    int reason)
{
    char json[384]{};
    std::snprintf(
        json,
        sizeof(json),
        "{\"schema\":1,\"event\":\"ble_state\",\"uptime_ms\":%u,"
        "\"peer\":\"ASN\",\"connected\":%s,\"connection_count\":%u,"
        "\"reason\":%d}",
        static_cast<unsigned>(esp_timer_get_time() / 1000),
        connected ? "true" : "false",
        static_cast<unsigned>(connection_count),
        reason);
    emit_line(json);
}

extern "C" void system_host_report_sampling_state(
    bool automatic,
    bool asn_synchronised,
    const char *source)
{
    char json[384]{};
    std::snprintf(
        json,
        sizeof(json),
        "{\"schema\":1,\"event\":\"sampling_state\",\"uptime_ms\":%u,"
        "\"mode\":\"%s\",\"vsn_ready\":true,\"asn_synchronised\":%s,"
        "\"source\":\"%s\"}",
        static_cast<unsigned>(esp_timer_get_time() / 1000),
        automatic ? "automatic" : "manual",
        asn_synchronised ? "true" : "false",
        source == nullptr ? "unknown" : source);
    emit_line(json);
}

extern "C" void system_host_report_node_output(
    const char *node,
    const char *source,
    const char *transport,
    uint32_t sequence,
    uint32_t source_uptime_ms,
    uint32_t received_uptime_ms,
    float risk_score,
    float confidence,
    uint8_t status,
    float capture_ms,
    float decode_ms,
    float preprocess_ms,
    float inference_ms,
    float total_ms)
{
    char json[768]{};
    std::snprintf(
        json,
        sizeof(json),
        "{\"schema\":1,\"event\":\"node_output\",\"uptime_ms\":%u,"
        "\"node\":\"%s\",\"phase\":\"%s\",\"source\":\"%s\","
        "\"transport\":\"%s\",\"sequence\":%u,\"source_uptime_ms\":%u,"
        "\"received_uptime_ms\":%u,\"risk_score\":%.6f,\"confidence\":%.6f,"
        "\"status\":\"%s\",\"latency_ms\":{\"capture\":%.3f,"
        "\"decode\":%.3f,\"preprocess\":%.3f,\"inference\":%.3f,"
        "\"total\":%.3f}}",
        static_cast<unsigned>(esp_timer_get_time() / 1000),
        node,
        std::strcmp(node, "VSN") == 0 ? "local_inference_complete" : "summary_received",
        source,
        transport,
        static_cast<unsigned>(sequence),
        static_cast<unsigned>(source_uptime_ms),
        static_cast<unsigned>(received_uptime_ms),
        risk_score,
        confidence,
        status_name(status),
        capture_ms,
        decode_ms,
        preprocess_ms,
        inference_ms,
        total_ms);
    emit_line(json);
}

extern "C" bool system_host_marh_active(void)
{
    return g_marh_active.load();
}

extern "C" void system_host_report_fusion(
    uint32_t fusion_index,
    uint32_t uptime_ms,
    uint32_t visual_sequence,
    float visual_risk,
    float visual_confidence,
    uint8_t visual_status,
    bool acoustic_available,
    uint16_t acoustic_sequence,
    uint32_t acoustic_age_ms,
    float acoustic_risk,
    float acoustic_confidence,
    uint8_t acoustic_status,
    float fused_risk,
    float fused_confidence,
    int risk_level,
    const char *risk_trend,
    int active_nodes,
    uint8_t triggered_mask,
    const char *alert_action,
    bool marh_recommended,
    const char *marh_reason)
{
    const RuntimeHeadSnapshot head = runtime_shared_head_snapshot();
    HostSnapshot snapshot{};
    snapshot.valid = true;
    snapshot.fusion_index = fusion_index;
    snapshot.uptime_ms = uptime_ms;
    snapshot.visual_sequence = visual_sequence;
    snapshot.visual_risk = visual_risk;
    snapshot.visual_confidence = visual_confidence;
    snapshot.visual_status = visual_status;
    snapshot.acoustic_available = acoustic_available;
    snapshot.acoustic_sequence = acoustic_sequence;
    snapshot.acoustic_age_ms = acoustic_age_ms;
    snapshot.acoustic_risk = acoustic_risk;
    snapshot.acoustic_confidence = acoustic_confidence;
    snapshot.acoustic_status = acoustic_status;
    snapshot.fused_risk = fused_risk;
    snapshot.fused_confidence = fused_confidence;
    snapshot.risk_level = risk_level;
    std::snprintf(snapshot.risk_trend, sizeof(snapshot.risk_trend), "%s", risk_trend);
    snapshot.active_nodes = active_nodes;
    snapshot.triggered_mask = triggered_mask;
    std::snprintf(snapshot.alert_action, sizeof(snapshot.alert_action), "%s", alert_action);
    snapshot.marh_recommended = marh_recommended;
    std::snprintf(snapshot.marh_reason, sizeof(snapshot.marh_reason), "%s", marh_reason);
    if (g_snapshot_lock != nullptr) {
        xSemaphoreTake(g_snapshot_lock, portMAX_DELAY);
        g_snapshot = snapshot;
        xSemaphoreGive(g_snapshot_lock);
    }

    char message[384]{};
    build_alert_message(
        message,
        sizeof(message),
        risk_level,
        risk_trend,
        active_nodes,
        triggered_mask,
        acoustic_available,
        visual_status,
        acoustic_status,
        marh_reason);
    const bool attention_required = alert_attention_required(
        risk_level,
        active_nodes,
        marh_recommended);
    const char *severity = alert_severity(
        risk_level,
        active_nodes,
        marh_recommended);
    const char *headline = alert_headline(
        risk_level,
        risk_trend,
        active_nodes,
        triggered_mask,
        marh_reason);

    char json[kJsonBufferBytes]{};
    std::snprintf(
        json,
        sizeof(json),
        "{\"schema\":1,\"event\":\"fusion\",\"uptime_ms\":%u,"
        "\"fusion_index\":%u,\"mode\":\"%s\","
        "\"vsn\":{\"sequence\":%u,\"risk_score\":%.6f,\"confidence\":%.6f,"
        "\"status\":\"%s\"},"
        "\"asn\":{\"available\":%s,\"sequence\":%u,\"age_ms\":%u,"
        "\"risk_score\":%.6f,\"confidence\":%.6f,\"status\":\"%s\"},"
        "\"fusion\":{\"risk_score\":%.6f,\"confidence\":%.6f,\"risk_level\":%d,"
        "\"risk_trend\":\"%s\",\"active_nodes\":%d,\"method\":\"quality_gate\"},"
        "\"alert\":{\"attention_required\":%s,\"severity\":\"%s\","
        "\"headline\":\"%s\",\"triggered_mask\":%u,\"evidence\":\"%s\","
        "\"action\":\"%s\",\"message\":\"%s\"},"
        "\"marh\":{\"active\":%s,\"recommended\":%s,\"reason\":\"%s\"},"
        "\"model\":{\"version\":%u,\"generation\":%u,\"format\":\"%s\"}}",
        static_cast<unsigned>(uptime_ms),
        static_cast<unsigned>(fusion_index),
        acoustic_available ? "dual" : "vsn_only",
        static_cast<unsigned>(visual_sequence),
        visual_risk,
        visual_confidence,
        status_name(visual_status),
        acoustic_available ? "true" : "false",
        acoustic_sequence,
        static_cast<unsigned>(acoustic_age_ms),
        acoustic_risk,
        acoustic_confidence,
        status_name(acoustic_status),
        fused_risk,
        fused_confidence,
        risk_level,
        risk_trend,
        active_nodes,
        attention_required ? "true" : "false",
        severity,
        headline,
        triggered_mask,
        alert_evidence(
            triggered_mask,
            acoustic_available,
            visual_status,
            acoustic_status),
        alert_action,
        message,
        g_marh_active.load() ? "true" : "false",
        marh_recommended ? "true" : "false",
        marh_reason,
        head.version,
        static_cast<unsigned>(head.generation),
        model_exchange::format_name(head.format));
    emit_line(json);
}
