// Host protocol interface and event-reporting functions.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool system_host_protocol_init(void);
void system_host_report_boot(bool camera_ready, bool model_ready, bool ble_ready);
void system_host_report_ble(bool connected, uint32_t connection_count, int reason);
bool system_host_marh_active(void);
void system_host_report_sampling_state(
    bool automatic,
    bool asn_synchronised,
    const char *source);
void system_host_report_node_output(
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
    float total_ms);

void system_host_report_fusion(
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
    const char *marh_reason);

#ifdef __cplusplus
}
#endif
