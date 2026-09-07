// VSN BLE client state and control interface.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool vsn_ble_init(void);
bool vsn_ble_rescan(void);
bool vsn_ble_set_asn_sampling_automatic(bool automatic);
bool vsn_ble_trigger_asn_sample(void);

typedef struct {
    bool valid;
    bool connected;
    uint16_t sequence;
    uint32_t source_uptime_ms;
    uint32_t received_local_uptime_ms;
    float risk_score;
    float confidence;
    uint8_t status;
} vsn_ble_summary_snapshot_t;

bool vsn_ble_get_latest_summary(vsn_ble_summary_snapshot_t *snapshot);

#ifdef __cplusplus
}
#endif
