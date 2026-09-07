// Public interface for the ASN BLE service.

#pragma once

#include <cstdint>

bool asn_ble_init();
bool asn_ble_publish(float risk_score, float confidence, uint8_t status);
