// Sampling-control interface used by the ASN runtime and BLE service.

#pragma once

#include <cstdint>

#include "sampling_control_protocol.h"

bool asn_sampling_control_init();
bool asn_sampling_apply(sampling_control::Command command, uint16_t request_id);
bool asn_sampling_wait_for_window();
bool asn_sampling_is_automatic();
