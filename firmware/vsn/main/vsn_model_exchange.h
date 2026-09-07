// Control interface for model push, pull and rollback operations.

#pragma once

#include <cstdint>

#include "shared_head_payloads.h"

struct VsnModelExchangeHandles {
    uint16_t conn_handle;
    uint16_t control_handle;
    uint16_t data_rx_handle;
    uint16_t status_handle;
    uint16_t data_tx_handle;
};

bool vsn_model_exchange_init();
void vsn_model_exchange_start(const VsnModelExchangeHandles &handles);
void vsn_model_exchange_disconnect();
bool vsn_model_exchange_ready();
bool vsn_model_exchange_push_vsn_head();
bool vsn_model_exchange_push_package(
    const model_exchange::SharedHeadPackage &package,
    float remote_threshold);
bool vsn_model_exchange_pull_asn_head();
bool vsn_model_exchange_rollback_local();
