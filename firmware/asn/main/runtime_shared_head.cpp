// Double-buffered runtime shared head with validation, commit and rollback.
// The inactive slot is prepared first; the active slot is changed only after a
// golden-vector check, leaving the previous valid head available for rollback.

#include "runtime_shared_head.h"

#include <cmath>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

namespace {
struct HeadSlot {
    uint8_t *payload = nullptr;
    size_t payload_bytes = 0;
    model_exchange::SharedHeadFormat format = model_exchange::SharedHeadFormat::kFp32;
    uint16_t version = 0;
    float threshold = 0.5f;
    uint32_t generation = 0;
    bool valid = false;
};
SemaphoreHandle_t g_lock = nullptr;
HeadSlot g_slots[2]{};
int g_active_slot = 0;
int g_previous_slot = -1;
float g_last_representation[64]{};
bool g_has_last_representation = false;
uint32_t g_generation = 0;
RuntimeHeadSnapshot snapshot(const HeadSlot &slot)
{
    return {slot.version, slot.format, slot.threshold, slot.generation};
}
bool allocate_slots()
{
    for (auto &slot : g_slots) {
        if (slot.payload == nullptr) {
            slot.payload = static_cast<uint8_t *>(heap_caps_malloc(
                model_exchange::kSharedHeadFp32Bytes,
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        }
        if (slot.payload == nullptr) return false;
    }
    return true;
}
bool evaluate_slot(const HeadSlot &slot, const float *representation, float &logit)
{
    return slot.valid && model_exchange::evaluate_shared_head_with_embedding(
        slot.payload, slot.payload_bytes, slot.format, representation, logit);
}
}  // namespace

bool runtime_shared_head_init(const model_exchange::SharedHeadPackage &package, float threshold)
{
    if (g_lock == nullptr) g_lock = xSemaphoreCreateMutex();
    if (g_lock == nullptr || !allocate_slots() ||
        !model_exchange::valid_package_contract(package.format, package.size)) return false;
    float golden = 0.0f;
    if (!model_exchange::evaluate_shared_head(package.data, package.size, package.format, golden) ||
        std::fabs(golden - package.expected_golden_output) >= 1.0e-4f) return false;
    xSemaphoreTake(g_lock, portMAX_DELAY);
    std::memcpy(g_slots[0].payload, package.data, package.size);
    g_slots[0].payload_bytes = package.size;
    g_slots[0].format = package.format;
    g_slots[0].version = package.model_version;
    g_slots[0].threshold = threshold;
    g_slots[0].generation = ++g_generation;
    g_slots[0].valid = true;
    g_slots[1].valid = false;
    g_active_slot = 0;
    g_previous_slot = -1;
    g_has_last_representation = false;
    xSemaphoreGive(g_lock);
    return true;
}

bool runtime_shared_head_activate(
    const uint8_t *payload, size_t payload_bytes,
    model_exchange::SharedHeadFormat format, uint16_t version,
    float threshold, float expected_golden, RuntimeHeadTransition &transition)
{
    transition = {};
    const int64_t started = esp_timer_get_time();
    float golden = 0.0f;
    if (g_lock == nullptr || payload == nullptr ||
        !model_exchange::evaluate_shared_head(payload, payload_bytes, format, golden)) return false;
    transition.golden_output = golden;
    transition.golden_error = std::fabs(golden - expected_golden);
    if (transition.golden_error >= 1.0e-4f) return false;
    xSemaphoreTake(g_lock, portMAX_DELAY);
    HeadSlot &current = g_slots[g_active_slot];
    transition.before = snapshot(current);
    transition.had_live_representation = g_has_last_representation;
    if (g_has_last_representation &&
        !evaluate_slot(current, g_last_representation, transition.live_logit_before)) {
        xSemaphoreGive(g_lock);
        return false;
    }
    float candidate_logit = 0.0f;
    if (g_has_last_representation &&
        !model_exchange::evaluate_shared_head_with_embedding(
            payload, payload_bytes, format, g_last_representation, candidate_logit)) {
        xSemaphoreGive(g_lock);
        return false;
    }
    // Build the candidate in the inactive slot. No inference observes a
    // partially copied payload.
    const int candidate_index = 1 - g_active_slot;
    HeadSlot &candidate = g_slots[candidate_index];
    std::memcpy(candidate.payload, payload, payload_bytes);
    candidate.payload_bytes = payload_bytes;
    candidate.format = format;
    candidate.version = version;
    candidate.threshold = threshold;
    candidate.generation = ++g_generation;
    candidate.valid = true;
    g_previous_slot = g_active_slot;
    g_active_slot = candidate_index;
    transition.live_logit_after = candidate_logit;
    transition.after = snapshot(candidate);
    xSemaphoreGive(g_lock);
    transition.elapsed_us = esp_timer_get_time() - started;
    return true;
}

bool runtime_shared_head_rollback(RuntimeHeadTransition &transition)
{
    transition = {};
    const int64_t started = esp_timer_get_time();
    if (g_lock == nullptr) return false;
    xSemaphoreTake(g_lock, portMAX_DELAY);
    if (g_previous_slot < 0 || !g_slots[g_previous_slot].valid) {
        xSemaphoreGive(g_lock);
        return false;
    }
    HeadSlot &current = g_slots[g_active_slot];
    HeadSlot &previous = g_slots[g_previous_slot];
    transition.before = snapshot(current);
    transition.after = snapshot(previous);
    transition.had_live_representation = g_has_last_representation;
    if (g_has_last_representation &&
        (!evaluate_slot(current, g_last_representation, transition.live_logit_before) ||
         !evaluate_slot(previous, g_last_representation, transition.live_logit_after))) {
        xSemaphoreGive(g_lock);
        return false;
    }
    const int old_active = g_active_slot;
    g_active_slot = g_previous_slot;
    g_previous_slot = old_active;
    xSemaphoreGive(g_lock);
    transition.elapsed_us = esp_timer_get_time() - started;
    return true;
}

bool runtime_shared_head_evaluate(
    const float *representation, size_t values, float &logit,
    RuntimeHeadSnapshot &head_snapshot)
{
    if (g_lock == nullptr || representation == nullptr || values != 64) return false;
    xSemaphoreTake(g_lock, portMAX_DELAY);
    std::memcpy(g_last_representation, representation, sizeof(g_last_representation));
    g_has_last_representation = true;
    const HeadSlot &active = g_slots[g_active_slot];
    const bool result = evaluate_slot(active, representation, logit);
    head_snapshot = snapshot(active);
    xSemaphoreGive(g_lock);
    return result;
}

RuntimeHeadSnapshot runtime_shared_head_snapshot()
{
    RuntimeHeadSnapshot result{};
    if (g_lock == nullptr) return result;
    xSemaphoreTake(g_lock, portMAX_DELAY);
    result = snapshot(g_slots[g_active_slot]);
    xSemaphoreGive(g_lock);
    return result;
}
