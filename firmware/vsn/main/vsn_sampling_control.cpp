// Thread-safe control of automatic and coordinated VSN sampling.

#include "vsn_sampling_control.h"

#include <atomic>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

namespace {

constexpr char kTag[] = "vsn_sampling";
SemaphoreHandle_t g_sample_trigger = nullptr;
std::atomic_bool g_automatic{true};
std::atomic_bool g_waiting_for_asn{false};

void drain_trigger()
{
    if (g_sample_trigger != nullptr) {
        while (xSemaphoreTake(g_sample_trigger, 0) == pdTRUE) {
        }
    }
}

}  // namespace

bool vsn_sampling_control_init()
{
    if (g_sample_trigger == nullptr) {
        g_sample_trigger = xSemaphoreCreateBinary();
    }
    g_automatic.store(true);
    g_waiting_for_asn.store(false);
    ESP_LOGI(kTag, "SAMPLING_CONTROL_READY mode=automatic");
    return g_sample_trigger != nullptr;
}

void vsn_sampling_set_automatic(bool automatic)
{
    g_automatic.store(automatic);
    g_waiting_for_asn.store(false);
    if (automatic) {
        xSemaphoreGive(g_sample_trigger);
    } else {
        drain_trigger();
    }
    ESP_LOGI(kTag, "SAMPLING_MODE mode=%s", automatic ? "automatic" : "manual");
}

bool vsn_sampling_trigger_once()
{
    return g_sample_trigger != nullptr && !g_automatic.load() &&
        xSemaphoreGive(g_sample_trigger) == pdTRUE;
}

bool vsn_sampling_schedule_after_asn()
{
    bool expected = false;
    return g_sample_trigger != nullptr && !g_automatic.load() &&
        g_waiting_for_asn.compare_exchange_strong(expected, true);
}

void vsn_sampling_cancel_after_asn()
{
    g_waiting_for_asn.store(false);
}

void vsn_sampling_notify_asn_ready()
{
    if (g_waiting_for_asn.exchange(false) && g_sample_trigger != nullptr) {
        xSemaphoreGive(g_sample_trigger);
        ESP_LOGI(kTag, "SAMPLING_TRIGGER node=VSN source=fresh_asn_summary");
    }
}

bool vsn_sampling_wait_for_frame()
{
    if (g_automatic.load()) {
        return true;
    }
    ESP_LOGI(kTag, "SAMPLING_WAIT node=VSN mode=manual");
    return g_sample_trigger != nullptr &&
        xSemaphoreTake(g_sample_trigger, portMAX_DELAY) == pdTRUE;
}

bool vsn_sampling_is_automatic()
{
    return g_automatic.load();
}
