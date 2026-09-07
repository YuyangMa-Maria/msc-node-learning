// Thread-safe control of automatic and host-triggered ASN sampling.

#include "asn_sampling_control.h"

#include <atomic>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

namespace {

constexpr char kTag[] = "asn_sampling";
SemaphoreHandle_t g_sample_trigger = nullptr;
std::atomic_bool g_automatic{true};

void drain_trigger()
{
    if (g_sample_trigger != nullptr) {
        while (xSemaphoreTake(g_sample_trigger, 0) == pdTRUE) {
        }
    }
}

}  // namespace

bool asn_sampling_control_init()
{
    if (g_sample_trigger == nullptr) {
        g_sample_trigger = xSemaphoreCreateBinary();
    }
    g_automatic.store(true);
    ESP_LOGI(kTag, "SAMPLING_CONTROL_READY mode=automatic");
    return g_sample_trigger != nullptr;
}

bool asn_sampling_apply(sampling_control::Command command, uint16_t request_id)
{
    if (g_sample_trigger == nullptr) {
        return false;
    }
    switch (command) {
    case sampling_control::Command::kAutomatic:
        g_automatic.store(true);
        xSemaphoreGive(g_sample_trigger);
        break;
    case sampling_control::Command::kManual:
        g_automatic.store(false);
        drain_trigger();
        break;
    case sampling_control::Command::kSampleOnce:
        if (g_automatic.load()) {
            ESP_LOGW(kTag, "SAMPLING_REQUEST_REJECT request=%u reason=automatic_mode", request_id);
            return false;
        }
        xSemaphoreGive(g_sample_trigger);
        break;
    default:
        return false;
    }
    ESP_LOGI(
        kTag,
        "SAMPLING_CONTROL command=%s request=%u result=PASS mode=%s",
        sampling_control::command_name(command),
        request_id,
        g_automatic.load() ? "automatic" : "manual");
    return true;
}

bool asn_sampling_wait_for_window()
{
    if (g_automatic.load()) {
        return true;
    }
    ESP_LOGI(kTag, "SAMPLING_WAIT node=ASN mode=manual");
    return xSemaphoreTake(g_sample_trigger, portMAX_DELAY) == pdTRUE;
}

bool asn_sampling_is_automatic()
{
    return g_automatic.load();
}
