// VSN application entry point and camera initialisation for XIAO ESP32-S3 Sense.
// Startup performs camera and model parity checks before enabling the continuous
// sensing task, so a failed deployment contract does not emit normal decisions.

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

#include "esp_camera.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sensor.h"
#include "vsn_ble_client.h"
#include "vsn_model_runtime.h"
#include "system_host_protocol.h"

#define CAM_PIN_PWDN (-1)
#define CAM_PIN_RESET (-1)
#define CAM_PIN_XCLK 10
#define CAM_PIN_SIOD 40
#define CAM_PIN_SIOC 39
#define CAM_PIN_D7 48
#define CAM_PIN_D6 11
#define CAM_PIN_D5 12
#define CAM_PIN_D4 14
#define CAM_PIN_D3 16
#define CAM_PIN_D2 18
#define CAM_PIN_D1 17
#define CAM_PIN_D0 15
#define CAM_PIN_VSYNC 38
#define CAM_PIN_HREF 47
#define CAM_PIN_PCLK 13
#define CAMERA_BENCHMARK_FRAMES 30

static const char *TAG = "vsn_bringup";

void vsn_run_model_parity(void);

static camera_config_t make_camera_config(void)
{
    camera_config_t config = {
        .pin_pwdn = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_QVGA,
        .jpeg_quality = 12,
        .fb_count = 1,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    };
    return config;
}

static void log_memory(const char *stage)
{
    const size_t internal_free =
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const size_t internal_largest =
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    ESP_LOGI(
        TAG,
        "memory stage=%s internal_free=%u internal_largest=%u psram_free=%u",
        stage,
        (unsigned)internal_free,
        (unsigned)internal_largest,
        (unsigned)psram_free);
}

void app_main(void)
{
    const bool host_ready = system_host_protocol_init();
    ESP_LOGI(TAG, "HOST_PROTOCOL_INIT=%s", host_ready ? "PASS" : "FAIL");
    ESP_LOGI(TAG, "XIAO ESP32S3 Sense VSN camera bring-up");
    ESP_LOGI(
        TAG,
        "psram_detected=%s psram_size=%u",
        esp_psram_is_initialized() ? "yes" : "no",
        (unsigned)esp_psram_get_size());
    log_memory("before_camera");

    camera_config_t camera_config = make_camera_config();
    ESP_ERROR_CHECK(esp_camera_init(&camera_config));

    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor == NULL) {
        ESP_LOGE(TAG, "Camera sensor handle is null");
        return;
    }

    if (sensor->id.PID == OV3660_PID) {
        sensor->set_vflip(sensor, 1);
        sensor->set_brightness(sensor, 1);
        sensor->set_saturation(sensor, -2);
    }

    ESP_LOGI(TAG, "camera_pid=0x%04x", sensor->id.PID);
    log_memory("after_camera");

    camera_fb_t *warmup_frame = esp_camera_fb_get();
    if (warmup_frame == NULL) {
        ESP_LOGE(TAG, "Warm-up frame capture failed");
        return;
    }
    esp_camera_fb_return(warmup_frame);

    int64_t capture_wait_us[CAMERA_BENCHMARK_FRAMES] = {0};
    uint32_t captured_frames = 0;
    size_t total_jpeg_bytes = 0;

    for (uint32_t frame_index = 0; frame_index < CAMERA_BENCHMARK_FRAMES; frame_index++) {
        const int64_t capture_start_us = esp_timer_get_time();
        camera_fb_t *frame = esp_camera_fb_get();
        const int64_t frame_wait_us = esp_timer_get_time() - capture_start_us;

        if (frame == NULL) {
            ESP_LOGE(TAG, "Frame capture failed at benchmark frame=%" PRIu32, frame_index + 1);
            break;
        }

        capture_wait_us[captured_frames] = frame_wait_us;
        total_jpeg_bytes += frame->len;
        captured_frames++;
        ESP_LOGI(
            TAG,
            "frame=%" PRIu32 " width=%u height=%u bytes=%u format=%d capture_wait_ms=%.3f",
            frame_index + 1,
            frame->width,
            frame->height,
            (unsigned)frame->len,
            frame->format,
            frame_wait_us / 1000.0);

        esp_camera_fb_return(frame);
    }

    if (captured_frames > 0) {
        for (uint32_t i = 1; i < captured_frames; i++) {
            const int64_t value = capture_wait_us[i];
            uint32_t j = i;
            while (j > 0 && capture_wait_us[j - 1] > value) {
                capture_wait_us[j] = capture_wait_us[j - 1];
                j--;
            }
            capture_wait_us[j] = value;
        }

        int64_t total_wait_us = 0;
        for (uint32_t i = 0; i < captured_frames; i++) {
            total_wait_us += capture_wait_us[i];
        }
        const uint32_t p50_index = (captured_frames - 1) / 2;
        const uint32_t p95_index = (captured_frames * 95 + 99) / 100 - 1;

        ESP_LOGI(
            TAG,
            "benchmark frames=%" PRIu32
            " capture_mean_ms=%.3f capture_p50_ms=%.3f capture_p95_ms=%.3f"
            " capture_min_ms=%.3f capture_max_ms=%.3f mean_jpeg_bytes=%.1f",
            captured_frames,
            total_wait_us / (1000.0 * captured_frames),
            capture_wait_us[p50_index] / 1000.0,
            capture_wait_us[p95_index] / 1000.0,
            capture_wait_us[0] / 1000.0,
            capture_wait_us[captured_frames - 1] / 1000.0,
            total_jpeg_bytes / (double)captured_frames);
    }

    log_memory("after_benchmark");
    ESP_LOGI(TAG, "Camera benchmark complete; starting embedded ESP-DL parity test.");
    vsn_run_model_parity();
    log_memory("camera_and_model_resident");
    ESP_LOGI(TAG, "Camera and model tests complete; starting BLE central.");
    const bool ble_ready = vsn_ble_init();
    ESP_LOGI(TAG, "VSN_BLE_INIT=%s", ble_ready ? "PASS" : "FAIL");
    log_memory("camera_model_and_ble_resident");

    const bool online_ready = ble_ready && vsn_start_online_runtime();
    ESP_LOGI(TAG, "VSN_ONLINE_RUNTIME=%s", online_ready ? "PASS" : "FAIL");
    system_host_report_boot(true, online_ready, ble_ready);

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
