// Standalone microphone and model bring-up retained as a development diagnostic.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "dl_model_base.hpp"
#include "driver/i2s_pdm.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace {

constexpr char kTag[] = "asn_bringup";
constexpr char kModelPartition[] = "model";
constexpr int kSampleRate = 16000;
constexpr int kWindowSeconds = 5;
constexpr int kWindowSamples = kSampleRate * kWindowSeconds;
constexpr gpio_num_t kPdmClock = GPIO_NUM_42;
constexpr gpio_num_t kPdmData = GPIO_NUM_41;
constexpr int kBenchmarkWarmup = 3;
constexpr int kBenchmarkRuns = 30;
constexpr size_t kReadChunkSamples = 1024;

void log_memory(const char *stage)
{
    ESP_LOGI(
        kTag,
        "MEMORY stage=%s internal_free=%u internal_largest=%u psram_free=%u",
        stage,
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
}

dl::Model *load_and_test_model()
{
    log_memory("before_model_load");
    const int64_t start_us = esp_timer_get_time();
    auto *model = new dl::Model(
        kModelPartition,
        fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
        0,
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,
        false);
    ESP_LOGI(kTag, "MODEL_LOAD elapsed_ms=%.3f", (esp_timer_get_time() - start_us) / 1000.0);
    log_memory("after_model_load");

    const int64_t test_start_us = esp_timer_get_time();
    const esp_err_t result = model->test();
    ESP_LOGI(
        kTag,
        "MODEL_PARITY=%s result=%s elapsed_ms=%.3f",
        result == ESP_OK ? "PASS" : "FAIL",
        esp_err_to_name(result),
        (esp_timer_get_time() - test_start_us) / 1000.0);
    model->profile_memory();
    return model;
}

void benchmark_model(dl::Model *model)
{
    for (int i = 0; i < kBenchmarkWarmup; ++i) {
        model->run(dl::RUNTIME_MODE_MULTI_CORE);
    }
    std::array<int64_t, kBenchmarkRuns> latency_us{};
    int64_t total_us = 0;
    for (int i = 0; i < kBenchmarkRuns; ++i) {
        const int64_t start_us = esp_timer_get_time();
        model->run(dl::RUNTIME_MODE_MULTI_CORE);
        latency_us[i] = esp_timer_get_time() - start_us;
        total_us += latency_us[i];
    }
    std::sort(latency_us.begin(), latency_us.end());
    const int p50 = (kBenchmarkRuns - 1) / 2;
    const int p95 = (kBenchmarkRuns * 95 + 99) / 100 - 1;
    ESP_LOGI(
        kTag,
        "MODEL_BENCHMARK runs=%d mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f min_ms=%.3f max_ms=%.3f",
        kBenchmarkRuns,
        total_us / (1000.0 * kBenchmarkRuns),
        latency_us[p50] / 1000.0,
        latency_us[p95] / 1000.0,
        latency_us.front() / 1000.0,
        latency_us.back() / 1000.0);

    dl::TensorBase *output = model->get_output();
    if (output != nullptr) {
        ESP_LOGI(
            kTag,
            "MODEL_OUTPUT dtype=%d exponent=%d size=%u",
            static_cast<int>(output->get_dtype()),
            output->get_exponent(),
            static_cast<unsigned>(output->get_size()));
        if (output->get_dtype() == dl::DATA_TYPE_INT8) {
            for (int index = 0; index < std::min<int>(4, output->get_size()); ++index) {
                const int raw = static_cast<int>(output->get_element<int8_t>(index));
                const float value = std::ldexp(static_cast<float>(raw), output->get_exponent());
                ESP_LOGI(kTag, "MODEL_OUTPUT index=%d raw=%d dequant=%.6f", index, raw, value);
            }
        }
    }
}

i2s_chan_handle_t initialise_microphone()
{
    i2s_chan_handle_t rx = nullptr;
    i2s_chan_config_t channel_config = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&channel_config, nullptr, &rx));

    i2s_pdm_rx_config_t config = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(kSampleRate),
        .slot_cfg = I2S_PDM_RX_SLOT_PCM_FMT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = kPdmClock,
            .dins = {kPdmData, GPIO_NUM_NC, GPIO_NUM_NC, GPIO_NUM_NC},
            .invert_flags = {.clk_inv = false},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx, &config));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
    ESP_LOGI(
        kTag,
        "MIC_INITIALISED format=PDM_MONO_PCM16 sample_rate=%d clock_gpio=%d data_gpio=%d",
        kSampleRate,
        static_cast<int>(kPdmClock),
        static_cast<int>(kPdmData));
    return rx;
}

bool capture_window(i2s_chan_handle_t rx, int16_t *window)
{
    std::array<int16_t, kReadChunkSamples> chunk{};
    size_t captured = 0;
    const int64_t started_us = esp_timer_get_time();
    while (captured < kWindowSamples) {
        const size_t requested_samples = std::min(kReadChunkSamples, static_cast<size_t>(kWindowSamples) - captured);
        size_t bytes_read = 0;
        const esp_err_t result = i2s_channel_read(
            rx,
            chunk.data(),
            requested_samples * sizeof(int16_t),
            &bytes_read,
            pdMS_TO_TICKS(1000));
        if (result != ESP_OK || bytes_read == 0) {
            ESP_LOGE(kTag, "MIC_READ_FAIL result=%s captured=%u", esp_err_to_name(result), static_cast<unsigned>(captured));
            return false;
        }
        const size_t samples_read = bytes_read / sizeof(int16_t);
        std::memcpy(window + captured, chunk.data(), samples_read * sizeof(int16_t));
        captured += samples_read;
    }

    int16_t minimum = std::numeric_limits<int16_t>::max();
    int16_t maximum = std::numeric_limits<int16_t>::min();
    int64_t sum = 0;
    double sum_squares = 0.0;
    size_t zeros = 0;
    size_t clipped = 0;
    for (size_t i = 0; i < captured; ++i) {
        const int32_t sample = window[i];
        minimum = std::min(minimum, window[i]);
        maximum = std::max(maximum, window[i]);
        sum += sample;
        sum_squares += static_cast<double>(sample) * sample;
        zeros += sample == 0 ? 1 : 0;
        clipped += std::abs(sample) >= 32000 ? 1 : 0;
    }
    const double mean = static_cast<double>(sum) / captured;
    const double rms = std::sqrt(sum_squares / captured);
    ESP_LOGI(
        kTag,
        "MIC_WINDOW_COMPLETE samples=%u elapsed_ms=%.3f mean=%.3f rms=%.3f min=%d max=%d zero_rate=%.6f clip_rate=%.6f",
        static_cast<unsigned>(captured),
        (esp_timer_get_time() - started_us) / 1000.0,
        mean,
        rms,
        static_cast<int>(minimum),
        static_cast<int>(maximum),
        static_cast<double>(zeros) / captured,
        static_cast<double>(clipped) / captured);
    return true;
}

}  // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(kTag, "=== XIAO ESP32S3 ASN OFFLINE BRING-UP ===");
    ESP_LOGI(
        kTag,
        "PSRAM detected=%s size_bytes=%u",
        esp_psram_is_initialized() ? "YES" : "NO",
        static_cast<unsigned>(esp_psram_get_size()));
    log_memory("boot");

    dl::Model *model = load_and_test_model();
    benchmark_model(model);

    i2s_chan_handle_t microphone = initialise_microphone();
    auto *window = static_cast<int16_t *>(heap_caps_malloc(kWindowSamples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (window == nullptr) {
        ESP_LOGE(kTag, "PCM_WINDOW_ALLOCATION_FAIL bytes=%u", static_cast<unsigned>(kWindowSamples * sizeof(int16_t)));
    } else {
        log_memory("model_mic_pcm_resident");
        const bool captured = capture_window(microphone, window);
        ESP_LOGI(kTag, "BRINGUP_RESULT=%s", captured ? "PASS" : "FAIL");
        heap_caps_free(window);
    }
    log_memory("bringup_complete");
    ESP_LOGI(kTag, "Offline bring-up complete; Wi-Fi and BLE remain disabled.");
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
