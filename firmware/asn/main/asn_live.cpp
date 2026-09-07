// Final ASN pipeline: PDM capture, log-Mel frontend, INT8 inference and BLE output.
// The risk head and signal-validity head are evaluated separately so invalid
// microphone input can withdraw its vote without being mistaken for low risk.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "asn_ble_server.h"
#include "asn_model_contract.h"
#include "asn_sampling_control.h"
#include "dl_fbank.hpp"
#include "dl_model_base.hpp"
#include "driver/i2s_pdm.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "node_summary_protocol.h"
#include "runtime_shared_head.h"

namespace {

constexpr char kTag[] = "asn_live";
constexpr char kEncoderPartition[] = "encoder";
constexpr char kValidityPartition[] = "validity";
constexpr char kGoldenPartition[] = "golden";
constexpr gpio_num_t kPdmClock = GPIO_NUM_42;
constexpr gpio_num_t kPdmData = GPIO_NUM_41;
constexpr size_t kReadChunkSamples = 1024;
constexpr int kPadSamples = ASN_N_FFT / 2;
constexpr int kPaddedSamples = ASN_WINDOW_SAMPLES + 2 * kPadSamples;
constexpr int kFeatureValues = ASN_N_MELS * ASN_FEATURE_FRAMES;
constexpr float kTopDbNaturalLogRange = 80.0f * 2.302585093f / 10.0f;
constexpr int kGoldenVectors = 3;
constexpr int kBenchmarkWarmup = 3;
constexpr int kBenchmarkRuns = 30;

struct NodeOutput {
    std::array<float, ASN_REPRESENTATION_COUNT> representation{};
    std::array<float, ASN_VALIDITY_OUTPUT_COUNT> validity_logits{};
    std::array<float, 3> validity_probabilities{};
    RuntimeHeadSnapshot head{};
    float logit = 0.0f;
    float risk_score = 0.0f;
    float confidence = 0.0f;
    int validity_index = ASN_VALIDITY_INVALID_INDEX;
    bool risk_triggered = false;
    bool accepted = false;
};

// The validity model can withdraw an ASN vote before risk fusion.
uint8_t operational_status(const NodeOutput &node)
{
    using node_learning::NodeStatus;
    if (!node.accepted) {
        return static_cast<uint8_t>(NodeStatus::kInvalid);
    }
    if (node.risk_score >= 0.85f) {
        return static_cast<uint8_t>(NodeStatus::kCritical);
    }
    if (node.risk_triggered) {
        return static_cast<uint8_t>(NodeStatus::kWarning);
    }
    return static_cast<uint8_t>(NodeStatus::kNormal);
}

void log_memory(const char *stage)
{
    ESP_LOGI(
        kTag,
        "MEMORY stage=%s internal_free=%u internal_largest=%u psram_free=%u "
        "main_stack_high_water=%u",
        stage,
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
        static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr)));
}

float sigmoid(float value)
{
    if (value >= 0.0f) {
        const float exp_negative = std::exp(-value);
        return 1.0f / (1.0f + exp_negative);
    }
    const float exp_value = std::exp(value);
    return exp_value / (1.0f + exp_value);
}

float binary_entropy_confidence(float probability)
{
    const float bounded = std::clamp(probability, 1e-7f, 1.0f - 1e-7f);
    const float entropy = -(bounded * std::log2(bounded) + (1.0f - bounded) * std::log2(1.0f - bounded));
    return 1.0f - entropy;
}

dl::Model *load_and_test_model(
    const char *partition,
    const char *role,
    size_t expected_output_values)
{
    const int64_t start_us = esp_timer_get_time();
    auto *model = new dl::Model(
        partition,
        fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
        0,
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,
        false);
    ESP_LOGI(kTag, "MODEL_LOAD role=%s partition=%s elapsed_ms=%.3f",
        role, partition, (esp_timer_get_time() - start_us) / 1000.0);
    const esp_err_t result = model->test();
    ESP_LOGI(kTag, "MODEL_EMBEDDED_PARITY role=%s result=%s error=%s",
        role, result == ESP_OK ? "PASS" : "FAIL", esp_err_to_name(result));
    model->profile_memory();

    dl::TensorBase *input = model->get_input();
    dl::TensorBase *output = model->get_output();
    const bool contract_ok = result == ESP_OK && input != nullptr && output != nullptr &&
        input->get_size() == kFeatureValues && output->get_size() == expected_output_values;
    ESP_LOGI(
        kTag,
        "MODEL_CONTRACT role=%s input_dtype=%d input_exponent=%d input_size=%u "
        "output_dtype=%d output_exponent=%d output_size=%u result=%s",
        role,
        input != nullptr ? static_cast<int>(input->get_dtype()) : -1,
        input != nullptr ? input->get_exponent() : 0,
        input != nullptr ? static_cast<unsigned>(input->get_size()) : 0,
        output != nullptr ? static_cast<int>(output->get_dtype()) : -1,
        output != nullptr ? output->get_exponent() : 0,
        output != nullptr ? static_cast<unsigned>(output->get_size()) : 0,
        contract_ok ? "PASS" : "FAIL");
    if (!contract_ok) {
        delete model;
        return nullptr;
    }
    return model;
}

void benchmark_model(dl::Model *model, const char *role)
{
    for (int index = 0; index < kBenchmarkWarmup; ++index) {
        model->run(dl::RUNTIME_MODE_MULTI_CORE);
    }
    std::array<int64_t, kBenchmarkRuns> latency_us{};
    int64_t total_us = 0;
    for (int index = 0; index < kBenchmarkRuns; ++index) {
        const int64_t start_us = esp_timer_get_time();
        model->run(dl::RUNTIME_MODE_MULTI_CORE);
        latency_us[index] = esp_timer_get_time() - start_us;
        total_us += latency_us[index];
    }
    std::sort(latency_us.begin(), latency_us.end());
    const int p50 = (kBenchmarkRuns - 1) / 2;
    const int p95 = (kBenchmarkRuns * 95 + 99) / 100 - 1;
    ESP_LOGI(
        kTag,
        "MODEL_BENCHMARK role=%s runs=%d mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f min_ms=%.3f max_ms=%.3f",
        role,
        kBenchmarkRuns,
        total_us / (1000.0 * kBenchmarkRuns),
        latency_us[p50] / 1000.0,
        latency_us[p95] / 1000.0,
        latency_us.front() / 1000.0,
        latency_us.back() / 1000.0);
}

dl::audio::SpeechFeatureConfig frontend_config()
{
    dl::audio::SpeechFeatureConfig config;
    config.sample_rate = ASN_SAMPLE_RATE_HZ;
    config.frame_length = ASN_N_FFT * 1000 / ASN_SAMPLE_RATE_HZ;
    config.frame_shift = ASN_HOP_LENGTH * 1000 / ASN_SAMPLE_RATE_HZ;
    config.num_mel_bins = ASN_N_MELS;
    config.preemphasis = 0.0f;
    config.window_type = dl::audio::WinType::HANN;
    config.low_freq = 0.0f;
    config.high_freq = ASN_SAMPLE_RATE_HZ / 2.0f;
    config.log_epsilon = 1e-10f;
    config.use_log_fbank = 1;
    config.use_power = true;
    config.use_energy = false;
    config.remove_dc_offset = false;
    return config;
}

void reflect_pad_pcm(const int16_t *pcm, int16_t *padded)
{
    std::memcpy(padded + kPadSamples, pcm, ASN_WINDOW_SAMPLES * sizeof(int16_t));
    for (int index = 0; index < kPadSamples; ++index) {
        padded[kPadSamples - 1 - index] = pcm[index + 1];
        padded[kPadSamples + ASN_WINDOW_SAMPLES + index] = pcm[ASN_WINDOW_SAMPLES - 2 - index];
    }
}

bool write_model_input(dl::Model *model, const float *frame_major_features, float &feature_mean, float &feature_std)
{
    dl::TensorBase *input = model->get_input();
    if (input == nullptr ||
        (input->get_dtype() != dl::DATA_TYPE_INT8 && input->get_dtype() != dl::DATA_TYPE_INT16) ||
        input->get_size() != kFeatureValues) {
        ESP_LOGE(kTag, "MODEL_INPUT_CONTRACT_FAIL");
        return false;
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (int index = 0; index < kFeatureValues; ++index) {
        maximum = std::max(maximum, frame_major_features[index]);
    }
    const float floor = maximum - kTopDbNaturalLogRange;

    double sum = 0.0;
    double sum_squares = 0.0;
    for (int index = 0; index < kFeatureValues; ++index) {
        const float value = std::max(frame_major_features[index], floor);
        sum += value;
        sum_squares += static_cast<double>(value) * value;
    }
    feature_mean = static_cast<float>(sum / kFeatureValues);
    const double variance = std::max(
        (sum_squares - static_cast<double>(kFeatureValues) * feature_mean * feature_mean) / (kFeatureValues - 1),
        1e-10);
    feature_std = static_cast<float>(std::sqrt(variance));

    const float input_scale = std::ldexp(1.0f, input->get_exponent());
    size_t clipped = 0;
    if (input->get_dtype() == dl::DATA_TYPE_INT16) {
        int16_t *destination = input->get_element_ptr<int16_t>();
        for (int mel = 0; mel < ASN_N_MELS; ++mel) {
            for (int frame = 0; frame < ASN_FEATURE_FRAMES; ++frame) {
                const float value = std::max(frame_major_features[frame * ASN_N_MELS + mel], floor);
                const float normalised = (value - feature_mean) / feature_std;
                int quantised = static_cast<int>(std::nearbyint(normalised / input_scale));
                clipped += (quantised < -32768 || quantised > 32767) ? 1 : 0;
                quantised = std::clamp(quantised, -32768, 32767);
                destination[mel * ASN_FEATURE_FRAMES + frame] = static_cast<int16_t>(quantised);
            }
        }
    } else {
        int8_t *destination = input->get_element_ptr<int8_t>();
        for (int mel = 0; mel < ASN_N_MELS; ++mel) {
            for (int frame = 0; frame < ASN_FEATURE_FRAMES; ++frame) {
                const float value = std::max(frame_major_features[frame * ASN_N_MELS + mel], floor);
                const float normalised = (value - feature_mean) / feature_std;
                int quantised = static_cast<int>(std::nearbyint(normalised / input_scale));
                clipped += (quantised < -128 || quantised > 127) ? 1 : 0;
                quantised = std::clamp(quantised, -128, 127);
                destination[mel * ASN_FEATURE_FRAMES + frame] = static_cast<int8_t>(quantised);
            }
        }
    }
    ESP_LOGI(
        kTag,
        "FRONTEND_NORMALISATION mean=%.6f std=%.6f input_scale=%.6f clipped=%u/%d",
        feature_mean,
        feature_std,
        input_scale,
        static_cast<unsigned>(clipped),
        kFeatureValues);
    return true;
}

bool decode_output(
    dl::Model *encoder_model,
    dl::Model *validity_model,
    NodeOutput &node)
{
    dl::TensorBase *encoder_output = encoder_model->get_output();
    dl::TensorBase *validity_output = validity_model->get_output();
    if (encoder_output == nullptr || validity_output == nullptr ||
        (encoder_output->get_dtype() != dl::DATA_TYPE_INT8 &&
         encoder_output->get_dtype() != dl::DATA_TYPE_INT16) ||
        validity_output->get_dtype() != dl::DATA_TYPE_INT8 ||
        encoder_output->get_size() != ASN_REPRESENTATION_COUNT ||
        validity_output->get_size() != ASN_VALIDITY_OUTPUT_COUNT) {
        return false;
    }
    for (int index = 0; index < ASN_REPRESENTATION_COUNT; ++index) {
        const float raw = encoder_output->get_dtype() == dl::DATA_TYPE_INT16
            ? static_cast<float>(encoder_output->get_element<int16_t>(index))
            : static_cast<float>(encoder_output->get_element<int8_t>(index));
        node.representation[index] = std::ldexp(raw, encoder_output->get_exponent());
    }
    for (int index = 0; index < ASN_VALIDITY_OUTPUT_COUNT; ++index) {
        node.validity_logits[index] = std::ldexp(
            static_cast<float>(validity_output->get_element<int8_t>(index)),
            validity_output->get_exponent());
    }
    if (!runtime_shared_head_evaluate(
            node.representation.data(),
            node.representation.size(),
            node.logit,
            node.head)) {
        return false;
    }
    node.risk_score = sigmoid(node.logit);
    node.risk_triggered = node.risk_score >= node.head.threshold;

    float maximum = -std::numeric_limits<float>::infinity();
    for (int index = 0; index < 3; ++index) {
        maximum = std::max(
            maximum,
            node.validity_logits[index + 1] / ASN_VALIDITY_TEMPERATURE);
    }
    float denominator = 0.0f;
    for (int index = 0; index < 3; ++index) {
        node.validity_probabilities[index] = std::exp(
            node.validity_logits[index + 1] / ASN_VALIDITY_TEMPERATURE - maximum);
        denominator += node.validity_probabilities[index];
    }
    if (!std::isfinite(denominator) || denominator <= 0.0f) {
        return false;
    }
    for (float &probability : node.validity_probabilities) {
        probability /= denominator;
    }
    node.validity_index = static_cast<int>(std::max_element(
        node.validity_probabilities.begin(),
        node.validity_probabilities.end()) - node.validity_probabilities.begin());
    node.accepted = node.validity_index != ASN_VALIDITY_INVALID_INDEX;
    const float validity_confidence = 1.0f -
        node.validity_probabilities[ASN_VALIDITY_INVALID_INDEX];
    // Confidence is reduced when the validity head assigns probability to the
    // invalid class, even if the binary risk output itself is very decisive.
    node.confidence = binary_entropy_confidence(node.risk_score) * validity_confidence;
    return std::isfinite(node.risk_score) && std::isfinite(node.confidence);
}

bool infer_pcm(
    dl::Model *encoder_model,
    dl::Model *validity_model,
    dl::audio::Fbank &frontend,
    const int16_t *pcm,
    NodeOutput &node,
    double &frontend_ms,
    double &inference_ms)
{
    auto *padded = static_cast<int16_t *>(heap_caps_malloc(kPaddedSamples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    auto *features = static_cast<float *>(heap_caps_malloc(kFeatureValues * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (padded == nullptr || features == nullptr) {
        ESP_LOGE(kTag, "FRONTEND_ALLOCATION_FAIL");
        if (padded != nullptr) heap_caps_free(padded);
        if (features != nullptr) heap_caps_free(features);
        return false;
    }
    log_memory("frontend_peak_buffers_allocated");

    const int64_t frontend_start_us = esp_timer_get_time();
    reflect_pad_pcm(pcm, padded);
    const esp_err_t frontend_result = frontend.process(padded, kPaddedSamples, features);
    float feature_mean = 0.0f;
    float feature_std = 0.0f;
    const bool input_ok = frontend_result == ESP_OK &&
        write_model_input(encoder_model, features, feature_mean, feature_std) &&
        write_model_input(validity_model, features, feature_mean, feature_std);
    frontend_ms = (esp_timer_get_time() - frontend_start_us) / 1000.0;
    heap_caps_free(features);
    heap_caps_free(padded);
    if (!input_ok) {
        ESP_LOGE(kTag, "FRONTEND_FAIL result=%s", esp_err_to_name(frontend_result));
        return false;
    }

    const int64_t inference_start_us = esp_timer_get_time();
    encoder_model->run(dl::RUNTIME_MODE_MULTI_CORE);
    validity_model->run(dl::RUNTIME_MODE_MULTI_CORE);
    inference_ms = (esp_timer_get_time() - inference_start_us) / 1000.0;
    return decode_output(encoder_model, validity_model, node);
}

void log_node_output(const char *source, int index, const NodeOutput &node, double frontend_ms, double inference_ms)
{
    ESP_LOGI(
        kTag,
        "NODE_OUTPUT source=%s index=%d risk_score=%.6f confidence=%.6f risk_triggered=%s "
        "logit=%.6f head_version=0x%04x generation=%u format=%s threshold=%.3f "
        "validity=[%.6f,%.6f,%.6f] validity_index=%d status=%s "
        "frontend_ms=%.3f inference_ms=%.3f",
        source,
        index,
        node.risk_score,
        node.confidence,
        node.risk_triggered ? "yes" : "no",
        node.logit,
        node.head.version,
        static_cast<unsigned>(node.head.generation),
        model_exchange::format_name(node.head.format),
        node.head.threshold,
        node.validity_probabilities[0],
        node.validity_probabilities[1],
        node.validity_probabilities[2],
        node.validity_index,
        node.accepted ? "accepted" : "invalid",
        frontend_ms,
        inference_ms);
}

bool run_golden_frontend_parity(
    dl::Model *encoder_model,
    dl::Model *validity_model,
    dl::audio::Fbank &frontend,
    int16_t *pcm)
{
    const esp_partition_t *partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        ESP_PARTITION_SUBTYPE_ANY,
        kGoldenPartition);
    if (partition == nullptr) {
        ESP_LOGE(kTag, "GOLDEN_PARTITION_NOT_FOUND");
        return false;
    }
    constexpr std::array<int, kGoldenVectors> expected_validity = {0, 0, 2};
    bool all_agree = true;
    for (int index = 0; index < kGoldenVectors; ++index) {
        ESP_ERROR_CHECK(esp_partition_read(
            partition,
            index * ASN_WINDOW_SAMPLES * sizeof(int16_t),
            pcm,
            ASN_WINDOW_SAMPLES * sizeof(int16_t)));
        NodeOutput node;
        double frontend_ms = 0.0;
        double inference_ms = 0.0;
        const bool ran = infer_pcm(
            encoder_model,
            validity_model,
            frontend,
            pcm,
            node,
            frontend_ms,
            inference_ms);
        const bool agreement = ran && std::isfinite(node.logit) &&
            node.validity_index == expected_validity[index];
        log_node_output("golden", index, node, frontend_ms, inference_ms);
        ESP_LOGI(kTag, "GOLDEN_FRONTEND vector=%d expected_validity=%d result=%s",
            index, expected_validity[index], agreement ? "PASS" : "FAIL");
        all_agree = all_agree && agreement;
    }
    ESP_LOGI(kTag, "GOLDEN_FRONTEND_PARITY=%s", all_agree ? "PASS" : "FAIL");
    return all_agree;
}

i2s_chan_handle_t initialise_microphone()
{
    i2s_chan_handle_t rx = nullptr;
    i2s_chan_config_t channel_config = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&channel_config, nullptr, &rx));
    i2s_pdm_rx_config_t config = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(ASN_SAMPLE_RATE_HZ),
        .slot_cfg = I2S_PDM_RX_SLOT_PCM_FMT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = kPdmClock,
            .dins = {kPdmData, GPIO_NUM_NC, GPIO_NUM_NC, GPIO_NUM_NC},
            .invert_flags = {.clk_inv = false},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx, &config));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
    ESP_LOGI(kTag, "MIC_INITIALISED sample_rate=%d clock_gpio=%d data_gpio=%d", ASN_SAMPLE_RATE_HZ, kPdmClock, kPdmData);
    return rx;
}

bool capture_window(i2s_chan_handle_t rx, int16_t *window)
{
    std::array<int16_t, kReadChunkSamples> chunk{};
    size_t captured = 0;
    const int64_t started_us = esp_timer_get_time();
    while (captured < ASN_WINDOW_SAMPLES) {
        const size_t requested = std::min(kReadChunkSamples, static_cast<size_t>(ASN_WINDOW_SAMPLES) - captured);
        size_t bytes_read = 0;
        const esp_err_t result = i2s_channel_read(rx, chunk.data(), requested * sizeof(int16_t), &bytes_read, pdMS_TO_TICKS(1000));
        if (result != ESP_OK || bytes_read == 0) return false;
        const size_t samples_read = bytes_read / sizeof(int16_t);
        std::memcpy(window + captured, chunk.data(), samples_read * sizeof(int16_t));
        captured += samples_read;
    }

    int16_t minimum = std::numeric_limits<int16_t>::max();
    int16_t maximum = std::numeric_limits<int16_t>::min();
    double sum_squares = 0.0;
    for (int index = 0; index < ASN_WINDOW_SAMPLES; ++index) {
        minimum = std::min(minimum, window[index]);
        maximum = std::max(maximum, window[index]);
        sum_squares += static_cast<double>(window[index]) * window[index];
    }
    ESP_LOGI(
        kTag,
        "MIC_WINDOW samples=%d elapsed_ms=%.3f rms=%.3f min=%d max=%d",
        ASN_WINDOW_SAMPLES,
        (esp_timer_get_time() - started_us) / 1000.0,
        std::sqrt(sum_squares / ASN_WINDOW_SAMPLES),
        minimum,
        maximum);
    return true;
}

}  // namespace

extern "C" void app_main(void)
{
    ESP_LOGI(kTag, "=== XIAO ESP32S3 ASN LIVE BLE NODE ===");
    ESP_LOGI(kTag, "PSRAM detected=%s size_bytes=%u", esp_psram_is_initialized() ? "YES" : "NO", esp_psram_get_size());
    log_memory("boot");

    dl::Model *encoder_model = load_and_test_model(
        kEncoderPartition,
        "private_encoder",
        ASN_REPRESENTATION_COUNT);
    dl::Model *validity_model = load_and_test_model(
        kValidityPartition,
        "signal_validity",
        ASN_VALIDITY_OUTPUT_COUNT);
    if (encoder_model == nullptr || validity_model == nullptr) {
        ESP_LOGE(kTag, "ASN_MODEL_LOAD_FAIL");
        return;
    }
    const bool head_ready = runtime_shared_head_init(
        model_exchange::kAsnFederatedINT8,
        0.945f);
    ESP_LOGI(kTag, "RUNTIME_SHARED_HEAD_INIT=%s version=0x%04x threshold=0.945",
        head_ready ? "PASS" : "FAIL",
        model_exchange::kAsnFederatedINT8.model_version);
    if (!head_ready) {
        return;
    }
    benchmark_model(encoder_model, "private_encoder");
    benchmark_model(validity_model, "signal_validity");
    dl::audio::Fbank frontend(frontend_config());
    const std::vector<int> shape = frontend.get_output_shape(kPaddedSamples);
    ESP_LOGI(kTag, "FRONTEND_SHAPE frames=%d mels=%d expected=%dx%d", shape[0], shape[1], ASN_FEATURE_FRAMES, ASN_N_MELS);

    auto *pcm = static_cast<int16_t *>(heap_caps_malloc(ASN_WINDOW_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (pcm == nullptr) {
        ESP_LOGE(kTag, "PCM_ALLOCATION_FAIL");
        return;
    }
    const bool golden_pass = run_golden_frontend_parity(
        encoder_model,
        validity_model,
        frontend,
        pcm);

    i2s_chan_handle_t microphone = initialise_microphone();
    if (!asn_sampling_control_init()) {
        ESP_LOGE(kTag, "SAMPLING_CONTROL_INIT=FAIL");
        return;
    }
    const bool ble_ready = asn_ble_init();
    ESP_LOGI(kTag, "ASN_BLE_INIT=%s", ble_ready ? "PASS" : "FAIL");
    log_memory("ble_ready");

    uint32_t window_index = 0;
    bool deployment_reported = false;
    while (true) {
        if (!asn_sampling_wait_for_window()) {
            ESP_LOGE(kTag, "SAMPLING_WAIT_FAIL");
            continue;
        }
        ++window_index;
        const bool captured = capture_window(microphone, pcm);
        NodeOutput live_node;
        double frontend_ms = 0.0;
        double inference_ms = 0.0;
        const bool live_ran = captured && infer_pcm(
            encoder_model,
            validity_model,
            frontend,
            pcm,
            live_node,
            frontend_ms,
            inference_ms);
        if (live_ran) {
            log_node_output(
                "microphone",
                static_cast<int>(window_index),
                live_node,
                frontend_ms,
                inference_ms);
            asn_ble_publish(
                live_node.risk_score,
                live_node.confidence,
                operational_status(live_node));
        } else {
            ESP_LOGE(kTag, "LIVE_WINDOW_FAIL index=%u", window_index);
        }

        if (!deployment_reported) {
            const bool deployment_pass = golden_pass && captured && live_ran && ble_ready;
            ESP_LOGI(kTag, "ASN_BLE_DEPLOYMENT_RESULT=%s", deployment_pass ? "PASS" : "FAIL");
            ESP_LOGI(
                kTag,
                "Live microphone outputs are operational observations, not labelled accuracy evidence.");
            log_memory("first_ble_window_complete");
            deployment_reported = true;
        }
    }
}
