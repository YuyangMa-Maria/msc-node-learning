// Final VSN inference, status-aware fusion and representative-node decision path.
// The camera model produces local evidence; fresh ASN summaries may contribute
// to fusion, but invalid, low-power or stale acoustic evidence receives no vote.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "dl_image_jpeg.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "esp_camera.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "node_summary_protocol.h"
#include "runtime_shared_head.h"
#include "system_host_protocol.h"
#include "vsn_ble_client.h"
#include "vsn_model_runtime.h"
#include "vsn_sampling_control.h"

namespace {

constexpr char kModelPartition[] = "model";
constexpr char kTag[] = "vsn_model";
constexpr int kBenchmarkWarmupRuns = 3;
constexpr int kBenchmarkRuns = 50;
constexpr int kLiveWarmupRuns = 2;
constexpr int kLiveBenchmarkRuns = 20;
constexpr float kVsnBaseSharedThreshold = 0.165f;
constexpr float kCriticalRiskThreshold = 0.85f;
constexpr uint32_t kVisualPeriodMs = 1000;
constexpr uint32_t kAsnFreshnessMs = 7500;
constexpr uint32_t kFallbackPeriodMs = 5000;
constexpr uint32_t kOnlineTaskStackBytes = 12288;
constexpr UBaseType_t kOnlineTaskPriority = 4;
constexpr std::array<float, 3> kImageNetMean = {
    123.675f,
    116.28f,
    103.53f,
};
constexpr std::array<float, 3> kImageNetStd = {
    58.395f,
    57.12f,
    57.375f,
};

dl::Model *g_model = nullptr;
TaskHandle_t g_online_task = nullptr;

struct FusionNode {
    float risk_score;
    float confidence;
    node_learning::NodeStatus status;
};

struct FusionState {
    std::array<float, 3> history{};
    size_t history_count = 0;
    uint32_t fusion_index = 0;
    bool had_acoustic_node = false;
};

void log_memory(const char *stage)
{
    ESP_LOGI(
        kTag,
        "memory stage=%s internal_free=%u internal_largest=%u psram_free=%u",
        stage,
        static_cast<unsigned>(
            heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        static_cast<unsigned>(
            heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
}

void benchmark_model(
    dl::Model *model,
    dl::runtime_mode_t mode,
    const char *mode_name)
{
    for (int i = 0; i < kBenchmarkWarmupRuns; ++i) {
        model->run(mode);
    }

    int64_t latency_us[kBenchmarkRuns] = {};
    int64_t total_us = 0;
    for (int i = 0; i < kBenchmarkRuns; ++i) {
        const int64_t start_us = esp_timer_get_time();
        model->run(mode);
        latency_us[i] = esp_timer_get_time() - start_us;
        total_us += latency_us[i];
    }

    for (int i = 1; i < kBenchmarkRuns; ++i) {
        const int64_t value = latency_us[i];
        int j = i;
        while (j > 0 && latency_us[j - 1] > value) {
            latency_us[j] = latency_us[j - 1];
            --j;
        }
        latency_us[j] = value;
    }

    const int p50_index = (kBenchmarkRuns - 1) / 2;
    const int p95_index = (kBenchmarkRuns * 95 + 99) / 100 - 1;
    dl::TensorBase *output = model->get_output();
    const int output_exponent = output != nullptr ? output->get_exponent() : 0;
    const int output_raw = output != nullptr && output->get_dtype() == dl::DATA_TYPE_INT8
        ? static_cast<int>(output->get_element<int8_t>(0)) : 0;

    ESP_LOGI(
        kTag,
        "INFERENCE_BENCHMARK mode=%s warmup_runs=%d runs=%d mean_ms=%.3f "
        "p50_ms=%.3f p95_ms=%.3f min_ms=%.3f max_ms=%.3f "
        "output_raw=%d output_exponent=%d",
        mode_name,
        kBenchmarkWarmupRuns,
        kBenchmarkRuns,
        total_us / (1000.0 * kBenchmarkRuns),
        latency_us[p50_index] / 1000.0,
        latency_us[p95_index] / 1000.0,
        latency_us[0] / 1000.0,
        latency_us[kBenchmarkRuns - 1] / 1000.0,
        output_raw,
        output_exponent);
}

float sigmoid(float value)
{
    if (value >= 0.0f) {
        const float exp_neg = std::exp(-value);
        return 1.0f / (1.0f + exp_neg);
    }
    const float exp_pos = std::exp(value);
    return exp_pos / (1.0f + exp_pos);
}

float binary_confidence(float probability)
{
    constexpr float kEpsilon = 1.0e-6f;
    constexpr float kLogTwo = 0.6931471805599453f;
    const float bounded = std::fmax(
        kEpsilon,
        std::fmin(1.0f - kEpsilon, probability));
    const float entropy =
        -(bounded * std::log(bounded) +
          (1.0f - bounded) * std::log(1.0f - bounded)) /
        kLogTwo;
    return 1.0f - entropy;
}

void log_latency_summary(const char *stage, int64_t *values_us, int count)
{
    int64_t total_us = 0;
    for (int i = 0; i < count; ++i) {
        total_us += values_us[i];
    }

    for (int i = 1; i < count; ++i) {
        const int64_t value = values_us[i];
        int j = i;
        while (j > 0 && values_us[j - 1] > value) {
            values_us[j] = values_us[j - 1];
            --j;
        }
        values_us[j] = value;
    }

    const int p50_index = (count - 1) / 2;
    const int p95_index = (count * 95 + 99) / 100 - 1;
    ESP_LOGI(
        kTag,
        "LIVE_LATENCY stage=%s runs=%d mean_ms=%.3f p50_ms=%.3f "
        "p95_ms=%.3f min_ms=%.3f max_ms=%.3f",
        stage,
        count,
        total_us / (1000.0 * count),
        values_us[p50_index] / 1000.0,
        values_us[p95_index] / 1000.0,
        values_us[0] / 1000.0,
        values_us[count - 1] / 1000.0);
}

bool process_live_frame(
    dl::Model *model,
    dl::image::ImagePreprocessor &preprocessor,
    int sample_index,
    bool log_result,
    int64_t &capture_us,
    int64_t &decode_us,
    int64_t &preprocess_us,
    int64_t &inference_us,
    int64_t &total_us,
    float *risk_score_out = nullptr,
    float *confidence_out = nullptr)
{
    const int64_t total_start_us = esp_timer_get_time();

    const int64_t capture_start_us = esp_timer_get_time();
    camera_fb_t *frame = esp_camera_fb_get();
    capture_us = esp_timer_get_time() - capture_start_us;
    if (frame == nullptr) {
        ESP_LOGE(kTag, "LIVE_PIPELINE frame capture failed sample=%d", sample_index);
        return false;
    }

    if (frame->format != PIXFORMAT_JPEG) {
        ESP_LOGE(
            kTag,
            "LIVE_PIPELINE expected JPEG frame sample=%d format=%d",
            sample_index,
            frame->format);
        esp_camera_fb_return(frame);
        return false;
    }

    const size_t jpeg_bytes = frame->len;
    const dl::image::jpeg_img_t jpeg_image = {
        .data = frame->buf,
        .data_len = frame->len,
    };
    const int64_t decode_start_us = esp_timer_get_time();
    dl::image::img_t rgb_image = dl::image::sw_decode_jpeg(
        jpeg_image,
        dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    decode_us = esp_timer_get_time() - decode_start_us;
    esp_camera_fb_return(frame);

    if (rgb_image.data == nullptr || rgb_image.width == 0 || rgb_image.height == 0) {
        ESP_LOGE(kTag, "LIVE_PIPELINE JPEG decode failed sample=%d", sample_index);
        if (rgb_image.data != nullptr) {
            heap_caps_free(rgb_image.data);
        }
        return false;
    }

    const int64_t preprocess_start_us = esp_timer_get_time();
    preprocessor.preprocess(rgb_image);
    preprocess_us = esp_timer_get_time() - preprocess_start_us;
    const unsigned decoded_width = rgb_image.width;
    const unsigned decoded_height = rgb_image.height;
    heap_caps_free(rgb_image.data);

    const int64_t inference_start_us = esp_timer_get_time();
    model->run(dl::RUNTIME_MODE_MULTI_CORE);
    inference_us = esp_timer_get_time() - inference_start_us;

    dl::TensorBase *output = model->get_output();
    if (output == nullptr || output->get_dtype() != dl::DATA_TYPE_INT8 ||
        output->get_size() != 64) {
        ESP_LOGE(kTag, "LIVE_PIPELINE invalid model output sample=%d", sample_index);
        return false;
    }

    const int output_exponent = output->get_exponent();
    float representation[64]{};
    float representation_norm_sq = 0.0f;
    for (size_t index = 0; index < 64; ++index) {
        representation[index] = std::ldexp(
            static_cast<float>(output->get_element<int8_t>(index)),
            output_exponent);
        representation_norm_sq += representation[index] * representation[index];
    }
    float logit = 0.0f;
    RuntimeHeadSnapshot head{};
    if (!runtime_shared_head_evaluate(representation, 64, logit, head)) {
        ESP_LOGE(kTag, "LIVE_PIPELINE shared-head evaluation failed sample=%d", sample_index);
        return false;
    }
    const float risk_score = sigmoid(logit);
    const float confidence = binary_confidence(risk_score);
    if (risk_score_out != nullptr) {
        *risk_score_out = risk_score;
    }
    if (confidence_out != nullptr) {
        *confidence_out = confidence;
    }
    total_us = esp_timer_get_time() - total_start_us;

    if (log_result) {
        ESP_LOGI(
            kTag,
            "LIVE_RISK sample=%d source=%ux%u jpeg_bytes=%u representation_l2=%.6f "
            "exponent=%d head_version=0x%04x generation=%u format=%s threshold=%.6f "
            "logit=%.6f risk_score=%.6f confidence=%.6f above_threshold=%s total_ms=%.3f",
            sample_index,
            decoded_width,
            decoded_height,
            static_cast<unsigned>(jpeg_bytes),
            std::sqrt(representation_norm_sq),
            output_exponent,
            head.version,
            static_cast<unsigned>(head.generation),
            model_exchange::format_name(head.format),
            head.threshold,
            logit,
            risk_score,
            confidence,
            risk_score >= head.threshold ? "yes" : "no",
            total_us / 1000.0);
    }
    return true;
}

void benchmark_live_pipeline(dl::Model *model)
{
    ESP_LOGI(
        kTag,
        "LIVE_PIPELINE_START warmup_runs=%d runs=%d input=128x128x3 "
        "normalization=ImageNet output=64d_shared_risk_representation",
        kLiveWarmupRuns,
        kLiveBenchmarkRuns);

    dl::image::ImagePreprocessor preprocessor(
        model,
        kImageNetMean,
        kImageNetStd,
        false);
    log_memory("live_preprocessor_ready");

    int64_t capture_us[kLiveBenchmarkRuns] = {};
    int64_t decode_us[kLiveBenchmarkRuns] = {};
    int64_t preprocess_us[kLiveBenchmarkRuns] = {};
    int64_t inference_us[kLiveBenchmarkRuns] = {};
    int64_t total_us[kLiveBenchmarkRuns] = {};

    for (int i = 0; i < kLiveWarmupRuns; ++i) {
        int64_t warmup_capture = 0;
        int64_t warmup_decode = 0;
        int64_t warmup_preprocess = 0;
        int64_t warmup_inference = 0;
        int64_t warmup_total = 0;
        if (!process_live_frame(
                model,
                preprocessor,
                -(i + 1),
                false,
                warmup_capture,
                warmup_decode,
                warmup_preprocess,
                warmup_inference,
                warmup_total)) {
            ESP_LOGE(kTag, "LIVE_PIPELINE warm-up failed");
            return;
        }
    }

    int completed_runs = 0;
    for (int i = 0; i < kLiveBenchmarkRuns; ++i) {
        if (!process_live_frame(
                model,
                preprocessor,
                i + 1,
                true,
                capture_us[completed_runs],
                decode_us[completed_runs],
                preprocess_us[completed_runs],
                inference_us[completed_runs],
                total_us[completed_runs])) {
            break;
        }
        ++completed_runs;
    }

    if (completed_runs == 0) {
        ESP_LOGE(kTag, "LIVE_PIPELINE no benchmark frame completed");
        return;
    }

    log_latency_summary("capture", capture_us, completed_runs);
    log_latency_summary("jpeg_decode_rgb888", decode_us, completed_runs);
    log_latency_summary("resize_normalize_quantize", preprocess_us, completed_runs);
    log_latency_summary("dual_core_inference", inference_us, completed_runs);
    log_latency_summary("end_to_end", total_us, completed_runs);
    log_memory("live_pipeline_complete");
    ESP_LOGI(
        kTag,
        "LIVE_PIPELINE_COMPLETE completed_runs=%d result=PASS "
        "risk_semantics=structural_risk_proxy",
        completed_runs);
}

node_learning::NodeStatus visual_status(float risk_score)
{
    if (risk_score >= kCriticalRiskThreshold) {
        return node_learning::NodeStatus::kCritical;
    }
    if (risk_score >= runtime_shared_head_snapshot().threshold) {
        return node_learning::NodeStatus::kWarning;
    }
    return node_learning::NodeStatus::kNormal;
}

float clamp01(float value)
{
    return std::fmax(0.0f, std::fmin(1.0f, value));
}

float effective_weight(const FusionNode &node)
{
    switch (node.status) {
    case node_learning::NodeStatus::kInvalid:
        return 0.0f;
    case node_learning::NodeStatus::kLowPower:
    case node_learning::NodeStatus::kDegraded:
        return 0.5f * clamp01(node.confidence);
    case node_learning::NodeStatus::kCritical:
        return std::fmax(clamp01(node.confidence), 0.8f);
    default:
        return clamp01(node.confidence);
    }
}

bool is_quality_gate_active(const FusionNode &node)
{
    return node.status != node_learning::NodeStatus::kInvalid &&
        node.status != node_learning::NodeStatus::kLowPower;
}

struct FusionOutcome {
    float risk_score = 0.0f;
    float confidence = 0.0f;
    int active_nodes = 0;
};

FusionOutcome weighted_subset(
    const FusionNode *nodes,
    size_t count,
    node_learning::NodeStatus required_status,
    bool filter_status)
{
    float weighted_score = 0.0f;
    float total_weight = 0.0f;
    float unweighted_score = 0.0f;
    int included = 0;
    for (size_t index = 0; index < count; ++index) {
        const FusionNode &node = nodes[index];
        if (!is_quality_gate_active(node) ||
            (filter_status && node.status != required_status)) {
            continue;
        }
        const float weight = effective_weight(node);
        weighted_score += weight * clamp01(node.risk_score);
        total_weight += weight;
        unweighted_score += clamp01(node.risk_score);
        ++included;
    }
    if (included == 0) {
        return {};
    }
    FusionOutcome outcome;
    outcome.active_nodes = included;
    outcome.risk_score = total_weight > 1.0e-9f
        ? weighted_score / total_weight
        : unweighted_score / included;
    outcome.confidence = total_weight > 1.0e-9f
        ? clamp01(total_weight / included)
        : 0.0f;
    return outcome;
}

FusionOutcome quality_gate_fusion(const FusionNode *nodes, size_t count)
{
    // Preserve explicit critical evidence. Otherwise combine nodes from the
    // strongest available status band instead of averaging healthy and invalid
    // observations together.
    int critical_count = 0;
    int warning_count = 0;
    FusionOutcome critical_outcome;
    float best_critical_priority = -1.0f;
    for (size_t index = 0; index < count; ++index) {
        const FusionNode &node = nodes[index];
        if (!is_quality_gate_active(node)) {
            continue;
        }
        if (node.status == node_learning::NodeStatus::kCritical) {
            ++critical_count;
            const float priority = clamp01(node.risk_score) *
                std::fmax(effective_weight(node), 1.0e-6f);
            if (priority > best_critical_priority) {
                best_critical_priority = priority;
                critical_outcome.risk_score = clamp01(node.risk_score);
                critical_outcome.confidence = effective_weight(node);
            }
        } else if (node.status == node_learning::NodeStatus::kWarning) {
            ++warning_count;
        }
    }
    if (critical_count > 0) {
        critical_outcome.active_nodes = critical_count;
        return critical_outcome;
    }
    if (warning_count > 0) {
        return weighted_subset(
            nodes,
            count,
            node_learning::NodeStatus::kWarning,
            true);
    }
    return weighted_subset(
        nodes,
        count,
        node_learning::NodeStatus::kNormal,
        false);
}

int risk_level(float score)
{
    score = clamp01(score);
    if (score < 0.20f) return 1;
    if (score < 0.40f) return 2;
    if (score < 0.60f) return 3;
    if (score < 0.80f) return 4;
    return 5;
}

const char *risk_trend(const FusionState &state, float current)
{
    if (state.history_count < 2) {
        return "unknown";
    }
    float previous = 0.0f;
    for (size_t index = 0; index < state.history_count; ++index) {
        previous += state.history[index];
    }
    previous /= state.history_count;
    const float delta = current - previous;
    if (delta > 0.05f) return "rising";
    if (delta < -0.05f) return "falling";
    return "stable";
}

void append_history(FusionState &state, float current)
{
    if (state.history_count < state.history.size()) {
        state.history[state.history_count++] = current;
        return;
    }
    state.history[0] = state.history[1];
    state.history[1] = state.history[2];
    state.history[2] = current;
}

bool triggers(const FusionNode &node)
{
    if (!is_quality_gate_active(node)) {
        return false;
    }
    return node.status == node_learning::NodeStatus::kWarning ||
        node.status == node_learning::NodeStatus::kCritical ||
        (node.risk_score >= 0.60f && node.confidence >= 0.50f);
}

const char *alert_action(
    int level,
    const char *trend,
    int triggered_count,
    int active_nodes,
    bool acoustic_lost,
    bool disagreement)
{
    if (active_nodes == 0) return "insufficient_evidence";
    if (level >= 5) {
        return triggered_count >= 2 ? "move_away_multi_node" : "move_away";
    }
    if (level == 4) return "avoid_entry";
    if (level == 3) {
        return std::strcmp(trend, "rising") == 0
            ? "keep_distance_and_monitor"
            : "maintain_caution";
    }
    if (disagreement) return "repeat_sampling_and_request_assessment";
    if (acoustic_lost) return "continue_monitoring_limited_evidence";
    return "continue_monitoring";
}

void emit_fusion(
    FusionState &state,
    const FusionNode &visual,
    const FusionNode *acoustic,
    uint16_t acoustic_sequence,
    uint32_t acoustic_age_ms,
    uint32_t visual_sequence)
{
    // MARH is recommended from current evidence and availability. This flag is
    // advisory and does not transfer control away from the local sensing loop.
    std::array<FusionNode, 2> nodes = {visual, {}};
    size_t node_count = 1;
    if (acoustic != nullptr) {
        nodes[1] = *acoustic;
        node_count = 2;
    }
    const FusionOutcome outcome = quality_gate_fusion(nodes.data(), node_count);
    const char *trend = risk_trend(state, outcome.risk_score);
    const int level = risk_level(outcome.risk_score);
    const bool visual_triggered = triggers(visual);
    const bool acoustic_triggered = acoustic != nullptr && triggers(*acoustic);
    const int triggered_count = static_cast<int>(visual_triggered) +
        static_cast<int>(acoustic_triggered);
    const unsigned triggered_mask = (visual_triggered ? 0x01u : 0u) |
        (acoustic_triggered ? 0x02u : 0u);
    const bool acoustic_lost = acoustic == nullptr && state.had_acoustic_node;
    const bool disagreement = acoustic != nullptr &&
        visual.confidence >= 0.50f && acoustic->confidence >= 0.50f &&
        std::fabs(visual.risk_score - acoustic->risk_score) >= 0.50f;
    const bool insufficient_evidence = outcome.active_nodes == 0;
    const bool high_risk = level >= 4;
    const bool rising_risk = level >= 3 && std::strcmp(trend, "rising") == 0;
    const bool marh_recommended = insufficient_evidence || high_risk ||
        rising_risk || acoustic_lost || disagreement;
    const char *marh_reason = insufficient_evidence
        ? "insufficient_evidence"
        : high_risk
            ? "high_risk"
            : rising_risk
                ? "rising_risk"
                : acoustic_lost
                    ? "node_loss"
                    : disagreement
                        ? "node_disagreement"
                        : "none";
    ++state.fusion_index;

    const char *action = alert_action(
        level,
        trend,
        triggered_count,
        outcome.active_nodes,
        acoustic_lost,
        disagreement);
    ESP_LOGI(
        kTag,
        "FUSION_OUTPUT index=%u mode=%s vsn_sequence=%u asn_sequence=%u "
        "asn_age_ms=%u vsn_risk=%.6f vsn_confidence=%.6f vsn_status=%u "
        "asn_risk=%.6f asn_confidence=%.6f asn_status=%u "
        "fused_risk_score=%.6f fused_confidence=%.6f risk_level=%d "
        "risk_trend=%s active_nodes=%d method=quality_gate",
        state.fusion_index,
        acoustic != nullptr ? "dual" : "vsn_only",
        visual_sequence,
        acoustic_sequence,
        acoustic_age_ms,
        visual.risk_score,
        visual.confidence,
        static_cast<unsigned>(visual.status),
        acoustic != nullptr ? acoustic->risk_score : 0.0f,
        acoustic != nullptr ? acoustic->confidence : 0.0f,
        acoustic != nullptr ? static_cast<unsigned>(acoustic->status) :
            static_cast<unsigned>(node_learning::NodeStatus::kInvalid),
        outcome.risk_score,
        outcome.confidence,
        level,
        trend,
        outcome.active_nodes);
    ESP_LOGI(
        kTag,
        "ALERT_OUTPUT fusion_index=%u risk_level=%d risk_trend=%s "
        "triggered_mask=0x%02x action=%s",
        state.fusion_index,
        level,
        trend,
        triggered_mask,
        action);
    system_host_report_fusion(
        state.fusion_index,
        static_cast<uint32_t>(esp_timer_get_time() / 1000),
        visual_sequence,
        visual.risk_score,
        visual.confidence,
        static_cast<uint8_t>(visual.status),
        acoustic != nullptr,
        acoustic_sequence,
        acoustic_age_ms,
        acoustic != nullptr ? acoustic->risk_score : 0.0f,
        acoustic != nullptr ? acoustic->confidence : 0.0f,
        acoustic != nullptr
            ? static_cast<uint8_t>(acoustic->status)
            : static_cast<uint8_t>(node_learning::NodeStatus::kInvalid),
        outcome.risk_score,
        outcome.confidence,
        level,
        trend,
        outcome.active_nodes,
        static_cast<uint8_t>(triggered_mask),
        action,
        marh_recommended,
        marh_reason);
    if (acoustic != nullptr) {
        state.had_acoustic_node = true;
    }
    append_history(state, outcome.risk_score);
}

void online_runtime_task(void *)
{
    // Automatic and manual sampling share this task so camera ownership and the
    // fusion history remain serialised in one place.
    dl::image::ImagePreprocessor preprocessor(
        g_model,
        kImageNetMean,
        kImageNetStd,
        false);
    log_memory("online_preprocessor_ready");
    ESP_LOGI(
        kTag,
        "ONLINE_RUNTIME_START visual_period_ms=%u asn_freshness_ms=%u "
        "fusion_method=quality_gate task_stack_bytes=%u",
        kVisualPeriodMs,
        kAsnFreshnessMs,
        kOnlineTaskStackBytes);

    TickType_t last_wake = xTaskGetTickCount();
    uint32_t visual_sequence = 0;
    uint16_t last_fused_asn_sequence = 0;
    bool have_fused_asn_sequence = false;
    uint32_t last_fallback_ms = 0;
    bool previous_automatic = true;
    FusionState fusion_state;

    while (true) {
        if (!vsn_sampling_wait_for_frame()) {
            ESP_LOGE(kTag, "SAMPLING_WAIT_FAIL");
            continue;
        }
        const bool frame_automatic = vsn_sampling_is_automatic();
        if (frame_automatic && !previous_automatic) {
            // A manual pause makes the old periodic deadline stale. Reset it
            // so automatic mode resumes at 1 Hz without a catch-up burst.
            last_wake = xTaskGetTickCount();
        }
        previous_automatic = frame_automatic;
        ++visual_sequence;
        int64_t capture_us = 0;
        int64_t decode_us = 0;
        int64_t preprocess_us = 0;
        int64_t inference_us = 0;
        int64_t total_us = 0;
        float risk_score = 0.0f;
        float confidence = 0.0f;
        const bool frame_ok = process_live_frame(
            g_model,
            preprocessor,
            static_cast<int>(visual_sequence),
            false,
            capture_us,
            decode_us,
            preprocess_us,
            inference_us,
            total_us,
            &risk_score,
            &confidence);
        const uint32_t now_ms = static_cast<uint32_t>(esp_timer_get_time() / 1000);
        if (!frame_ok) {
            ESP_LOGE(kTag, "VSN_LOCAL_FAIL sequence=%u", visual_sequence);
            if (vsn_sampling_is_automatic()) {
                vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(kVisualPeriodMs));
            }
            continue;
        }

        const FusionNode visual = {
            .risk_score = risk_score,
            .confidence = confidence,
            .status = visual_status(risk_score),
        };
        ESP_LOGI(
            kTag,
            "VSN_LOCAL sequence=%u risk_score=%.6f confidence=%.6f status=%u "
            "capture_ms=%.3f decode_ms=%.3f preprocess_ms=%.3f "
            "inference_ms=%.3f total_ms=%.3f",
            visual_sequence,
            visual.risk_score,
            visual.confidence,
            static_cast<unsigned>(visual.status),
            capture_us / 1000.0,
            decode_us / 1000.0,
            preprocess_us / 1000.0,
            inference_us / 1000.0,
            total_us / 1000.0);
        system_host_report_node_output(
            "VSN",
            "camera",
            "local",
            visual_sequence,
            now_ms,
            now_ms,
            visual.risk_score,
            visual.confidence,
            static_cast<uint8_t>(visual.status),
            static_cast<float>(capture_us / 1000.0),
            static_cast<float>(decode_us / 1000.0),
            static_cast<float>(preprocess_us / 1000.0),
            static_cast<float>(inference_us / 1000.0),
            static_cast<float>(total_us / 1000.0));

        vsn_ble_summary_snapshot_t snapshot{};
        const bool has_snapshot = vsn_ble_get_latest_summary(&snapshot);
        const uint32_t acoustic_age_ms = has_snapshot
            ? now_ms - snapshot.received_local_uptime_ms
            : UINT32_MAX;
        const bool acoustic_fresh = has_snapshot && snapshot.connected &&
            acoustic_age_ms <= kAsnFreshnessMs;
        const bool new_acoustic_sequence = acoustic_fresh &&
            (!have_fused_asn_sequence || snapshot.sequence != last_fused_asn_sequence);

        if (new_acoustic_sequence) {
            const FusionNode acoustic = {
                .risk_score = snapshot.risk_score,
                .confidence = snapshot.confidence,
                .status = static_cast<node_learning::NodeStatus>(snapshot.status),
            };
            emit_fusion(
                fusion_state,
                visual,
                &acoustic,
                snapshot.sequence,
                acoustic_age_ms,
                visual_sequence);
            last_fused_asn_sequence = snapshot.sequence;
            have_fused_asn_sequence = true;
        } else if ((!acoustic_fresh || !snapshot.connected) &&
                   now_ms - last_fallback_ms >= kFallbackPeriodMs) {
            emit_fusion(
                fusion_state,
                visual,
                nullptr,
                0,
                acoustic_age_ms,
                visual_sequence);
            last_fallback_ms = now_ms;
        }

        if (visual_sequence % 20 == 0) {
            ESP_LOGI(
                kTag,
                "ONLINE_MEMORY sequence=%u internal_free=%u internal_largest=%u "
                "psram_free=%u task_stack_high_water=%u",
                visual_sequence,
                static_cast<unsigned>(
                    heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
                static_cast<unsigned>(
                    heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
                static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
                static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr)));
        }
        if (vsn_sampling_is_automatic()) {
            vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(kVisualPeriodMs));
        } else {
            last_wake = xTaskGetTickCount();
        }
    }
}

}  // namespace

extern "C" void vsn_run_model_parity(void)
{
    ESP_LOGI(kTag, "Loading ESP32-S3 INT8 VSN model from partition '%s'", kModelPartition);
    log_memory("before_model_load");

    const int64_t load_start_us = esp_timer_get_time();
    auto *model = new dl::Model(
        kModelPartition,
        fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
        0,
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,
        false);
    const int64_t load_us = esp_timer_get_time() - load_start_us;

    ESP_LOGI(kTag, "model_load_ms=%.3f", load_us / 1000.0);
    log_memory("after_model_load");

    const int64_t test_start_us = esp_timer_get_time();
    const esp_err_t test_result = model->test();
    const int64_t test_us = esp_timer_get_time() - test_start_us;

    if (test_result == ESP_OK) {
        g_model = model;
        ESP_LOGI(kTag, "MODEL_PARITY=PASS test_ms=%.3f", test_us / 1000.0);
        const bool head_ready = runtime_shared_head_init(
            model_exchange::kVsnBaseINT8,
            kVsnBaseSharedThreshold);
        ESP_LOGI(kTag, "RUNTIME_SHARED_HEAD_INIT=%s version=0x%04x threshold=%.3f",
            head_ready ? "PASS" : "FAIL",
            model_exchange::kVsnBaseINT8.model_version,
            kVsnBaseSharedThreshold);
        if (!head_ready) {
            return;
        }
        benchmark_model(model, dl::RUNTIME_MODE_SINGLE_CORE, "single_core");
        benchmark_model(model, dl::RUNTIME_MODE_MULTI_CORE, "dual_core");
        benchmark_live_pipeline(model);
    } else {
        ESP_LOGE(
            kTag,
            "MODEL_PARITY=FAIL error=%s test_ms=%.3f",
            esp_err_to_name(test_result),
            test_us / 1000.0);
    }

    model->profile_memory();
    log_memory("after_model_test");
}

extern "C" bool vsn_start_online_runtime(void)
{
    if (g_model == nullptr) {
        ESP_LOGE(kTag, "ONLINE_RUNTIME_START_FAIL reason=model_not_ready");
        return false;
    }
    if (g_online_task != nullptr) {
        return true;
    }
    const BaseType_t result = xTaskCreatePinnedToCore(
        online_runtime_task,
        "vsn_online",
        kOnlineTaskStackBytes,
        nullptr,
        kOnlineTaskPriority,
        &g_online_task,
        1);
    if (result != pdPASS) {
        ESP_LOGE(kTag, "ONLINE_RUNTIME_START_FAIL reason=task_create");
        g_online_task = nullptr;
        return false;
    }
    return true;
}
