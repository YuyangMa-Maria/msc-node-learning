// Fixed-width wire representation of a node risk summary.
//
// The fast decision path carries no raw sensor data or feature vectors. Q15
// encoding keeps risk and confidence portable while the final CRC protects the
// complete 20-byte record used by the default BLE ATT payload.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace node_learning {

constexpr uint16_t kSummaryMagic = 0x4E4C;
constexpr uint8_t kProtocolVersion = 1;

enum class NodeId : uint8_t {
    kVsn = 1,
    kAsn = 2,
    kVbn = 3,
    kMarh = 4,
};

enum class Modality : uint8_t {
    kVisual = 1,
    kAcoustic = 2,
    kVibration = 3,
};

enum class NodeStatus : uint8_t {
    kNormal = 0,
    kWarning = 1,
    kCritical = 2,
    kDegraded = 3,
    kInvalid = 4,
    kLowPower = 5,
};

#pragma pack(push, 1)
struct NodeSummaryV1 {
    uint16_t magic;
    uint8_t version;
    uint8_t node_id;
    uint8_t modality;
    uint8_t status;
    uint16_t sequence;
    uint32_t uptime_ms;
    uint16_t risk_q15;
    uint16_t confidence_q15;
    uint16_t model_version;
    uint16_t crc16;
};
#pragma pack(pop)

static_assert(sizeof(NodeSummaryV1) == 20, "Summary must fit the default ATT payload");

inline uint16_t crc16_ccitt(const uint8_t *data, size_t length)
{
    uint16_t crc = 0xFFFF;
    for (size_t index = 0; index < length; ++index) {
        crc ^= static_cast<uint16_t>(data[index]) << 8;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000) != 0
                ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

inline uint16_t encode_q15(float value)
{
    if (!std::isfinite(value)) {
        return 0;
    }
    const float bounded = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    return static_cast<uint16_t>(std::lround(bounded * 32767.0f));
}

inline float decode_q15(uint16_t value)
{
    return static_cast<float>(value) / 32767.0f;
}

inline void seal(NodeSummaryV1 &summary)
{
    summary.magic = kSummaryMagic;
    summary.version = kProtocolVersion;
    summary.crc16 = crc16_ccitt(
        reinterpret_cast<const uint8_t *>(&summary),
        offsetof(NodeSummaryV1, crc16));
}

inline bool validate(const NodeSummaryV1 &summary)
{
    return summary.magic == kSummaryMagic &&
        summary.version == kProtocolVersion &&
        summary.crc16 == crc16_ccitt(
            reinterpret_cast<const uint8_t *>(&summary),
            offsetof(NodeSummaryV1, crc16));
}

}  // namespace node_learning
