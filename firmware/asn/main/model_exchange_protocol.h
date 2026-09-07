// Packed, versioned BLE records for partial-model transfer and acknowledgement.
//
// Control and status records fit one default ATT payload. The larger shared head
// is chunked and is not executable until ordering, length, CRC, version and
// golden-output checks have all passed.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "shared_head_payloads.h"

namespace model_exchange {

constexpr uint16_t kControlMagic = 0x4d58;
constexpr uint16_t kChunkMagic = 0x4d43;
constexpr uint16_t kStatusMagic = 0x4d53;
constexpr uint8_t kProtocolVersion = 1;
constexpr uint16_t kPreferredMtu = 247;
constexpr size_t kChunkOverheadBytes = 10;
constexpr size_t kMaximumChunkPayloadBytes = kPreferredMtu - 3 - kChunkOverheadBytes;

enum class ControlOpcode : uint8_t {
    kBeginPush = 1,
    kCommitPush = 2,
    kPreparePull = 3,
    kSelectPullChunk = 4,
    kPreparePushMetadata = 5,
};

enum class TransferDirection : uint8_t {
    kVsnToAsn = 1,
    kAsnToVsn = 2,
};

enum class TransferStatus : uint8_t {
    kIdle = 0,
    kReady = 1,
    kReceiving = 2,
    kComplete = 3,
    kCrcFailure = 4,
    kProtocolFailure = 5,
    kExecutionFailure = 6,
    kTimeout = 7,
    kReplayFailure = 8,
    kVersionFailure = 9,
    kAborted = 10,
};

inline const char *status_name(TransferStatus status)
{
    switch (status) {
    case TransferStatus::kIdle: return "IDLE";
    case TransferStatus::kReady: return "READY";
    case TransferStatus::kReceiving: return "RECEIVING";
    case TransferStatus::kComplete: return "COMPLETE";
    case TransferStatus::kCrcFailure: return "CRC_FAILURE";
    case TransferStatus::kProtocolFailure: return "PROTOCOL_FAILURE";
    case TransferStatus::kExecutionFailure: return "EXECUTION_FAILURE";
    case TransferStatus::kTimeout: return "TIMEOUT";
    case TransferStatus::kReplayFailure: return "REPLAY_FAILURE";
    case TransferStatus::kVersionFailure: return "VERSION_FAILURE";
    case TransferStatus::kAborted: return "ABORTED";
    }
    return "UNKNOWN";
}

#pragma pack(push, 1)
struct ModelControlV1 {
    uint16_t magic;
    uint8_t version;
    uint8_t opcode;
    uint16_t transfer_id;
    uint16_t model_version;
    uint8_t format;
    uint8_t direction;
    uint16_t payload_bytes;
    uint16_t chunk_index;
    uint32_t payload_crc32;
    uint16_t crc16;
};

struct ModelChunkHeaderV1 {
    uint16_t magic;
    uint16_t transfer_id;
    uint16_t chunk_index;
    uint16_t payload_bytes;
};

struct ModelStatusV1 {
    uint16_t magic;
    uint8_t version;
    uint8_t status;
    uint16_t transfer_id;
    uint16_t model_version;
    uint8_t format;
    uint8_t direction;
    uint16_t chunks_received;
    uint16_t payload_bytes;
    uint32_t computed_crc32;
    uint16_t crc16;
};
#pragma pack(pop)

static_assert(sizeof(ModelControlV1) == 20, "Control must fit the default ATT payload");
static_assert(sizeof(ModelChunkHeaderV1) == 8, "Unexpected chunk header size");
static_assert(sizeof(ModelStatusV1) == 20, "Status must fit the default ATT payload");

inline uint16_t crc16_ccitt(const uint8_t *data, size_t length)
{
    uint16_t crc = 0xffff;
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

inline uint32_t crc32_ieee(const uint8_t *data, size_t length)
{
    uint32_t crc = 0xffffffffu;
    for (size_t index = 0; index < length; ++index) {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 1u) != 0 ? (crc >> 1) ^ 0xedb88320u : crc >> 1;
        }
    }
    return crc ^ 0xffffffffu;
}

inline void seal(ModelControlV1 &control)
{
    control.magic = kControlMagic;
    control.version = kProtocolVersion;
    control.crc16 = crc16_ccitt(
        reinterpret_cast<const uint8_t *>(&control),
        offsetof(ModelControlV1, crc16));
}

inline bool validate(const ModelControlV1 &control)
{
    return control.magic == kControlMagic && control.version == kProtocolVersion &&
        control.crc16 == crc16_ccitt(
            reinterpret_cast<const uint8_t *>(&control),
            offsetof(ModelControlV1, crc16));
}

inline void seal(ModelStatusV1 &status)
{
    status.magic = kStatusMagic;
    status.version = kProtocolVersion;
    status.crc16 = crc16_ccitt(
        reinterpret_cast<const uint8_t *>(&status),
        offsetof(ModelStatusV1, crc16));
}

inline bool validate(const ModelStatusV1 &status)
{
    return status.magic == kStatusMagic && status.version == kProtocolVersion &&
        status.crc16 == crc16_ccitt(
            reinterpret_cast<const uint8_t *>(&status),
            offsetof(ModelStatusV1, crc16));
}

inline size_t chunk_payload_capacity(uint16_t mtu)
{
    if (mtu <= 3 + kChunkOverheadBytes) {
        return 0;
    }
    return std::min(
        static_cast<size_t>(mtu - 3 - kChunkOverheadBytes),
        kMaximumChunkPayloadBytes);
}

inline size_t chunk_count(size_t payload_bytes, size_t chunk_payload_bytes)
{
    return chunk_payload_bytes == 0
        ? 0
        : (payload_bytes + chunk_payload_bytes - 1) / chunk_payload_bytes;
}

inline size_t build_chunk(
    uint8_t *destination,
    size_t destination_capacity,
    uint16_t transfer_id,
    uint16_t chunk_index,
    const uint8_t *payload,
    size_t payload_bytes)
{
    const size_t packet_bytes = sizeof(ModelChunkHeaderV1) + payload_bytes + sizeof(uint16_t);
    if (destination == nullptr || payload == nullptr ||
        payload_bytes > UINT16_MAX || packet_bytes > destination_capacity) {
        return 0;
    }
    ModelChunkHeaderV1 header{
        kChunkMagic,
        transfer_id,
        chunk_index,
        static_cast<uint16_t>(payload_bytes),
    };
    std::memcpy(destination, &header, sizeof(header));
    std::memcpy(destination + sizeof(header), payload, payload_bytes);
    const uint16_t crc = crc16_ccitt(destination, sizeof(header) + payload_bytes);
    std::memcpy(destination + sizeof(header) + payload_bytes, &crc, sizeof(crc));
    return packet_bytes;
}

inline bool parse_chunk(
    const uint8_t *packet,
    size_t packet_bytes,
    ModelChunkHeaderV1 &header,
    const uint8_t *&payload)
{
    if (packet == nullptr || packet_bytes < kChunkOverheadBytes) {
        return false;
    }
    std::memcpy(&header, packet, sizeof(header));
    const size_t expected = sizeof(header) + header.payload_bytes + sizeof(uint16_t);
    if (header.magic != kChunkMagic || expected != packet_bytes) {
        return false;
    }
    uint16_t supplied_crc = 0;
    std::memcpy(&supplied_crc, packet + packet_bytes - sizeof(supplied_crc), sizeof(supplied_crc));
    if (supplied_crc != crc16_ccitt(packet, packet_bytes - sizeof(supplied_crc))) {
        return false;
    }
    payload = packet + sizeof(header);
    return true;
}

inline bool valid_package_contract(SharedHeadFormat format, size_t payload_bytes)
{
    return (format == SharedHeadFormat::kFp32 && payload_bytes == kSharedHeadFp32Bytes) ||
        (format == SharedHeadFormat::kInt8Symmetric && payload_bytes == kSharedHeadInt8Bytes);
}

inline float read_float(const uint8_t *data)
{
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

inline bool evaluate_shared_head_with_embedding(
    const uint8_t *payload,
    size_t payload_bytes,
    SharedHeadFormat format,
    const float *embedding,
    float &output)
{
    if (payload == nullptr || embedding == nullptr ||
        !valid_package_contract(format, payload_bytes)) {
        return false;
    }

    float hidden[32]{};
    if (format == SharedHeadFormat::kFp32) {
        const size_t first_bias_offset = 2048 * sizeof(float);
        const size_t second_weight_offset = (2048 + 32) * sizeof(float);
        const size_t second_bias_offset = (2048 + 32 + 32) * sizeof(float);
        for (size_t row = 0; row < 32; ++row) {
            float sum = read_float(payload + first_bias_offset + row * sizeof(float));
            for (size_t column = 0; column < 64; ++column) {
                const size_t index = row * 64 + column;
                sum += read_float(payload + index * sizeof(float)) * embedding[column];
            }
            hidden[row] = std::max(sum, 0.0f);
        }
        output = read_float(payload + second_bias_offset);
        for (size_t index = 0; index < 32; ++index) {
            output += read_float(payload + second_weight_offset + index * sizeof(float)) * hidden[index];
        }
    } else {
        float scales[4]{};
        for (size_t index = 0; index < 4; ++index) {
            scales[index] = read_float(payload + index * sizeof(float));
        }
        const int8_t *parameters = reinterpret_cast<const int8_t *>(payload + 4 * sizeof(float));
        for (size_t row = 0; row < 32; ++row) {
            float sum = static_cast<float>(parameters[2048 + row]) * scales[1];
            for (size_t column = 0; column < 64; ++column) {
                const size_t index = row * 64 + column;
                sum += static_cast<float>(parameters[index]) * scales[0] * embedding[column];
            }
            hidden[row] = std::max(sum, 0.0f);
        }
        output = static_cast<float>(parameters[2112]) * scales[3];
        for (size_t index = 0; index < 32; ++index) {
            output += static_cast<float>(parameters[2080 + index]) * scales[2] * hidden[index];
        }
    }
    return std::isfinite(output);
}

inline bool evaluate_shared_head(
    const uint8_t *payload,
    size_t payload_bytes,
    SharedHeadFormat format,
    float &output)
{
    return evaluate_shared_head_with_embedding(
        payload, payload_bytes, format, kGoldenEmbedding, output);
}

inline const char *format_name(SharedHeadFormat format)
{
    return format == SharedHeadFormat::kFp32 ? "FP32" : "INT8";
}

}  // namespace model_exchange
