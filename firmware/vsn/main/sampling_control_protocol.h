// Packed BLE commands for coordinated manual sampling.

#pragma once

#include <cstddef>
#include <cstdint>

namespace sampling_control {

constexpr uint16_t kMagic = 0x4e4c;
constexpr uint8_t kVersion = 1;

enum class Command : uint8_t {
    kAutomatic = 1,
    kManual = 2,
    kSampleOnce = 3,
};

#pragma pack(push, 1)
struct MessageV1 {
    uint16_t magic;
    uint8_t version;
    uint8_t command;
    uint16_t request_id;
    uint16_t checksum;
};
#pragma pack(pop)

static_assert(sizeof(MessageV1) == 8, "Sampling control message must remain 8 bytes");

inline uint16_t checksum16(const uint8_t *data, size_t bytes)
{
    uint16_t crc = 0xffff;
    for (size_t index = 0; index < bytes; ++index) {
        crc ^= static_cast<uint16_t>(data[index]) << 8;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000) != 0
                ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

inline void seal(MessageV1 &message)
{
    message.magic = kMagic;
    message.version = kVersion;
    message.checksum = 0;
    message.checksum = checksum16(
        reinterpret_cast<const uint8_t *>(&message),
        offsetof(MessageV1, checksum));
}

inline bool validate(const MessageV1 &message)
{
    const auto command = static_cast<Command>(message.command);
    return message.magic == kMagic && message.version == kVersion &&
        (command == Command::kAutomatic || command == Command::kManual ||
         command == Command::kSampleOnce) &&
        message.checksum == checksum16(
            reinterpret_cast<const uint8_t *>(&message),
            offsetof(MessageV1, checksum));
}

inline const char *command_name(Command command)
{
    switch (command) {
    case Command::kAutomatic: return "automatic";
    case Command::kManual: return "manual";
    case Command::kSampleOnce: return "sample_once";
    default: return "unknown";
    }
}

}  // namespace sampling_control
