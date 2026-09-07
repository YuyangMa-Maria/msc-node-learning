# asn_bringup.cpp is a retained diagnostic; asn_live.cpp is the final application.
idf_component_register(
    SRCS
        "asn_live.cpp"
        "asn_ble_server.cpp"
        "asn_sampling_control.cpp"
        "runtime_shared_head.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES bt esp-dl esp_driver_i2s esp_partition esp_psram nvs_flash
)

set(asn_encoder_model_file
    "${CMAKE_CURRENT_SOURCE_DIR}/models/asn_private_encoder_projection64_esp32s3_int16.espdl")
esptool_py_flash_to_partition(flash "encoder" "${asn_encoder_model_file}")

set(asn_validity_model_file
    "${CMAKE_CURRENT_SOURCE_DIR}/models/asn_signal_validity_seed99_esp32s3_int8.espdl")
esptool_py_flash_to_partition(flash "validity" "${asn_validity_model_file}")

set(asn_golden_pcm_file
    "${CMAKE_CURRENT_SOURCE_DIR}/golden/golden_pcm16.bin")
esptool_py_flash_to_partition(flash "golden" "${asn_golden_pcm_file}")
