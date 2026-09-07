idf_component_register(
    SRCS
        "vsn_camera_bringup.c"
        "vsn_model_parity.cpp"
        "vsn_ble_client.cpp"
        "vsn_model_exchange.cpp"
        "vsn_sampling_control.cpp"
        "runtime_shared_head.cpp"
        "host_model_update.cpp"
        "system_host_protocol.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES bt esp-dl esp_psram nimble_central_utils nvs_flash
)

set(vsn_model_file
    "${CMAKE_CURRENT_SOURCE_DIR}/models/vsn_private_encoder_projection64_esp32s3_int8.espdl")
esptool_py_flash_to_partition(flash "model" "${vsn_model_file}")
