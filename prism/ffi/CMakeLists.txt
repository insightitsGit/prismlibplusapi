cmake_minimum_required(VERSION 3.22)
project(prism_driver VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Source files ──────────────────────────────────────────────────────────────
set(SOURCES
    src/prism_driver.cpp
    src/chorus_client.cpp
    src/tensor_cipher.cpp
)

# ── Shared library (DLL / .so / .dylib) ──────────────────────────────────────
add_library(prism_driver SHARED ${SOURCES})
target_compile_definitions(prism_driver PRIVATE PRISM_DRIVER_EXPORTS)
target_include_directories(prism_driver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# ── gRPC / Protobuf ───────────────────────────────────────────────────────────
find_package(Protobuf REQUIRED)
find_package(gRPC    REQUIRED)

target_link_libraries(prism_driver PRIVATE
    gRPC::grpc++
    protobuf::libprotobuf
)

# ── Platform-specific output naming ──────────────────────────────────────────
if(WIN32)
    set_target_properties(prism_driver PROPERTIES
        OUTPUT_NAME "prism_driver"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
elseif(APPLE)
    set_target_properties(prism_driver PROPERTIES
        OUTPUT_NAME "prism_driver"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        MACOSX_RPATH ON
    )
else()
    set_target_properties(prism_driver PROPERTIES
        OUTPUT_NAME "prism_driver"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS prism_driver
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES prism_driver.h DESTINATION include/prism)

# ── CMake presets (multi-platform) ───────────────────────────────────────────
# Defined in CMakePresets.json (see below).
# Quick build commands:
#   Linux x86_64:    cmake --preset linux-x64   && cmake --build --preset linux-x64
#   Linux arm64:     cmake --preset linux-arm64  && cmake --build --preset linux-arm64
#   macOS arm64:     cmake --preset macos-arm64  && cmake --build --preset macos-arm64
#   Windows x64:     cmake --preset windows-x64  && cmake --build --preset windows-x64
#   Windows arm64:   cmake --preset windows-arm64 && cmake --build --preset windows-arm64
