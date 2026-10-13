cmake_minimum_required(VERSION 3.18.4)
find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
project(AscendAddPipeline LANGUAGES ASC AICPU CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
if(NOT TORCH_NPU_PATH OR NOT EXISTS "${TORCH_NPU_PATH}/include")
    message(FATAL_ERROR "Set TORCH_NPU_PATH to the installed torch_npu package")
endif()
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Source the CANN set_env.sh before configuring")
endif()
set(ASCEND_DIR "$ENV{ASCEND_HOME_PATH}")
set(BISHENG "${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/ccec_compiler/bin/bisheng")
if(NOT EXISTS "${BISHENG}")
    message(FATAL_ERROR "Bisheng compiler not found: ${BISHENG}")
endif()
set(NPU_ARCH "dav-3510" CACHE STRING "Bisheng NPU architecture (950)")

set(ASCEND_INCLUDES
    "${ASCEND_DIR}/include"
    "${ASCEND_DIR}/pkg_inc"
    "${ASCEND_DIR}/pkg_inc/base"
    "${ASCEND_DIR}/include/hcomm"
    "${ASCEND_DIR}/compiler/tikcpp/include"
    "${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl"
    "${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface"
    "${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/impl"
    "${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/tiling"
    "${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api"
)

# The Add source contains both the Ascend C kernel and the PyTorch registration.
# Compile it with Bisheng, then link the resulting object into the Python module.
set(CMAKE_CXX_COMPILER "${BISHENG}")
set(CMAKE_LINKER "${BISHENG}")
add_library(add_pipeline_obj OBJECT csrc/add_pipeline.cpp)
set_source_files_properties(csrc/add_pipeline.cpp PROPERTIES COMPILE_FLAGS "--npu-arch=${NPU_ARCH} -xasc")
target_include_directories(add_pipeline_obj PRIVATE
    ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_PATH}/include" ${ASCEND_INCLUDES})
target_compile_options(add_pipeline_obj PRIVATE -O3 -DTORCH_MODE)
separate_arguments(TORCH_COMPILE_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(add_pipeline_obj PRIVATE ${TORCH_COMPILE_FLAGS})

add_library(_C SHARED csrc/extension.cpp $<TARGET_OBJECTS:add_pipeline_obj>)
set_target_properties(_C PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME "_C")
target_include_directories(_C PRIVATE ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${ASCEND_INCLUDES})
target_compile_options(_C PRIVATE ${TORCH_COMPILE_FLAGS})
target_link_directories(_C PRIVATE "${TORCH_NPU_PATH}/lib" "${ASCEND_DIR}/lib64")
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_npu ascendcl platform register tiling_api runtime hccl hcomm)
add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_C>
            "${CMAKE_CURRENT_SOURCE_DIR}/ascend_add_pipeline/$<TARGET_FILE_NAME:_C>")
