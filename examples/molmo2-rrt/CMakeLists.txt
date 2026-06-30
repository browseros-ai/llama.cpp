find_package(Threads REQUIRED)

add_library(molmo2-rrt-preprocess-obj OBJECT molmo2-rrt-preprocess.cpp)
target_compile_definitions(molmo2-rrt-preprocess-obj PRIVATE MOLMO2_RRT_NO_MAIN)
target_include_directories(molmo2-rrt-preprocess-obj SYSTEM PRIVATE ../../vendor)
target_compile_features(molmo2-rrt-preprocess-obj PRIVATE cxx_std_17)

add_library(molmo2-rrt-vision-obj OBJECT molmo2-rrt-vision.cpp)
target_compile_definitions(molmo2-rrt-vision-obj PRIVATE MOLMO2_RRT_NO_MAIN)
target_link_libraries(molmo2-rrt-vision-obj PRIVATE ggml)
target_compile_features(molmo2-rrt-vision-obj PRIVATE cxx_std_17)

add_library(molmo2-rrt-compose-obj OBJECT molmo2-rrt-compose-embd.cpp)
target_compile_definitions(molmo2-rrt-compose-obj PRIVATE MOLMO2_RRT_NO_MAIN)
target_link_libraries(molmo2-rrt-compose-obj PRIVATE llama)
target_compile_features(molmo2-rrt-compose-obj PRIVATE cxx_std_17)

add_library(molmo2-rrt-decode-obj OBJECT molmo2-rrt-decode.cpp)
target_compile_definitions(molmo2-rrt-decode-obj PRIVATE MOLMO2_RRT_NO_MAIN)
target_link_libraries(molmo2-rrt-decode-obj PRIVATE llama)
target_compile_features(molmo2-rrt-decode-obj PRIVATE cxx_std_17)

set(TARGET llama-molmo2-rrt-run)
add_executable(${TARGET}
    molmo2-rrt-run.cpp
    $<TARGET_OBJECTS:molmo2-rrt-preprocess-obj>
    $<TARGET_OBJECTS:molmo2-rrt-vision-obj>
    $<TARGET_OBJECTS:molmo2-rrt-compose-obj>
    $<TARGET_OBJECTS:molmo2-rrt-decode-obj>)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
