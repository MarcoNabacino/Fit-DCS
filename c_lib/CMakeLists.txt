cmake_minimum_required(VERSION 3.31)
project(Fit_DCS C)

set(CMAKE_C_STANDARD 11)

# Output directory for shared libraries
set(LIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/../src/fit_dcs/lib)
# Create 'lib' folder if it doesn't exist
file(MAKE_DIRECTORY ${LIB_OUTPUT_DIR})

# Asynchronous correlation
add_library(async_corr SHARED async_corr.c)
add_custom_command(TARGET async_corr POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:async_corr>
        ${LIB_OUTPUT_DIR}/$<TARGET_FILE_NAME:async_corr>
)

# Bilayer with GSL
find_package(GSL REQUIRED)
add_library(bilayer SHARED bilayer.c)
target_include_directories(bilayer PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(bilayer PRIVATE ${GSL_LIBRARIES})
add_custom_command(TARGET bilayer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:bilayer>
        ${LIB_OUTPUT_DIR}/$<TARGET_FILE_NAME:bilayer>
)
# Copy also the GSL shared libraries to the output directory
file(GLOB GSL_DLLS "C:/msys64/mingw64/bin/libgsl*.dll" "C:/msys64/mingw64/bin/libgslcblas*.dll")
foreach(dll ${GSL_DLLS})
    get_filename_component(dll_name ${dll} NAME)
    add_custom_command(TARGET bilayer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${dll}"
        "${LIB_OUTPUT_DIR}/${dll_name}"
    )
endforeach()
