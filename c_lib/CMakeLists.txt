cmake_minimum_required(VERSION 3.31)
project(Fit_DCS C)

set(CMAKE_C_STANDARD 11)

add_library(async_corr SHARED async_corr.c)

# Create 'lib' folder if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/../src/fit_dcs/lib)

# Post-build command to copy the shared library to the 'lib' folder
add_custom_command(TARGET async_corr POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:async_corr>
        ${CMAKE_SOURCE_DIR}/../src/fit_dcs/lib/$<TARGET_FILE_NAME:async_corr>
)