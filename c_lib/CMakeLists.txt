cmake_minimum_required(VERSION 3.31)
project(Fit_DCS C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output directory for shared libraries
set(LIB_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/../src/fit_dcs/lib")
file(MAKE_DIRECTORY ${LIB_OUTPUT_DIR})

# Helper function for creating shared libraries and copying them
function(add_shared_lib name)
    add_library(${name} SHARED ${name}.c)
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:${name}>
        ${LIB_OUTPUT_DIR}/$<TARGET_FILE_NAME:${name}>
    )
endfunction()

# ----------------------------------------------------------------
# async_corr - no external dependencies
# ----------------------------------------------------------------
add_shared_lib(async_corr)

# ----------------------------------------------------------------
# bilayer - requires GSL
# ----------------------------------------------------------------
find_package(GSL QUIET)

if (GSL_FOUND)
    message(STATUS "GSL found: ${GSL_LIBRARIES}")
    add_library(bilayer SHARED bilayer.c)
    target_include_directories(bilayer PRIVATE ${GSL_INCLUDE_DIRS})
    target_link_libraries(bilayer PRIVATE ${GSL_LIBRARIES})

    add_custom_command(TARGET bilayer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:bilayer>
            ${LIB_OUTPUT_DIR}/$<TARGET_FILE_NAME:bilayer>
    )

    # Windows only - copy GSL DLLs to output directory
    if (WIN32)
        file(GLOB GSL_DLLS "${GSL_DIR}/bin/libgsl*.dll" "${GSL_DIR}/bin/libgslcblas*.dll")
        foreach(dll ${GSL_DLLS})
            get_filename_component(dll_name ${dll} NAME)
            add_custom_command(TARGET bilayer POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${dll}"
                    "${LIB_OUTPUT_DIR}/${dll_name}"
            )
        endforeach()
    endif()
else()
    message(WARNING "GSL not found. Skipping bilayer library build.")
endif()