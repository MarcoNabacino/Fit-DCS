cmake_minimum_required(VERSION 3.31)
project(Fit_DCS C)

set(CMAKE_C_STANDARD 11)

add_library(async_corr SHARED utils/async_corr.c)