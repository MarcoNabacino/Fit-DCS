"""
 Fit-DCS: A Python toolbox for Diffuse Correlation Spectroscopy analysis
 Copyright (C) 2025  Marco Nabacino

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import platform
import warnings
from pathlib import Path
import ctypes
from ctypes import POINTER, c_int64, c_double, c_int, c_void_p


def get_lib_path(lib_name: str) -> str:
    """
    Returns the path to a shared library with platform-specific extension.
    Assumes the library is located in the top-level 'lib' directory.

    :param lib_name: Name of the shared library.
    :return: Absolute path to the shared library.
    """
    ext_map = {'Windows': '.dll', 'Linux': '.so', 'Darwin': '.dylib'}

    system = platform.system()
    ext = ext_map.get(system)
    if ext is None:
        raise RuntimeError(f"Unsupported operating system: {system}")

    project_root = Path(__file__).parent.parent.resolve()
    lib_path = project_root / 'lib' / f"{lib_name}{ext}"

    if not lib_path.is_file():
        raise FileNotFoundError(f"Library {lib_name}{ext} not found in {lib_path.parent}")

    return str(lib_path)


# Attempt to load the C libraries
# Load libasync_corr
try:
    lib_path = get_lib_path("libasync_corr")
    ASYNC_CORR_LIB = ctypes.CDLL(lib_path)
    ASYNC_CORR_LIB.async_corr.argtypes = [
        POINTER(c_int64),  # t
        c_int64,  # n_tags
        c_int64,  # p
        c_int64,  # m
        c_int64,  # s
        c_double,  # tau_start
        c_double,  # t0
        POINTER(c_double),  # g2_out
        POINTER(c_double),  # tau_out
    ]
    ASYNC_CORR_LIB.async_corr.restype = None
except FileNotFoundError:
    warnings.warn("Fast C implementation of async_corr not found. Using slower Python implementation."
                  "macOS/Linux users need to compile the C library manually. See the README for instructions.")
    ASYNC_CORR_LIB = None

# Load libbilayer
try:
    lib_path = get_lib_path("libbilayer")
    BILAYER_LIB = ctypes.CDLL(lib_path)
    BILAYER_LIB.integrand.argtypes = [
        c_int,  # Number of variables
        POINTER(c_double),  # Integration variables
        c_void_p  # Additional parameters
    ]
    BILAYER_LIB.integrand.restype = c_double
except FileNotFoundError:
    warnings.warn("Fast C implementation of bilayer model not found. Using slower Python implementation.")
    BILAYER_LIB = None