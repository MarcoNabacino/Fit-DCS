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
    _HAS_CORR_LIB = True
except FileNotFoundError:
    warnings.warn("Fast C implementation of async_corr not found. Using slower Python implementation."
                  "macOS/Linux users need to compile the C library manually. See the README for instructions.")
    ASYNC_CORR_LIB = None
    _HAS_CORR_LIB = False

# Load libbilayer
try:
    lib_path = get_lib_path("libbilayer")
    BILAYER_LIB = ctypes.CDLL(lib_path)
    BILAYER_LIB.integrand.argtypes = [
        c_int,
        POINTER(c_double),
        c_void_p
    ]
    BILAYER_LIB.integrand.restype = c_double
    _HAS_BILAYER_LIB = True
except FileNotFoundError:
    warnings.warn("Fast C implementation of bilayer model not found. Using slower Python implementation.")
    BILAYER_LIB = None
    _HAS_BILAYER_LIB = False