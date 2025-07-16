import os
import platform


def get_lib_path(lib_name: str) -> str:
    """
    Returns the path to a shared library with platform-specific extension.
    Assumes the library is located in the top-level 'lib' directory.

    :param lib_name: Name of the shared library.
    :return: Absolute path to the shared library.
    """
    ext_map = {
        'Windows': '.dll',
        'Linux': '.so',
        'Darwin': '.dylib'  # macOS
    }

    system = platform.system()
    ext = ext_map.get(system)
    if ext is None:
        raise RuntimeError(f"Unsupported operating system: {system}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    lib_dir = os.path.join(project_root, 'lib')
    lib_path = os.path.join(lib_dir, lib_name + ext)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Library {lib_name}{ext} not found in {lib_dir}")

    return lib_path