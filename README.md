<p align="left">
    <img src="assets/logo.png" alt="Fit-DCS Logo" width="200">
</p>

# Fit-DCS

A Python package for analysis of Diffuse Correlation Spectroscopy (DCS) data.

---

## Features
- Forward models for several geometries
- Inverse solvers (nonlinear fitting, Modified Beer-Lambert law) for analysis of experimental data
- Utility functions for data loading, noise modeling, motion artifact correction, and correlation calculation
- C library integration for performance-critical computations (e.g., calculation of autocorrelation from time-tag data)

---

## Installation

1. Clone the repository and navigate into the directory:
    ```bash
    git clone https://github.com/marconabacino/Fit-DCS.git
    cd Fit-DCS
    ```
   
2. (Optional but recommended) Activate the virtual environment where you want to install the package,
creating it if necessary. For example, if you are using anaconda/miniconda:
    ```bash
    conda create -n fitdcs # Skip if the environment already exists
    conda activate fitdcs
    ```

3. Install the package using pip:
    ```bash
    pip install .
    ```

### Optional dependencies

Fit-DCS can be used without additional libraries for most features.
However, if you want to use the **time-tag data loading** functionality (necessary for the software correlator), you
need the `TimeTagger` library from Swabian Instruments, which is available at
[https://www.swabianinstruments.com/time-tagger/downloads](https://www.swabianinstruments.com/time-tagger/downloads).

If `TimeTagger` is not installed, time-tag data loading will not be available, but the rest of Fit-DCS will work
normally.

### Optional C libraries (Linux and macOS)

Some performance-critical components are implemented in C for speed. For Windows users, these libraries are already
precompiled and shipped with the package, which will load them automatically.

Linux and macOS users need to compile the C libraries manually. Until then, Fit-DCS will automatically fall back to
slower Python implementations with a warning message. Note that the libraries should be built before installing the
package. If you build or rebuild them after installation, you'll need to reinstall the package in order for Python to
pick up the new binaries:
```bash
pip install --force-reinstall .
```

#### Prerequisites
- CMake (version 3.31 or higher)
  - macOS: `brew install cmake`
  - Ubuntu/Debian: `sudo apt install cmake`
- A C compiler
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: `gcc` or `clang` (usually pre-installed)
- (only for `bilayer`) GNU Scientific Library (GSL)
  - macOS: `brew install gsl`
  - Ubuntu/Debian: `sudo apt install libgsl-dev`

#### Build steps
1. Open a terminal in the [`c_lib`](./c_lib) folder.
2. Create and enter a build directory:
   ```bash
   mkdir build
   cd build
   ```
3. Configure CMake:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
4. Build the libraries:
   ```bash
   cmake --build .
   ```
5. After a successful build, the compiled library will be automatically copied to the `../src/fit_dcs/lib/` folder.
Fit-DCS will then detect and use the libraries automatically at runtime.

---

## Getting started

Extensive examples are available as Jupyter notebooks in the [`examples/`](./examples) folder:
- [Forward modeling](./examples/1-forward_modeling.ipynb)
- [Inverse solvers](./examples/2-analyzing.ipynb)
- [Data loading](./examples/3-loading.ipynb)

Note that they need the `examples/data/` folder. Run the notebooks from their location to ensure the correct
relative paths.

### Command-line scripts
Fit-DCS is intended to be used as a library, but it also provides some command-line scripts for common tasks. These
are installed automatically with the package:
- `fitdcs-corr`: Compute autocorrelations from time-tagger data files.
- `fitdcs-fit`: Fit DCS data created by `fitdcs-corr` with the nonlinear solver, using the semi-infinite model.

Keep in mind that these scripts are basic and do not cover all use cases. For more complex analyses, it's recommended
to use the library directly in your own Python scripts or Jupyter notebooks.

The scripts are installed with the package, so they are available in your environment's PATH and you can run them
directly from the command line:
```bash
fitdcs-corr --config path/to/config.yaml
fitdcs-fit --config path/to/config.yaml
```

#### Configuration files
All scripts require a YAML configuration file specifying parameters such as input files, analysis settings, and output
directories. Example configuration files showing the available options are provided in the
[`examples/yaml/`](./examples/yaml) folder for reference, but you should create your own based on your
specific data and analysis needs.

---

## License and citation

Fit-DCS is distributed under the **GNU General Public License v3.0 (GPLv3)**.
This means you are free to use, modify, and redistribute the software, provided that derivative works are also released
under the same license.

If you use this software in your research, please cite:

**Nabacino, M. (2025). Fit-DCS: A Python toolbox for Diffuse Correlation Spectroscopy analysis.**
GitHub: [https://github.com/marconabacino/Fit-DCS](https://github.com/marconabacino/Fit-DCS)

A formal citation entry is also available in the [`CITATION.cff`](./CITATION.cff) file.