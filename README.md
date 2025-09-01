# Fit-DCS

A Python package for analysis of Diffuse Correlation Spectroscopy (DCS) data.

---

## Features
- Forward models for several geometries
- Inverse solvers (nonlinear fitting, Modified Beer-Lambert law) for analysis of experimental data
- Utility functions for data loading, noise modeling, and correlation calculation
- C library integration for performance-critical computations (i.e., calculation of autocorrelation from time-tag data)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/marconabacino/Fit-DCS.git
cd Fit-DCS
```
(Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

Install the package using pip:

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

### Optional C software correlator

The performance-critical software correlator is implemented in C for speed.
For Windows users, a precompiled version of the C library is already included in the repository and gets automatically
used by the package. Of course, you can also compile it yourself if you want.

Linux and macOS users need to compile it manually using CMake. Until then, the package will automatically fall back
to a slower Python implementation.

#### Prerequisites
- CMake (version 3.31 or higher)
  - macOS: `brew install cmake`
  - Ubuntu/Debian: `sudo apt install cmake`
- A C compiler
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: `gcc` or `clang` (usually pre-installed)

#### Compilation steps
1. Open a terminal in the [`c_lib`](./c_lib) folder.
2. Create a build directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```
3. Run CMake to configure the build:
   ```bash
   cmake ..
   ```
4. Compile the library:
   ```bash
   cmake --build . --config Release
   ```
5. After a successful build, the compiled library will be automatically copied into the `lib/` folder:
- macOS: `../src/fit_dcs/lib/libasync_corr.dylib`
- Linux: `../src/fit_dcs/lib/libasync_corr.so`
- Windows: `../src/fit_dcs/lib/async_corr.dll` (already included)

Fit-DCS will then detect and use the compiled library automatically.

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
are installed automatically when you install the package via `pip`:
- `fitdcs-corr`: Compute autocorrelations from time-tagger data files.
- `fitdcs-fit`: Fit DCS data created by `fitdcs-corr` with the nonlinear solver, using the semi-infinite forward model.

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
directories. Example configuration files are provided in the [`examples/yaml/`](./examples/yaml) folder for reference,
but you should create your own based on your specific data and analysis needs.

