# cosipy.nonimaging

The cosipy library is [COSI](https://cosi.ssl.berkeley.edu)'s high-level analysis software. 

COSI is a single mission with multiple detectors: 
- The Germanium (Ge) detector, an imaging-capable Compton telescope
- Non-imaging (count-based) detectors:
  - Bismuth Germanium Oxide (BGO): which serves as the Anti-Coincidence Shield (ACS)
  - Background and Transient Observer (BTO): a Student Collaboration experiment that will fly on the COSI satellite
  - The Ge detector in single-site event mode: data with no identified tracks, only individual hits.

The main [cosipy](https://github.com/cositools/cosipy) library can only analyze Compton data from the Ge detector.
This extension --a namespace package-- will allow (work in progress) to analyze data from the non-imaging detectors.

## Installation

Clone or download this repo and run

```commandline
pip install .
```

If you are a developer, it is recommended that you use the `-e` to make it "editable" --i.e. you will execute the code
in your working direction, as opposed as the code as it was when you run `pip install .` for the last time.

This will install the main cosipy automatically for you. By default, it will fetch the latest released cosipy version.
If you are a developer, it is recommended that install cosipy from source before installing cosipy.nonimaging. The
cosipy library is not yet stable, and this will allow you to better control your environment.

For the main cosipy installation and usage instructions please refer to the main [cosipy documentation](https://cositools.github.io/cosipy/).

# Usage

Since the main cosipy is a dependency of this submodule, you'll be able to import the main cosipy, However, just 
importing the main cosipy won't allow you to use `cosipy.nonimaging`. For example, the following won't work

```python
import cosipy

print(cosipy.nonimaging.__version__) # This will fail
```

You need to import the non-imaging module explicitly:

```python
import cosipy.nonimaging # Explicit module import

print(cosipy.nonimaging.__version__) # Now this works!
```
