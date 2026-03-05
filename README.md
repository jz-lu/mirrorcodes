# mirrorcodes
Repository for numerics of mirror codes. At some point in the hopefully not too far future, this README will document the entire repository well.
For now, however, a basic introduction to working with mirror codes is given in `getting_started.ipynb`, with the basic functions of how to generate a mirror code in a few lines, and how to evaluate it.


## Dependencies
* `stim`: must be at least version 1.15
* `sinter`: note that if you are using newer versions of numpy, you may get an assertion error when running `stim` or `sinter`. This can be fixed by upgrading to dev version 1.16, as discussed [here](https://quantumcomputing.stackexchange.com/questions/44224/why-is-there-this-assertion-error-with-sinter-collect-in-the-stim-tutorial-wher).
* `pysat`: install with `examples` via `pip install git+https://github.com/pysathq/pysat.git`
* `tesseract-decoder`

