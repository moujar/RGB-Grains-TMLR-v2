`morpho.py` and `iterprogress.py` in this directory are vendored from the
[gala](https://github.com/janelia-flyem/gala) project (BSD-3-Clause), copied
directly rather than installed as a dependency (see the original French
comment in `__init__.py`). They provide the h-minima / watershed-with-dams
helpers used by `rgb_grains/segmentation/grain_hs/segmentation.py`.

Only these two files are used; the rest of the upstream `gala` package is not
vendored here.
