from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("homotopy._ipm_ct", ["homotopy/_ipm_ct.pyx"], include_dirs=[np.get_include()]),
               Extension("homotopy._loggame_ct", ["homotopy/_loggame_ct.pyx"], include_dirs=[np.get_include()]),
               Extension("homotopy._qre_ct", ["homotopy/_qre_ct.pyx"], include_dirs=[np.get_include()]),
               Extension("homotopy._qre_ct_2", ["homotopy/_qre_ct_2.pyx"], include_dirs=[np.get_include()]),
               Extension("homotopy._tracing_ct", ["homotopy/_tracing_ct.pyx"], include_dirs=[np.get_include()]),
               ]

setup(name="test",
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules,
      zip_safe=False,)
