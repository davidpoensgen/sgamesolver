from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension("homotopies.qre_ct", ["homotopies/qre_ct.pyx"], include_dirs=[np.get_include()]),
    Extension("homotopies.tracing_ct", ["homotopies/tracing_ct.pyx"], include_dirs=[np.get_include()]),
]

setup(
    name="test",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    zip_safe=False
)

# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import numpy as np
#
# # print(np.get_include())
# #
# # setup(
# #     name='qre_ct_app',
# #     ext_modules=cythonize("qre_ct.pyx", include_path=[np.get_include()]),
# #     zip_safe=False,
# # )
# extensions = [Extension("qre_ct", ["qre_ct.c"], include_dirs=[np.get_include()])]
# setup(name="qre", ext_modules=extensions)
