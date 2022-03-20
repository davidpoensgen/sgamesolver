from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np


with open('README.md', 'r', encoding='utf-8') as readme:
    long_description = readme.read()


# TODO: could later use optional for extensions - to continue installation if building exts fails.
# Ideally, we'll wheel anyways.
# https://docs.python.org/3/distutils/setupscript.html -> 2.3.5

ext_modules = [Extension('dsgamesolver.homotopy._ipm_ct', ['dsgamesolver/homotopy/_ipm_ct.pyx'],
                         include_dirs=[np.get_include()]),
               Extension('dsgamesolver.homotopy._loggame_ct', ['dsgamesolver/homotopy/_loggame_ct.pyx'],
                         include_dirs=[np.get_include()]),
               Extension('dsgamesolver.homotopy._qre_ct', ['dsgamesolver/homotopy/_qre_ct.pyx'],
                         include_dirs=[np.get_include()]),
               Extension('dsgamesolver.homotopy._tracing_ct', ['dsgamesolver/homotopy/_tracing_ct.pyx'],
                         include_dirs=[np.get_include()]),
               ]

setup(name='dsgamesolver',
      version='0.1',
      description='A homotopy-based solver for stochastic games',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='TO-BE-PUT-HERE',
      project_urls={
          'Bug Tracker': 'TO-BE-PUT-HERE-2',
          'Please cite this paper': 'TO-BE-PUT-HERE-3'
      },
      author='Steffen Eibelshäuser, David Poensgen',
      author_email='eibelshaeuser@econ.uni-frankfurt.de, davidpoensgenecon@gmail.com',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics'
      ],
      keywords='game theory, stochastic games, homotopy method, computational economics',

      cmdclass={'build_ext': build_ext},
      packages=['dsgamesolver', 'dsgamesolver.homotopy'],
      ext_modules=ext_modules,

      python_requires='>=3.6',
      install_requires=['numpy', 'scipy', 'cython'],

      zip_safe=False,
      )
