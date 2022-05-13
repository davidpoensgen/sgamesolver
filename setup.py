from setuptools import setup, Extension
from setuptools.command.install import install
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    cython = True
except ImportError:
    from setuptools.command.build_ext import build_ext
    cythonize = None
    cython = False
import numpy as np

with open('README.md', 'r', encoding='utf-8') as readme:
    long_description = readme.read()
no_openmp = False
no_cython = False


class install_with_options(install):

    user_options = install.user_options + [
        ('no-cython', None, 'Skips installing the cython extensions completely.'),
        ('no-openmp', None, 'Installs cython extensions without openMP support (for parallel computing).'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.no_cython = False
        self.no_openmp = False

    def run(self):
        global no_cython
        global no_openmp
        no_cython = self.no_cython
        no_openmp = self.no_openmp
        install.run(self)


class build_ext_openmp(build_ext):
    def build_extensions(self):
        if no_cython:
            self.extensions = []
        if not no_openmp:
            openmp_extensions = [
                    'sgamesolver.homotopy._shared_ct',
                    'sgamesolver.homotopy._qre_ct',
                    'sgamesolver.homotopy._logtracing_ct',
                    ]
            c = self.compiler.compiler_type
            if c == 'msvc':
                extra_compile_args = ['/openmp']
                extra_link_args = ['/openmp']
            else:
                extra_compile_args = ['-fopenmp']
                extra_link_args = ['-fopenmp']
            for e in self.extensions:
                if e.name in openmp_extensions:
                    e.extra_compile_args += extra_compile_args
                    e.extra_link_args += extra_link_args
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        'sgamesolver.homotopy._shared_ct',
        ['sgamesolver/homotopy/_shared_ct.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'sgamesolver.homotopy._ipm_ct',
        ['sgamesolver/homotopy/_ipm_ct.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'sgamesolver.homotopy._loggame_ct',
        ['sgamesolver/homotopy/_loggame_ct.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'sgamesolver.homotopy._qre_ct',
        ['sgamesolver/homotopy/_qre_ct.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'sgamesolver.homotopy._logtracing_ct',
        ['sgamesolver/homotopy/_logtracing_ct.pyx'],
        include_dirs=[np.get_include()]
    ),
]

setup(
    name='sgamesolver',
    version='0.1',
    description='A homotopy-based solver for stochastic games',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/seibelsh/sgamesolver',
    project_urls={
        'Bug Tracker': 'TO-BE-PUT-HERE-2',
        'Please cite this paper': 'TO-BE-PUT-HERE-3'
    },
    author='Steffen Eibelshäuser, David Poensgen',
    author_email='steffen.eibelshaeuser@gmail.com, davidpoensgenecon@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
    ],
    keywords='game theory, stochastic games, stationary equilibrium, homotopy method, computational economics',

    cmdclass={
        'build_ext': build_ext_openmp,
        'install': install_with_options
    },
    packages=['sgamesolver', 'sgamesolver.homotopy'],
    ext_modules=cythonize(ext_modules, language_level="3") if cython else ext_modules,

    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'cython'],

    zip_safe=False,
)
