from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
import numpy as np

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    cython = True  # flag tells setup() to cythonize Extensions
except ImportError:
    from setuptools.command.build_ext import build_ext
    cythonize = None
    cython = False


with open('README.md', 'r', encoding='utf-8') as readme:
    long_description = readme.read()

no_openmp = False
no_cython = False


class OptionsMixin:

    user_options = [
        ('no-cython', None, 'Skips installing the cython extensions completely.'),
        ('no-openmp', None, 'Installs cython extensions without openMP (i.e. without support for parallel computing).'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.no_cython = False
        self.no_openmp = False

    def run(self):
        global no_cython
        no_cython = self.no_cython
        global no_openmp
        no_openmp = self.no_openmp
        super().run()


class install_with_options(OptionsMixin, install):
    user_options = OptionsMixin.user_options + getattr(install, 'user_options', [])


class develop_with_options(OptionsMixin, develop):
    user_options = OptionsMixin.user_options + getattr(develop, 'user_options', [])


class build_ext_openmp(build_ext):

    user_options = build_ext.user_options + [
        ('no-openmp', None, 'Installs cython extensions without openMP (i.e. without support for parallel computing).'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.no_openmp = False

    def build_extensions(self):
        # no_cython: skip all extensions
        if no_cython:
            print('~'*10 + ' not compiling any extensions ' + '~'*10)
            self.extensions = []
        # no_openmp: build extensions, but without openmp compiler flags
        # passed either via global (if install or develop were called, e.g. from pip) or as option to build_ext itself
        elif no_openmp or self.no_openmp:
            print('~'*10 + ' compiling extensions without openmp ' + '~'*10)
        elif not (no_openmp or self.no_openmp):
            openmp_extensions = [
                    'sgamesolver.homotopy._shared_ct',
                    'sgamesolver.homotopy._qre_ct',
                    'sgamesolver.homotopy._logtracing_ct',
                    ]
            compiler = self.compiler.compiler_type
            if compiler == 'msvc':
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
    version='1.0',
    description='A homotopy-based solver for stochastic games',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/davidpoensgen/sgamesolver',
    project_urls={
        'github': 'https://github.com/davidpoensgen/sgamesolver/',
        'Bug Tracker': 'https://github.com/davidpoensgen/sgamesolver/issues',
        'Documentation': 'https://sgamesolver.readthedocs.io/en/latest/',
        'Please cite this paper': 'https://dx.doi.org/10.2139/ssrn.3316631'
    },
    author='Steffen Eibelshäuser, David Poensgen',
    author_email='steffen.eibelshaeuser@gmail.com, davidpoensgenecon@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Cython',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
    ],
    keywords='game theory, stochastic games, stationary equilibrium, Markov perfect equilibrium,'
             ' homotopy method, computational economics',

    cmdclass={
        'build_ext': build_ext_openmp,
        'install': install_with_options,
        'develop': develop_with_options,
    },
    packages=['sgamesolver', 'sgamesolver.homotopy'],
    ext_modules=cythonize(ext_modules, language_level="3") if cython else ext_modules,

    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'cython',
        'pandas',
        'matplotlib'
    ],

    entry_points={
        'console_scripts': [
            'sgamesolver-timings = sgamesolver.utility.excel_timings:main',
        ],
    },

    zip_safe=False,
)
