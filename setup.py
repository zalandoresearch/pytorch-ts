from setuptools import setup, find_packages

setup(
    name='pytorch-ts',
    version='0.1.0',
    description="PyTorch Probabilistic Time Series Modeling framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",

    url='https://github.com/kashif/pytorch-ts',
    license='Apache License Version 2.0',

    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires = [
        'torch>=1.3.0',
        'holidays',
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
        'ujson',
        'pydantic',
        'matplotlib',
    ],

    test_suite='tests',
    tests_require = [
        'flake8',
        'pytest'
    ],
)
