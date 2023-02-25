from setuptools import setup, find_packages

setup(
    name="pytorchts",
    version="0.7.0",
    description="PyTorch Probabilistic Time Series Modeling framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kashif Rasul",
    author_email="kashif.rasul@zalando.de",
    url="https://github.com/zalandoresearch/pytorch-ts",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "pytorch-lightning~=1.5",
        "protobuf~=3.20.3",
        "gluonts>=0.12.0",
        "holidays",
        "matplotlib",
    ],
    test_suite="tests",
    tests_require=["flake8", "pytest"],
)
