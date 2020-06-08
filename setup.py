from setuptools import find_packages, setup

VERSION = "0.0.1"
setup(
    name="effdepth",
    version=VERSION,
    license="MIT",
    classifiers=[
        "License :: MIT",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(include=["effdepth"]),
    python_requires=">=3.8",
    setup_requires=[
        "pip>=19.1",
        "setuptools>=41",
    ],
    install_requires=[
        "torch>=1.5",
        "pytorch_lightning>=0.7",
        "scikit-image>=0.17",
        "scikit-video>=1.1",
    ],
    extras_require={},
    tests_require=[],
)
