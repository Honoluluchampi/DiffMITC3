from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension("DiffMITC3", ["src/pybind.cpp"])
]

setup(
    name="DiffMITC3Layer",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["numpy"],
    author="Riku TOYOTA",
    description="An implementation of the differentiable MITC3 element layer for Pytorch."
)