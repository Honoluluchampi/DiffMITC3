from setuptools import setup, find_packages

setup(
    name="DiffMITC3",
    version='0.0.1',
    packages=find_packages(),
    install_requires=['DiffMITC3Impl @ git+https://github.com/Honoluluchampi/DiffMITC3Impl'],
    author="Riku TOYOTA",
    description="An implementation of the differentiable MITC3 element layer for Pytorch."
)