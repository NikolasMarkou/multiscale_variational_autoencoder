# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="mvae",
    version="0.1.0",
    description="Multidimensional Variational Autoencoder",
    long_description=readme,
    author="Nikolas Markou",
    author_email="nikolasmarkou@gmail.com",
    url="https://github.com/NikolasMarkou/multiscale_variational_autoencoder",
    license=license,
    packages=find_packages(exclude=("tests", "docs"))
)