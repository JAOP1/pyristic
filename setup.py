"""
Module: Configuration required to publish package.

Created: 2023-06-01
Author: Jesus Armando Ortiz
__________________________________________________
"""
from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", "r", encoding="utf-8") as dependencies:
    list_dependencies = [
        dependency.replace("\n", "") for dependency in dependencies.readlines()
    ]

setup(
    name="pyristic",
    version="v2.0.0",
    license="MIT",
    description="Set of metaheuristic for solve optimization problems.",
    author="Jesús Armando Ortíz Peñafiel",
    author_email="armandopenafiel12@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/JAOP1/",
    download_url="https://github.com/JAOP1/pyristic/archive/refs/tags/v1.4.1.tar.gz",
    keywords=["Optimization", "Metaheuristic", "Python"],
    install_requires=list_dependencies,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("examples", "docs")),
)
