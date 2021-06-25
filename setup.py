# from distutils.core import setup
from setuptools import find_packages,setup

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
  name = 'pyristic',\
  version = '0.1.0',\
  license='MIT',\
  description = 'Set of metaheuristic for solve optimization problems.',\
  author = 'Jesús Armando Ortíz Peñafiel',\
  author_email = 'armandopenafiel12@gmail.com',\
  long_description = LONG_DESCRIPTION,\
  long_description_content_type="text/markdown",\
  url = 'https://github.com/JAOP1/',\
  download_url = 'https://github.com/JAOP1/pyristic/archive/refs/tags/0.1.0.tar.gz',\
  keywords = ['Optimization', 'Metaheuristic', 'Python'],\
  install_requires=[
          'numpy>=1.15.4',
          'tqdm>=4.28.1',
          'numba>=0.50.1'
      ],\
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7'
  ],\
  packages=find_packages(exclude=("examples",))
)