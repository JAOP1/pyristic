from distutils.core import setup

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
  name = 'pyristic',\
  packages = ['pyristic'],\
  version = '0.1.2',\
  license='MIT',\
  description = 'Set of metaheuristic for solve optimization problems.',\
  author = 'Jesús Armando Ortíz Peñafiel',\
  author_email = 'armandopenafiel12@gmail.com',\
  long_description = LONG_DESCRIPTION,\
  url = 'https://github.com/JAOP1/',\
  download_url = 'https://github.com/JAOP1/pyristic/archive/refs/tags/0.1.2.tar.gz',\
  keywords = ['Optimization', 'Metaheuristic', 'Python'],\
  install_requires=[
          'numpy>=1.15.4',
          'tqdm>=4.28.1',
          'numba==0.41.0'
      ],\
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7'
  ],
)