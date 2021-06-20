from distutils.core import setup
setup(
  name = 'pyristic',\
  packages = ['pyristic'],\
  version = '0.1',\
  license='MIT',\
  description = 'TYPE YOUR DESCRIPTION HERE',\
  author = 'Jesús Armando Ortíz Peñafiel',\
  author_email = 'armandopenafiel12@gmail.com',\
  url = 'https://github.com/JAOP1/',\
  download_url = 'https://github.com/JAOP1/pyristic/archive/refs/tags/0.1.tar.gz',\
  keywords = ['Optimization', 'Metaheuristic', 'Python'],\
  install_requires=[
          'numpy',
          'tqdm',
          'numba'
      ],\
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research'
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7'
  ],
)