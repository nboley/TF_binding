"""
Copyright (c) 2011-2015 Nathan Boley

This file is part of pyTFbindtools.

pyTFbindtools is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyTFbindtools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyTFbindtools.  If not, see <http://www.gnu.org/licenses/>.
"""
from distutils.core import setup, Extension

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = cythonize([
    Extension("pyTFbindtools.sequence", 
              ["pyTFbindtools/sequence.pyx", ]),
])

config = {
    'include_package_data': True,
    'ext_modules': extensions, 
    'description': 'pyTFbindtools',
    'author': 'Nathan Boley',
    'url': 'NA',
    'download_url': 'http://github.com/nboley/TF_binding/',
    'author_email': 'npboley@gmail.com',
    'version': '0.1.1',
    'packages': ['pyTFbindtools', 
                 'pyTFbindtools.selex'
             ],
    'setup_requires': [],
    'install_requires': [ 'scipy', 'numpy' ],
    'scripts': [],
    'name': 'pyTFbindtools'
}

if __name__== '__main__':
    setup(**config)
