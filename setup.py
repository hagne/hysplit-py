import sys

required_verion = (3,)
if sys.version_info < required_verion:
    raise ValueError('At least python {} needed! You are trying to install under python {}'.format('.'.join(str(i) for i in required_verion), sys.version))

# import ez_setup
# ez_setup.use_setuptools()

from setuptools import setup
# from distutils.core import setup
setup(
    name='hysplit-py',
    version="0.1",
    packages=['hysplit_py'],
    author="Hagen Telg",
    author_email="hagen@hagnet.net",
    description="HYSPLIT wrapper",
    license="MIT",
    keywords="hysplit",
    url="https://github.com/hagne/hysplit-py",
    # install_requires = ['numpy', 'pandas', 'matplotlib', 'mpl_toolkits', 'geopy', 'netCDF4', 'magic', 'gdal'],
    # extras_require={'plotting': ['matplotlib'],
    #                 'testing': ['scipy']},
    # test_suite='nose.collector',
    # tests_require=['nose'],
)
