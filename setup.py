from distutils.core import setup

setup(
    name='hysplit-py',
    version='',
    packages=['hysplit_py'],
    url='',
    license='',
    author='Hagen Telg',
    author_email='hagen@hagnet.net',
    description='',
    install_requires = ['numpy', 'pandas', 'matplotlib', 'mpl_toolkits', 'geopy', 'netCDF4', 'magic', 'gdal'],
    # extra_requires=['newrelic'],
)
