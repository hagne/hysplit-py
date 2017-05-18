from setuptools import setup#, find_packages

setup(
    name='hysplit-py',
    version='',
    packages=['hysplit_py'], #find_packages()
    url='',
    license='',
    author='Hagen Telg',
    author_email='hagen@hagnet.net',
    description='',
    install_requires = ['numpy', 'pandas', 'matplotlib', 'mpl_toolkits', 'geopy', 'ftplib', 'netCDF4', 'magic',
                        'osgeo',
                        # 'gdal'
                        ],
    # extra_requires=['newrelic'],
)
