


def read_land_use_file(fname, filetype = None):
    allowed_filetypes = ['TIFF', 'netCDF']
    if not filetype:
#         if filetype not in allowed_filetypes:
#             txt = 'Filetype {} not known'
#     else:
        if 'TIFF image data' in magic.from_file(fname):
            filetype = 'TIFF'
        elif 'Hierarchical Data Format (version 5) data' in magic.from_file(fname):
            filetype = 'netCDF'

    if filetype == 'TIFF':
        data = gdal.Open(fname)

        geotrans = data.GetGeoTransform()

        lon_orig = geotrans[0]
        lon_sw = geotrans[1]

        lat_orig = geotrans[3]
        lat_sw = geotrans[5]

        lon_fin = lon_orig + (data.RasterXSize * lon_sw)
        lon_arr = np.arange(lon_orig, lon_fin, lon_sw)

        lat_fin = lat_orig + (data.RasterYSize * lat_sw)
        lat_arr = np.arange(lat_orig, lat_fin, lat_sw)

        arr = data.ReadAsArray()

        arr = pd.DataFrame(arr, index = lat_arr, columns=lon_arr)

    # These are saved LandUseMap instances
    elif filetype == 'netCDF':
        nc = Dataset(fname, 'r')
        lat = nc.variables['lat'][:]
        lon = nc.variables['lon'][:]
        land_use_data = nc.variables['land_use'][:]
        arr = pd.DataFrame(land_use_data, index=lat, columns=lon)

    else:
        txt = 'Filetype "{}" not known or not recognized. Try to set the "filetype" kwarg. Allowed options are {}'.format(filetype, allowed_filetypes)
        raise ValueError(txt)
    return LandUseMap(arr)