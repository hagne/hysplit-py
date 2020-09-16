from netCDF4 import Dataset
import magic
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import warnings
try:
    import gdal
except ModuleNotFoundError:
    warnings.warn('gdal not found, this will probably result in prublems (with respect to landuse) further down the road')



def read_file(fname, filetype=None):
    """I think that will work only with the particular I originally use, which I found somewhere here: https://landcover.usgs.gov/landcoverdata.php"""
    #######
    # this is in order to read the internal map ... not used anymore
    # lat = np.linspace(89, -90, 180)
    # lon = np.linspace(-180, 179, 360)
    #
    # fname = '/Users/htelg/Hysplit4/bdyfiles/LANDUSE.ASC'
    # land_use_map = pd.read_fwf(fname, names=lon)
    # land_use_map.index = lat


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

        arr = pd.DataFrame(arr, index=lat_arr, columns=lon_arr)

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

    # legend
    land_types = ['Water',
             'Evergreen Needle leaf Forest',
             'Evergreen Broadleaf Forest',
             'Deciduous Needle leaf Forest',
             'Deciduous Broadleaf Forest',
             'Mixed Forests',
             'Closed Shrublands',
             'Open Shrublands',
             'Woody Savannas',
             'Savannas',
             'Grasslands',
             'Permanent Wetland',
             'Croplands',
             'Urban and Built-Up',
             'Cropland/Natural Vegetation Mosaic',
             'Snow and Ice',
             'Barren or Sparsely Vegetated']

    land_types_legend = pd.DataFrame(land_types)
    land_types_legend.columns = ['land_use_type']
    land_types_legend['color'] = np.nan

    return LandUseMap(arr, land_types_legend)


def save2netCDF(land_use_map, fname, leave_open=False):
    nc = Dataset(fname, 'w')

    ### Dimensions
    lat_dim = nc.createDimension('lat', land_use_map.land_use_data.index.shape[0])
    lon_dim = nc.createDimension('lon', land_use_map.land_use_data.columns.shape[0])

    ### Variables
    lat_var = nc.createVariable('lat', land_use_map.land_use_data.index.dtype, 'lat')
    lat_var[:] = land_use_map.land_use_data.index.values

    lon_var = nc.createVariable('lon', land_use_map.land_use_data.columns.dtype, 'lon')
    lon_var[:] = land_use_map.land_use_data.columns.values

    land_use_var = nc.createVariable('land_use', land_use_map.land_use_data.values.dtype, ('lat', 'lon',))
    land_use_var[:] = land_use_map.land_use_data.values
    if not leave_open:
        nc.close()
    return


def plot_land_use_map(self, **kwargs):
    lon_2d, lat_2d = np.meshgrid(self.land_use_data.columns, self.land_use_data.index)
    f, a = plt.subplots()
    pc = a.pcolormesh(lon_2d, lat_2d, self.land_use_data)
    return f, a, pc


class LandUseMap(object):
    def __init__(self, df, legend = None):
        self.land_use_data = df
        self.legend = legend

    plot = plot_land_use_map
    save = save2netCDF

    def get_resolution(self):
        it = self.land_use_data.index.values
        res_it = (it[:-1] - it[1:]).mean()
        ct = self.land_use_data.columns.values
        res_ct = (ct[1:] - ct[:-1]).mean()
        return (res_it, res_ct)

    def down_sample(self, nrows=2, ncols=2):
        mod_row = self.land_use_data.shape[0] % nrows
        mod_col = self.land_use_data.shape[1] % ncols
        if mod_row or mod_col:

            nrows_h = nrows
            nrows_l = nrows
            while self.land_use_data.shape[0] % nrows_h:
                nrows_h += 1
            while self.land_use_data.shape[0] % nrows_l:
                nrows_l -= 1
            nrows_sug = (nrows_l, nrows_h)

            ncols_h = ncols
            ncols_l = ncols
            while self.land_use_data.shape[0] % ncols_h:
                ncols_h += 1
            while self.land_use_data.shape[0] % ncols_l:
                ncols_l -= 1
            ncols_sug = (ncols_l, ncols_h)

            txt = 'Non-integer number of blocksizes sizes. Adjust nrows and ncols so an integer number of blocks fit into current grid. Suggestions: nrows = {}, ncols = {}'.format(nrows_sug,
                                                                                                                                                                                    ncols_sug)
            raise ValueError(txt)

        def get_number_of_max_occurence(at):
            counts = np.bincount(at.reshape(at.size))

            # randomize in case argmax ambiguous
            # if np.argmax(counts) != (len(counts) - 1  - np.argmax(counts[::-1])):
            counts[counts < counts.max()] = 0
            counts = counts * np.random.random(counts.size)

            return np.argmax(counts)

        a = self.land_use_data.values
        h, w = a.shape
        a1 = a.reshape(h // nrows, nrows, -1, ncols)
        a2 = a1.swapaxes(1, 2)
        a3 = a2.reshape(-1, nrows, ncols)

        res = np.zeros(a3.shape[0])
        for e, block in enumerate(a3):
            res[e] = get_number_of_max_occurence(block)
        # res = res.reshape(np.array(a.shape) // 10)
        res = res.reshape((a.shape[0] // nrows, a.shape[1] // ncols))

        #     nrows = 10
        #     ncols = 10

        idx_ds = np.apply_along_axis(lambda x: x.mean(), 1, self.land_use_data.index.values.reshape(-1, nrows))
        col_ds = np.apply_along_axis(lambda x: x.mean(), 1, self.land_use_data.columns.values.reshape(-1, ncols))
        df = pd.DataFrame(res.astype(int), index=idx_ds, columns=col_ds)

        return LandUseMap(df)


