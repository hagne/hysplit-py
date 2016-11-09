import pandas as _pd
from copy import deepcopy
import numpy as _np
from mpl_toolkits.basemap import Basemap as _Basemap
import matplotlib.pylab as _plt
from geopy.distance import vincenty
import subprocess
import os
from matplotlib.colors import LinearSegmentedColormap as _LinearSegmentedColormap
from matplotlib.colors import LogNorm as _LogNorm
import ftplib
from netCDF4 import Dataset as _Dataset
import copy

def save_result_netCDF(result, fname, leave_open=False):
    file_mode = 'w'

    try:
        ni = _Dataset(fname, file_mode)
    except RuntimeError:
        if os.path.isfile(fname):
            os.remove(fname)
            ni = _Dataset(fname, file_mode)

    # result = run.result

    # concentration
    data_v = result.concentration.values

    # concentration.dimensions
    lat_dim, lon_dim = data_v.shape
    lat_dim = ni.createDimension('latitude', lat_dim)
    lon_dim = ni.createDimension('longitude', lon_dim)

    # concentration.vaiables
    concentration_var = ni.createVariable('concentration', data_v.dtype, ('latitude', 'longitude'))
    concentration_var[:] = data_v
    concentration_var.units = 'particle concentration no/m^3'

    lat_var = ni.createVariable('latitudes', result.concentration.index.dtype, 'latitude')
    lat_var[:] = result.concentration.index.values
    lat_var.units = 'degrees in decimals'

    lon_var = ni.createVariable('longitudes', result.concentration.columns.dtype, 'longitude')
    lon_var[:] = result.concentration.columns.values
    lon_var.units = 'degrees in decimals'

    # attributes
    ni.run_settings = result.run_settings
    ni.qc_reports = result.qc_reports

    # todo: get rid of following once entire project is saved instead of result only
    ni.start_loc = result._parent.parameters.starting_loc  # When saving the entire project (run) this should not be needed anymore

    if leave_open:
        #         pass
        return ni
    else:
        ni.close()


def load_result_netCDF(fname, leave_open=False):
    ni = _Dataset(fname, 'r')

    lat = ni.variables['latitudes']
    lon = ni.variables['longitudes']

    concentration = ni.variables['concentration']
    data = _pd.DataFrame(concentration[:], index=lat[:], columns=lon[:])
    data.columns.name = 'lon'
    data.index.name = 'lat'

    settings = ni.getncattr('run_settings')

    start_loc = ni.getncattr('start_loc')
    qc_reports = [ni.getncattr('qc_reports')]

    # todo: get rid of following once entire project is saved instead of result only
    parent = type("parent", (object,), {"parameters": type('parameters', (object,), {'starting_loc': [start_loc]})})

    if leave_open:
        pass
    # return ni
    else:
        ni.close()

    res = HySplitConcentration(parent, data)
    res.qc_reports = qc_reports
    return res


def datetime_str2hysplittime(time_string = '2010-01-01 00:00:00'):
    t = _pd.to_datetime(time_string)
    return '{} {:02d} {:02d} {:02d} {:02d}'.format(str(t.year)[-2:], t.month, t.day, t.hour, t.minute)


def date_str2file_name_list(start, stop, data_format):
    """
    Parameters
    ----------
    data_format: str [gdas1]
    """
    month = ['XXX', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    format_options = ['gdas1', 'gdas0p5', 'test']
    if data_format not in format_options:
        txt = 'data_formate has to be one of %s' % (format_options)
        raise ValueError(txt)

    if data_format == 'test':
        return ['oct1618.BIN']

    elif data_format == 'gdas1':
        tdt = _pd.to_datetime(start)
        year = str(tdt.year)[-2:]
        week = ((tdt.day - 1) // 7) + 1
        fname_s = 'gdas1.{}{}.w{}'.format(month[tdt.month], year, week)

        stt = _pd.to_datetime(stop)
        year = str(stt.year)[-2:]
        week = ((stt.day - 1) // 7) + 1
        fname_e = 'gdas1.{}{}.w{}'.format(month[stt.month], year, week)

        if (stt - tdt) / _pd.to_timedelta(1, 's') >= 0:
            direction = 1
        else:
            direction = -1

        fname = [fname_s]
        if fname_s != fname_e:
            fname = [fname_s]
            i = 0
            stt = _pd.to_datetime(start)
            while i in range(1000):
                i += 1
                stt = _pd.to_datetime(stt) + _pd.to_timedelta(direction, 'D')
                year = str(stt.year)[-2:]
                week = ((stt.day - 1) // 7) + 1
                fname_t = 'gdas1.{}{}.w{}'.format(month[stt.month], year, week)
                if fname_t not in fname:
                    fname.append(fname_t)
                # print('{}\t{}'.format(fname_t, fname_e))
                if fname_t == fname_e:
                    break

    elif data_format == 'gdas0p5':
        tdt = _pd.to_datetime(start)
        fbase = '{}{:02d}{:02d}_gdas0p5'
        fname_s = fbase.format(tdt.year, tdt.month, tdt.day)

        stt = _pd.to_datetime(stop)
        fname_e = fbase.format(stt.year, stt.month, stt.day) #'gdas1.{}{}.w{}'.format(month[stt.month], year, week)

        if (stt - tdt) / _pd.to_timedelta(1, 's') >= 0:
            direction = 1
        else:
            direction = -1

        fname = [fname_s]
        if fname_s != fname_e:
            fname = [fname_s]
            i = 0
            stt = _pd.to_datetime(start)
            while i in range(1000):
                i += 1
                stt = _pd.to_datetime(stt) + _pd.to_timedelta(direction, 'D')
                fname_t = fbase.format(stt.year, stt.month, stt.day) #'gdas1.{}{}.w{}'.format(month[stt.month], year, week)
                if fname_t not in fname:
                    fname.append(fname_t)
                # print('{}\t{}'.format(fname_t, fname_e))
                if fname_t == fname_e:
                    break

    return fname

def date_str2file_name(date_str, data_format):
    """
    Parameters
    ----------
    data_format: str [gdas1]
    """
    format_options = ['gdas1', 'test']
    if data_format not in format_options:
        txt = 'data_formate has to be one of %s' % (format_options)
        raise ValueError(txt)

    if data_format == 'test':
        return ''

    if data_format == 'gdas1':
        tdt = _pd.to_datetime(date_str)
        month = ['XXX', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec']
        year = str(tdt.year)[-2:]
        week = ((tdt.day - 1) // 7) + 1
        fname = 'gdas1.{}{}.w{}'.format(month[tdt.month], year, week)
    return fname

def read_hysplit_conc_output_file(fname='/Users/htelg/Hysplit4/working/cdump'):
    # os.chdir(self.settings.path2working_directory)
    path2executable = ['./con2asc', '-icdump', '-s', '-d']
    process = subprocess.check_output(path2executable, stderr=subprocess.STDOUT)
    stdout = process.decode()
    if stdout:
        txt = 'Stdout produced a non empty string ... something is probably wrong?!?'
        raise HysplitError(txt)

    dfin = _pd.read_csv(fname + '.txt')
    dfin.columns = [i.strip() for i in dfin.columns]

    thedata = dfin.columns[-1]

    lon = dfin.LON.unique()
    lon.sort()
    lat = dfin.LAT.unique()
    lat.sort()
    lat[:] = lat[::-1]

    matrix = _pd.DataFrame(index=lat, columns=lon, dtype=float)
    matrix.index.name = 'lat'
    matrix.columns.name = 'lon'

    for index, row in dfin.iterrows():
        matrix.loc[row['LAT'], row['LON']] = row[thedata]

    return matrix


def read_hysplit_traj_output_file(fname='/Users/htelg/Hysplit4/working/tdump'):
    # fname_traj = '/Users/htelg/Hysplit4/working/tdump'
    traj_rein = open(fname)

    output_dict = {}

    # Record #1
    no_met_grids = traj_rein.readline()
    no_met_grids = int(no_met_grids.split()[0])

    if no_met_grids > 1:
        txt = 'Programming requrired! Till now only one meterologic grid is allowed. The output indicates %s were used.' % no_met_grids
        raise ValueError(txt)

    # Record #2

    rec2 = traj_rein.readline()
    rec2 = rec2.split()
    met_model_id = rec2[0]

    year = int(rec2[1])
    if year > 50:
        year = 1900 + year
    else:
        year = 2000 + year
    month = int(rec2[2])
    day = int(rec2[3])
    hour = int(rec2[4])
    forecast_hour = rec2[5]

    date_time_start = '{}-{:02}-{:02} {:02}:00:00'.format(year, month, day, hour)
    date_time_start = _pd.Timestamp(date_time_start)

    met_model_id, date_time_start

    output_dict['met_model_id'] = met_model_id
    output_dict['date_time_start'] = date_time_start

    # Record 3

    rec3 = traj_rein.readline()
    rec3 = rec3.split()
    num_of_trajs = int(rec3[0])
    direction_of_traj = rec3[1]
    vertical_motion_method = rec3[2]

    num_of_trajs, direction_of_traj, vertical_motion_method

    output_dict['num_of_trajs'] = num_of_trajs
    output_dict['direction_of_traj'] = direction_of_traj
    output_dict['vertical_motion_method'] = vertical_motion_method

    # record 4

    rec4 = [traj_rein.readline() for i in range(num_of_trajs)]

    # start_date_times = np.zeros(num_of_trajs, dtype = '<U19' )
    start_date_times = _np.zeros(num_of_trajs, dtype='datetime64[ns]')
    lans_start = _np.zeros(num_of_trajs)
    lons_start = _np.zeros(num_of_trajs)
    alts_start = _np.zeros(num_of_trajs)
    for e, line in enumerate(rec4):
        l = line.split()
        year = int(l[0])
        if year > 50:
            year = 1900 + year
        else:
            year = 2000 + year
        month = int(l[1])
        day = int(l[2])
        hour = int(l[3])
        start_date_times[e] = _pd.Timestamp('{}-{:02}-{:02} {:02}:00:00'.format(year, month, day, hour))
        lans_start[e] = float(l[4])
        lons_start[e] = float(l[5])
        alts_start[e] = float(l[6])
    # break

    start_conditions = _pd.DataFrame()
    start_conditions['date_time'] = _pd.Series(start_date_times)
    start_conditions['latitude'] = _pd.Series(lans_start)
    start_conditions['longitude'] = _pd.Series(lons_start)
    start_conditions['altitude_above_ground(m)'] = _pd.Series(alts_start)
    start_conditions

    output_dict['start_conditions'] = start_conditions

    # Record #5
    rec5 = traj_rein.readline()

    rec5 = rec5.split()
    num_of_output_variables = int(rec5[0])
    output_variables = []
    for i in range(num_of_output_variables):
        output_variables.append(rec5[i + 1])

    output_variables

    output_dict['output_variables'] = output_variables

    # Record Loop #6 through the number of hours in the simulation
    # I6 - trajectory number
    # I6 - meteorological grid number or antecedent trajectory number
    # 5I6 - year month day hour minute of the point
    # I6 - forecast hour at point
    # F8.1 - age of the trajectory in hours
    # 2F9.3 - position latitude and longitude
    # 1X,F8.1 - position height in meters above ground
    # n(1X,F8.1) - n diagnostic output variables (1st to be output is always pressure)

    import io
    rec6 = traj_rein.read()

    rec6_clean = '\n'.join([','.join(e) for e in [i.split() for i in rec6.split('\n')]])

    names = ['trajectory_num', 'met_grid_num', 'year', 'month', 'day', 'hour', 'minute', 'forcast_hour', 'age_of_trajectory(h)', 'latitude', 'longitude', 'altitude_above_ground(m)']
    for ov in output_variables:
        names.append(ov)

    buffer = io.StringIO(rec6_clean)
    trajectory = _pd.read_csv(buffer, names=names)

    trajectory.year.loc[trajectory.year > 50] += 1900
    trajectory.year.loc[trajectory.year <= 50] += 2000

    ts_cols = ['year', 'month', 'day', 'hour', 'minute']
    time_df = trajectory[ts_cols]
    trajectory.index = _pd.to_datetime(time_df)

    trajectory.drop(ts_cols, axis=1, inplace=True)

    output_list = []
    for i in range(num_of_trajs):
        odt = output_dict.copy()
        odt['trajectory'] = (trajectory[trajectory.trajectory_num == i + 1])
        output_list.append(HySplitTrajectory(odt))
    return output_list

from matplotlib.collections import LineCollection


def plot_conc_on_map(concentration, resolution='c', back_ground = None, lat_c='auto', lon_c='auto', w='auto', h='auto', zoom_out=2, bmap=None, verbose=False, **plt_kwargs):
    """Plots a map of the flight path

    Note
    ----
    packages: matplotlib-basemap,

    Arguments
    ---------
    color_gradiant: bool or colormap.
        The trajectory can be plotted so it changes color in time. If True the standard cm map is used but you can also pass a cm map here, e.g. color_gradiant = plt.cm.jet
    zoom_out: float or array-like of len==2
    back_ground: str [shadedrelief, bluemarble, etopo]
    """
    data = concentration.concentration
    x_lon, y_lat = _np.meshgrid(data.columns, data.index)

    if bmap:
        a = _plt.gca()
        f = a.get_figure()
    else:
        f, a = _plt.subplots()

        if _np.any(_np.array([lat_c, lon_c, w, h]) == 'auto'):
            # if 1:

            lon_center = (x_lon.max() + x_lon.min()) / 2.
            lat_center = (y_lat.max() + y_lat.min()) / 2.

            if not hasattr(zoom_out, '__iter__'):
                zoom_out = [zoom_out, zoom_out]
            height = vincenty((y_lat.max(), lon_center), (y_lat.min(), lon_center)).m * zoom_out[0]
            width = vincenty((lat_center, x_lon.max()), (lat_center, x_lon.min())).m * zoom_out[1]

        if lat_c != 'auto':
            lat_center = lat_c
        if lon_c != 'auto':
            lon_center = lon_c
        if w != 'auto':
            width = w
        if h != 'auto':
            height = h

        if verbose:
            print(('lat_center: %s\n'
                   'lon_center: %s\n'
                   'width: %s\n'
                   'height: %s' % (lat_center, lon_center, width, height)))

        bmap = _Basemap(projection='aeqd',
                       lat_0=lat_center,
                       lon_0=lon_center,
                       width=width,
                       height=height,
                       resolution=resolution,
                       ax=a)

        if not back_ground:
            # Fill the globe with a blue color
            wcal = _np.array([161., 190., 255.]) / 255.
            boundary = bmap.drawmapboundary(fill_color=wcal)

            grau = 0.9
            continents = bmap.fillcontinents(color=[grau, grau, grau], lake_color=wcal)
        elif back_ground == 'shadedrelief':
            bmap.shadedrelief()
        elif back_ground == 'bluemarble':
            bmap.bluemarble()
            # bmap.lsmask[bmap.lsmask == 1] = 0

        elif back_ground == 'etopo':
            bmap.etopo()

        costlines = bmap.drawcoastlines()
        bmap.drawstates()
        bmap.drawcountries()


    # #draw paralles
    #         parallels = np.arange(l,90,10.)
    #         m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    #         # draw meridians
    #         meridians = np.arange(180.,360.,10.)
    #         m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)



    z = _np.ma.masked_invalid(data.values)
    xm_lon, ym_lat = bmap(x_lon, y_lat)
    pc = a.pcolormesh(xm_lon, ym_lat, z, zorder=100, norm=_LogNorm(), alpha=0.6, cmap=_plt.cm.Accent, linewidth=0, rasterized=True, **plt_kwargs)
    f.colorbar(pc)

    return bmap

def source_attribution_angular(concentration_instance, noang = 8, raise_error = False):
    """Calculates the ratio of aerosols comming form that particular solid angle."""

    def calculate_initial_compass_bearing_vec(pointA, pointB):
        """
        Calculates the bearing between two points.
        The formulae used is the following:
            θ = atan2(sin(Δlong).cos(lat2),
                      cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
        :Parameters:
          - `pointA: The tuple representing the latitude/longitude for the
            first point. Latitude and longitude must be in decimal degrees
          - `pointB: The tuple representing the latitude/longitude for the
            second point. Latitude and longitude must be in decimal degrees
        :Returns:
          The bearing in degrees
        :Returns Type:
          float
        """
        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError("Only tuples are supported as arguments")

        lat1 = _np.deg2rad(pointA[0])
        lat2 = _np.deg2rad(pointB[0])

        diffLong = _np.deg2rad(pointB[1] - pointA[1])

        x = _np.sin(diffLong) * _np.cos(lat2)
        y = _np.cos(lat1) * _np.sin(lat2) - (_np.sin(lat1)
                * _np.cos(lat2) * _np.cos(diffLong))

        initial_bearing = _np.arctan2(x, y)
        # Now we have the initial bearing but math.atan2 return values
        # from -180° to + 180° which is not what we want for a compass bearing
        # The solution is to normalize the initial bearing as shown below
        initial_bearing = _np.rad2deg(initial_bearing)

        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    data_conc = concentration_instance.concentration.copy()
    pointA = tuple(concentration_instance._parent.parameters.starting_loc[0][0:2])
    x_lon, y_lat = _np.meshgrid(data_conc.columns, data_conc.index)
    cbv = calculate_initial_compass_bearing_vec(pointA, (y_lat, x_lon))
    cbv[_np.logical_and(y_lat == pointA[0], x_lon == pointA[1])] = _np.nan # the origin will always be 0 and therefore gives a bias to 0 ... so I get rid of it

    concentration_instance.compass_bearings = cbv


    data_conc[_np.logical_and(y_lat == pointA[0], x_lon == pointA[1])] = _np.nan # the origin will always be 0 and therefore gives a bias to 0 ... so I get rid of it
    ############## new
    data_conc[_np.isnan(data_conc)] = 0
    data_conc /= data_conc.values.sum()

    # noofbins = 8
    hist, bedges = _np.histogram(cbv, weights=data_conc, bins=noang * 2, range=(0, 360))

    # cycle and downsample
    hist_sds = _np.roll(hist, 1).reshape(-1, 2).sum(axis=1)
    bedges_sds = _np.append(bedges[1::2][-1], bedges[1::2])

    center_sds = (bedges_sds[1:] + bedges_sds[:-1]) / 2
    center_sds[0] = 0

    df = _pd.DataFrame(hist_sds, index=center_sds)

    ############### old
    # data_conc /= data_conc.values[~_np.isnan(data_conc.values)].sum() #normalized
    # for e, center in enumerate(angbincent):
    #     angle_range = angbins[e:e+2]
    #     data = data_conc.copy()
    #     with _np.errstate(invalid='ignore'):
    #         data.values[_np.logical_or(cbv <= angle_range[0], cbv > angle_range[1])] = _np.nan
    #     z = _np.ma.masked_invalid(data)
    #     df.loc[center] = z.sum()

    tolerance = 0.05
    if not (1 - tolerance) < df.values.sum() < (1. + tolerance):
        if raise_error:
            raise ValueError('This value has to be equal to 1. It is %s'%(df.sum()))
        else:
            txt = 'Sum over all probabilities has to be equal to 1. It is %s' % (df.sum())
            # print(concentration_instance.qc_report)
            # import pdb
            # pdb.set_trace()
            concentration_instance.qc_reports.append(txt)

    return Source_Attribution_Angular(df)


def source_attribution_land_use(res, land_use):
    """
    Parameters
    ----------
    land_use: LandUseMap instance
    """
    ### generate land use map with the same dimentions like the concentration grid
    res_tmp = res.concentration.copy()
    res_tmp[:] = _np.nan
    res_land_use = _pd.concat([land_use.land_use_data, res_tmp])
    res_land_use.sort_index(inplace=True, ascending=False)
    res_land_use.interpolate(method='nearest', axis=0, inplace=True)
    res_land_use.interpolate(method='nearest', axis=1, inplace=True)
    res_land_use_ds = res_land_use[res_tmp.columns]
    res_land_use_ds = res_land_use_ds.loc[res_tmp.index]

    ###

    res_norm = res.concentration.copy()
    res_norm[_np.isnan(res_norm)] = 0
    res_norm /= res_norm.values.sum()

    res_lu = _pd.DataFrame(land_use.legend.land_use_type.copy())
    res_lu['concentration'] = _np.nan
    for e, lu in enumerate(land_use.legend.land_use_type):
        res_tmp = res_norm.copy()
        res_tmp[res_land_use_ds != e] = 0
        res_lu.loc[e, 'concentration'] = res_tmp.values.sum()
    # print(bla)
    #     break
    return res_lu

class Source_Attribution_Angular(object):
    def __init__(self, data):
        self.source_attribution_angular = data

    def plot(self):
        df = self.source_attribution_angular
        f, a = _plt.subplots(subplot_kw=dict(projection='polar'))
        a.set_theta_zero_location("N")
        a.set_theta_direction(-1)
        bars = a.bar(_np.deg2rad(df.index), df.values, width= 2* _np.pi / df.shape[0],  align='center')
        imax = df.values.max()
        for i, bar in zip(df.values[:, 0], bars):
            bar.set_facecolor(_plt.cm.jet(i / imax))
            bar.set_alpha(0.5)


class Source_Attribution_Land_use(object):
    def __init__(self, parent):
        self._parent = parent
        self.land_use_map = None
        self.__source_attribution_land_use = None

    @property
    def source_attribution_land_use(self):
        if not _np.any(self.__source_attribution_land_use):
            if not self.land_use_map:
                raise AttributeError('please set the attribute land_use_map with an LandUseMap instance.')
            self.__source_attribution_land_use = source_attribution_land_use(self._parent, self.land_use_map)
        return self.__source_attribution_land_use

    def plot(self, low_lim=0.02, **kwargs):
        #     lowlim = 0.02
        res_lu = self.source_attribution_land_use.copy()
        res_lu_other = res_lu[res_lu.concentration < low_lim].copy()
        res_lu[res_lu.concentration < low_lim] = _np.nan
        res_lu.dropna(inplace=True)
        res_lu.set_value(res_lu.shape[1] + 1, 'land_use_type', 'Other')
        res_lu.set_value(res_lu.shape[1] + 1, 'concentration', res_lu_other.concentration.sum())
        res_lu.sort_values('concentration', inplace=True)

        f, a = _plt.subplots()
        out = a.pie(res_lu.concentration.values, labels=res_lu.land_use_type, autopct='%1.1f%%', **kwargs)
        centre_circle = _plt.Circle((0, 0), 0.75, color='black', fc='white', linewidth=1.25)
        a.add_artist(centre_circle)
        a.set_aspect(1)
        return a

def plot_traj_on_map(self, intensity ='time', resolution='c', lat_c='auto', lon_c='auto', w='auto', h='auto', bmap=None, color_gradiant = True, verbose=False, **plt_kwargs):
    """Plots a map of the flight path

    Note
    ----
    packages: matplotlib-basemap,

    Arguments
    ---------
    color_gradiant: bool or colormap.
        The trajectory can be plotted so it changes color in time. If True the standard cm map is used but you can also pass a cm map here, e.g. color_gradiant = plt.cm.jet
    """

    def get_colorMap():
        blue = _np.array([0., 13., 120.]) / 255.
        orange = _np.array([255., 102., 0.]) / 255.
        green = _np.array([9., 84., 0.]) / 255.

        color1 = _np.append(blue, 1)
        color2 = _np.append(orange, 1)
        color3 = _np.append(green, 1)
        color4 = _np.array([0, 0, 0, 0])

        # color1 = _np.array([133, 12, 0, 255.]) / 255.
        # color2 = _np.array([10, 90, 0, 255. / 2]) / 255
        # color3 = _np.array([0, 0, 0, 0])
        steps = [0, 0.4, 0.8, 1]
        cdict = {'red': ((steps[0], color1[0], color1[0]),
                         (steps[1], color2[0], color2[0]),
                         (steps[2], color3[0], color3[0]),
                         (steps[3], color4[0], color4[0])
                         ),

                 'green': ((steps[0], color1[1], color1[1]),
                           (steps[1], color2[1], color2[1]),
                           (steps[2], color3[1], color3[1]),
                           (steps[3], color4[1], color4[1])
                           ),

                 'blue': ((steps[0], color1[2], color1[2]),
                          (steps[1], color2[2], color2[2]),
                          (steps[2], color3[2], color3[2]),
                          (steps[3], color4[2], color4[2])
                          ),

                 'alpha': ((steps[0], color1[3], color1[3]),
                           (steps[1], color2[3], color2[3]),
                           (steps[2], color3[3], color3[3]),
                           (steps[3], color4[3], color4[3])
                           ),
                 }

        hag_cmap = _LinearSegmentedColormap('hag_cmap', cdict)
        hag_cmap.set_bad(_np.array([0, 0, 0, 0]))
        return hag_cmap


    def make_segments(x, y):
        '''
        Create list of line segments from x and y coordinates, in the correct format for LineCollection:
        an array of the form   numlines x (points per line) x 2 (x and y) array
        '''

        points = _np.array([x, y]).T.reshape(-1, 1, 2)
        segments = _np.concatenate([points[:-1], points[1:]], axis=1)

        return segments


    def colorline(x, y, z=None, zmax = 1, cmap=_plt.get_cmap('copper'), norm=_plt.Normalize(0.0, 1.0), alpha=None, **kwargs):
        '''
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        '''

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = _np.linspace(0.0, zmax, len(x))

        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
            z = _np.array([z])

        z = _np.asarray(z)

        segments = make_segments(x, y)
        lc = LineCollection(segments, array=z, cmap=cmap, norm=norm,
                            alpha=alpha,
                            **kwargs
                            )

        ax = _plt.gca()
        ax.add_collection(lc)

        return lc


    three_d = False
    data = self.trajectory.copy()
    data = data.loc[:, ['longitude', 'latitude']]
    data = data.dropna()

    if not bmap:

        if _np.any(_np.array([lat_c, lon_c, w, h]) == 'auto'):
            lon_center = (data.longitude.values.max() + data.longitude.values.min()) / 2.
            lat_center = (data.latitude.values.max() + data.latitude.values.min()) / 2.

            points = _np.array([data.latitude.values, data.longitude.values]).transpose()
            distances_from_center_lat = _np.zeros(points.shape[0])
            distances_from_center_lon = _np.zeros(points.shape[0])
            for e, p in enumerate(points):
                distances_from_center_lat[e] = vincenty(p, (lat_center, p[1])).m
                distances_from_center_lon[e] = vincenty(p, (p[0], lon_center)).m

            lat_radius = distances_from_center_lat.max()
            lon_radius = distances_from_center_lon.max()
            scale = 1
            border = scale * 2 * _np.array([lat_radius, lon_radius]).max()

            height = border + lat_radius
            width = border + lon_radius

        if lat_c != 'auto':
            lat_center = lat_c
        if lon_c != 'auto':
            lon_center = lon_c
        if w != 'auto':
            width = w
        if h != 'auto':
            height = h

        if verbose:
            print(('lat_center: %s\n'
                   'lon_center: %s\n'
                   'width: %s\n'
                   'height: %s' % (lat_center, lon_center, width, height)))
    if not three_d:
        if not bmap:
            bmap = _Basemap(projection='aeqd',
                            lat_0=lat_center,
                            lon_0=lon_center,
                            width=width,
                            height=height,
                            resolution=resolution)

            # Fill the globe with a blue color
            wcal = _np.array([161., 190., 255.]) / 255.
            boundary = bmap.drawmapboundary(fill_color=wcal)

            grau = 0.9
            continents = bmap.fillcontinents(color=[grau, grau, grau], lake_color=wcal)
            costlines = bmap.drawcoastlines()


        x, y = bmap(data.longitude.values, data.latitude.values)

        if color_gradiant:
            if type(color_gradiant).__name__ == 'bool':
                cm = get_colorMap()
            else:
                cm = color_gradiant
            if intensity == 'time':
                print('time')
                hours = (self.trajectory.index[-1] - self.trajectory.index[0]) / _np.timedelta64(1,'h')
                path = colorline(x, y, zmax = hours,
                                 norm = _plt.Normalize(0.0, hours) ,
                                 cmap = cm)
            else:
                try:
                    zt = self.trajectory[intensity]
                except KeyError:
                    opt = list(self.trajectory.columns.values)
                    opt.append('time')
                    txt = '{} not an option. Try {}'.format(intensity, opt )
                    raise KeyError(txt)
                path = colorline(x, y, zt,
                             norm=_plt.Normalize(zt.min(), zt.max()),
                             cmap=cm,
                             **plt_kwargs)

            f = path.get_figure()
            cb = f.colorbar(path)

        else:
            path = bmap.plot(x, y,
                             color='m', **plt_kwargs)
        return bmap


    # else:
    #     bmap = _Basemap(projection='aeqd',
    #                     lat_0=lat_center,
    #                     lon_0=lon_center,
    #                     width=width,
    #                     height=height,
    #                     resolution=resolution)
    #
    #     fig = _plt.figure()
    #     ax = Axes3D(fig)
    #     ax.add_collection3d(bmap.drawcoastlines())
    #     x, y = bmap(self.trajectory.longitude.values, self.trajectory.latitude.values)
    #     # ax.plot(x, y,self.trajectory.Altitude.values,
    #     #           color='m')
    #     N = len(x)
    #     for i in range(N - 1):
    #         color = _plt.cm.jet(i / N)
    #         ax.plot(x[i:i + 2], y[i:i + 2], self.trajectory.Altitude.values[i:i + 2],
    #                 color=color)
    #     return bmap, ax



class HySplitTrajectory(object):
    def __init__(self, trajectory_dict):
        #         met_model_id
        self.start_date_time = trajectory_dict['date_time_start']
        self.direction_of_traj = trajectory_dict['direction_of_traj']
        #         vertical_motion_method
        self.start_conditions = trajectory_dict['start_conditions']
        self.output_variables = trajectory_dict['output_variables']
        self.trajectory = trajectory_dict['trajectory'][['latitude', 'longitude', 'altitude_above_ground(m)'] + trajectory_dict['output_variables']]
        self.trajectory_dropped = trajectory_dict['trajectory'][['trajectory_num', 'met_grid_num', 'forcast_hour', 'age_of_trajectory(h)']]

    plot_on_map = plot_traj_on_map


class HySplitConcentration(object):
    def __init__(self, parent, matrix):
        self._parent = parent
        self.concentration = matrix
        self.qc_reports = []
        #####
        self.__source_attribution_angular = None
        # self.__source_attribution_land_use = None
        #####
        self.source_attribution_land_use = Source_Attribution_Land_use(self)

    def __len__(self):
        return 0

    plot = plot_conc_on_map
    _get_source_attribution_angular = source_attribution_angular
    _get_source_attribution_land_use = source_attribution_land_use
    save_netCDF = save_result_netCDF

    @property
    def source_attribution_angular(self):
        if not self.__source_attribution_angular:
            self.__source_attribution_angular = self._get_source_attribution_angular()
        return self.__source_attribution_angular

    # @property
    # def source_attribution_land_use(self):
    #     if not self.__source_attribution_land_use:
    #         self.__source_attribution_land_use = self._get_source_attribution_land_use()
    #     return self.__source_attribution_land_use

class Parameter(object):
    def __init__(self, parent, what):
        self._parent = parent
        self._what = what
        self._dict = self._parent._parent._settings[what]

        if 'doc' in self._dict.keys():
            self.__doc__ = self._dict['doc']

    def info(self):
        if 'doc' in self._dict.keys():
            out = self._dict['doc']
        else:
            out = None
        return out

    def __repr__(self):
        return str(self._dict['value'])

    def __bool__(self):
        return self._dict['value'].__bool__()

    def __len__(self):
        return self._dict['value'].__len__()

    def reset2default(self):
        self._dict['value'] = self._dict['default']

    def _get_value(self):
        return self._dict['value']

    def _set_value(self, value):
        if 'options' in self._dict.keys():
            if value not in self._dict['options']:
                txt = '{} is not an option for parameter {}. Choose one from {}'.format(value, self.what, self._dict['options'])
                raise ValueError(txt)
        self._dict['value'] = value

    ###############

    def __add__(self, other):
        return self._dict['value'] + other

    def __radd__(self, other):
        return self._dict['value'] + other

    def __sub__(self, other):
        return self._dict['value'] - other

    def __rsub__(self, other):
        return other - self._dict['value']

    def __mul__(self, other):
        return self._dict['value'] * other

    def __rmul__(self, other):
        return self._dict['value'] * other

    def __truediv__(self, other):
        return self._dict['value'] / other

    def __rtruediv__(self, other):
        return other / self._dict['value']

    def __getitem__(self, item):
        return self._dict['value'][item]

    def __format__(self, format_spec):
        return self._dict['value'].__format__(format_spec)

    def __abs__(self):
        return abs(self._dict['value'])

    ###############
    def copy(self):
        return deepcopy(self)

    @property
    def default_value(self):
        return self._dict['default']


class Parameters(object):

    def __init__(self, parent):
        self._parent = parent
        self.pollutants = Pollutants(self)
        self.concentration_grids = ConcentrationGrids(self)
        # self.pollutants.add_pollutant('default')
        # self.concentration_grids.add_grid('default')
        self.predefined_scenes = Scenes(parent)

    def __repr__(self):
        return all_attributes2string(self, ignore=['predefined_scenes'])

    @property
    def start_time(self):
        return Parameter(self, 'control.start_time')

    @start_time.setter
    def start_time(self, data):
        Parameter(self, 'control.start_time')._set_value(data)

    @property
    def num_starting_loc(self):
        return Parameter(self, 'control.num_starting_loc')

    @num_starting_loc.setter
    def num_starting_loc(self, data):
        Parameter(self, 'control.num_starting_loc')._set_value(data)

    @property
    def starting_loc(self):
        return Parameter(self, 'control.starting_loc')

    @starting_loc.setter
    def starting_loc(self, data):
        Parameter(self, 'control.starting_loc')._set_value(data)

    @property
    def run_time(self):
        return Parameter(self, 'control.run_time')

    @run_time.setter
    def run_time(self, data):
        old = Parameter(self, 'control.concentration_grid.sampling_interval')
        Parameter(self, 'control.concentration_grid.sampling_interval')._set_value([old[0], abs(data), old[-1]])
        Parameter(self, 'control.run_time')._set_value(data)

    @property
    def vertical_motion_option(self):
        return Parameter(self, 'control.vertical_motion_option')

    @vertical_motion_option.setter
    def vertical_motion_option(self, data):
        Parameter(self, 'control.vertical_motion_option')._set_value(data)

    @property
    def top_of_model_domain(self):
        return Parameter(self, 'control.top_of_model_domain')

    @top_of_model_domain.setter
    def top_of_model_domain(self, data):
        Parameter(self, 'control.top_of_model_domain')._set_value(data)

    @property
    def meterologic_data_format(self):
        return Parameter(self, 'control.meterologic_data_format')

    @meterologic_data_format.setter
    def meterologic_data_format(self, data):
        Parameter(self, 'control.meterologic_data_format')._set_value(data)

    @property
    def input_met_file_names(self):
        stop_time = _pd.to_datetime(self.start_time._get_value()) + _pd.to_timedelta(self.run_time._get_value(), 'H')
        list = date_str2file_name_list(self.start_time._get_value(), stop_time, self.meterologic_data_format._get_value())
        return list

    @property
    def input_met_data_folder(self):
        """This is the base folder for all met files"""
        return Parameter(self, 'control.input_met_data_folder')

    # @input_met_file_names.setter
    # def input_met_file_names(self, data):
    #     Parameter(self, 'control.input_met_file_names')._set_value(data)

    @property
    def output_path(self):
        return Parameter(self, 'control.output_path')

    @output_path.setter
    def output_path(self, data):
        Parameter(self, 'control.output_path')._set_value(data)


class Settings(object):
    @property
    def path2executable(self):
        return self.__path2compiler

    @path2executable.setter
    def path2executable(self, data):
        self.__path2compiler = data

    @property
    def path2working_directory(self):
        return self.__path2working_directory

    @path2working_directory.setter
    def path2working_directory(self, data):
        self.__path2working_directory = data

class HysplitError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


settings = {}
# settings[''] = {'default': , 'doc': ''}


#####
## control ... general
settings['control.start_time'] = {'value': '2010-01-01 00:00:00',
                                  'default': '2010-01-01 00:00:00',
                                  'doc': ''}
settings['control.num_starting_loc'] = {'value': 1,
                                        'default': 1,
                                        'doc': ''}
settings['control.starting_loc'] = {'value': [[40., -90., 10.]],
                                    'default': [[40., -90., 10.]],
                                    'doc': ''}
settings['control.run_time'] = {'value': 48,
                                'default': 48,
                                'doc': ''}
settings['control.vertical_motion_option'] = {'value': 0,
                                              'default': 0,
                                              'doc': ''}
settings['control.top_of_model_domain'] = {'value': 10000.0,
                                           'default': 10000.0,
                                           'doc': ''}
# settings['control.input_met_file_names'] = {'value': ['/Users/htelg/Hysplit4/working/oct1618.BIN'],
#                                             'default': ['/Users/htelg/Hysplit4/working/oct1618.BIN'],
#                                             'doc': ''}
settings['control.input_met_data_folder'] = {'value': '/Volumes/HTelg_4TB_Backup/hysplit_met_files/',
                                            'default': '/Volumes/HTelg_4TB_Backup/hysplit_met_files/',
                                            'doc': ''}

settings['control.meterologic_data_format'] = {'value': 'test',
                                               'default': 'test',
                                               'doc': """Meteorology / ARL Data FTP / Archive


The ARL web server contains several meteorological model data sets already converted into a HYSPLIT compatible format on the public directories. Direct access via FTP to these data files is "hardwired" into the GUI. Only an email address is required for the password to access the server. The "FTP menu" is locked until the FTP process is completed, at which point a message box will appear indicating the file name that was just transferred.

The ARL analysis data archive consists of output from the Global Data Analysis System (GDAS) and the NAM Data Analysis System (NDAS - previously called EDAS) covering much of North America. Both data archives are available from 1997 in semi-monthly files (SM). The EDAS was saved at 80-km resolution every 3-h through 2003, and then at 40-km resolution (AWIPS 212 grid) starting in 2004. The GDAS (every 6-h) was saved on a hemispheric projection at 191-km resolution through the end of 2006. The new 1-degree and half-degree GDAS archive is available every three hours. See notes below for a more detailed discussion of each file.


	1	EDAS :: 80 km 3P ( EDAS :: 40 km 3P (>=2004 SM 600 Mb) Semi-Monthly data files at three hour intervals on pressure surfaces
	2	GDAS1 :: 1-deg 3P (>=2005 WK 600 Mb) Weekly files (W1=1-7; W2=8-14; W3=15-21; W4=22-28; W5=29-end) every three hours on pressure surfaces
	3	GDAS0p5 :: 0.5-deg 3S (>=2010 DA 468 Mb) Daily files every three hours on the native GFS hybrid sigma coordinate system. Beginning July 28, 2010.
	4	NAM12 :: 12-km 3P (>=2007 DA 395 Mb) Composite archive 0 to +6 hour forecasts appended into daily files for the CONUS at three hour intervals on pressure surfaces
	5	NAMs :: 12 km 1S (>=2008 DA 994 Mb) Composite archive 0 to +6 hour forecasts appended into daily files for the CONUS at one hour intervals on sigma surfaces


From the GUI it is necessary to enter the year, month, and data file period. Semi-monthly files are designated by 001 (1st through the 15th) and 002 (16th through the end of the month). Weekly files by the week number and the two-digit day for the daily files. The file names are created automatically.

For earlier global archive data see  Meteorology - ARL Data FTP - Reanalysis.

Additional information about the data archives on the ARL server can be found at the ARL web page.  """}

settings['control.output_path'] = {'value': './tdump',
                                   'default': './tdump',
                                   'doc': ''}

#####
## Pollutants

settings['control.pollutant.emission_rate'] = {'default': 1.,
                                               'doc': ('Emission rate (per hour)\n'
                                                       'Default: 1.0\n'
                                                       'Mass units released each hour. Units are arbitrary except when specific chemical transformation subroutines are associated with the calculation. Output air concentration units will be in the same units as specified on this line. For instance an input of kg/hr results in an output of kg/m3. When multiple sources are defined this rate is assigned to all sources unless the optional parameters are present on line 3(1).')}
settings['control.pollutant.hours_of_emission'] = {'default': 1.,
                                                   'doc': ('Hours of emission\n'
                                                           'Default: 1.0\n'
                                                           'The duration of emission may be defined in fractional hours. An emission duration of less than one time-step will be emitted over one time-step with a total emission that would yield the requested rate over the emission duration.')}
#####
## Pollutants.deposition
settings['control.pollutant.deposition_particle_diameter'] = {'default': 0.0,
                                                              'doc': ('Particle: Diameter (μm), Density (g/cc), and Shape\n'
                                                                      'Default: 0.0 0.0 0.0\n,'
                                                                      'These three entries are used to define the pollutant as a particle for gravitational settling and wet removal calculations. A value of zero in any field will cause the pollutant to be treated as a gas. All three fields must be defined (>0) for particle deposition calculations. However, these values only need to be correct only if gravitational settling or resistance deposition is to be computed by the model. Otherwise a nominal value of 1.0 may be assigned as the default for each entry to define the pollutant as a particle. If a dry deposition velocity is specified as the first entry in the next line (28), then that value is used as the particle settling velocity rather than the value computed from the particle diameter and density.\n'
                                                                      'If gravitational settling is on and the Shape is set to a negative value then the Ganser (1993) calculation is used to replace Stokes equation for estimating particle fallspeeds. The absolute value of the Shape factor is used for the calculation. The Stokes equation overestimates particle fallspeeds for particles larger than about 20 micron diameter. As this diameter often lies within size distributions of volcanic ash particles, it is desirable to use the Ganser formulation so that particle fallspeeds can be computed accurately. Ganser, G.H., 1993: A rational approach to drag prediction of spherical and nonspherical particles. Powder Technology, 77, 143-152. .\n'
                                                                      'The particle definitions can be used in conjunction with a special namelist parameter NBPTYP that determines if the model will just release the above defined particles or create a continuous particle distribution using the particle definitions as fixed points within the distribution. This option is only valid if the model computes the gravitational settling velocity rather than pre-defining a velocity for each particle size.')}

settings['control.pollutant.deposition_particle_density'] = {'default': 0.0, 'doc': settings['control.pollutant.deposition_particle_diameter']['doc']}
settings['control.pollutant.deposition_particle_shape'] = {'default': 0.0, 'doc': settings['control.pollutant.deposition_particle_diameter']['doc']}

settings['control.pollutant.deposition_velocity'] = {'default': 0.0,
                                                     'doc': (
                                                     'Deposition velocity (m/s), Pollutant molecular weight (Gram/Mole), Surface Reactivity Ratio, Diffusivity Ratio, Effective Henrys Constant\n'
                                                     'Default: 0.0 0.0 0.0 0.0 0.0\n'
                                                     'Dry deposition calculations are performed in the lowest model layer based upon the relation that the deposition flux equals the velocity times the ground-level air concentration. This calculation is available for gases and particles. The dry deposition velocity can be set directly for each pollutant by entering a non-zero value in the first field. In the special case where the dry deposition velocity is set to a value less than zero, the absolute value will be used to compute gravitational settling but with no mass removal. The dry deposition velocity can also be calculated by the model using the resistance method which requires setting the remaining four parameters (molecular weight, surface reactivity, diffusivity, and the effective Henrys constant). See the table below for more information.')}
settings['control.pollutant.deposition_pollutant_molecular_weight'] = {'default': 0.0, 'doc': settings['control.pollutant.deposition_velocity']['doc']}
settings['control.pollutant.deposition_surface_reactivity_ratio'] = {'default': 0.0, 'doc': settings['control.pollutant.deposition_velocity']['doc']}
settings['control.pollutant.deposition_diffusivity_ratio'] = {'default': 0.0, 'doc': settings['control.pollutant.deposition_velocity']['doc']}
settings['control.pollutant.deposition_effective_henrys_constant'] = {'default': 0.0, 'doc': settings['control.pollutant.deposition_velocity']['doc']}

settings['control.pollutant.wet_removal_henrys_constant'] = {'default': 0.0, 'doc': ('Wet Removal: Actual Henrys constant, In-cloud (GT 1 =L/L; LT 1 =1/s), Below-cloud (1/s)\n'
                                                                                     'Default: 0.0 0.0 0.0\n'
                                                                                     'Suggested: 0.0 8.0E-05 8.0E-05\n'
                                                                                     'Henrys constant defines the wet removal process for soluble gases. It is defined only as a first-order process by a non-zero value in the field. Wet removal of particles is defined by non-zero values for the in-cloud and below-cloud parameters. In-cloud removal can be defined as a ratio of the pollutant in rain (g/liter) measured at the ground to that in air (g/liter of air in the cloud layer) when the value in the field is greater than one. For within-cloud values less than one, the removal is defined as a time constant. Below-cloud removal is always defined through a removal time constant. The default cloud bottom and top RH values can be changed through the SETUP.CFG namelist file. Wet removal only occurs in grid cells with both a non-zero precipitation value and a defined cloud layer.')}
settings['control.pollutant.wet_removal_in_cloud'] = {'default': 0.0, 'doc': settings['control.pollutant.wet_removal_henrys_constant']['doc']}
settings['control.pollutant.wet_removal_below_cloud'] = {'default': 0.0, 'doc': settings['control.pollutant.wet_removal_henrys_constant']['doc']}

settings['control.pollutant.radioactive_decay'] = {'default': 0.0,
                                                   'doc': ('Radioactive decay half-life (days)\n'
                                                           'Default: 0.0\n'
                                                           'A non-zero value in this field initiates the decay process of both airborne and deposited pollutants. The particle mass decays as well as the deposition that has been accumulated on the internal sampling grid. The deposition array (but not air concentration) is decayed until the values are written to the output file. Therefore, the decay is applied only the the end of each output interval. Once the values are written to the output file, the values are fixed. The default is to decay deposited material. This can be turned off so that decay only occurs to the particle mass while airborne by setting the decay namelist variable to zero.')}
settings['control.pollutant.pollutant_resuspension'] = {'default': 0.0, 'doc': ('Pollutant Resuspension (1/m)\n'
                                                                                'Default: 0.0\n'
                                                                                'Suggested: 1.0E-06'
                                                                                'A non-zero value for the re-suspension factor causes deposited pollutants to be re-emitted based upon soil conditions, wind velocity, and particle type. Pollutant re-suspension requires the definition of a deposition grid, as the pollutant is re-emitted from previously deposited material. Under most circumstances, the deposition should be accumulated on the grid for the entire duration of the simulation. Note that the air concentration and deposition grids may be defined at different temporal and spatial scales.')}

######
## Concentration Grid
# setting['control.concentration_grid.center_lat_lon'] = {default': , 'doc': ''}
settings['control.concentration_grid.spacing'] = {'default': [1., 1.],
                                                  'doc': ('Grid spacing (degrees) Latitude, Longitude\n'
                                                          'Default: 1.0 1.0\n'
                                                          'Sets the interval in degrees between nodes of the sampling grid. Puffs must pass over a node to contribute concentration to that point and therefore if the spacing is too wide, they may pass between intersection points. Particle model calculations represent grid-cell averages, where each cell is centered on a node position, with its dimensions equal to the grid spacing. Finer resolution concentration grids require correspondingly finer integration time-steps. This may be mitigated to some extent by limiting fine resolution grids to only the first few hours of the simulation.\n'
                                                          'In the special case of a polar (arc,distance) concentration grid, defined when the namelist variable cpack=3, the definition changes such that the latitude grid spacing equals the sector angle in degrees and the longitude grid spacing equals the sector distance spacing in kilometers.')}

settings['control.concentration_grid.span'] = {'default': [180., 360.],
                                               'doc': ('Grid span (deg) Latitude, Longitude\n'
                                                       'Default: [180.0] [360.0]\n'
                                                       'Sets the total span of the grid in each direction. For instance, a span of 10 degrees would cover 5 degrees on each side of the center grid location. A plume that goes off the grid would have cutoff appearance, which can sometimes be mitigated by moving the grid center further downwind.\n'
                                                       'In the special case of a polar (arc,distance) concentration grid, defined when the namelist variable cpack=3, the definition changes such that the latitude span always equals 360.0 degrees and the longitude span equals the total downwind distance in kilometers. Note that the number of grid points equals 360/arc-angle or the total-distance divided by the sector-distance.')}

settings['control.concentration_grid.output_path'] = {'default': './cdump', 'doc': 'path to file'}

settings['control.concentration_grid.vertical_concentration_levels_number'] = {'default': 1,
                                                                               'doc': ('Number of vertical concentration levels\n'
                                                                                       'Default: 1\n'
                                                                                       'The number of vertical levels in the concentration grid including the ground surface level if deposition output is required.')}

settings['control.concentration_grid.vertical_concentration_levels_height_of_each'] = {'default': 50,
                                                                                       'doc': ('Height of each level (m)\n'
                                                                                               'Default: 50\n'
                                                                                               '￼￼￼￼￼￼￼￼Output grid levels may be defined in any order for the puff model as long as the deposition level (0) comes first (aheight of zero indicates deposition output). Air concentrations must have a non-zero height defined. A height for the puff model indicates the concentration at that level. A height for the particle model indicates the average concentration between that level and the previous level (or the ground for the first level). Therefore heights for the particle model need to be defined in ascending order. Note that the default is to treat the levels as above-ground-level (AGL) unless the MSL (above Mean-Sea-Level) flag has been set (see advanced configuration).')}
settings['control.concentration_grid.sampling_stop_time'] = {'default': '12 31 24 60',
                                                             'doc': ('Sampling stop time: year month day hour minute\n'
                                                                     'Default: 12 31 24 60\n'
                                                                     'After this time no more concentration records are written. Early termination on a high resolution grid (after the plume has moved away from the source) is an effective way of speeding up the computation for high resolution output near the source because once turned-off that particular grid resolution is no longer used for time-step computations.')}
settings['control.concentration_grid.sampling_interval'] = {'value': [0,24, 0],
                                                            'default': [0,24, 0],
                                                            'doc': ('Sampling interval: type hour minute\n'
                                                                    'Default: 0 24 0\n'
                                                                    'Each grid may have its own sampling or averaging interval. The interval can be of three different types: averaging (type=0), snapshot (type=1), or maximum (type=2). Averaging will produce output averaged over the specified interval. For instance, you may want to define a concentration grid that produces 24-hour average air concentrations for the duration of the simulation, which in the case of a 2-day simulation will result in 2 output maps, one for each day. Each defined grid can have a different output type and interval. Snapshot (or now) will give the instantaneous output at the output interval, and maximum will save the maximum concentration at each grid point over the duration of the output interval. Therefore, when a maximum concentration grid is defined, it is also required to define an identical snapshot or average grid over which the maximum will be computed. There is also the special case when the type value is less than zero. In that case the value represents the averaging time in hours and the output interval time represents the interval at which the average concentration is output. For instance, a setting of {-1 6 0} would output a one-hour average concentration every six hours.')}


def all_attributes2string(obj, ignore = []):
    att_list = []
    max_length = 0
    # print('before loop')
    # print('dir', dir(obj))
    # print('===========')
    for i in dir(obj):
        if i[0] == '_':
            continue
        # print(i)
        if i in ignore:
            continue
        if len(i) > max_length:
            max_length = len(i)

    for i in dir(obj):
        if i[0] == '_':
            continue
        if i in ignore:
            continue
        # no_char_full = 100
        att_list.append('{i:<{max_len}}:  {value}'.format(i=i, max_len = max_length + 1, value=getattr(obj, i)))
    out = '\n'.join(att_list)
    return out


class ConcentrationGrid(object):
    """ To create test scenario:
    run.parameters.run_time = 12
run.parameters.concentration_grids.default.spacing = [0.05, 0.05]
run.parameters.concentration_grids.default.sampling_interval = [0, 12, 0]
run.parameters.concentration_grids.default.span = [30., 30.]
run.parameters.concentration_grids.default.vertical_concentration_levels_height_of_each = 100
run.parameters.concentration_grids.default"""
    def __repr__(self):
        return all_attributes2string(self)

    def __init__(self, parent, name):
        self._name = name
        self._parent = parent
        self.center = self._parent._parent.starting_loc[0][:-1]  # starting location of first trajectory, not sure if you can run multiple location at once ... then this would have to be fixed (todo)!
        self.spacing = settings['control.concentration_grid.spacing']['default']
        self.span = settings['control.concentration_grid.span']['default']
        # self.output_path = settings['control.concentration_grid.output_path']['default']
        self.vertical_concentration_levels_number = settings['control.concentration_grid.vertical_concentration_levels_number']['default']
        self.vertical_concentration_levels_height_of_each = settings['control.concentration_grid.vertical_concentration_levels_height_of_each']['default']
        # self.sampling_start_time = self._parent._parent.start_time
        # self.sampling_stop_time = settings['control.concentration_grid.sampling_stop_time']['default']

    @property
    def output_path(self):
        return self._parent._parent.output_path

    @property
    def sampling_start_time(self):
        return self._parent._parent.start_time

    @property
    def sampling_stop_time(self):
        st_dt = _pd.to_datetime(self._parent._parent.start_time._get_value()) + _pd.to_timedelta(self._parent._parent.run_time._get_value(), 'H')
        return '{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(st_dt.year, st_dt.month, st_dt.day, st_dt.hour, st_dt.minute, st_dt.second)

    @property
    def sampling_interval(self):
        return Parameter(self._parent._parent, 'control.concentration_grid.sampling_interval')
        # self.sampling_interval = [0, self._parent._parent.run_time, 0] #settings['control.concentration_grid.sampling_interval']['default']

    @sampling_interval.setter
    def sampling_interval(self, data):
        Parameter(self._parent._parent, 'control.concentration_grid.sampling_interval')._set_value(data)

class ConcentrationGrids(object):
    def __init__(self, parent):
        self._grid_dict = {}
        self._parent = parent

    def __repr__(self):
        out = "\n==========\n"
        if len(self._grid_dict.keys()) == 0:
            out = 'no grids defined'
        else:
            for k in self._grid_dict.keys():
                out += k + '\n'
                out += "--------\n"
                out += self._grid_dict[k].__repr__()
        out += "\n==========\n"
        return out

    def __len__(self):
        out = len(self._grid_dict.keys())
        return out

    def __iter__(self):
        out = []
        for i in self._grid_dict.items():
            out.append(i[1])
        return iter(out)

    def add_grid(self, name):
        if name in self._grid_dict.keys():
            txt = 'Grid with name "{}" already exists ... pick other name!'
            raise KeyError(txt)
        grid = ConcentrationGrid(self, name)
        setattr(self, name, grid)
        self._grid_dict[name] = grid


class Pollutant(object):
    def __repr__(self):
        out =  all_attributes2string(self, ignore= ['remove_pollutant'])
        return out

    def __init__(self, parent, name):
        self._parent = parent
        self._name = name
        self.emission_rate = settings['control.pollutant.emission_rate']['default']
        self.hours_of_emission =  settings['control.pollutant.hours_of_emission']['default'] #self._parent._parent.run_time
        self.release_start_time = self._parent._parent.start_time

        self.deposition_particle_diameter = settings['control.pollutant.deposition_particle_diameter']['default']
        self.deposition_particle_density = settings['control.pollutant.deposition_particle_density']['default']
        self.deposition_particle_shape = settings['control.pollutant.deposition_particle_shape']['default']

        self.deposition_velocity = settings['control.pollutant.deposition_velocity']['default']
        self.deposition_pollutant_molecular_weight = settings['control.pollutant.deposition_pollutant_molecular_weight']['default']
        self.deposition_surface_reactivity_ratio = settings['control.pollutant.deposition_surface_reactivity_ratio']['default']
        self.deposition_diffusivity_ratio = settings['control.pollutant.deposition_diffusivity_ratio']['default']
        self.deposition_effective_henrys_constant = settings['control.pollutant.deposition_effective_henrys_constant']['default']

        self.wet_removal_henrys_constant = settings['control.pollutant.wet_removal_henrys_constant']['default']
        self.wet_removal_in_cloud = settings['control.pollutant.wet_removal_in_cloud']['default']
        self.wet_removal_below_cloud = settings['control.pollutant.wet_removal_below_cloud']['default']

        self.radioactive_decay = settings['control.pollutant.radioactive_decay']['default']
        self.pollutant_resuspension = settings['control.pollutant.pollutant_resuspension']['default']


    def remove_pollutant(self):
        pass
        # self._parent._pollutant_dict.pop(self._name)
        # delattr(self._parent, self._name)


class Pollutants(object):
    def __init__(self, parent):
        self._pollutant_dict = {}
        self._parent = parent

    def __repr__(self):
        out = "\n==========\n"
        if len(self._pollutant_dict.keys()) == 0:
            out += 'no pollutants defined'
        else:
            for k in self._pollutant_dict.keys():
                out += k + '\n'
                out += "----------\n"
                out += self._pollutant_dict[k].__repr__()
        out += "\n==========\n"
        return out

    def __len__(self):
        out = len(self._pollutant_dict.keys())
        return out

    def __iter__(self):
        out = []
        for i in self._pollutant_dict.items():
            out.append(i[1])
        return iter(out)

    def add_pollutant_gas(self, name):
        if name in self._pollutant_dict.keys():
            txt = 'Pollutant with name "{}" already exists ... pick other name!'
            raise KeyError(txt)
        pollutant = Pollutant(self, name)
        setattr(self, name, pollutant)
        self._pollutant_dict[name] = pollutant

    def add_pollutant_particle(self, name, mode = 'accumulation', deposition = 'dry'):
        if name in self._pollutant_dict.keys():
            txt = 'Pollutant with name "{}" already exists ... pick other name!'
            raise KeyError(txt)
        pollutant = Pollutant(self, name)
        setattr(self, name, pollutant)
        self._pollutant_dict[name] = pollutant
        if mode == 'accumulation':
            pollutant.deposition_particle_diameter = 0.2
            pollutant.deposition_particle_density = 1.8
            pollutant.deposition_particle_shape = 1.
        elif mode == 'coarse':
            pollutant.deposition_particle_diameter = 2.5
            pollutant.deposition_particle_density = 2.6 # density of SiO2
            pollutant.deposition_particle_shape = 1.

        if deposition == 'wet':
            pollutant.wet_removal_in_cloud = 8.0e-5
            pollutant.wet_removal_below_cloud = 8.0e-5

        elif deposition != 'dry':
            txt = 'sorry, depositions other then dry are not implemented yet'
            raise ValueError(txt)

class Scenes(object):
    def __init__(self, run_instance):
        self.run = run_instance

    def hysplit_example(self):
        """this is the default hysplit example for concentration simulations"""
        self.run.parameters.pollutants.add_pollutant_gas('default')
        self.run.parameters.concentration_grids.add_grid('default')

        self.run.parameters.start_time = '1995-10-16 00:00:00'
        self.run.parameters.starting_loc = [[40.0, -90.0, 10.0]]
        self.run.parameters.run_time = 12
        self.run.parameters.vertical_motion_option = 0
        self.run.parameters.top_of_model_domain = 10000.0
        # self.run.parameters.input_met_file_names = ['/Users/htelg/Hysplit4/working/oct1618.BIN']
        # self.run.parameters.concentration_grids.default.sampling_interval = [0, 12, 0]
        self.run.parameters.concentration_grids.default.spacing = [0.05, 0.05]
        self.run.parameters.concentration_grids.default.span = [30., 30.]
        self.run.parameters.concentration_grids.default.vertical_concentration_levels_height_of_each = 100
        self.run.parameters.pollutants.default.deposition_diffusivity_ratio = 0.0
        self.run.parameters.pollutants.default.deposition_effective_henrys_constant = 0.0
        self.run.parameters.pollutants.default.deposition_particle_density = 0.0
        self.run.parameters.pollutants.default.deposition_particle_diameter = 0.0
        self.run.parameters.pollutants.default.deposition_particle_shape = 0.0
        self.run.parameters.pollutants.default.deposition_pollutant_molecular_weight = 0.0
        self.run.parameters.pollutants.default.deposition_surface_reactivity_ratio = 0.0
        self.run.parameters.pollutants.default.deposition_velocity = 0.0
        self.run.parameters.pollutants.default.emission_rate = 1.
        self.run.parameters.pollutants.default.hours_of_emission = 1.0
        self.run.parameters.pollutants.default.pollutant_resuspension = 0.0
        self.run.parameters.pollutants.default.radioactive_decay = 0.0
        # self.run.parameters.pollutants.default.release_start_time = "00 00 00 00"
        self.run.parameters.pollutants.default.wet_removal_below_cloud = 0.0
        self.run.parameters.pollutants.default.wet_removal_henrys_constant = 0.0
        self.run.parameters.pollutants.default.wet_removal_in_cloud = 0.0

    def sgp_aerosol_accu_backwards_gdas1(self):
        """ARM's SGP central facility, concentration for accumulation mode particles including wet deposition"""
        self.run.parameters.meterologic_data_format = 'gdas1'
        self.run.parameters.pollutants.add_pollutant_gas('default')
        self.run.parameters.concentration_grids.add_grid('default')

        self.run.parameters.start_time = '2012-01-01 00:00:00'
        self.run.parameters.starting_loc = [[36.605, -97.485, 10.0]]
        self.run.parameters.run_time = -12
        self.run.parameters.vertical_motion_option = 0
        self.run.parameters.top_of_model_domain = 10000.0
        # self.run.parameters.input_met_file_names = ['/Users/htelg/Hysplit4/working/oct1618.BIN']
        # self.run.parameters.concentration_grids.default.sampling_interval = [0, 12, 0]
        self.run.parameters.concentration_grids.default.spacing = [0.05, 0.05]
        self.run.parameters.concentration_grids.default.span = [60., 120.]
        self.run.parameters.concentration_grids.default.vertical_concentration_levels_height_of_each = 100
        self.run.parameters.pollutants.default.deposition_diffusivity_ratio = 0.0
        self.run.parameters.pollutants.default.deposition_effective_henrys_constant = 0.0
        self.run.parameters.pollutants.default.deposition_particle_density = 1.8
        self.run.parameters.pollutants.default.deposition_particle_diameter = 0.29 #center of accumulation mode in Volume moment
        self.run.parameters.pollutants.default.deposition_particle_shape = 1
        self.run.parameters.pollutants.default.deposition_pollutant_molecular_weight = 0.0
        self.run.parameters.pollutants.default.deposition_surface_reactivity_ratio = 0.0
        self.run.parameters.pollutants.default.deposition_velocity = 0.0
        self.run.parameters.pollutants.default.emission_rate = 1.
        self.run.parameters.pollutants.default.hours_of_emission = 1.0
        self.run.parameters.pollutants.default.pollutant_resuspension = 0.0
        self.run.parameters.pollutants.default.radioactive_decay = 0.0
        # self.run.parameters.pollutants.default.release_start_time = "00 00 00 00"
        self.run.parameters.pollutants.default.wet_removal_below_cloud = 8.0e-05
        self.run.parameters.pollutants.default.wet_removal_henrys_constant = 0.0
        self.run.parameters.pollutants.default.wet_removal_in_cloud = 8.0e-05

class Run(object):
    def __init__(self,
                 hysplit_mode,
                 start_time='00 00 00 00',
                 num_starting_loc=1,
                 starting_loc=[[40., -90., 10.]],
                 run_time=48,
                 vertical_motion_option=0,
                 top_of_model_domain=10000.0,
                 input_met_file_names=['/Users/htelg/Hysplit4/working/oct1618.BIN'],
                 output_path='./tdump', ):

        """
        This class sets up a Hysplit run

        Parameters
        ----------
        hysplit_mode: str ["trajectory", "concentration"]
            If HYSPLIT is calculating trajectories or concentrations
        start_time: string
            Starting time (year, month, day, hour, {minutes optional})
            Default: 00 00 00 00 {00}
            Enter the two digit values for the UTC time that the calculation is to
            start. Use 0's to start at the beginning (or end) of the file according to
            the direction of the calculation. All zero values in this field will force
            the calculation to use the time of the first (or last) record of the
            meteorological data file. In the special case where year and month are
            zero, day and hour are treated as relative to the start or end of the file.
            For example, the first record of the meteorological data file usually
            starts at 0000 UTC. An entry of "00 00 01 12" would start the calculation
            36 hours from the start of the data file. The minutes field is optional but
            should only be used when the start time is explicitly set to a value.
        num_starting_loc: int
            Number of starting locations
            Default: 1
            Simultaneous trajectories can be calculated at multiple levels or starting
            locations. Specification of additional locations for certain types of
            simulations can also be accomplished through the Special Simulations menu
            tab, or through manual editing of the CONTROL file and running the model
            outside of the GUI. When multiple starting locations are specified, all
            trajectories start at the same time. A multiple trajectory in time option
            is available through the Advanced menu through a namelist file parameter
            setting.
        starting_loc: nested list of shape(num_starting_loc, 3)
            Enter starting location (lat, lon, meters)
            Default: 40.0 -90.0 50.0
            Trajectory starting position in degrees and decimal (West and South are
            negative). Height is entered as meters above ground-level. An option to
            treat starting heights as relative to mean-sea-level is available through
            the Advanced menu through a namelist file parameter setting.
        run_time: int
            Total run time (hours)
            Default: 48
            Specifies the duration of the calculation in hours. Backward calculations
            are entered as negative hours. A backward trajectory starts from the
            trajectory termination point and proceeds upwind. Meteorological data are
            processed in reverse-time order.
        vertical_motion_option: int
            Vertical motion option (0:data 1:isob 2:isen 3:dens 4:sigma 5:diverg
                6:msl2agl 7:average)
            Default: 0
            Indicates the vertical motion calculation method. The default "data"
            selection will use the meteorological model's vertical velocity fields;
            other options include {isob}aric, {isen}tropic, constant {dens}ity,
            constant internal {sigma} coordinate, computed from the velocity
            {diverg}ence, vertical coordinate remapping from MSL to AGL, and a special
            option (7) to spatially average the vertical velocity. The averaging
            distance is automatically computed from the ratio of the temporal frequency
            of the data to the horizontal grid resolution.
        top_of_model_domain: float
            Top of model domain (internal coordinates m-agl)
            Default: 10000.0
            Sets the vertical limit of the internal meteorological grid. If
            calculations are not required above a certain level, fewer meteorological
            data are processed thus speeding up the computation. Trajectories will
            terminate when they reach this level. A secondary use of this parameter is
            to set the model's internal scaling height - the height at which the
            internal sigma surfaces go flat relative to terrain. The default internal
            scaling height is set to 25 km but it is set to the top of the model domain
            if the entry exceeds 25 km. Further, when meteorological data are provided
            on terrain sigma surfaces it is assumed that the input data were scaled to
            a height of 20 km (RAMS) or 34.8 km (COAMPS). If a different height is
            required to decode the input data, it should be entered on this line as the
            negative of the height. HYSPLIT's internal scaling height remains at 25 km
            unless the absolute value of the domain top exceeds 25 km.
        input_met_file_names: list of str
            Meteorological data grid # 1 file name
            Default: file_name
            Name of the file containing meteorological data. Located in the previous directory.
        ￼￼￼￼output_traj_file_name: str
            Name of the trajectory endpoints file
            ￼Default: file_name
            The trajectory end-points output file is named in this entry line.

        Deprecated
        ----------
        num_met_files: int
            deprecated: will be determined from input_met_file_names
            Number of input data grids
            Default: 1
            Number of simultaneous input meteorological files. The following two entries (directory and name) will be repeated this number of times. A simulation will terminate when the computation is off all of the grids in either space or time. Trajectory calculations will check the grid each time step and use the finest resolution input data available at that location at that time. When multiple meteorological grids have different resolution, there is an additional restriction that there should be some overlap between the grids in time, otherwise it is not possible to transfer a trajectory position from one grid to another. If multiple grids are defined and the model has trouble automatically transferring the calculation from one grid to another, the sub-grid size may need to be increased to its maximum value.
        input_met_file_folder: str
            deprecated: will be determined from input_met_file_names
            Meteorological data grid # 1 directory
            Default: ( \main\sub\data\ )
            Directory location of the meteorological file on the grid specified. Always terminate with the appropriate slash (\ or /).
        output_traj_file_folder: str
            deprecated: folder is taken from filename path
            Directory of trajectory output file
            Default: ( \main\trajectory\output\ )
            Directory location to which the text trajectory end-points file will be written. Always terminate with the appropriate slash (\ or /).
        """
        self._settings = settings.copy()
        self.hysplit_mode = hysplit_mode
        self.parameters = Parameters(self)


        ######
        # self.parameters.start_time = start_time
        # self.parameters.num_starting_loc = num_starting_loc
        # self.parameters.starting_loc = starting_loc
        # self.parameters.run_time = run_time
        # self.parameters.vertical_motion_option = vertical_motion_option
        # self.parameters.top_of_model_domain = top_of_model_domain
        # self.parameters.input_met_file_names = input_met_file_names
        # self.parameters.output_path = output_path
        #######

        self.settings = Settings()
        hysplit_mode_options = ['trajectory', 'concentration']
        if self.hysplit_mode == 'trajectory':
            self.settings.path2executable = '/Users/htelg/Hysplit4/working/hyts_std'
        elif self.hysplit_mode == 'concentration':
            self.settings.path2executable = '/Users/htelg/Hysplit4/working/hycs_std'
            self.parameters.output_path = './cdump'
        else:
            txt = '{} is not an option for the hysplit_mode input parameter. Please choose one from {}'.format(self.hysplit_mode, str(hysplit_mode_options))
            raise ValueError(txt)

        self.settings.path2working_directory = '/Users/htelg/Hysplit4/working/'
        self.settings.project_directory = os.getcwd()

    def copy(self):
        return copy.deepcopy(self)

    def run_test(self, verbose = True):
        test_result = True
        all_found = True
        txtl = ['Met files available:']
        missing_files = []
        for fn in self.parameters.input_met_file_names:
            file = '{}{}/{}'.format(self.parameters.input_met_data_folder, self.parameters.meterologic_data_format, fn)
            found = os.path.isfile(file)
            if not found:
                all_found = False
                test_result = False
                missing_files.append(fn)
            txtl.append('{} ..... {}'.format(file, found))

        txt = 'All met files present: %s' % all_found
        txtl = [txt] + txtl

        txt = 'Test result: %s' % test_result
        txtl = [txt] + txtl
        test_text = '\n'.join(txtl)
        if verbose:
            print(test_text)
        return test_result, test_text, missing_files


    def download_missing_meterologic_files(run, max_files = 10, verbose=False):
        missing_files = run.run_test(verbose=False)[2]
        if verbose:
            print(missing_files)

        if len(missing_files) == 0:
            if verbose:
                print('no missing files')
            return

        if len(missing_files) > max_files:
            txt = 'the number of files needed to be downloaded exeeds the maximum set by the argument max_files. (#: {}, max_files: {}).'.format(len(missing_files), max_files)
            raise ValueError(txt)

        # Open ftp connection and change to particular folder
        server = 'arlftp.arlhq.noaa.gov'
        user = 'anonymous'
        email = 'hagen.telg@noaa.gov'

        ftp = ftplib.FTP(server, user, email)
        base_dir = 'pub/archives/'
        ftp_folder = '{}{}'.format(base_dir, run.parameters.meterologic_data_format)
        ftp.cwd(ftp_folder)

        listing = []
        ftp.retrlines("LIST ./", listing.append)
        listing = [i.split()[-1] for i in listing]

        #     return ftp

        # download missing files
        for fn in missing_files:
            # destination where to save file
            if fn not in listing:
                ftp.close()
                txt = "{} not among available files: \n {}".format(fn, '\n'.join(listing))
                raise ValueError(txt)

            save2fname = '{}{}/{}'.format(run.parameters.input_met_data_folder, run.parameters.meterologic_data_format, fn)
            dest_file = open(save2fname, 'wb')

            # download and write to file
            ftp.retrbinary('RETR ' + fn, dest_file.write)

            dest_file.close()
        ftp.close()
        return

    def _create_control_file(self):
        raus = open(self.settings.path2working_directory + 'CONTROL', 'w')
        raus.write(datetime_str2hysplittime(self.parameters.start_time._get_value()) + '\n')  # 1
        raus.write('{}\n'.format(self.parameters.num_starting_loc))  # 2
        raus.write('\n'.join(['{:0.2f} {:0.2f} {:0.1f}'.format(*i) for i in self.parameters.starting_loc]) + '\n')  # 3
        raus.write('{:d}\n'.format(self.parameters.run_time))  # 4
        raus.write('{:d}\n'.format(self.parameters.vertical_motion_option))  # 5
        raus.write('{:0.1f}\n'.format(self.parameters.top_of_model_domain))  # 6
        raus.write('{:d}\n'.format(len(self.parameters.input_met_file_names)))  # 7
        for fn in self.parameters.input_met_file_names:
            # folder, file = os.path.split(fn)
            raus.write('{}{}/\n'.format(self.parameters.input_met_data_folder, self.parameters.meterologic_data_format))  # 8
            # raus.write(folder + '/' + '\n')  # 8
            raus.write(fn + '\n')  # 9
        if self.hysplit_mode == 'concentration':

            # Pollutants
            raus.write('{}\n'.format(len(self.parameters.pollutants)))
            for pol in self.parameters.pollutants:
                raus.write(pol._name + '\n')
                raus.write('{}\n'.format(pol.emission_rate))
                raus.write('{}\n'.format(pol.hours_of_emission))                            # line 13 in control
                raus.write('{}\n'.format(datetime_str2hysplittime(pol.release_start_time._get_value()))) # line 14 in control

            # Grids
            raus.write('{}\n'.format(len(self.parameters.concentration_grids)))             # line 15 in control
            for grid in self.parameters.concentration_grids:
                raus.write('{} {}\n'.format(*grid.center))                                  # line 16 in control
                raus.write('{} {}\n'.format(*grid.spacing))                                 # line 17 in control
                raus.write('{} {}\n'.format(*grid.span))                                    # line 18 in control
                folder, file = os.path.split(str(grid.output_path))
                raus.write('{}/\n'.format(folder))                                          # line 19 in control
                raus.write('{}\n'.format(file))                                             # line 20 in control
                raus.write('{}\n'.format(grid.vertical_concentration_levels_number))        # line 21 in control
                raus.write('{}\n'.format(grid.vertical_concentration_levels_height_of_each))# line 22 in control
                raus.write('{}\n'.format(datetime_str2hysplittime(grid.sampling_start_time._get_value())))                         # line 23 in control
                raus.write('{}\n'.format(datetime_str2hysplittime(grid.sampling_stop_time)))                          # line 24 in control
                raus.write('{} {} {}\n'.format(*grid.sampling_interval))                    # line 25 in control

            # Deposition ... has to be one for each pollutant, thats why I put them together
            raus.write('{}\n'.format(len(self.parameters.pollutants)))
            for pol in self.parameters.pollutants:
                raus.write('{} '.format(pol.deposition_particle_diameter))
                raus.write('{} '.format(pol.deposition_particle_density))
                raus.write('{}\n'.format(pol.deposition_particle_shape))

                raus.write('{} '.format(pol.deposition_velocity))
                raus.write('{} '.format(pol.deposition_pollutant_molecular_weight))
                raus.write('{} '.format(pol.deposition_surface_reactivity_ratio))
                raus.write('{} '.format(pol.deposition_diffusivity_ratio))
                raus.write('{}\n'.format(pol.deposition_effective_henrys_constant))

                raus.write('{} '.format(pol.wet_removal_henrys_constant))
                raus.write('{} '.format(pol.wet_removal_in_cloud))
                raus.write('{}\n'.format(pol.wet_removal_below_cloud))

                raus.write('{}\n'.format(pol.radioactive_decay))
                raus.write('{}\n'.format(pol.pollutant_resuspension))
        elif self.hysplit_mode == 'trajectory':
            folder, file = os.path.split(self.parameters.output_path)
            raus.write(folder + '/' + '\n')
            raus.write(file + '\n')  # 11
        else:
            raise ValueError()

        raus.close()

    # def _run_hysplit_conc(self):



    def _run_hysplit_traj(self, verbose=False):
        os.chdir(self.settings.path2working_directory)
        try:
            process = subprocess.check_output(self.settings.path2executable, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            if verbose:
                print('======')
                print(e.output.decode())
                print('======')
                print(e.returncode)
                print('======')
                print(dir(e))
            txt = 'An error occured when trying to run HYSPLIT.'
            if e.returncode == 24:
                txt += '\nError code is 24, this might mean that the control file was not found.'

            txt += '\n' + e.output.decode()
            # raise subprocess.CalledProcessError(txt, process)
            raise ValueError(txt)

        self.result_stdout = process.decode()

        if self.result_stdout.split('\n')[-3] == ' Percent complete: 100.0':
            if verbose:
                print(self.result_stdout)
            return True

        elif 'ERROR' in self.result_stdout.split('\n')[-3]:
            txt = self.result_stdout.split('\n')[-3]
            if 'ASCDATA.CFG file not found' in self.result_stdout.split('\n')[-3]:
                txt = 'ASCDATA.CFG not found. This usually means the folder to the path2working_directory is wrong.\n Hysplit return:\n' + txt

            raise HysplitError(txt)

    def run(self, verbose=False):
        test = self.run_test(verbose = False)
        if not test[0]:
            txt = 'Test run failed. Make sure the test_run goes through before etempting a real run.'
            raise ValueError(txt)

        self._create_control_file()
        self._run_hysplit_traj(verbose=verbose)
        if self.hysplit_mode == 'trajectory':
            result = read_hysplit_traj_output_file()
        elif self.hysplit_mode == 'concentration':
            result = read_hysplit_conc_output_file()
            result = HySplitConcentration(self, result)

        if len(result) == 1:
            result = result[0]
        self.result = result
        self.result.run_settings = self.parameters.__repr__()

class Trajectory_project_notuesedyet(object):
    def __init__(self,
                 start_time='00 00 00 00',
                 num_starting_loc=1,
                 starting_loc=[[40., -90., 10.]],
                 run_time=48,
                 vertical_motion_option=0,
                 top_of_model_domain=10000.0,
                 input_met_file_names=['/Users/htelg/Hysplit4/working/oct1618.BIN'],
                 output_traj_file_name='./tdump', ):

        """
        This class sets up a Hysplit run

        Parameters
        ----------
        start_time: string
            Starting time (year, month, day, hour, {minutes optional})
            Default: 00 00 00 00 {00}
            Enter the two digit values for the UTC time that the calculation is to start. Use 0's to start at the beginning (or end) of the file according to the direction of the calculation. All zero values in this field will force the calculation to use the time of the first (or last) record of the meteorological data file. In the special case where year and month are zero, day and hour are treated as relative to the start or end of the file. For example, the first record of the meteorological data file usually starts at 0000 UTC. An entry of "00 00 01 12" would start the calculation 36 hours from the start of the data file. The minutes field is optional but should only be used when the start time is explicitly set to a value.
        num_starting_loc: int
            Number of starting locations
            Default: 1
            Simultaneous trajectories can be calculated at multiple levels or starting locations. Specification of additional locations for certain types of simulations can also be accomplished through the Special Simulations menu tab, or through manual editing of the CONTROL file and running the model outside of the GUI. When multiple starting locations are specified, all trajectories start at the same time. A multiple trajectory in time option is available through the Advanced menu through a namelist file parameter setting.
        starting_loc: nested list of shape(num_starting_loc, 3)
            Enter starting location (lat, lon, meters)
            Default: 40.0 -90.0 50.0
            Trajectory starting position in degrees and decimal (West and South are negative). Height is entered as meters above ground-level. An option to treat starting heights as relative to mean-sea-level is available through the Advanced menu through a namelist file parameter setting.
        run_time: int
            Total run time (hours)
            Default: 48
            Specifies the duration of the calculation in hours. Backward calculations are entered as negative hours. A backward trajectory starts from the trajectory termination point and proceeds upwind. Meteorological data are processed in reverse-time order.
        vertical_motion_option: int
            Vertical motion option (0:data 1:isob 2:isen 3:dens 4:sigma 5:diverg 6:msl2agl 7:average)
            Default: 0
            Indicates the vertical motion calculation method. The default "data" selection will use the meteorological model's vertical velocity fields; other options include {isob}aric, {isen}tropic, constant {dens}ity, constant internal {sigma} coordinate, computed from the velocity {diverg}ence, vertical coordinate remapping from MSL to AGL, and a special option (7) to spatially average the vertical velocity. The averaging distance is automatically computed from the ratio of the temporal frequency of the data to the horizontal grid resolution.
        top_of_model_domain: float
            Top of model domain (internal coordinates m-agl)
            Default: 10000.0
            Sets the vertical limit of the internal meteorological grid. If calculations are not required above a certain level, fewer meteorological data are processed thus speeding up the computation. Trajectories will terminate when they reach this level. A secondary use of this parameter is to set the model's internal scaling height - the height at which the internal sigma surfaces go flat relative to terrain. The default internal scaling height is set to 25 km but it is set to the top of the model domain if the entry exceeds 25 km. Further, when meteorological data are provided on terrain sigma surfaces it is assumed that the input data were scaled to a height of 20 km (RAMS) or 34.8 km (COAMPS). If a different height is required to decode the input data, it should be entered on this line as the negative of the height. HYSPLIT's internal scaling height remains at 25 km unless the absolute value of the domain top exceeds 25 km.
        input_met_file_names: list of str
            Meteorological data grid # 1 file name
            Default: file_name
            Name of the file containing meteorological data. Located in the previous directory.
        ￼￼￼￼output_traj_file_name: str
            Name of the trajectory endpoints file
            ￼Default: file_name
            The trajectory end-points output file is named in this entry line.

        Deprecated
        ----------
        num_met_files: int
            deprecated: will be determined from input_met_file_names
            Number of input data grids
            Default: 1
            Number of simultaneous input meteorological files. The following two entries (directory and name) will be repeated this number of times. A simulation will terminate when the computation is off all of the grids in either space or time. Trajectory calculations will check the grid each time step and use the finest resolution input data available at that location at that time. When multiple meteorological grids have different resolution, there is an additional restriction that there should be some overlap between the grids in time, otherwise it is not possible to transfer a trajectory position from one grid to another. If multiple grids are defined and the model has trouble automatically transferring the calculation from one grid to another, the sub-grid size may need to be increased to its maximum value.
        input_met_file_folder: str
            deprecated: will be determined from input_met_file_names
            Meteorological data grid # 1 directory
            Default: ( \main\sub\data\ )
            Directory location of the meteorological file on the grid specified. Always terminate with the appropriate slash (\ or /).
        output_traj_file_folder: str
            deprecated: folder is taken from filename path
            Directory of trajectory output file
            Default: ( \main\trajectory\output\ )
            Directory location to which the text trajectory end-points file will be written. Always terminate with the appropriate slash (\ or /).
        """
        self.parameters = Parameters()
        self.parameters.start_time = start_time
        self.parameters.num_starting_loc = num_starting_loc
        self.parameters.starting_loc = starting_loc
        self.parameters.run_time = run_time
        self.parameters.vertical_motion_option = vertical_motion_option
        self.parameters.top_of_model_domain = top_of_model_domain
        self.parameters.input_met_file_names = input_met_file_names
        self.parameters.output_path = output_traj_file_name

        self.settings = Settings()
        self.settings.path2executable = '/Users/htelg/Hysplit4/working/hyts_std'
        self.settings.path2working_directory = '/Users/htelg/Hysplit4/working/'
        self.settings.project_directory = os.getcwd()

    def _create_control_file(self):
        raus = open(self.settings.path2working_directory + 'CONTROL', 'w')
        raus.write(self.parameters.start_time + '\n')  # 1
        raus.write('{}\n'.format(self.parameters.num_starting_loc))  # 2
        raus.write('\n'.join(['{:0.2f} {:0.2f} {:0.1f}'.format(*i) for i in self.parameters.starting_loc]) + '\n')  # 3
        raus.write('{:d}\n'.format(self.parameters.run_time))  # 4
        raus.write('{:d}\n'.format(self.parameters.vertical_motion_option))  # 5
        raus.write('{:0.1f}\n'.format(self.parameters.top_of_model_domain))  # 6
        raus.write('{:d}\n'.format(len(self.parameters.input_met_file_names)))  # 7
        for fn in self.parameters.input_met_file_names:
            folder, file = os.path.split(fn)
            raus.write(folder + '/' + '\n')  # 8
            raus.write(file + '\n')  # 9
        folder, file = os.path.split(self.parameters.output_path)
        raus.write(folder + '/' + '\n')
        raus.write(file + '\n')  # 11
        raus.close()

    def _run_hysplit(self, verbose=False):
        os.chdir(self.settings.path2working_directory)
        try:
            process = subprocess.check_output(self.settings.path2executable, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            if verbose:
                print('======')
                print(e.output.decode())
                print('======')
                print(e.returncode)
                print('======')
                print(dir(e))
            txt = 'An error occured when trying to run HYSPLIT.'
            if e.returncode == 24:
                txt += '\nError code is 24, this might mean that the control file was not found.'

            txt += '\n' + e.output.decode()
            raise subprocess.CalledProcessError(txt)

        self.result_stdout = process.decode()

        if self.result_stdout.split('\n')[-3] == ' Percent complete: 100.0':
            if verbose:
                print(self.result_stdout)
            return True

        elif 'ERROR' in self.result_stdout.split('\n')[-3]:
            txt = self.result_stdout.split('\n')[-3]
            if 'ASCDATA.CFG file not found' in self.result_stdout.split('\n')[-3]:
                txt = 'ASCDATA.CFG not found. This usually means the folder to the path2working_directory is wrong.\n Hysplit return:\n' + txt

            raise HysplitError(txt)

    def run(self, verbose=False):
        self._create_control_file()
        self._run_hysplit(verbose=verbose)
        result = read_hysplit_traj_output_file()
        if len(result) == 1:
            result = result[0]
        self.result = result

class Concentration_projectnotusedyet(object):
    def __init__(self,
                 start_time='00 00 00 00',
                 num_starting_loc=1,
                 starting_loc=[[40., -90., 10.]],
                 run_time=48,
                 vertical_motion_option=0,
                 top_of_model_domain=10000.0,
                 input_met_file_names=['/Users/htelg/Hysplit4/working/oct1618.BIN'],
                 output_traj_file_name='./cdump', ):

        """
        This class sets up a Hysplit run

        Parameters
        ----------
        start_time: string
            Starting time (year, month, day, hour, {minutes optional})
            Default: 00 00 00 00 {00}
            Enter the two digit values for the UTC time that the calculation is to start. Use 0's to start at the beginning (or end) of the file according to the direction of the calculation. All zero values in this field will force the calculation to use the time of the first (or last) record of the meteorological data file. In the special case where year and month are zero, day and hour are treated as relative to the start or end of the file. For example, the first record of the meteorological data file usually starts at 0000 UTC. An entry of "00 00 01 12" would start the calculation 36 hours from the start of the data file. The minutes field is optional but should only be used when the start time is explicitly set to a value.
        num_starting_loc: int
            Number of starting locations
            Default: 1
            Simultaneous trajectories can be calculated at multiple levels or starting locations. Specification of additional locations for certain types of simulations can also be accomplished through the Special Simulations menu tab, or through manual editing of the CONTROL file and running the model outside of the GUI. When multiple starting locations are specified, all trajectories start at the same time. A multiple trajectory in time option is available through the Advanced menu through a namelist file parameter setting.
        starting_loc: nested list of shape(num_starting_loc, 3)
            Enter starting location (lat, lon, meters)
            Default: 40.0 -90.0 50.0
            Trajectory starting position in degrees and decimal (West and South are negative). Height is entered as meters above ground-level. An option to treat starting heights as relative to mean-sea-level is available through the Advanced menu through a namelist file parameter setting.
        run_time: int
            Total run time (hours)
            Default: 48
            Specifies the duration of the calculation in hours. Backward calculations are entered as negative hours. A backward trajectory starts from the trajectory termination point and proceeds upwind. Meteorological data are processed in reverse-time order.
        vertical_motion_option: int
            Vertical motion option (0:data 1:isob 2:isen 3:dens 4:sigma 5:diverg 6:msl2agl 7:average)
            Default: 0
            Indicates the vertical motion calculation method. The default "data" selection will use the meteorological model's vertical velocity fields; other options include {isob}aric, {isen}tropic, constant {dens}ity, constant internal {sigma} coordinate, computed from the velocity {diverg}ence, vertical coordinate remapping from MSL to AGL, and a special option (7) to spatially average the vertical velocity. The averaging distance is automatically computed from the ratio of the temporal frequency of the data to the horizontal grid resolution.
        top_of_model_domain: float
            Top of model domain (internal coordinates m-agl)
            Default: 10000.0
            Sets the vertical limit of the internal meteorological grid. If calculations are not required above a certain level, fewer meteorological data are processed thus speeding up the computation. Trajectories will terminate when they reach this level. A secondary use of this parameter is to set the model's internal scaling height - the height at which the internal sigma surfaces go flat relative to terrain. The default internal scaling height is set to 25 km but it is set to the top of the model domain if the entry exceeds 25 km. Further, when meteorological data are provided on terrain sigma surfaces it is assumed that the input data were scaled to a height of 20 km (RAMS) or 34.8 km (COAMPS). If a different height is required to decode the input data, it should be entered on this line as the negative of the height. HYSPLIT's internal scaling height remains at 25 km unless the absolute value of the domain top exceeds 25 km.
        input_met_file_names: list of str
            Meteorological data grid # 1 file name
            Default: file_name
            Name of the file containing meteorological data. Located in the previous directory.
        ￼￼￼￼output_traj_file_name: str
            Name of the trajectory endpoints file
            ￼Default: file_name
            The trajectory end-points output file is named in this entry line.

        Deprecated
        ----------
        num_met_files: int
            deprecated: will be determined from input_met_file_names
            Number of input data grids
            Default: 1
            Number of simultaneous input meteorological files. The following two entries (directory and name) will be repeated this number of times. A simulation will terminate when the computation is off all of the grids in either space or time. Trajectory calculations will check the grid each time step and use the finest resolution input data available at that location at that time. When multiple meteorological grids have different resolution, there is an additional restriction that there should be some overlap between the grids in time, otherwise it is not possible to transfer a trajectory position from one grid to another. If multiple grids are defined and the model has trouble automatically transferring the calculation from one grid to another, the sub-grid size may need to be increased to its maximum value.
        input_met_file_folder: str
            deprecated: will be determined from input_met_file_names
            Meteorological data grid # 1 directory
            Default: ( \main\sub\data\ )
            Directory location of the meteorological file on the grid specified. Always terminate with the appropriate slash (\ or /).
        output_traj_file_folder: str
            deprecated: folder is taken from filename path
            Directory of trajectory output file
            Default: ( \main\trajectory\output\ )
            Directory location to which the text trajectory end-points file will be written. Always terminate with the appropriate slash (\ or /).
        """
        self.parameters = Parameters()
        self.parameters.start_time = start_time
        self.parameters.num_starting_loc = num_starting_loc
        self.parameters.starting_loc = starting_loc
        self.parameters.run_time = run_time
        self.parameters.vertical_motion_option = vertical_motion_option
        self.parameters.top_of_model_domain = top_of_model_domain
        self.parameters.input_met_file_names = input_met_file_names
        self.parameters.output_path = output_traj_file_name

        self.settings = Settings()
        self.settings.path2executable = '/Users/htelg/Hysplit4/working/hyts_std'
        self.settings.path2working_directory = '/Users/htelg/Hysplit4/working/'
        self.settings.project_directory = os.getcwd()

    def _create_control_file(self):
        raus = open(self.settings.path2working_directory + 'CONTROL', 'w')
        raus.write(self.parameters.start_time + '\n')  # 1
        raus.write('{}\n'.format(self.parameters.num_starting_loc))  # 2
        raus.write('\n'.join(['{:0.2f} {:0.2f} {:0.1f}'.format(*i) for i in self.parameters.starting_loc]) + '\n')  # 3
        raus.write('{:d}\n'.format(self.parameters.run_time))  # 4
        raus.write('{:d}\n'.format(self.parameters.vertical_motion_option))  # 5
        raus.write('{:0.1f}\n'.format(self.parameters.top_of_model_domain))  # 6
        raus.write('{:d}\n'.format(len(self.parameters.input_met_file_names)))  # 7
        for fn in self.parameters.input_met_file_names:
            folder, file = os.path.split(fn)
            raus.write(folder + '/' + '\n')  # 8
            raus.write(file + '\n')  # 9
        folder, file = os.path.split(self.parameters.output_path)
        raus.write(folder + '/' + '\n')
        raus.write(file + '\n')  # 11
        raus.close()

    def _run_hysplit(self, verbose=False):
        os.chdir(self.settings.path2working_directory)
        try:
            process = subprocess.check_output(self.settings.path2executable, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            if verbose:
                print('======')
                print(e.output.decode())
                print('======')
                print(e.returncode)
                print('======')
                print(dir(e))
            txt = 'An error occured when trying to run HYSPLIT.'
            if e.returncode == 24:
                txt += '\nError code is 24, this might mean that the control file was not found.'

            txt += '\n' + e.output.decode()
            raise subprocess.CalledProcessError(txt)

        self.result_stdout = process.decode()

        if self.result_stdout.split('\n')[-3] == ' Percent complete: 100.0':
            if verbose:
                print(self.result_stdout)
            return True

        elif 'ERROR' in self.result_stdout.split('\n')[-3]:
            txt = self.result_stdout.split('\n')[-3]
            if 'ASCDATA.CFG file not found' in self.result_stdout.split('\n')[-3]:
                txt = 'ASCDATA.CFG not found. This usually means the folder to the path2working_directory is wrong.\n Hysplit return:\n' + txt

            raise HysplitError(txt)

    def run(self, verbose=False):
        self._create_control_file()
        self._run_hysplit(verbose=verbose)
        result = read_hysplit_traj_output_file()
        if len(result) == 1:
            result = result[0]
        self.result = result
