import os, shutil
import pandas as pd
import numpy as np
import cdflib
from gemsem.utils import _to_mjd2000, _sun_mjd2000


def data_loader(DATA_PATH, year=None, month=None, COV_PATH=None, DST_PATH=None, KP_PATH=None, **kwargs):
    """
    This function loads different relevant datatypes used within the field of geomagnetism. It can load raw L1
    Swarm data products, given as .cdf files, virtual observatory (VO) and ground observatory files (GO).
    The VO and GO data is often given without extension, if this is the case the function checks the extension
    and if not ".txt" then it makes a copy of the file with ".txt" extension.

    The function assumes the approximate format of the header of the VO and GO data:

    % Virtual Observatory Model - MF%
    % theta     |   phi    |  Year  Month |   Time       |     r    |        Br             Bt            Bp     | dBrdr dBtdt dBpdp dBtdr dBpdr dBtdp |  N_{data}  |
    % [deg]     |  [deg]   |              |  [mjd2000]   |    [km]  |     Predicted field - NEC frame [nT]       | Predicted gradient field - NEC frame [nT/km] |   # data   |
    %

    % Ground Observatory Revised Monthly Means - MF%
    % theta    |  phi   |  Year  Month |Mean Time |  r     |     Br        Bt         Bp      | OBS CODE
    % [deg]    | [deg]  |(+2000)        | [yr] | [km]  |                Main Field
    %

    It returns a dataframe with the above structue. In the L1 Swarm case the returned dataframe is close to the
    structure above.

    Important: If COV_PATH is given there will be added three columns to the returned dataframe containing the
    diagonal of the covariance in the r, theta and phi components.
    In order for VO/GO to match with covariances it is important that only one epoch is chosen e.g. month=1 or month=3.

    Args:
        DATA_PATH (str): path pointing to either: a) ground or virtual observatory data or b) a directory with
                         L1 Swarm data product
        year (int or float): selected year, if float month should not be set.
        month (int): selected month, defaults to None. for VO choose [3,7,11] for GO [1-12]
        COV_PATH (str): path pointing to covariance matrix. Uses only variance i diagonal. Not implemented for
                               L1 Swarm data product
        DST_PATH (str): path pointing to Dst indices. dDst will be linearly interpolated to fit data time
        KP_PATH (str): path pointing to Dst indices, which will be linearly interpolated to fit data time

    **kwargs (optional):
        kp_limit (int): set kp index limit for data. Data with kp <= limit will be selected = 30
        dst_limit: (float) = set dst index limit for data. Data with abs(dst) <= limit will be selected
        down_sample_step (int): select every i'th row in the dataframe, default 60
        remove_sunlit_degree (float): choose the degrees for computing zenith = 90 - remove_sunlit_degree
                                      default 10 degrees
        drop_poles (boolean): remove pole points (lat=0 or 180) if any, defaults to True
        drop_nans (boolean): drop rows with nans, default True
        drop_specific_nans (boolean, str): drop rows based on nans in specified column (str) e.g. 'r', default False
        remove_b_flags (list int): drop flag_b values given by a list of values, e.g. [1,2] default [255]
        remove_q_flags (list int): drop flag_q values given by a list of values, e.g. [1,2] default [255]
        skiprows_before_col_names (int): how many rows to skip in VO and GO data before row with column names, default 1
        NaN_marker (str, float, int): value which marks nans in data, default is 99999.
        print_status (boolean): print status in L1 swarm data processing, default True

    Returns:
        dataframe

    Examples:
        #Example1
        DAT_PATH = "swarm_data/"
        Dst_PATH = "disturbance_indices/Dst_MJD_1998.dat"
        Kp_PATH = "disturbance_indices/Kp_MJD_1998_QL.dat"

        df = data_loader(DATA_PATH=DAT_PATH, DST_PATH=Dst_PATH, KP_PATH=Kp_PATH, dst_limit=3, kp_limit=20)

        #Example2
        DAT_PATH = 'vo_data/VO_SWARM_MF.0109.txt'
        COV_PATH = 'vo_data/VO_MF_SWARM_COV_diag.0109'
        df = data_loader(DATA_PATH=DAT_PATH, year=2015, month=7, COVARIANCE_PATH=COV_PATH)

    @author: Eigil Y. H. Lippert, Student DTU Space, <s132561@student.dtu.dk>
    """


    # defaults:
    skiprows_before_col_names = 1
    NaN_marker = 99999.
    drop_nans = True
    drop_poles = True
    drop_specific_nans = False
    remove_b_flags = [255]
    remove_q_flags = [255]
    down_sample_step = 60
    remove_sunlit_degree = 10
    dst_limit = 3
    kp_limit = 30
    print_status = True
    convert_to_radians = True

    # check user input
    for key, value in kwargs.items():
        if key == 'down_sample_step':
            down_sample_step = value
        elif key == 'remove_sunlit_degree':
            remove_sunlit_degree = value
        elif key == 'dst_limit':
            dst_limit = value
        elif key == 'kp_limit':
            kp_limit = value
        elif key == 'remove_b_flags':
            remove_b_flags = value
        elif key == 'remove_q_flags':
            remove_q_flags = value
        elif key == 'skiprows_before_col_names':
            skiprows_before_col_names = value
        elif key == 'NaN_marker':
            NaN_marker = value
        elif key == 'drop_nans':
            drop_nans = value
        elif key == 'drop_poles':
            drop_poles = value
        elif key == 'drop_specific_nans':
            drop_specific_nans = value
        elif key == 'print_status':
            print_status = value
        elif key == 'convert_to_rad':
            convert_to_radians = value

    # constants
    rad = np.pi / 180

    # if DST_PATH and KP_PATH not provided
    if DST_PATH is None:
        dst_limit = None
    if KP_PATH is None:
        kp_limit = None

    if isinstance(year, float):
        month = None

    # cdf file loading
    if os.path.isdir(DATA_PATH):
        if DST_PATH is not None:
            # Load in Dst indices:
            dst_indices = pd.read_csv(DST_PATH, sep=r"\s*", skiprows=3, engine='python')

            # also reads in "#" from header, so shift colums:
            cols = dst_indices.columns[1::]
            dst_indices = dst_indices.drop('Flag', 1)
            dst_indices.columns = cols

        if KP_PATH is not None:
            # Load in Kp indices:
            kp_indices = pd.read_csv(KP_PATH, sep=r"\s*", skiprows=4, engine='python')

            # also reads in "#" from header, so shift colums:
            cols = kp_indices.columns[1::]
            kp_indices = kp_indices.drop('Flag', 1)
            kp_indices.columns = cols

        # Open cdf and save in arrays
        mjd2000_time = []
        radii = []
        theta = []
        phi = []
        b_nec = []
        flags_b = []
        flags_q = []

        i = 0
        # os.walk iterates through the folder in DATA_PATH and ignores non-cdf files.
        for folder, subfolder, files in os.walk(DATA_PATH):
            for file in sorted(list(files)):
                # if there is any non-cdf files in your folder they will be skipped:
                try:
                    cdf_file = cdflib.CDF(folder + file)
                    if print_status: print('Succesfully loaded:', "\n", file)
                    time_stamps = cdf_file.varget("Timestamp")  # CDF epoch is in miliseconds since 01-Jan-0000
                    mjd2000_time.extend((time_stamps - time_stamps[0]) / (1e3 * 60 * 60 * 24)
                                        + _to_mjd2000(2014, 9, 14 + i))
                    radii.extend(cdf_file.varget("Radius") / 1e3)
                    theta.extend(90 - cdf_file.varget("Latitude"))
                    phi.extend(cdf_file.varget("Longitude"))
                    b_nec.extend(cdf_file.varget("B_NEC"))
                    flags_b.extend(cdf_file.varget("Flags_b"))
                    flags_q.extend(cdf_file.varget("Flags_q"))
                    i += 1
                    cdf_file.close()
                except OSError:
                    print('Error could not open file:', "\n", file)
                    pass

        # set up list and column names for dataframe
        b_nec = np.array(b_nec)
        data = [theta, phi, mjd2000_time, radii, -b_nec[:, 2], -b_nec[:, 0], b_nec[:, 1], flags_b, flags_q]
        column_names = ["theta", "phi", "time",  "r",  "Br", "Bt", "Bp", "flags_b", "flags_q"]

        # place data in dataframe for easier nan-removal etc.
        dataframe = pd.DataFrame()
        for index, col in enumerate(column_names):
            dataframe[col] = data[index]

        if print_status: print('Full dataframe created')

        # save memory
        del data, mjd2000_time, radii, theta, phi, b_nec, flags_q, flags_b

        # drop nans, if any
        if drop_nans:
            dataframe = dataframe.dropna()

        if drop_specific_nans:
            dataframe = dataframe[pd.notnull(dataframe[drop_specific_nans])]

        # check for error flags.
        if remove_b_flags is not None:
            for flags in remove_b_flags:
                # drops rows where flag_b == some value
                dataframe.drop(dataframe[dataframe.flags_b == flags].index, inplace=True)

        if remove_q_flags is not None:
            for flags in remove_q_flags:
                # drops rows where flag_q == some value
                dataframe.drop(dataframe[dataframe.flags_q == flags].index, inplace=True)

        if down_sample_step is not None:
            # downsample from 1 to 60 second intervals
            dataframe = dataframe.iloc[::down_sample_step, :]


        # for data reduction status
        n_data_all = dataframe.shape[0]

        # sunlit data selection
        if remove_sunlit_degree is not None:
            zenith = 90 - remove_sunlit_degree

            # to make cos_zeta expression more readable:
            colat = dataframe.theta.values
            lon = dataframe.phi.values
            time = dataframe.time.values

            # threshold for dark time observation
            cos_zeta_0 = np.cos((zenith) * rad)
            _, declination = _sun_mjd2000(dataframe.time.values)
            cos_zeta = np.cos(colat * rad) * np.sin(declination) \
                       + np.sin(colat * rad) * np.cos(declination) * np.cos(np.mod(time + .5, 1) * 2 * np.pi + lon * rad)

            # rows with cos_zeta higher than cos_zeta_0 will be dropped.
            sunlit_data_mask = cos_zeta < cos_zeta_0

            # select rows where sunlit_data_mask==True
            dataframe = dataframe[sunlit_data_mask]

            del colat, lon, time  # save memory

        # quiet time data selection
        if dst_limit is not None:
            # normalize d_dst by difference in time in case two time rows have longer timestep than sample rate.
            d_dst = np.diff(dst_indices.Dst.values)
            d_dst = np.hstack((d_dst[0], d_dst))  # for interpolation

            # Interpolate in order to obtain d_Dst at times of interest:
            d_dst = np.interp(dataframe.time.values, dst_indices.MJD2000, d_dst)

            # create boolean masks for data selection, we keep data where both conditions are met:
            dst_kp_mask = (abs(d_dst) <= dst_limit)

            if kp_limit is not None:
                # Interpolate in order to obtain Kp at times of interest:
                kp = np.interp(dataframe.time.values, kp_indices.MJD2000.values, kp_indices.Kp.values)

                # create boolean masks for data selection, we keep data where both conditions are met:
                dst_kp_mask = (abs(d_dst) <= dst_limit) & (kp <= kp_limit)

            # select rows where dst_kp_mask==True
            dataframe = dataframe[dst_kp_mask]

        elif kp_limit is not None:
            # Interpolate in order to obtain Kp at times of interest:
            kp = np.interp(dataframe.time.values, kp_indices.MJD2000.values, kp_indices.Kp.values)

            # create boolean masks for data selection, we keep data where both conditions are met:
            dst_kp_mask = (kp <= kp_limit)

            # select rows where dst_kp_mask==True
            dataframe = dataframe[dst_kp_mask]

        # for data reduction status
        n_data_dark_quiet = dataframe.shape[0]
        if print_status: print("Data reduced by: {0:.2f} %".format(n_data_dark_quiet / n_data_all * 100))

    else:
        if month is not None:
            year = 'Year==' + str(year)
            month = 'Month==' + str(month)
        elif year is not None:
            year = 'Time==' + str(year)
        else: pass

        # pandas cannot handle standard extension, so make a copy of file with .txt
        if DATA_PATH.split('.')[-1] != 'txt':
            shutil.copy2(DATA_PATH, DATA_PATH + '.txt')
            DATA_PATH = DATA_PATH + '.txt'
        else:
            DATA_PATH = DATA_PATH

        if COV_PATH.split('.')[-1] != 'txt':
            shutil.copy2(COV_PATH, COV_PATH + '.txt')
            COV_PATH = COV_PATH + '.txt'
        else:
            COV_PATH = COV_PATH

        # read in column names alone
        dataframe = pd.read_table(DATA_PATH, sep=' ', skiprows=skiprows_before_col_names, header=None, nrows=1)
        dataframe = dataframe.dropna(axis=1)

        # column names are not consistent, so we have to work around
        column_names = []
        for col_name in dataframe.loc[0]:
            for string in col_name.split('|'):
                if (string != '') and (string != '%'):
                    # in GO data there is Mean Time and OBS CODE, which had to be handled explicitly
                    if (string == 'Mean') or (string == 'CODE'):
                        pass
                    else:
                        column_names.append(string)

        # read in data without header, set % as comment symbol
        dataframe = pd.read_table(DATA_PATH, comment='%', header=None, delim_whitespace=True)
        dataframe.columns = column_names

        # set NaN-marker to NaN and drop nan rows
        dataframe = dataframe.replace(NaN_marker, np.nan)  # set all 99999 values to NaN

        # selects chosen data time (year, epoch)
        if year is not None:
            dataframe = dataframe.query(year)
        if month is not None:
            dataframe = dataframe.query(month)

        if COV_PATH is None:
            # empty errors object
            errors = None
        else:
            # Read in uncertainty estimates, corresponding to chosen year
            covariance_matrix = pd.read_table(COV_PATH, header=None)

            # Extract diagonals
            cov_r = np.diag(covariance_matrix)[0:dataframe.shape[0]]
            cov_t = np.diag(covariance_matrix)[dataframe.shape[0]:dataframe.shape[0] * 2]
            cov_p = np.diag(covariance_matrix)[dataframe.shape[0] * 2::]

            # Add errors to existing data frame
            dataframe = dataframe.assign(var_r=cov_r)
            dataframe = dataframe.assign(var_t=cov_t)
            dataframe = dataframe.assign(var_p=cov_p)

        if drop_poles:
            dataframe = dataframe[dataframe['theta'] != 0]  # drop rows with theta=0
            dataframe = dataframe[dataframe['theta'] != 180]  # drop rows with theta=180

        # drop all rows with nans
        if drop_nans:
            dataframe = dataframe.dropna(axis=0, how='any')  # drop rows with NaNs

        # drop only rows with nans in specific columns
        if drop_specific_nans:
            dataframe = dataframe[pd.notnull(dataframe[drop_specific_nans])]

    if convert_to_radians:
        dataframe['theta'] *= rad
        dataframe['phi'] *= rad

    return dataframe


#%%
# DAT_PATH = "swarm_data/"
# Dst_PATH = "disturbance_indices/Dst_MJD_1998.dat"
# Kp_PATH = "disturbance_indices/Kp_MJD_1998_QL.dat"


# DAT_PATH = 'vo_data/VO_SWARM_MF.0109.txt'
# COV_PATH = 'vo_data/VO_MF_SWARM_COV_diag.0109'
# DAT_PATH = 'go_data/GR_OBS_RMM_MF_less_bias_V27.10.txt'
# COV_PATH = 'go_data/GR_OBS_RMM_MF_COV_V27.10.txt'



# df = data_loader(DATA_PATH=DAT_PATH)


