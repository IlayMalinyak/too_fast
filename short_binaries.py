import os
import pandas as pd
from utils import merge_all_cats, apply_constraints, aggregate_results, fill_nan_np, convert_cols_to_float
from classical_analysis import analyze_lc
import cgs_const as cgs
from wide_binaries import get_kepler_gaia_wide_binary_samples, get_kepler_gaia_primary_binaries
import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
from sklearn.linear_model import LinearRegression
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

import itertools
from astropy.table import Table, join
from scipy.signal import savgol_filter as savgol
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp
from scipy.optimize import curve_fit
from matplotlib.cm import get_cmap
import re
from astropy.table import Table
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from typing import Optional, Union, List, Tuple
from sklearn.metrics import mean_squared_error
from math import sqrt
import requests
from astropy.io import ascii


JUPYTER_RAD = 11.2

MORTON_TEFF = {'Kepler-508 b': 6025, 'Kepler-1521 b': 5042, 'Kepler-1184 b': 5983, 'Kepler-448 b': 6779}


CLUSTERS = {'{alpha} Per':80, 'Pleiades':120, 'Blanco-1':120, 'Psc-Eri':120, 'NGC-3532':300,
       'Group-X':300, 'Praesepe':670, 'NGC- 6866':794, 'NGC-6811':1000, 'NGC-6819':2500, 'Ruprecht-147':2700,
       'cep_her':40, 'melange3':150, 'theia':300}

MIST_PATH = 'tables/ks_mist_catalog_kepler_1e9_feh_interp_all.csv'

from astropy.io import ascii


mpl.rcParams['axes.linewidth'] = 4
plt.rcParams.update({'font.size': 32, 'figure.figsize': (16,10), 'lines.linewidth': 4})
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "gray", "c", 'm', 'brown', 'yellow'])
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r",  "black", "brown", "blue", "green",  "y",  "purple", "pink"])
plt.rcParams.update({'xtick.labelsize': 30, 'ytick.labelsize': 30})
plt.rcParams.update({'legend.fontsize': 26})


def download_machine_readable_table(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        # Save the downloaded content to a local file
        with open(f'{save_path}.txt', 'wb') as f:
            f.write(response.content)
        print("ASCII table downloaded successfully.")
    else:
        print(f"Failed to download ASCII table. Status code: {response.status_code}")

def read_machine_readable_table(table_path):
    try:
        table = ascii.read(table_path)
    except Exception as e:
        print(f"Error loading table: {e}")
        table = None
    return table


def fractional_error(x, y, dx, dy):
    return ((dx/y) ** 2 + (dy * x / y ** 2) ** 2 ) ** 0.5



def find_density_boundary(teff, kmag_diff,
                          teff_bins=15,
                          kmag_bins=10,
                          kmag_range=(-2, 0.5),
                          teff_range=(4000,6000)):
    """
    Find the boundary between two populations by identifying lowest density regions.

    Parameters:
    teff (array): Effective temperature values
    kmag_diff (array): K magnitude difference values
    teff_bins (int): Number of temperature bins
    kmag_bins (int): Number of K magnitude bins for each temperature slice
    kmag_range (tuple): (min, max) range for kmag_diff to search for boundary
    smooth (bool): Whether to smooth the final boundary

    Returns:
    dict: Dictionary with temperature points as keys and boundary kmag values as values
    """
    # Create temperature bins
    teff_edges = np.linspace(teff_range[0], teff_range[1], teff_bins + 1)
    boundary_dict = {}

    # Iterate through temperature bins
    for i in range(teff_bins):
        # Get data in this temperature slice
        mask = (teff >= teff_edges[i]) & (teff < teff_edges[i + 1])
        slice_kmag = kmag_diff[mask]

        if len(slice_kmag) > 0:
            # Create histogram for this slice
            hist, bin_edges = np.histogram(slice_kmag,
                                           bins=kmag_bins,
                                           range=kmag_range)

            # Find the minimum density point within the range
            valid_bins = (bin_edges[:-1] >= kmag_range[0]) & (bin_edges[:-1] <= kmag_range[1])
            if np.any(valid_bins):
                min_density_idx = np.argmin(hist[valid_bins])
                density_grad = np.gradient(hist[valid_bins])
                max_density_idx = np.argmax(density_grad)
                boundary_point = bin_edges[:-1][valid_bins][min_density_idx]

                # Store the midpoint of the temperature bin and the boundary point
                temp_midpoint = (teff_edges[i] + teff_edges[i + 1]) / 2
                boundary_dict[temp_midpoint] = boundary_point
    coeff = np.polyfit(list(boundary_dict.keys()), list(boundary_dict.values()), 2)
    poly = np.poly1d(coeff)
    return poly


def Teff_r_mass_mainsequence(Teff, r):
    """
    Calculate mass of a star on the main sequence given its effective temperature and radius.
    We use  L \propto M ** 4

    Parameters:
    -----------
    Teff : float
        Effective temperature in Kelvin
    r : float
        Radius in solar radii

    Returns:
    --------
    float : Mass in solar masses
    """
    const = (cgs.sigma_sb * 4 * np.pi / cgs.L_sun) ** 0.25
    return Teff * (r * cgs.R_sun) ** 0.5 * const


def Teff_mass_mainsequence(Teff):
    """
    Calculate mass of a star on the main sequence given its effective temperature.
    We use R \propto M ** 0.8, L \propto M ** 4

    Parameters:
    -----------
    Teff : float
        Effective temperature in Kelvin

    Returns:
    --------
    float : Mass in solar masses
    """
    const = (cgs.sigma_sb * 4 * np.pi / cgs.L_sun) ** 0.4167
    return Teff ** 1.667 * cgs.R_sun ** 0.833 * const

def plot_density_boundary(teff, kmag_diff, boundary_fn, smooth=True):
    """
    Plot the data density and the detected boundary.

    Parameters:
    teff (array): Effective temperature values
    kmag_diff (array): K magnitude difference values
    boundary_dict (dict): Dictionary with temperature points and boundary values
    smooth (bool): Whether to smooth the boundary line
    """

    # Plot density
    hb1 = plt.hexbin(teff, kmag_diff, mincnt=1, cmap='YlGnBu')

    # Sort boundary points for plotting
    temps = np.sort(teff)
    boundaries = boundary_fn(temps)

    # if smooth:
    #     # Create smoothed interpolation
    #     f = interp1d(temps, boundaries, kind='cubic', fill_value='extrapolate')
    #     temps = np.linspace(temps.min(), temps.max(), 200)
    #     boundaries = f(temps)
    #     # Plot linear interpolation between points
    #
    # res_dict = dict(zip(temps, boundaries))
    plt.plot(temps, boundaries, 'r-', label='Detected Boundary')

    plt.colorbar(label='Density')
    plt.xlabel('Teff (K)')
    plt.ylabel('$\Delta K_{iso}$')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.savefig("imgs/boundaries.png")
    plt.show()

def parse_table_description(description_file, data_file):
    """
    Parses a machine-readable table description and reads the data into a DataFrame.

    Parameters:
        description_file (str): Path to the file containing the table description.
        data_file (str): Path to the file containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing the table data.
    """
    # Step 1: Parse the description file to extract column names and byte ranges
    column_specs = []
    column_names = []

    with open(description_file, 'r') as desc:
        description_lines = desc.readlines()

    # Extract byte ranges and column names using regex
    for line in description_lines:
        match = re.match(r'\s*(\d+)-\s*(\d+)\s+\S+\s+\S+\s+(\S+)', line)
        if match:
            start = int(match.group(1)) - 1  # Convert to 0-based indexing
            end = int(match.group(2))       # End is inclusive in pandas.read_fwf
            name = match.group(3)
            column_specs.append((start, end))
            column_names.append(name)

    # Step 2: Use pandas.read_fwf to read the data file with extracted specs
    df = pd.read_fwf(data_file, colspecs=column_specs, names=column_names, skiprows=3)

    return df


def detrend_df(df, trend_hot, x_thresh, x_col, y_col):
    df_detrended = df.copy()

    mask = df[x_col] >= x_thresh
    df_detrended.loc[mask, f'{y_col}_detrended'] = df.loc[mask, y_col] - trend_hot(df.loc[mask, x_col])

    # For values below threshold, keep original values
    df_detrended.loc[~mask, f'{y_col}_detrended'] = df.loc[~mask, y_col]

    return df_detrended

def binary_boundary(x):
    result = np.zeros_like(x, dtype=float)
    result[x < 5800] = -0.28
    result[x >= 5800] = 0.23/1200 * x[x >= 5800] - 1.4
    result[result > 0] = np.nan
    return result


def is_binary(boundary_fn, teff, kmag_diff, teff_range=[3000,7000], buffer=0):
    """
    Determine if a star is in the binary (upper) population based on its temperature
    and K magnitude difference.

    Parameters:
    boundary_fn (np.polyfit): Boundary function
    teff (float): Effective temperature of the star
    kmag_diff (float): K magnitude difference of the star

    Returns:
    bool: True if the star is in the upper population (binary), False otherwise
    """
    if (teff > teff_range[1]) or (teff < teff_range[0]):
        return np.nan
    # Convert boundary dictionary keys to array for efficient searching
    # Get the boundary value at this temperature
    boundary_value = boundary_fn(teff)
    if kmag_diff < boundary_value + buffer:
        position = 'binary'
    else:
        position = 'single'

    # Return True if the point is above the boundary (smaller kmag_diff)
    return position

def is_inside_shape(points):
    """
    Determines if points are inside the shape defined by binary_boundary
    Args:
        points: numpy array of shape (N,2) where each row is [x,y]
    Returns:
        numpy array of booleans, True if point is inside shape
    """
    x = points[:, 0]
    y = points[:, 1]
    boundary_y = binary_boundary(x)
    # Points where boundary is nan are considered outside (False)
    return (y >= boundary_y) & (y < 0) & ~np.isnan(boundary_y)

def triple_critical_stability(e_out, i_mut=0, e_in=0, q_out=0.5):
    """
    stability criteria from :
    'Algebraic and machine learning approach to hierarchical triple-star
    stability' P. Vaynatheya et al. 2022
    :param e_out:
    :param i_mut:
    :param e_in:
    :param q_out:
    :return:
    """
    e_in_max = np.sqrt(1 - 5 / 3 * np.cos(i_mut) ** 2)
    e_in_avg = 0.5 * e_in_max ** 2
    e_in_tilde = max(e_in, e_in_avg)
    first_term = 2.4 * ((1 + q_out) / ((1 + e_in_tilde) * (1 - e_out)) ** 0.5) ** (2 / 5)
    second_term = (1 - 0.2 * e_in_tilde + e_out) / 8 * (np.cos(i_mut - 1)) + 1
    return first_term * second_term

def triple_stability(a_out, a_in, e_out, e_in=0, i_mut=0, q_out=0.5):
    Y = a_out * (1 - e_out) / ((a_in) * (1 + e_in))
    Y_crit = triple_critical_stability(e_out, i_mut=i_mut, e_in=e_in, q_out=q_out)
    return Y > Y_crit

def T_from_M(M):
    return (cgs.L_sun / (4 * np.pi * cgs.sigma_sb * cgs.R_sun ** 2)) ** (1/4) * M ** 0.575

def M_from_T(T):
    return ((4 * np.pi * cgs.sigma_sb * cgs.R_sun ** 2)  / cgs.L_sun) ** (1/2.4) * T ** (4/2.4)


def synchronization_period(teff, tau, lambda_2=0.006, q=0.5, beta=1):
    A = 0.707 * ((tau  * lambda_2) ** (1/4) * 10 ** (-1/2) * beta ** (-1/2) * (q ** (1/2) / (1 + q) ** (1/2))
         * (4 * np.pi * cgs.sigma_sb * cgs.R_sun**2 / cgs.L_sun) ** 0.34)
    return A * (teff ** 1.35)

def tau_circ_convective(p, M, lambda_2=0.006):
    p_year = p
    return 1.99 * 1e3 * 2 ** (5/3) * lambda_2 ** (-1) * M ** (-4.16) * p_year ** (16/3)

def tau_circ_convective_T(p, T, lambda_2=0.006):
    p_year = p
    M = M_from_T(T)
    return 1.99 * 1e3 * 2 ** (5/3) * lambda_2 ** (-1) * M ** (-4.16) * p_year ** (16/3)

def tau_circ_radiative(p, M, E_2=1e-7):
    p_year = p
    return 1.71 * 1e1 * 2 ** (5/3) * E_2 ** (-1) * M ** (-4.2) * p_year ** 7

def tidal_torque(p_rot, p_orb, M, R, d, q=0.5, tau_diss=1):
    return - (p_rot - p_orb) / tau_diss * q ** 2 * M * R ** 2 * (R / d) ** 6

def vsini_sat_threshold(poly_path, teff, vsini):
    coeffs = np.load(poly_path)
    poly = np.poly1d(coeffs)
    thresh_val = poly(teff)
    return thresh_val > vsini

def vsini_sat_threshold_pandas(row, poly):
    if row['Teff'] > 4800:
        thresh_val = poly(row['Teff'])
    else:
        thresh_val = poly(4800)
    return thresh_val > row['vsini']

def cutoff_threshold_pandas(row, poly, p_val):
    if row['Teff'] > 4800:
        thresh_val = poly(row['Teff'])
    else:
        thresh_val = poly(4800)
    return thresh_val < row[p_val]

def create_density_data(x,y):
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    idx = density.argsort()
    x, y, density = x[idx], y[idx], density[idx]
    return x,y,density


def create_catalog(df_path, conf=0.86, low_p=1.5, self_kmag=False):
    df = merge_all_cats(aggregate_results(pd.read_csv(df_path), selected_numbers=None))
    df = apply_constraints(df,
                            contaminants=True,
                            teff=7000,
                            doubles=True,
                            conf=conf,
                            low_p=low_p,
                            error=None)
    df['relative error'] = (df['sigma error'] / (df['predicted period']))
    df['total error'] = (df['relative error'] / (df['mean_period_confidence']))
    df['main_seq'] = df.apply(giant_cond,axis=1)
    # df = df[df['main_seq']==True]
    df = get_mag_data(df, mist_path=MIST_PATH)
    if self_kmag:
        df = self_kmag_diff(df)
    df['GRCOLOR_abs'] = df['GMAG_abs'] - df['RMAG_abs']
    df = get_vsini(df, p_val='predicted period')
    df['msi'] = df.apply(lambda x: Teff_mass_mainsequence(x['Teff']), axis=1)
    df['m_diff'] = df['msi'] - df['Mstar']

    return df


def self_kmag_diff(
        df: pd.DataFrame,
        df_long: pd.DataFrame,
        teff_threshold: Optional[float] = 50,
        feh_threshold: Optional[float] = 0.05,
        teff_bins: Optional[List[float]] = None,
        feh_bins: Optional[List[float]] = None,
        min_stars_per_bin: int = 5,
        return_stats: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculate the difference between 'kmag_abs' values and their corresponding reference
    medians based on Teff and FeH binning. The reference medians are calculated from
    the long-period stars (df_long) for each (Teff, FeH) bin.

    Parameters:
    - df: pandas.DataFrame
        Target dataframe containing 'Teff', 'FeH', and 'kmag_abs' columns
    - df_long: pandas.DataFrame
        Reference dataframe containing long-period stars with 'Teff', 'FeH', and 'kmag_abs' columns
    - teff_threshold: float, optional
        Bin size for 'Teff' grouping when custom bins aren't provided
    - feh_threshold: float, optional
        Bin size for 'FeH' grouping when custom bins aren't provided
    - teff_bins: List[float], optional
        Custom bin edges for Teff. If provided, overrides teff_threshold
    - feh_bins: List[float], optional
        Custom bin edges for FeH. If provided, overrides feh_threshold
    - min_stars_per_bin: int, optional
        Minimum number of stars required in a bin for median calculation
    - return_stats: bool, optional
        If True, returns additional statistics about the binning

    Returns:
    - Union[pandas.DataFrame, Tuple[pandas.DataFrame, pandas.DataFrame]]
        If return_stats=False: Input dataframe with additional 'kmag_diff_self' column
        If return_stats=True: Tuple of (processed dataframe, bin statistics dataframe)
    """
    # Ensure required columns exist
    required_cols = ['Teff', 'FeH', 'kmag_abs']
    for df_check in [df, df_long]:
        if not all(col in df_check.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")

    # Create working copies to avoid modifying original dataframes
    df = df.copy()
    df_long = df_long.copy()

    # Handle missing values
    df = df.dropna(subset=required_cols)
    df_long = df_long.dropna(subset=required_cols)

    # Create bins based on either custom edges or thresholds
    if teff_bins is not None:
        df['Teff_group'] = pd.cut(df['Teff'], bins=teff_bins, labels=False)
        df_long['Teff_group'] = pd.cut(df_long['Teff'], bins=teff_bins, labels=False)
    else:
        df['Teff_group'] = (df['Teff'] / teff_threshold).round().astype(int)
        df_long['Teff_group'] = (df_long['Teff'] / teff_threshold).round().astype(int)

    if feh_bins is not None:
        df['FeH_group'] = pd.cut(df['FeH'], bins=feh_bins, labels=False)
        df_long['FeH_group'] = pd.cut(df_long['FeH'], bins=feh_bins, labels=False)
    else:
        df['FeH_group'] = (df['FeH'] / feh_threshold).round().astype(int)
        df_long['FeH_group'] = (df_long['FeH'] / feh_threshold).round().astype(int)

    # Calculate bin statistics
    bin_stats = (
        df_long.groupby(['Teff_group', 'FeH_group'])
        .agg({
            'kmag_abs': ['count', 'median', 'std'],
            'Teff': ['min', 'max'],
            'FeH': ['min', 'max']
        })
        .reset_index()
    )
    bin_stats.columns = ['Teff_group', 'FeH_group', 'stars_in_bin',
                         'median_kmag_abs', 'kmag_abs_std',
                         'Teff_min', 'Teff_max', 'FeH_min', 'FeH_max']

    # Filter bins with too few stars
    valid_bins = bin_stats[bin_stats['stars_in_bin'] >= min_stars_per_bin]

    # Calculate median kmag_abs for valid bins using long-period stars
    median_kmag_abs = valid_bins[['Teff_group', 'FeH_group', 'median_kmag_abs']]

    # Merge medians with original dataframe
    df = pd.merge(df, median_kmag_abs, on=['Teff_group', 'FeH_group'], how='left')

    # Calculate differences and mark invalid bins
    df['kmag_diff_self'] = df['kmag_abs'] - df['median_kmag_abs']
    df['valid_bin'] = ~df['kmag_diff_self'].isna()

    # Add bin size information to the main dataframe
    df = pd.merge(
        df,
        bin_stats[['Teff_group', 'FeH_group', 'stars_in_bin']],
        on=['Teff_group', 'FeH_group'],
        how='left'
    )

    # Clean up intermediate columns
    final_columns = list(set(df.columns) - {'Teff_group', 'FeH_group', 'median_kmag_abs'})
    df = df[final_columns]

    if return_stats:
        return df, bin_stats
    return df


def get_mag_data(df, mist_path):
    kepler_meta = pd.read_csv('tables/kepler_dr25_meta_data.csv')
    mist_data = pd.read_csv(mist_path)
    mag_cols = [c for c in kepler_meta.columns if ('MAG' in c) or ('COLOR' in c)]
    meta_columns = mag_cols + ['KID', 'EBMINUSV']
    df = df.merge(kepler_meta[meta_columns], on='KID', how='left')
    df = df.merge(mist_data[['KID', 'Kmag_MIST']], on='KID', how='left')
    if 'Dist' not in df.columns:
        berger_cat = pd.read_csv('tables/berger_catalog.csv')
        df = df.merge(berger_cat, on='KID', how='left',
                      suffixes=['_old', ''])
    for c in mag_cols:
        df[f'{c}_abs'] = df.apply(lambda x: x[c] - 5 * np.log10(x['Dist']) + 5, axis=1)
    df['kmag_abs'] = df.apply(lambda x: x['KMAG'] - 5 * np.log10(x['Dist']) + 5, axis=1)
    df['kmag_diff'] = df['kmag_abs'] - df['Kmag_MIST']
    return df

def get_w(df, cols=['Teff', 'logg', 'FeH', 'Mstar']):
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    params = np.array(df[cols])
    params_0 = np.zeros(len(cols))
    w = (np.sum((params_0[None, :] - params) ** 2, axis=-1).astype(np.float32)) ** 0.5
    df['w'] = w
    return df

def keil_diagram(df):
    plt.scatter(df['Teff'], df['logg'], c=df['mean_period_confidence'])
    plt.colorbar()

    teff_values = np.linspace(3000, 7000, 1000)
    logg_values = []

    for teff in teff_values:
        # Apply the giant_cond threshold logic
        if teff >= 6000:
            thresh = 3.5
        elif teff <= 4250:
            thresh = 4
        else:
            thresh = 5.2 - (2.8 * 1e-4 * teff)
        logg_values.append(thresh)

    # Plot the boundary line
    plt.plot(teff_values, logg_values, color='red', linestyle='dashed')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.savefig('imgs/keil_diagram.png')
    plt.show()

def exponential_curve(x, tau):
    return 1 - (x ** (1 / tau))

def power_cuvre(x, a ,b):
    return a * x ** b

def fit_envelop(x,y, lower=True):
    bins = np.linspace(np.nanmin(x), np.nanmax(x), 40)  # Adjust number of bins if needed
    bin_indices = np.digitize(x, bins)
    env_x = []
    env_y = []

    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            env_x.append(bins[i - 1])
            if lower:
                env_y.append(y[bin_mask].min())
            else:
                env_y.append(y[bin_mask].max())

    env_x = np.array(env_x)
    env_y = np.array(env_y)
    # increasing_indices = np.where(np.diff(env_y) > 0)[0]
    # increasing_indices = np.concatenate(([0], increasing_indices + 1))
    # env_y = np.array(env_y)[increasing_indices]
    # env_x = np.array(env_x)[increasing_indices]
    if not lower:
        print('upper bounds')
        env_x = env_x[:np.argmax(env_y)+1]
        env_y = env_y[:np.argmax(env_y)+1]

    outliers = np.logical_and(env_x > 200, env_y < 0.6)
    env_x = env_x[~outliers]
    env_y = env_y[~outliers]

    # Fit the curve to the lower envelope points
    popt, _ = curve_fit(power_cuvre, env_x, env_y, p0=[0, 1])  # Initial guess for tau

    # Generate the fitted curve
    fitted_x = np.linspace(x.min(), x.max(), 500)
    # fitted_y = exponential_curve(fitted_x, *popt)
    fitted_y = power_cuvre(fitted_x, *popt)
    return fitted_x, fitted_y, popt


def hr_diagram(df, y_data='Lstar', color='w', other=None, density_levels=10, bins=50):
    # Initial setup
    df['main_seq'] = df.apply(giant_cond, axis=1)
    fig, ax = plt.subplots()

    # Calculate density using histogram2d (much faster than KDE)
    x = df['Teff'].values
    y = df[y_data].values
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Get the mesh grid for contour plotting
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)

    # Plot contours
    contours = ax.contour(X, Y, hist.T, levels=density_levels,
                          colors='gray', linewidths=1)

    # Plot scatter points with original coloring
    if color is not None:
        scatter = ax.scatter(df['Teff'], df[y_data], c=df[color],
                             alpha=0.05, s=20)
        # plt.colorbar(scatter, label=color)
    else:
        ax.scatter(df['Teff'], df[y_data], alpha=0.5, s=20)

    # Plot additional data if provided
    if other is not None:
        ax.scatter(other['Teff'], other[y_data], c='brown')

    # Customize axes
    ax.invert_xaxis()
    if y_data != 'Lstar':
        ax.invert_yaxis()

    # Labels
    ax.set_xlabel('$T_{eff}$ (K)')
    ax.set_ylabel('K band absolute magnitude (Mag)')

    # Save and show
    plt.savefig(f'imgs/hr_diagram_{y_data}_density.png',
                dpi=300, bbox_inches='tight')
    plt.show()

def giant_thresh_single(teff):
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return thresh

def giant_thresh(teffs):
    res = 5.2 - (2.8 * 1e-4 * teffs)
    res[teffs >= 6000] = 3.5
    res[teffs <= 4250] = 4
    return res

def giant_cond(x):
    """
    condition for red giants in kepler object.
    the criterion for red giant is given in Ciardi et al. 2011
    :param: x row in dataframe with columns - Teff, logg
    :return: boolean
    """
    logg, teff = x['logg'], x['Teff']
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return logg >= thresh



def plot_lightcurves(df):
    for i, row in df.iterrows():
        kid = row['KID']
        kep_name = row['kepler_name']
        if kep_name is np.nan:
            kep_name = row['kepoi_name']
        p_model = row['predicted period']
        p_ref = row['P_rot'] if 'P_rot' in row else np.nan
        planet_prot = row['planet_Prot']
        conf = row['mean_period_confidence']
        print("downloading ", kid)
        try:
            res = lk.search_lightcurve(f'KIC {kid}', cadence='long', author='kepler')
            lc = res.download_all().stitch()
        except Exception as e:
            print(e)
            continue
        lc = lc.flatten(window_length=401)
        # lc_folded = lc.fold(period=planet_prot).bin(binsize=10)
        flux = fill_nan_np(np.array(lc.flux), interpolate=True)
        time = np.linspace(0, len(lc)/48, len(lc))
        p_acf = analyze_lc(flux)[0]
        np.save(f'imgs/samples/{kid}.npy', flux)
        # flux_filled = fill_nan_np(flux, interpolate=True)
        # flux_clean_long = fill_nan_np(savgol(flux, window_length=49, polyorder=1), interpolate=True)
        lc_sub = flux[:-int(p_acf*48)] - flux[int(p_acf*48):]
        start_idx = int(np.random.randint(low=0, high=len(time) - int(planet_prot * 48 * 6) - 1, size=1))
        end_idx = start_idx + int(planet_prot * 48 * 6)
        fig, ax = plt.subplots(3,1, figsize=(40,24))
        ax[0].plot(time[start_idx:end_idx], flux[start_idx:end_idx])
        ax[0].set_xlabel('Time (days)')
        ax[0].set_ylabel('Relative Flux (%)')
        ax[0].grid()
        ax[1].plot(time, flux)
        ax[1].set_xlabel('Time (days)')
        ax[1].set_ylabel('Relative Flux (%)')
        ax[2].plot(time[start_idx:end_idx], lc_sub[start_idx:end_idx])
        ax[2].set_xlabel('Time (days)')
        ax[2].set_ylabel('Relative Flux (%)')
        fig.suptitle(f'model - {p_model:.2f},'
                     f' acf - {p_acf:.2f},'
                     f' ref - {p_ref:.2f},'
                     f' planet prot - {planet_prot:.2f},'
                     f' confidence - {conf}')
        plt.tight_layout()
        plt.savefig(f'imgs/imgs/{kep_name}.png')
        plt.close()

def plot_planets_binaries(df):
    # Create the scatter plot with a colormap
    unique_eb_values = df['eb'].unique()
    colormap = cm.get_cmap('viridis', len(unique_eb_values))
    scatter = plt.scatter(
        df['predicted period'],
        df['planet_Prot'] / df['predicted period'],
        c=df['eb'], s=100, cmap=colormap, label='Data points'
    )
    plt.semilogy()
    plt.xlabel(r"$P_{stellar}$ (days)")
    plt.ylabel(r"$\frac{P_{planet}}{P_{stellar}}$")

    # Custom legend using colors from the colormap
    labels = ['eclipsing binary', 'non-eclipsing binary']
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=label,
                   markerfacecolor=colormap(i / len(unique_eb_values)), markersize=10)
        for i, label in enumerate(labels)
    ]
    plt.legend(handles=legend_handles, title="Binary Types")
    plt.savefig('imgs/prot_vs_planet_prot_conf_0.95.png')
    plt.show()

def create_holman_table(position='inner'):
    holman_df = pd.read_csv(f'tables/holman_stability_{position}.csv')
    masses = holman_df.columns.values[1:].astype(np.float16)
    eccentricities = holman_df.iloc[:,0].values.astype(np.float16)
    semi_major_axis_table = holman_df.iloc[1:,1:].values.astype(np.float16)
    period_table = semi_major_axis_table ** (3/2)
    return period_table, eccentricities, masses


def plot_planet_stability(periods_df, xval='kepler_name', p_val='predicted period', name='CONFIRMED'):
    stability_table_out, eccentricities_out, masses_out = create_holman_table('outer')
    stability_table_in, eccentricities_in, masses_in = create_holman_table('inner')


    min_value_out = np.min(stability_table_out)
    max_value_in = np.max(stability_table_in)
    circum_candidates = periods_df[(periods_df['period_ratio'] > (min_value_out - min_value_out * 0.2)) |
                                   (periods_df['period_ratio'] < (max_value_in + max_value_in * 0.2))]
    fp_candidates = periods_df[(periods_df['period_ratio'] < 2)]

    # Plot the scatter plot below both heatmaps
    # ax2 = fig.add_subplot(gs[1, :])  # Span the entire bottom row
    periods_df['P_source_code'] = periods_df.apply(lambda row: row['P_source'] == 'Kamai24', axis=1)
    plt.scatter(periods_df[xval], periods_df['period_ratio' ], s=100,  edgecolors='k')
    # plt.scatter(fp_candidates[xval], fp_candidates['period_ratio'], s=100, c='khaki')

    # plt.xlabel(xval)
    plt.ylabel(r"$\frac{P_{orb}}{P_{rot}}$")


    # Set the x-axis tick labels to the 'kepler_name' values
    plt.xticks(ticks=range(len(periods_df[xval])), labels=periods_df[xval], rotation=90, fontsize=18)
    # plt.xticklabels(periods_df[xval], rotation=90)

    # Add horizontal lines at the minimum values of each stability table
    plt.axhline(min_value_out, color='green', linestyle='--', label='Min Outer Stability')
    plt.axhline(max_value_in, color='green', linestyle='--', label='Max Inner Stability')
    plt.axhline(1, color='gray', linestyle='--', label=r'$\frac{P_{orb}}{P_{rot}}=1$')


    # plt.legend()
    # plt.colorbar(label='$R_J$')
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.semilogy()
    # plt.yticks(ticks=np.logspace(np.log10(min_value_out), np.log10(max_value_in), num=10), labels=[])

    plt.savefig(f'imgs/stability_{name}.png')
    plt.show()

    plt.figure(figsize=(16,12))
    plt.scatter(fp_candidates['period_ratio'], fp_candidates['a/R'],
                c=fp_candidates['koi_prad'] / JUPYTER_RAD, s=100)
    plt.errorbar(fp_candidates['period_ratio'], fp_candidates['a/R'], yerr=fp_candidates['a/R_err'],
                 xerr=fp_candidates['P_err'], fmt='none', capsize=5, c='gray', alpha=0.5)
    plt.vlines(1, fp_candidates['a/R'].min(), fp_candidates['a/R'].max(), linestyles='--', color='gray')
    for i, txt in enumerate(fp_candidates['kepler_name']):
        print(txt, fp_candidates.iloc[i]['Teff'], fp_candidates.iloc[i]['a/R'],
              fp_candidates.iloc[i]['koi_prad'] / JUPYTER_RAD, fp_candidates.iloc[i]['period_ratio'],
              fp_candidates.iloc[i]['b_prob'])
        plt.text(fp_candidates.iloc[i]['period_ratio'] + 0.5, fp_candidates.iloc[i]['a/R'],
                 txt, fontsize=18, ha='left', va='bottom')
    plt.xlabel(r"$\frac{P_{orb}}{P_{rot}}$")
    plt.ylabel(r"$\frac{a}{R_\star}$")
    plt.colorbar(label=r"$\frac{R_{planet}}{R_J}$")
    plt.savefig('imgs/fp_candidates.png')
    plt.show()

    # plt.figure(figsize=(16,12))
    # plt.scatter(fp_candidates['period_ratio'], fp_candidates['koi_prad'] / JUPYTER_RAD, s=100)
    # plt.xlabel(r"$\frac{P_{orb}}{P_{rot}}$")
    # plt.ylabel(r"$\frac{R_{planet}}{R_J}$")
    # plt.show()

    plt.scatter(circum_candidates[xval], circum_candidates['period_ratio'], c=circum_candidates['Teff'], s=100)
    plt.ylabel(r"$\frac{P_{orb}}{P_{rot}}$")
    plt.colorbar(label='confidence')
    plt.show()
    return circum_candidates, fp_candidates


def plot_scatter(df, df_highlight, x_val='Teff', y_val='kmag'):
    # Correct calculation of 'y_val'
    if x_val == 'predicted period':
        df_highlight['x_val'] = df_highlight.apply(
            lambda x: x['P_rot'] if pd.notna(x['P_rot']) else x['predicted period'], axis=1
        )
    else:
        df_highlight['x_val'] = df_highlight[x_val]

    # Filter eclipsing binaries
    df_binaries = df[df['eb'] == True]

    # Hexbin plot for the entire dataset
    plt.hexbin(df[x_val], df[y_val], mincnt=1, alpha=0.3)

    # Scatter plot for binaries
    plt.scatter(df_binaries[x_val], df_binaries[y_val], alpha=0.3, label='Eclipsing Binaries')

    # Highlight points with text annotations
    for _, row in df_highlight.iterrows():
        if pd.notna(row['kmag']):
            plt.scatter(row['x_val'], row[y_val], s=100, color='red')  # Highlight point
            plt.text(row['x_val'], row[y_val], row['kepler_name'], fontsize=16, color='black')  # Add text

    # Labels and save plot
    plt.xlabel(f'{x_val}')
    plt.ylabel(f'{y_val}')
    if x_val == 'Teff':
        plt.gca().invert_xaxis()
    if y_val == 'kmag':
        plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(f'imgs/{x_val}_vs_{y_val}.png')
    plt.show()

    # Additional scatter plot for highlights only
    plt.scatter(df_highlight['x_val'], df_highlight[y_val])
    plt.show()

def plot_groups_ratio(df, short_p=7, mag_cutoff=0.6):
    df_short = df[df['predicted period'] < short_p]
    df_short_high = df_short[df_short['kmag_abs_detrended'] > mag_cutoff]
    df_short_low = df_short[df_short['kmag_abs_detrended'] < mag_cutoff]
    bins = np.linspace(df['Teff'].min(), df['Teff'].max(), 41)  # 40 bins
    hist_high, _ = np.histogram(df_short_high['Teff'], bins=bins)
    hist_low, _ = np.histogram(df_short_low['Teff'], bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ratio = np.divide(hist_high, hist_low, out=np.zeros_like(hist_low, dtype=float), where=hist_high != 0)
    plt.scatter(bin_centers, ratio, label='Ratio (Low/High)', color='black')
    plt.xlabel('Teff')
    plt.ylabel('Ratio (Low/High)')
    plt.axhline(1, color='gray', linestyle='--', linewidth=1, label='Ratio = 1')
    plt.legend()
    plt.grid(True)
    plt.savefig('imgs/teff_ratio_plot.png')
    plt.close()

    # Avoid division by zero
    plt.hist(df_short_high['Teff'], bins=40, histtype='step', linewidth=3)
    plt.hist(df_short_low['Teff'], bins=40, histtype='step', linewidth=3)
    plt.xlabel('Teff')
    plt.ylabel('Counts')
    plt.savefig('imgs/groups_teff_hist.png')
    plt.close()


def scatter_kmag(df, simonian, ebs, line_of_best_fit, color_col=None):
    teffs = [3000,4000,5200,5500,6000,6200]
    for p in range(7,2,-1):
        fig, axis = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        axis = axis.ravel()
        for i, teff in enumerate(teffs):
            if i < len(teffs) - 1:
                reduced_df = df[(df['Teff'] >= teff) & (df['Teff'] < teffs[i+1])]
                title = rf'{teff} $\leq Teff$ < {teffs[i+1]}'
            else:
                reduced_df = df[df['Teff'] >= teff]
                title = rf'{teff} $\leq Teff$'
            df_short = reduced_df[reduced_df['predicted period'] < p]
            df_long = reduced_df[reduced_df['predicted period'] > 10]
            axis[i].hist(df_short['kmag_abs'], histtype='step', cumulative=True,
                         linewidth=3, density=True, label=rf'$P$<{p} Days {len(df_short)} samples')
            axis[i].hist(df_long['kmag_abs'], histtype='step', cumulative=True,
                         linewidth=3, density=True, label=rf'$P$>10 Days {len(df_long)} samples')
            ks_test_res, ks_test_p = ks_2samp(df_short['kmag_abs'], df_long['kmag_abs'])
            axis[i].set_title(title + f' P-value: {ks_test_p:.4e}', fontsize=16)
            axis[i].legend(loc='best', fontsize=16)
        fig.supxlabel(r'$M_{ks}$')
        fig.supylabel(r'$CDF$')
        plt.tight_layout()
        plt.savefig(f'imgs/short_long_kmag_p_{p}.png')
        plt.close()

    df_long = df[df['predicted period'] > 10]
    df_short = df[df['predicted period'] < 7]
    teff_kmag(df_short, simonian, ebs)
    exit()
    # teff_kmag_scatter_detrend(color_col, df_short, ebs, line_of_best_fit, simonian)


def get_magnitude_stats(df, data_col, teff_thresh=5450):
    df_cool = df[df['Teff'] < teff_thresh]
    df_cool_low = df_cool[df_cool[data_col] < 0]
    df_hot = df[df['Teff'] > teff_thresh]
    df_hot_low = df_hot[df_hot[data_col] < 0]
    cool_ratio = len(df_cool_low) / len(df_cool) if len(df_cool) else np.nan
    hot_ratio = len(df_hot_low) / len(df_hot) if len(df_hot) else np.nan
    return cool_ratio, hot_ratio

def plot_tables(df_short, simonian=None, other=None,
               ax=None, fig=None, full_df=None,
                color_col=None, hline=None, vline=None,
              x_col='Teff', data_col='kmag_diff',
                line_of_best_fit=None, name='ebs'):
    show=False
    if ax is None:
        show=True
        fig, ax = plt.subplots()
    # if color_col is None:
    if full_df is not None:
        ax.hexbin(full_df[x_col], full_df[data_col], mincnt=1, cmap='autumn')
    if other is None and color_col is not None:
        hb1 = ax.scatter(df_short[x_col], df_short[data_col], c=df_short[color_col])
        color_label = color_col[::]
    else:
        if simonian is not None:
            ax.scatter(simonian[x_col], simonian[data_col], color='pink', marker='^')
        hb1 = ax.hexbin(df_short[x_col], df_short[data_col], mincnt=1, cmap='YlGnBu', gridsize=100)
        color_label = 'Density'
        if other is not None:
            if color_col is None:
                ax.scatter(other[x_col], other[data_col], color='brown', s=100, marker='*', alpha=1)
            else:
                hb1 = ax.scatter(other[x_col], other[data_col], c=other[color_col],s=100, cmap='YlGnBu', marker='*')
                color_label = color_col[::]
    if line_of_best_fit is not None:
        x = np.sort(df_short[x_col])
        ax.plot(x, line_of_best_fit(x), c='r')
    min_x, max_x = df_short[x_col].min(), df_short[x_col].max()
    max_y, min_y = df_short[data_col].min(), df_short[data_col].max()
    if hline is not None:
        for line in hline:
            ax.hlines(line, max_x, min_x, color='black', linestyle='--')
    if vline is not None:
        for line in vline:
            ax.vlines(line, min_y, max_y, color='black', linestyle='--')
    if show:
        fig.colorbar(hb1, ax=ax, label=color_label)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_ylabel(r'$\Delta K_{iso}$')
        ax.set_xlabel(r'$T_{eff} (K)$')
    # ax.set_xlim(max_x, min_x)
        plt.savefig(f"../binaries/{x_col}_kmag_diff_{name}.png")
        plt.show()


def teff_kmag_scatter_detrend(df_short, ebs, simonian ,
                              line_of_best_fit, data_col='kmag_abs',
                              color_col=None,
                              hline=0.6):
    fig, ax = plt.subplots(1, 2)
    if color_col is None:
        hb1 = ax[0].hexbin(df_short['Teff'], df_short[data_col], mincnt=1, cmap='YlGnBu')
        color_label = 'Density'
    else:
        hb1 = ax[0].scatter(df_short['Teff'], df_short[data_col], c=df_short[color_col])
        color_label = color_col[::]
    # ebs1 = ebs[ebs[data_col] > 0]
    # ebs2 = ebs[(ebs[data_col] < 0) & (ebs['Teff'] > 6000) & (ebs[data_col] > -0.3)]
    # ebs3 = ebs[(ebs[data_col] < 0) & (ebs['Teff'] < 6000)]
    # ebds_groups = [ebs1, ebs2, ebs3]
    # colors = ['brown', 'blue', 'magenta']
    fig.colorbar(hb1, ax=ax[0], label=color_label)
    ax[0].scatter(simonian['Teff'], simonian[data_col], color='pink', marker='^')
    ax[0].plot(df_short['Teff'], line_of_best_fit(df_short['Teff']), color='red', linestyle='--')
    ax[0].scatter(ebs['Teff'], ebs[data_col], color='brown', marker='*')
    ax[0].invert_yaxis()
    ax[0].invert_xaxis()
    ax[0].set_ylabel(r'$M_{ks} - M_{ks}(MIST; 1Gyr)$')
    ax[0].set_xlabel(r'$T_{eff} (K)$')
    ax[0].set_xlim(7000, 3000)
    if color_col is None:
        hb2 = ax[1].hexbin(df_short['Teff'], df_short[f'{data_col}_detrended'], mincnt=1, cmap='YlGnBu')
        color_label = 'Density'
    else:
        hb2 = ax[1].scatter(df_short['Teff'], df_short[f'{data_col}_detrended'], c=df_short[color_col])
        color_label = color_col
    fig.colorbar(hb2, ax=ax[1], label=color_label)
    ax[1].scatter(simonian['Teff'], simonian[f'{data_col}_detrended'], color='pink', marker='^')
    ax[1].scatter(ebs['Teff'], ebs[f'{data_col}_detrended'], color='brown', marker='*')
    ax[1].hlines(hline, 7000, 3000, colors='black', linestyles='--')
    ax[1].invert_yaxis()
    ax[1].invert_xaxis()
    ax[1].set_ylabel(r'$M_{ks} - M_{ks}(MIST; 1Gyr)$')
    ax[1].set_xlabel(r'$T_{eff} (K)$')
    ax[1].set_xlim(7000, 3000)
    plt.tight_layout()
    plt.savefig('imgs/kmag_scatter_detrend.png')
    plt.close()

def lithium_ages():
    df_full = Table.read("tables/bouma2024.txt", encoding='utf-8', format='ascii.cds').to_pandas()
    df = Table.read("tables/bouma2024_planets.txt", encoding='utf-8', format='ascii.cds').to_pandas()
    df_li_full = df.dropna(subset=['tLi'])
    print(len(df_li_full))
    plt.hist(df_li_full['Teff'], histtype='step', linewidth=4, density=True)
    plt.hist(df['Teff'], histtype='step', linewidth=4, density=True)
    plt.hist(df_full['Teff'], histtype='step', linewidth=4, density=True)
    plt.xlabel('$T_{eff}$')
    plt.ylabel('Density')
    plt.savefig('imgs/lithium_ages_full.png')
    plt.show()

    cat = pd.read_csv('tables/berger_catalog.csv')
    df = df.merge(cat[['KID', 'Rstar', 'Lstar', 'E_Rstar', 'e_Rstar']], right_on='KID', left_on='KIC', how='left')

    df[['Prot', 'e_Prot']] = df[['Prot', 'e_Prot']].astype(np.float64)
    # df = get_vsini(df)
    poly_coeff= np.load('imgs/tables/predicted period_cutoff_best_fits.npy').astype(np.float64)
    poly = np.poly1d(poly_coeff)
    df_reduced = df.dropna(subset=['Teff','tLi', 'tGyro'])
    df_reduced['gyro_valid'] = df_reduced.apply(lambda row: cutoff_threshold_pandas(row, poly, 'Prot'), axis=1)

    xs = df_reduced['Teff'].sort_values()
    ys = poly(xs)
    ys[xs < 4800] = np.nan
    plt.scatter(df_reduced['Teff'], df_reduced['Prot'])
    plt.errorbar(df_reduced['Teff'], df_reduced['Prot'], yerr=df_reduced['e_Prot'],
                 xerr=df_reduced['e_Teff'], fmt='none', capsize=1, ecolor='dimgray', alpha=0.7)
    plt.plot(xs, ys, color='red')
    plt.xlabel('$T_{eff}$')
    plt.ylabel('$P_{rot}$ (Days)')
    plt.savefig('imgs/lithium_vsini.png')
    plt.show()

    df['hot'] = df['Teff'] > 5200
    df['C'] = df['Consist'] == 'Yes'
    df_li = df.dropna(subset=['tLi', 'tGyro'])
    print("number of lithium samples with gyro: ", len(df_reduced))
    plt.hist(df_reduced['Teff'])
    plt.xlabel(r'$T_{eff} (K)$')
    plt.ylabel('Count')
    plt.savefig('imgs/lithium_ages_teff_hist.png')
    plt.show()

    errs_li = df_reduced[['E_tLi', 'e_tLi']].T
    errs_gyro = df_reduced[['E_tGyro', 'e_tGyro']].T
    plt.scatter(df_reduced['tGyro'], df_reduced['tLi'],
                         c=df_reduced['gyro_valid'], s=100)
    plt.errorbar(df_reduced['tGyro'], df_reduced['tLi'], yerr=errs_li,
            xerr=errs_gyro, fmt='none', color='gray', alpha=0.3)
    plt.plot(df_reduced['tGyro'], df_reduced['tGyro'], c='r')
    plt.show()

    # fig, ax = plt.subplots(1, 2)
    # sc1 = ax[0].scatter(df_li['tGyro'], df_li['tLi'],
    #                      c=df_li['C'], label=r'$T_{eff} > 5250$', s=100)
    # ax[0].errorbar(df_li['tGyro'], df_li['tLi'], yerr=errs_li,
    #                xerr=errs_gyro,fmt='none', color='gray', alpha=0.3)
    #
    # ax[0].plot(df_li['tGyro'], df_li['tGyro'], color='r')
    # sc2 = ax[1].scatter(df_li['tGyro'], df_li['tLi'], c=df_li['C'], label=r'$T_{eff} > 5250$', s=100)
    # ax[1].errorbar(df_li['tGyro'], df_li['tLi'], yerr=errs_li,
    #                xerr=errs_gyro, fmt='none', color='gray', alpha=0.3)
    # ax[1].plot(df_li['tGyro'], df_li['tGyro'], color='r')
    # ax[1].set_xlim(0,1000)
    # fig.colorbar(sc1, ax=ax[0], label=r'$T_{eff}$')
    # fig.colorbar(sc2, ax=ax[1], label=r'$T_{eff}$')
    # fig.supxlabel('Gyro age (Myr)')
    # fig.supylabel('Li age (Myr)')
    # plt.tight_layout()
    # plt.savefig('imgs/lithium_ages.png')
    # plt.show()

    # plt.scatter(df_li['tLi'], df_li['Prot'])
    # plt.xlabel('Li age (Myr)')
    # plt.ylabel('Prot')
    # plt.show()


def get_vsini(df, p_val='Prot'):
    df['vsini'] = 2 * np.pi * df['Rstar'] * cgs.R_sun / (df[p_val] * 24 * 60 * 60) * 1e-5
    delta_r = (df['E_Rstar'] - df['e_Rstar']) / 2 * cgs.R_sun * 1e-5
    err_r = 2 * np.pi / (df[p_val] * 24 * 60 * 60) * delta_r
    if f'e_{p_val}' in df.columns:
        err_p = - df['e_Prot'] * 2 * np.pi * df['Rstar'] * cgs.R_sun / ((df['Prot'] * 24 * 60 * 60) ** 2) * 1e-5
    else:
        err_p = 0
    df['e_vsini'] = np.sqrt(err_r ** 2 + err_p ** 2)
    return df


def scatter_ages(df, color_col='Teff'):
    df = df.dropna(subset=['age_bouma24_gyro', 'age_angus23'])
    df_long = df[df['predicted period'] > 10]
    df_short = df[df['predicted period'] < 7]
    df_short_high = df_short[df_short['kmag_diff'] > 0]
    df_short_low = df_short[df_short['kmag_diff'] < 0]
    print("minimum Teff for high group: ", df_short_high['Teff'].min())


    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(40,22), sharex=True, sharey=True)

    sc1 = axs[0].scatter(df_short_high['age_bouma24_gyro'], df_short_high['age_angus23'],
                         c=df_short_high[color_col],
                        )
    axs[0].plot(df_short_high['age_bouma24_gyro'], df_short_high['age_bouma24_gyro'], color='r', linestyle='--')
    fig.colorbar(sc1, ax=axs[0], label=color_col)
    axs[0].set_xlabel(r'$Age_{Gyro} (Myr)$')
    axs[0].set_ylabel(r'$Age_{Iso} (Myr)$')
    axs[0].set_title('kmag_diff > 0')

    sc2 = axs[1].scatter(df_short_low['age_bouma24_gyro'], df_short_low['age_angus23'],
                         c=df_short_low[color_col],
                        )
    axs[1].plot(df_short_low['age_bouma24_gyro'], df_short_low['age_bouma24_gyro'], color='r', linestyle='--')
    fig.colorbar(sc2, ax=axs[1], label=color_col)
    axs[1].set_xlabel(r'$Age_{Gyro} (Myr)$')
    axs[1].set_ylabel(r'$Age_{Iso} (Myr)$')
    axs[1].set_title('kmag_diff < 0')


    sc3 = axs[2].scatter(df_short['age_bouma24_li'], df_short['age_angus23'],
                         c=df_short[color_col],
                         label=r'Li')
    axs[2].legend(loc='best')
    axs[2].plot(df_short['age_bouma24_gyro'], df_short['age_bouma24_gyro'], color='r', linestyle='--')
    fig.colorbar(sc3, ax=axs[2], label=color_col)
    axs[2].set_xlabel(r'$Age_{Li} (Myr)$')
    axs[2].set_ylabel(r'$Age_{Iso} (Myr)$')
    plt.tight_layout()
    plt.savefig('imgs/iso_gyro_compare')
    plt.show()

def fit_kmag_mist(df, plot=False):
    df = df.dropna(subset=['kmag_diff', 'Teff'])
    df = df[~df[['Teff', 'kmag_diff']].isin([np.inf, -np.inf]).any(axis=1)]
    df = df[df['kmag_diff'].abs() < 1]
    df_hot = df[df['Teff'] > 5000]
    df_cold = df[df['Teff'] < 5000]
    coeffs_hot = np.polyfit(df_hot['Teff'], df_hot['kmag_diff'], 1)
    line_hot = np.poly1d(coeffs_hot)
    coeffs_cold = np.polyfit(df_cold['Teff'], df_cold['kmag_diff'], 1)
    line_cold = np.poly1d(coeffs_cold)
    if plot:
        plt.scatter(df['Teff'], df['kmag_diff'], color='dimgray')
        plt.plot(df_hot['Teff'], line_hot(df_hot['Teff']), color='r')
        plt.plot(df_cold['Teff'], line_cold(df_cold['Teff']), color='r')
        plt.savefig('imgs/kmag_fit.png')
        plt.show()
    return line_hot, line_cold


def calculate_rmse_for_threshold(df, threshold):
    """Calculate combined RMSE for a given temperature threshold"""
    df_hot = df[df['Teff'] > threshold]
    df_cold = df[df['Teff'] <= threshold]

    # Skip if either segment has too few points
    if len(df_hot) < 2 or len(df_cold) < 2:
        return float('inf')

    # Fit lines
    coeffs_hot = np.polyfit(df_hot['Teff'], df_hot['kmag_diff'], 1)
    line_hot = np.poly1d(coeffs_hot)
    coeffs_cold = np.polyfit(df_cold['Teff'], df_cold['kmag_diff'], 1)
    line_cold = np.poly1d(coeffs_cold)

    # Calculate RMSE for both segments
    rmse_hot = sqrt(mean_squared_error(df_hot['kmag_diff'],
                                       line_hot(df_hot['Teff'])))
    rmse_cold = sqrt(mean_squared_error(df_cold['kmag_diff'],
                                        line_cold(df_cold['Teff'])))

    # Return weighted average RMSE
    n_total = len(df)
    weighted_rmse = (rmse_hot * len(df_hot) + rmse_cold * len(df_cold)) / n_total
    return weighted_rmse


def fit_kmag_mist_optimized(df, plot=False, optimize=True):
    """Fit two lines with optimized temperature threshold"""
    # Data preprocessing
    df = df.dropna(subset=['kmag_diff', 'Teff'])
    df = df[~df[['Teff', 'kmag_diff']].isin([np.inf, -np.inf]).any(axis=1)]

    print("number of samples in kmag fit: ", len(df))
    if plot:
        plt.scatter(df['Teff'], df['kmag_diff'], color='dimgray')
        plt.xlabel('Teff (K)')
        plt.ylabel(r'$\Delta K_{iso}$')
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.savefig('imgs/kmag_before_fit.png')
        plt.show()

        plt.scatter(df['FeH'], df['kmag_diff'], color='dimgray')
        plt.xlabel('FeH')
        plt.ylabel(r'$\Delta K_{iso}$')
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        # plt.savefig('imgs/kmag_before_fit.png')
        plt.show()
    df = df[(df['kmag_diff'] > -0.75) & (df['kmag_diff'] < 0.75)]

    if optimize:
        temp_min = 4000
        temp_max = 5700
        thresholds = np.arange(temp_min , temp_max , 100)
        rmse_values = [calculate_rmse_for_threshold(df, threshold)
                       for threshold in thresholds]

        rmse_diff = np.diff(rmse_values)
        optimal_threshold = thresholds[np.argmin(rmse_diff) + 1]

        # optimal_threshold = thresholds[np.argmin(rmse_values)]
    else:
        optimal_threshold = 0

    # Fit final lines using optimal threshold
    df_hot = df[df['Teff'] > optimal_threshold]
    df_cold = df[df['Teff'] <= optimal_threshold]
    coeffs_hot = np.polyfit(df_hot['Teff'], df_hot['kmag_diff'], 1)
    line_hot = np.poly1d(coeffs_hot)
    if optimize:
        coeffs_cold = np.polyfit(df_cold['Teff'], df_cold['kmag_diff'], 1)
        line_cold = np.poly1d(coeffs_cold)
    else:
        line_cold = np.nan

    if plot:
        plt.figure()
        plt.scatter(df['Teff'], df['kmag_diff'], color='dimgray', alpha=0.5)
        plt.plot(df_hot['Teff'], line_hot(df_hot['Teff']), color='r',
                 label=r'($T_{eff} > $ %d K)' % optimal_threshold)
        # plt.plot(df_cold['Teff'], line_cold(df_cold['Teff']), color='b',
        #          label=f'Cold (T ≤ {optimal_threshold:.0f}K)')
        if optimize:
            plt.axvline(x=optimal_threshold, color='k', linestyle='--', alpha=0.3,
                        label='Threshold')
        plt.xlabel('Teff (K)')
        plt.ylabel(r'$\Delta K_{iso}$')
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.savefig('imgs/kmag_fit_optimized.png')
        plt.show()

        if optimize:
            # Plot RMSE vs threshold
            plt.figure()
            plt.plot(thresholds, rmse_values)
            plt.axvline(x=optimal_threshold, color='r', linestyle='--',
                        label=f'Optimal: {optimal_threshold:.0f}K')
            plt.xlabel('Temperature Threshold (K)')
            plt.ylabel('RMSE')
            plt.legend()
            plt.show()

    return line_hot, line_cold, optimal_threshold

def plot_period_kmag_diff(df, prot_thresh=20, save_name=''):
    df = df.dropna(subset=['kmag_diff'])
    df = df[df['predicted period'] <= prot_thresh]
    # Define bins and categorize the 'predicted period' into these bins
    bins = np.linspace(df['predicted period'].min(), df['predicted period'].max(), 20)  # Adjust bin count as needed
    df['prot_bin'] = pd.cut(df['predicted period'], bins)

    # Calculate bin midpoints for plotting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    teffs = [3900, 5200, 5700, 6000]
    for i, teff in enumerate(teffs):
        if i < len(teffs) - 1:
            df_reduced = df[(df['Teff'] >= teff) & (df['Teff'] < teffs[i+1])]
            label = f'{teff} <= Teff < {teffs[i+1]}'
        else:
            df_reduced = df[(df['Teff'] > teff)]
            label = f'{teff} <= Teff '


        # Group by the bins and calculate mean and std for 'kmag_diff'
        stats = df_reduced.groupby('prot_bin')['kmag_diff'].agg(['mean', 'std']).reset_index()

        print(teff, len(stats), len(df_reduced))

        stats['bin_center'] = stats['prot_bin'].apply(lambda b: 0.5 * (b.left + b.right))
        plt.errorbar(stats['bin_center'], stats['mean'], yerr=stats['std'], fmt='o',
                     markersize=10, capsize=2, label=label)
    plt.xlabel('Predicted Period (Binned)')
    plt.ylabel('Mean Kmag_diff')
    plt.title('Kmag_diff vs. Predicted Period')
    plt.legend()
    plt.grid()
    plt.savefig(f'imgs/p_bin_vs_kmag_diff_{save_name}.png')
    plt.show()

def gaia_binaries():
    gaia_kepler_wide = get_kepler_gaia_wide_binary_samples()
    # godoy = pd.read_csv('imgs/tables/GodoyRivera25_TableA1.csv')
    # gaia_kepler_nss = godoy[godoy['flag_NSS']]
    # gaia_kepler_nss.rename(columns={'KIC': 'KID'}, inplace=True)

    gaia_nss = pd.read_csv('../lightPred/tables/gaia_nss.csv')
    gaia_nss.rename(columns={'SOURCE_ID':'source_id'}, inplace=True)
    kepler_gaia_table = Table.read('tables/kepler_dr3_1arcsec.fits', format='fits').to_pandas()
    gaia_kepler_nss = kepler_gaia_table.merge(gaia_nss, on='source_id')
    gaia_kepler_nss.rename(columns={'kepid': 'KID'}, inplace=True)
    gaia_kepler_nss = get_mag_data(gaia_kepler_nss, mist_path=MIST_PATH)
    gaia_kepler_nss['GRCOLOR_abs'] = gaia_kepler_nss['GMAG_abs'] - gaia_kepler_nss['RMAG_abs']
    gaia_kepler_nss['main_seq'] = gaia_kepler_nss.apply(giant_cond, axis=1)
    # gaia_kepler_nss = gaia_kepler_nss[gaia_kepler_nss['main_seq'] == True]

    gaia_kepler_wide = get_kepler_gaia_primary_binaries()
    gaia_kepler_wide.rename(columns={'kepid': 'KID'}, inplace=True)
    gaia_kepler_wide = get_mag_data(gaia_kepler_wide,
                                     mist_path=MIST_PATH)
    gaia_kepler_wide['theta_arcsec'] = gaia_kepler_wide['sep_AU'] / gaia_kepler_wide['Dist']
    gaia_kepler_wide['GRCOLOR_abs'] = gaia_kepler_wide['GMAG_abs'] - gaia_kepler_wide['RMAG_abs']
    # gaia_kepler_nss['msi'] = gaia_kepler_nss.apply(lambda x: Teff_mass_mainsequence(x['Teff']))
    # gaia_kepler_nss['m_diff'] = gaia_kepler_nss['msi'], gaia_nss['Mstar']


    return gaia_kepler_nss, gaia_kepler_wide


def cluster_binaries():
    clusters = Table.read('tables/bouma2023_calibration.txt', encoding='utf-8', format='ascii.cds').to_pandas()
    # clusters = clusters[clusters['Binary'] == 'True']
    clusters['cluster_age'] = clusters.apply(lambda row: CLUSTERS[row['Cluster']], axis=1)
    kepler_gaia_table = Table.read('tables/kepler_dr3_1arcsec.fits', format='fits').to_pandas()
    clusters_kepler = kepler_gaia_table.merge(clusters, left_on='source_id', right_on='DR3')
    clusters_kepler.rename(columns={'kepid': 'KID'}, inplace=True)
    clusters_kepler = get_mag_data(clusters_kepler,
                            mist_path=MIST_PATH)
    clusters_kepler['GRCOLOR_abs'] = clusters_kepler['GMAG_abs'] - clusters_kepler['RMAG_abs']
    return clusters_kepler, clusters



def get_semi_major_axis(df, p_name='period', a_name='a', comp_mass=1, p_err_name='period_error'):
    p_sec = df[p_name] * 24 * 3600
    M_tot = (1 + comp_mass) * cgs.M_sun * df['Mstar']
    nom = M_tot * cgs.G * p_sec ** 2
    denom = 4 * np.pi ** 2 # factor 2 since we assume equal mass binary
    df[a_name] = (nom / denom) ** (1/3)
    df[a_name] /= cgs.au
    M_err = (df['E_Mstar'] - df['e_Mstar']) / 2 * (1 + comp_mass) * cgs.M_sun
    p_err = df[p_err_name]
    err_a = (1/3 * (cgs.G / (4 * np.pi ** 2)) ** (1/3) *
             ( p_sec ** 4 * M_err ** 2 + (2 * p_sec * M_tot * p_err) ** 2 ) ** 0.5)
    df[f'{a_name}_err'] = err_a / cgs.au
    return df

def get_pericenter(df, e_name='eccentricity'):
    df['r_p'] = df['a'] * (1 - df[e_name])
    return df

def plot_binary_props(df, binary_df, cat_name, p_name='period', p_err_name='period_error', e_name='eccentricity'):
    print("ploting binary properties for {}".format(cat_name))
    # gyro_ages = pd.read_csv('tables/gyrointerp_lightPred_doubles.csv')
    binary_df = binary_df.merge(df, on='KID', suffixes=('', '_'), how='left')
    # binary_df = binary_df.merge(gyro_ages, on='KID', how='left')
    binary_df = get_semi_major_axis(binary_df, 'predicted period', p_err_name=p_err_name, a_name='a_in')
    binary_df['triple_stable'] = binary_df.apply(lambda x: triple_stability(x['a'], x['a_in'], x[e_name]), axis=1)
    binary_df['period_ratio'] = binary_df[p_name] / binary_df['predicted period']
    binary_df['period_ratio_norm'] = binary_df['period_ratio'] * 10 ** binary_df['kmag_diff']
    binary_df['t_conv'] = binary_df.apply(lambda x: tau_circ_convective(x[p_name], x['Mstar']), axis=1)
    binary_df['t_rad'] = binary_df.apply(lambda x: tau_circ_radiative(x[p_name], x['Mstar']), axis=1)
    binary_df['t_r_p'] = ((binary_df['r_p'] * cgs.au) ** (3/2) / ((cgs.G * (binary_df['Mstar'] * cgs.M_sun)) ** 0.5)
                          / (24*3600*365))
    binary_df['tidal_torque'] = binary_df.apply(lambda x: np.log(np.abs(tidal_torque(x['predicted period'], x[p_name],
                                                                       x['Mstar'], x['Rstar'], x['r_p'], tau_diss=1))), axis=1)
    binary_df['theta_arcsec'] = binary_df['a'] / binary_df['Dist']

    specials = binary_df[(binary_df['period_ratio'] > 100) & (binary_df['r_p'] < 0.6)]
    print("number of binaries-df merge: ", len(binary_df[~binary_df['predicted period'].isna()]))
    binary_short = binary_df[binary_df[p_name] < 100].dropna(subset=['r_p', 'period_ratio'])

    scatter_synchronization_stability(binary_short, cat_name, e_name, p_name='period', y_name='predicted period',
                                      xlabel=r'$P_{orb}$', ylabel=r'$P_{rot}$')
    scatter_synchronization_stability(binary_df, cat_name, e_name, xlabel=r'$P_{orb} / P_{rot}$', ylabel='$r_p$ (AU)',
                                      p_name='period_ratio', y_name='r_p',
                                      logy=True, logx=True,
                                      plot_sync_line=False, suffix='all')


    binary_df = binary_df.dropna(subset=['r_p', 'period_ratio'])
    x = binary_df['period_ratio'].values
    y = binary_df['r_p'].values
    fitted_x, fitted_y, popt = fit_envelop(x,y, lower=True)
    fitted_x = fitted_x[fitted_y > 0]
    fitted_y = fitted_y[fitted_y > 0]

    fitted_x_up, fitted_y_up, popt_up = fit_envelop(x, y, lower=False)
    fitted_x_up = fitted_x_up[fitted_y_up > 0]
    fitted_y_up = fitted_y_up[fitted_y_up > 0]

    print("fit parameters: ", popt, "\n", popt_up)

    # plt.figure(figsize=(20, 12))
    markers = {True: '*', False: 'o'}  # Circle for True, Triangle for False
    fig, ax = plt.subplots()
    scatter_plots = []
    for key, group in binary_df.groupby('triple_stable'):
        scatter = ax.scatter(
            group['period_ratio'],
            group['r_p'],
            c=group[e_name],
            cmap='viridis',
            s=100,
            marker=markers[key],
            edgecolors='black',
            label='Stable' if key else 'Unstable'
        )
        scatter_plots.append(scatter)
    # plt.scatter(binary_df['period_ratio'], binary_df['r_p'], c=binary_df[e_name], s=100, edgecolors='black')
    plt.plot(fitted_x, fitted_y, color='red', label=f'Fit: $1-(x^{{1/{popt[0]:.2f}}})$')
    # plt.plot(fitted_x_up, fitted_y_up, color='red', label=f'Fit: $1-(x^{{1/{popt_up[0]:.2f}}})$')
    plt.semilogy()
    plt.semilogx()
    # plt.vlines(x=1, ymin=binary_df['r_p'].min(), ymax=binary_df['r_p'].max(), colors='gray', linestyles='--')
    plt.xlabel(r'$P_{orb}$/$P_{rot}$')
    plt.ylabel(r'$r_p$ (au)')
    cbar = plt.colorbar(scatter_plots[0], ax=ax, label='Eccentricity')
    plt.savefig(f'imgs/{cat_name}_p_ratio_vs_r_p_short_fit')
    plt.show()

    # binary_ages = binary_df[~binary_df['age'].isna()]
    # plt.scatter(binary_ages['period_ratio'], binary_ages['t_conv'] / (binary_ages['age'] * 1e6),
    #             c=binary_ages[e_name], s=100, edgecolors='black')
    # plt.xlabel(r'$P_{orb}$/$P_{rot}$')
    # plt.ylabel(r'$\frac{\tau_{circ}(conv)}{age_{gyro}}$ (yr)')
    # plt.colorbar(label='$Eccentricity$')
    # plt.hlines(1, binary_ages['period_ratio'].min(), binary_ages['period_ratio'].max(), colors='gray', linestyles='--')
    # plt.vlines(1, (binary_ages['t_conv'] / (binary_ages['age'] * 1e6)).min(),
    #            (binary_ages['t_conv'] / (binary_ages['age'] * 1e6)).max(), colors='gray', linestyles='--')
    # plt.semilogy()
    # plt.xlim(0,10)
    # plt.savefig(f'imgs/{cat_name}_p_ratio_vs_tau_conv_over_age')
    # plt.show()


def scatter_synchronization_stability(binary_short, cat_name,
                            c_name, p_name, y_name,
                            suffix='short', plot_sync_line=True,
                            xlabel='Orbital Period (Days)',
                            ylabel= 'Predicted Period (Days)',
                            clabel='Eccentricity',
                            logx=False, logy=False,):
    markers = {True: '*', False: 'o'}  # Circle for True, Triangle for False
    # Create the scatter plot
    fig, ax = plt.subplots()
    scatter_plots = []
    for key, group in binary_short.groupby('triple_stable'):
        scatter = ax.scatter(
            group[p_name],
            group[y_name],
            c=group[c_name],
            cmap='viridis',
            s=100,
            marker=markers[key],
            edgecolors='black',
            label='Stable' if key else 'Unstable'
        )
        scatter_plots.append(scatter)
    # Add a diagonal reference line
    if plot_sync_line:
        ax.plot(np.arange(40), np.arange(40), c='gray', linestyle='--')
    # Add labels, legend, and colorbar
    if logy:
        plt.semilogy()
    if logx:
        plt.semilogx()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(scatter_plots[0], ax=ax, label=clabel)
    ax.legend()
    # Save and show the plot
    plt.savefig(f'imgs/{cat_name}_{p_name}_vs_{y_name}_e_{suffix}.png')
    plt.show()
    return markers

def scatter_synchronization(binary_short, cat_name,
                            c_name, p_name, y_name,
                            suffix='short', plot_sync_line=True,
                            xlabel='Orbital Period (Days)',
                            ylabel= 'Predicted Period (Days)',
                            clabel= 'Eccenctricity',
                            logy=False,
                            logx=False,
                            fit_power=False):
    binary_short = binary_short.dropna(subset=[p_name, y_name])
    markers = {True: '*', False: 'o'}  # Circle for True, Triangle for False
    # Create the scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        np.log(binary_short[p_name]),
        np.log(binary_short[y_name]),
        c=binary_short[c_name],
        cmap='viridis',
        s=100,
        edgecolors='black',
    )
    # Add a diagonal reference line
    if plot_sync_line:
        ax.plot(np.arange(40), np.arange(40), c='gray', linestyle='--')
    # Add labels, legend, and colorbar
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(scatter, ax=ax, label=clabel)
    ax.legend()
    if logy:
        plt.semilogy()
    if logx:
        plt.semilogx()
    if fit_power:
        coeffs = np.polyfit(np.log(binary_short[p_name]), np.log(binary_short[y_name]), 1)
        print(coeffs)
        poly = np.poly1d(coeffs)
        xs = binary_short[p_name].sort_values()
        ax.plot(np.log(xs), poly(np.log(xs)),c='r')
        ax.legend()
    # Save and show the plot
    plt.savefig(f'imgs/{cat_name}_{p_name}_vs_{y_name}_e_{suffix}.png')
    plt.show()
    return markers

def plot_ebs_prop(df, ebs):
    ets = Table.read('imgs/tables/eclipsing_triples.fit', format='fits').to_pandas()
    all_ets = []
    for i in range(3,10):
        all_ets.append(Table.read(f'imgs/tables/eclipsing_triples_{i}.fit', format='fits').to_pandas())
    all_ets = pd.concat(all_ets, axis=0)
    ebs = ebs.merge(df, on='KID', suffixes=('', '_'), how='left')
    ebs = ebs.merge(ets[['KIC', 'T0','ETV_QTV']], left_on='KID', right_on='KIC', how='left')
    ebs = ebs.merge(all_ets[['KIC', 'P1', 'P2', 'e2']], left_on='KID', right_on='KIC', how='left')
    ebs['ets'] = ~ebs['P2'].isna()  # True if not NaN, False otherwise
    ebs = get_semi_major_axis(ebs, 'predicted period', 'a_in')
    ebs['triple_stable'] = ebs.apply(lambda x: triple_stability(x['a'], x['a_in'], x['e']), axis=1)
    print("number of eb-df merge: ", len(ebs[~ebs['predicted period'].isna()]))
    ebs_short = ebs[ebs['period'] < 100]
    ebs_p = ebs[~ebs['predicted period'].isna()]

    colors = {True: 'blue', False: 'red'}

    for key, group in ebs_short.groupby('ets'):
        plt.scatter(group['period'], group['predicted period'],
                    c=colors[key], label='Eclipsing Triple' if key else 'Eclipsing Binary', s=100)

        # If 'ets' is True, add an additional point and connect with a line
        if key:  # Check if 'ets' is True
            for _, row in group.iterrows():
                plt.scatter(row['P2'], row['predicted period'], c='green', s=100, label='P2 Point' if _ == 0 else "")
                plt.plot([row['period'], row['P2']], [row['predicted period'], row['predicted period']],
                         c='green', linestyle='--')

    # Add a reference line
    plt.plot(np.arange(40), np.arange(40), c='gray', linestyle='--')

    # Add labels, legend, and limits
    plt.xlabel('Orbital Period')
    plt.ylabel('Predicted Period')
    plt.ylim(0, 40)
    plt.legend()

    # Save and show the plot
    plt.savefig('imgs/ebs_p_vs_p')
    plt.show()

    markers = {True: '*', False: 'o'}  # Circle for True, Triangle for False

    # Create the scatter plot
    fig, ax = plt.subplots()
    scatter_plots = []

    for key, group in ebs_short.groupby('triple_stable'):
        scatter = ax.scatter(
            group['period'],
            group['predicted period'],
            c=group['e'],
            cmap='viridis',
            s=100,
            marker=markers[key],
            edgecolors='black',
            label='Stable' if key else 'Unstable'
        )
        scatter_plots.append(scatter)

    # Add a diagonal reference line
    ax.plot(np.arange(40), np.arange(40), c='gray', linestyle='--')

    # Add labels, legend, and colorbar
    ax.set_xlabel('Orbital Period')
    ax.set_ylabel('Predicted Period')
    cbar = plt.colorbar(scatter_plots[0], ax=ax, label='Eccentricity')
    ax.legend()

    # Save and show the plot
    plt.savefig('imgs/ebs_p_vs_p_combined')
    plt.show()


def get_ebs():
    ebs = pd.read_csv('tables/kepler_eb.txt')
    lurie = pd.read_csv('tables/lurie2017.txt', sep=';')
    ebs_full_params = Table.read('imgs/tables/IJspreet2024.fit', format='fits').to_pandas()
    ebs_full_params['KIC'] = ebs_full_params['SimbadName'].str.decode('utf-8').str.extract(r'KIC\s*(\d+)').astype(
        'int64')
    ebs = get_mag_data(ebs,
                  mist_path=MIST_PATH)
    ebs['GRCOLOR_abs'] = ebs['GMAG_abs'] - ebs['RMAG_abs']
    ebs['main_seq'] = ebs.apply(giant_cond, axis=1)
    ebs = ebs[ebs['main_seq'] == True]
    # ebs['msi'] = ebs.apply(lambda row: Teff_mass_mainsequence(row[x]), axis=1)
    # ebs['m_diff'] = ebs['msi'] - ebs['Mstar']
    ebs = ebs.merge(ebs_full_params, left_on='KID', right_on='KIC', how='left')
    ebs = ebs.merge(lurie[['KIC', 'Class']], left_on='KID', right_on='KIC', how='left')
    return ebs

def get_all_tables(df_path, p_thresh, read_csv=False):
    if read_csv:
        df = pd.read_csv('tables/lightpred_cat.csv')
        gaia_nss = pd.read_csv('tables/gaia_nss.csv')
        gaia_wide = pd.read_csv('tables/gaia_wide.csv')
        ebs = pd.read_csv('tables/ebs.csv')
        clusters_b = pd.read_csv('tables/clusters_binaries_kepler.csv')
        all_clusters = pd.read_csv('tables/clusters_binaries.csv')
    else:
        df = create_catalog(df_path, conf=0.86, low_p=3, self_kmag=False)
        df = df.dropna(subset=['kmag_diff', 'Teff', 'kmag_abs'])
        df = df[~df[['Teff', 'kmag_diff']].isin([np.inf, -np.inf]).any(axis=1)]
        gaia_nss, gaia_wide = gaia_binaries()
        gaia_nss = get_semi_major_axis(gaia_nss)
        gaia_nss = get_pericenter(gaia_nss)
        print("number of gaia samples: ", len(gaia_nss))
        ebs = get_ebs()
        ebs = get_semi_major_axis(ebs, p_err_name='period_err')
        ebs = get_pericenter(ebs, e_name='e')
        clusters_b, all_clusters = cluster_binaries()
        print("number of clusters kepler binaries: ", len(clusters_b))

        df.to_csv(('tables/lightpred_cat.csv'), index=False)
        gaia_nss.to_csv('tables/gaia_nss.csv')
        gaia_wide.to_csv('tables/gaia_wide.csv')
        ebs.to_csv('tables/ebs.csv')
        clusters_b.to_csv('tables/clusters_binaries_kepler.csv')
        all_clusters.to_csv('tables/clusters_binaries.csv')

    simonian = pd.read_csv('tables/simonian2019.txt', sep='\t')

    df_short = df[df['predicted period'] <= p_thresh]
    return df, df_short, simonian, gaia_nss, gaia_wide, ebs, clusters_b, all_clusters

def get_kmag_binaries(known_binaries, df, x_col='Teff', data_col='kmag_diff_detrended'):
    fig, ax = plt.subplots(1,2, figsize=(20,12))
    hb1 = ax[0].hexbin(df[x_col], df[data_col], mincnt=1, cmap='YlGnBu')
    bounds = binary_boundary(df[x_col].values)
    ax[0].scatter(df[x_col], bounds, c='r')
    ax[1].hexbin(known_binaries[x_col], known_binaries[data_col], mincnt=1, cmap='YlGnBu')
    bounds = binary_boundary(known_binaries[x_col].values)
    ax[1].scatter(known_binaries[x_col].values, bounds, c='r')

    x_smooth = np.linspace(df[x_col].min(), df[x_col].max(), 1000)
    bounds_smooth = binary_boundary(x_smooth)
    ax[0].fill_between(x_smooth, bounds_smooth, 0,
                       where=(~np.isnan(bounds_smooth)),  # Only shade where boundary is defined
                       alpha=0.2,  # Transparency
                       color='brown',
                       label='Binary Region')
    ax[0].invert_yaxis()
    ax[0].invert_xaxis()
    ax[1].invert_yaxis()
    ax[1].invert_xaxis()
    ax[1].set_ylim(0.5,-0.5)
    ax[0].set_ylim(0.5,-0.5)
    fig.supylabel(r'$\Delta K_{iso}$')
    fig.supxlabel(r'$T_{eff} (K)$')
    plt.tight_layout()
    plt.savefig('imgs/binaries_boundary.png')
    plt.show()

    points = np.column_stack([
        df[x_col].values,
        df[data_col].values
    ])

    points_b = np.column_stack([
        known_binaries[x_col].values,
        known_binaries[data_col].values
    ])

    # Add the new column
    df['in_binary_region'] = is_inside_shape(points)
    known_binaries['in_binary_region'] = is_inside_shape(points_b)
    df_in = df[df['in_binary_region'] == True]
    binaries_in = known_binaries[known_binaries['in_binary_region'] == True]
    color_cols = [c for c in known_binaries.columns if 'MAG_abs' in c]
    plt.hexbin(df[x_col], df[data_col], mincnt=1, cmap='YlGnBu')
    plt.scatter(df_in[x_col], df_in[data_col], c='brown', alpha=0.01)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.show()

    print("number of binaries: ", len(df_in), "number of samples: ", len(df))
    df_in_clean = df_in[['KID', 'Teff', 'kmag_diff_detrended', 'predicted period',
                'mean_period_confidence']].dropna(subset=['kmag_diff_detrended'])

    df_in_clean[['KID','Teff']] = df_in_clean[['KID','Teff']].astype(np.int64)
    df_in_clean[['kmag_diff_detrended', 'predicted period',
                'mean_period_confidence']] = df_in_clean[['kmag_diff_detrended', 'predicted period',
                'mean_period_confidence']].astype(np.float64).round(3)

    df_in_clean.to_csv('imgs/tables/kmag_binaries.csv', index=False)
    return df_in ,binaries_in


def find_linear_envelopes(x, y, low_cutoff=0, up_cutoff=0.8, upper=True):
    """
    Find upper and lower straight lines bounding the data for x < x_cutoff
    """
    # Filter data for x < x_cutoff
    mask = np.logical_and(x < up_cutoff, x > low_cutoff)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Reshape x for sklearn
    X = x_filtered.reshape(-1, 1)

    # Find points for upper envelope
    n_bins = 20
    bins = np.linspace(min(x_filtered), max(x_filtered), n_bins)
    env_x = []
    env_y = []

    for i in range(len(bins) - 1):
        mask = (x_filtered >= bins[i]) & (x_filtered < bins[i + 1])
        if np.sum(mask) > 0:
            if upper == True:
                idx = np.argmax(y_filtered[mask])
            else:
                idx = np.argmin(y_filtered[mask])
            env_x.append(x_filtered[mask][idx])
            env_y.append(y_filtered[mask][idx])

    # Fit upper line
    env_reg = LinearRegression()
    env_reg.fit(np.array(env_x).reshape(-1, 1), env_y)
    return env_reg

def color_diagram(df, mag1, mag2, known_binaries=None):
    df[f'{mag1}-{mag2}_abs'] = df[f'{mag1}MAG_abs'] - df[f'{mag2}MAG_abs']
    fig, ax = plt.subplots()
    sc1 = ax.hexbin(df[f'{mag1}-{mag2}_abs'], df[f'{mag2}MAG_abs'], mincnt=1, cmap='YlGnBu')
    if known_binaries is not None:
        known_binaries[f'{mag1}-{mag2}_abs'] = known_binaries[f'{mag1}MAG_abs'] - known_binaries[f'{mag2}MAG_abs']
        ax.scatter(known_binaries[f'{mag1}-{mag2}_abs'], known_binaries[f'{mag2}MAG_abs'], c='brown', alpha=0.05)
    ax.set_xlabel(f'{mag1}-{mag2} color (mag)')
    ax.set_ylabel(f'Absolute {mag2} magnitude (mag)')
    ax.invert_yaxis()
    plt.savefig(f'imgs/{mag1}_{mag2}_binaries.png')
    plt.show()
def g_r_binaries(known_binaries, df):
    known_binaries['GRCOLOR_abs'] = known_binaries['GMAG_abs'] - known_binaries['RMAG_abs']
    df['GRCOLOR_abs'] = df['GMAG_abs'] - df['RMAG_abs']
    x,y = known_binaries['GRCOLOR_abs'].values, known_binaries['RMAG_abs'].values
    upper_env = find_linear_envelopes(x, y, low_cutoff=0, up_cutoff=0.8)
    lower_env = find_linear_envelopes(x, y, low_cutoff=0.4, up_cutoff=1, upper=False)
    df['upper_env'] = df['GRCOLOR_abs']*upper_env.coef_[0] + upper_env.intercept_
    df['lower_env'] = df['GRCOLOR_abs']*lower_env.coef_[0] + lower_env.intercept_
    df['in_env'] = (df['RMAG_abs'] < df['upper_env']) & (df['RMAG_abs'] > df['lower_env'])

    fig, ax = plt.subplots()
    sc1 = ax.hexbin(df['GRCOLOR_abs'], df['RMAG_abs'], mincnt=1, cmap='YlGnBu')
    ax.scatter(x, y, c='brown', alpha=0.1)
    ax.plot(x, x*lower_env.coef_[0] + lower_env.intercept_, color='gray', linestyle='--')
    ax.plot(x, x*upper_env.coef_[0] + upper_env.intercept_, color='gray', linestyle='--')
    ax.set_xlabel('G-R color (mag)')
    ax.set_ylabel('Absolute R magnitude (mag)')
    ax.invert_yaxis()
    plt.savefig('imgs/g_r_binaries.png')
    plt.show()

    df_in = df[df['in_env']]
    df_in_clean = df_in[['KID', 'Teff', 'kmag_diff', 'GRCOLOR_abs', 'RMAG_abs', 'predicted period',
                         'mean_period_confidence']]

    df_in_clean[['KID', 'Teff']] = df_in_clean[['KID', 'Teff']].astype(np.int64)
    df_in_clean[['kmag_diff', 'GRCOLOR_abs', 'RMAG_abs', 'predicted period',
                 'mean_period_confidence']] = df_in_clean[['kmag_diff', 'GRCOLOR_abs', 'RMAG_abs',
                                                           'predicted period',
                                                           'mean_period_confidence']].astype(np.float64).round(3)

    df_in_clean.to_csv('imgs/tables/g_r_binaries.csv', index=False)
    print("number of g-r binaries : ", len(df_in), 'all samples: ', len(df))
    return df_in

def plot_kepler_cluster(clusters):
    clusters_b = clusters[clusters['Binary']==True]
    clusters_s = clusters[clusters['Binary']==False]
    plt.scatter(clusters_b['Teff'], clusters_b['kmag_diff'], c=clusters_b['cluster_age'], s=100)
    plt.scatter(clusters_s['Teff'], clusters_s['kmag_diff'], c=clusters_s['cluster_age'], marker='*', s=100)
    plt.hlines(0, clusters['Teff'].min(), clusters['Teff'].max(), linestyle='--', color='black')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.colorbar(label='Cluster Age (Myr)')
    plt.show()

def simulate_p_cutoff():
    ps = np.arange(1, 50, 0.1)
    Ms = np.arange(0.1, 2.6, 0.2)
    Ts = np.arange(4800, 6400, 200)
    p_cutoffs = []
    for T in Ts:
        tau_circ = tau_circ_convective_T(ps, T)
        plt.plot(np.log10(ps), tau_circ, label='T={:.2f}'.format(T))
        circ_idx = np.where(tau_circ > 4e7)[0]
        p_cutoff = ps[circ_idx[0]]
        p_cutoffs.append(p_cutoff)
    plt.hlines(4e7, 0, np.log10(50), linestyle='--', color='black')
    plt.semilogy()
    plt.xlabel(r'$\log_{10}P$')
    plt.ylabel(r'$\tau_{circ}$ (yr)')
    plt.legend(fontsize=10)
    plt.savefig('imgs/tau_conv_simulation.png')
    plt.show()
    plt.scatter(Ts, p_cutoffs)
    plt.xlabel('$P_{cutoff}$ (Days)')
    plt.ylabel(r'$\frac{M}{M_{\odot}}$')
    plt.savefig('imgs/p_cutoff_simulation.png')
    plt.show()

def reconstruct_van_heyleen():
    ps = np.arange(1, 50, 0.1)
    Ms_conv = [0.5,0.8,1]
    Ms_rad = [2,5,10]
    for M in Ms_conv:
        tau_circ = tau_circ_convective(ps, M)
        tau_star = 10 ** 10 * (M ** (-2.9))
        plt.plot(np.log10(ps), tau_circ / tau_star, label='M={:.2f}'.format(M))
        pos_idx =  np.where(tau_circ / tau_star > 1)[0]
        p_cross = ps[pos_idx[0]]
        print(M, np.log10(p_cross), (tau_circ / tau_star)[pos_idx[0]])
    for M in Ms_rad:
        if M > 2:
            E_2 = 1e-6
        else:
            E_2 = 1e-7
        tau_circ = tau_circ_radiative(ps, M, E_2=E_2)
        tau_star = 10 ** 10 * (M ** (-2.9))
        plt.plot(np.log10(ps), tau_circ / tau_star, linestyle='--', label='M={:.2f}'.format(M, E_2))
        plt.plot(np.log10(ps), tau_circ / tau_star, label='M={:.2f}'.format(M))
        pos_idx = np.where(tau_circ / tau_star > 1)[0]
        p_cross = ps[pos_idx[0]]
        print(M, np.log10(p_cross), (tau_circ / tau_star)[pos_idx[0]])
    plt.legend()
    plt.semilogy()
    plt.xlabel(r'$p$')
    plt.ylabel(r'$Circ. time \ Stellar lifetime$')
    plt.show()

def plot_percentile_mag(df, n_bins=15):
    df['Teff_bin'] = pd.cut(df['Teff'], bins=n_bins)

    # Step 2: Filter rows where 'kmag_diff' < 0
    filtered_df = df

    # Step 3: Calculate the 25th percentile for each bin
    percentiles = (
        filtered_df.groupby('Teff_bin')['kmag_diff']
        .quantile(0.25)
        .reset_index()
    )

    # Extract bin centers for the scatter plot
    bin_centers = percentiles['Teff_bin'].apply(lambda x: x.mid)

    # Step 4: Scatter plot
    plt.figure()
    plt.plot(bin_centers, percentiles['kmag_diff'], color='brown', label="25th Percentile")
    plt.hexbin(df['Teff'], df['kmag_diff'], mincnt=1, cmap='YlGnBu')
    plt.xlabel('Teff (Bin Centers)')
    plt.ylabel('25th Percentile of kmag_diff')
    plt.title('Scatter Plot of 25th Percentile vs Teff')
    plt.grid()
    plt.legend()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()

def plot_mag_and_mist(df):
    up = df[df['position']=='binary']
    down = df[df['position']=='single']
    middle = df[df['position']=='boundary']
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes[0].hexbin(df['Teff'], df['kmag_abs'], mincnt=1, cmap='viridis')
    axes[1].hexbin(df['Teff'], df['kmag_abs'], mincnt=1, cmap='viridis')

    hist, xedges, yedges = np.histogram2d(df['Teff'], df['Kmag_MIST'], bins=50)

    # Get the mesh grid for contour plotting
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)

    # Plot contours
    contours = axes[1].contour(X, Y, hist.T, levels=4,
                          colors='black', linewidths=2)

    hist, xedges, yedges = np.histogram2d(middle['Teff'], middle['kmag_abs'], bins=50)

    # Get the mesh grid for contour plotting
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)

    # Plot contours
    contours = axes[1].contour(X, Y, hist.T, levels=1,
                               colors='red', linewidths=2)


    # axes[1].hexbin(df['Teff'], df['Kmag_MIST'], mincnt=1, cmap='magma')
    # axes[1].scatter(middle['Teff'], middle['kmag_abs'], c='brown', alpha=0.1)
    # plt.scatter(down['Teff'], down['kmag_abs'], c='limegreen', alpha=0.01)
    # plt.scatter(middle['Teff'], middle['kmag_abs'], c='pink', alpha=0.01)

    axes[0].invert_yaxis()
    axes[0].invert_xaxis()
    # axes[1].invert_yaxis()
    # axes[1].invert_xaxis()
    plt.show()

def plot_feh_bins(df, n_bins=6):
    feh_min = df['FeH'].min()
    feh_max = df['FeH'].max()
    bins = np.linspace(feh_min, feh_max, n_bins + 1)
    fig, axes = plt.subplots(nrows=2, ncols=n_bins//2, sharex=True, sharey=True)
    axes = axes.ravel()
    for i, feh in enumerate(bins[:-1]):
        df_bin = df[(df['FeH'] > feh) & (df['FeH'] <= bins[i+1])]
        label = f'{feh:.3f} < FeH <= {bins[i+1]:.3f}'
        plot_tables(df_bin, simonian=None, other=None, ax=axes[i], fig=fig, data_col='kmag_diff')
        axes[i].set_title(label, fontsize=14)
    axes[0].invert_xaxis()
    axes[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig('imgs/feh_bins.png')
    plt.show()


def plot_hr_main_squence_line(df, other=None, n_bins=20, fast_thresh=5,
                              slow_thresh=20, min_T=4000, max_T=6200,
                              poly_ord=3):
    df = df[(df['Teff'] > min_T) & (df['Teff'] < max_T)]
    if other is not None:
        other = other[(other['Teff'] > min_T) & (other['Teff'] < max_T)]
    df_fast = df[df['predicted period'] < fast_thresh]
    df_slow = df[df['predicted period'] > slow_thresh]

    # Bin Teff values
    teff_bins = pd.cut(df['Teff'], bins=n_bins)
    if other is not None:
        other_bins = pd.cut(other['Teff'], bins=n_bins)
    else:
        other_bins = teff_bins


    # Find highest density bins for fast and slow rotators
    fast_densities = []
    slow_densities = []
    mist_slow_vals = []
    other_densities = []

    for teff_bin, other_bin in zip(teff_bins.unique(), other_bins.unique()):
        df_teff_bin = df[teff_bins == teff_bin]
        if other is not None:
            other_teff_bin = other[other_bins == other_bin]

        # Fast rotators
        try:
            fast_bin_kmag = pd.cut(df_fast[df_fast['Teff'].isin(df_teff_bin['Teff'])]['kmag_abs'], bins=n_bins)
            fast_counts = fast_bin_kmag.value_counts(normalize=True)
            fast_densities.append((teff_bin, fast_counts.idxmax()))
        except ValueError:
            fast_densities.append((teff_bin, np.nan))

        # Slow rotators
        try:
            slow_subset = df_slow[df_slow['Teff'].isin(df_teff_bin['Teff'])]
            slow_bin_kmag = pd.cut(slow_subset['kmag_abs'], bins=n_bins)
            slow_counts = slow_bin_kmag.value_counts(normalize=True)
            max_bin = slow_counts.idxmax()
            slow_densities.append((teff_bin, max_bin))

            # Get median MIST K magnitude for this bin
            mist_val = slow_subset[(slow_subset['kmag_abs'] >= max_bin.left) &
                                   (slow_subset['kmag_abs'] < max_bin.right)]['Kmag_MIST'].median()
            mist_slow_vals.append(mist_val)

        except ValueError:
            slow_densities.append((teff_bin, np.nan))
            mist_slow_vals.append(np.nan)

        if other is not None and len(other_teff_bin):
            print(len(other_teff_bin))
            other_bin_kmag = pd.cut(other_teff_bin['kmag_abs'], bins=n_bins//2)
            other_counts = other_bin_kmag.value_counts(normalize=True)
            other_max = other_counts.idxmax()
            other_densities.append((teff_bin, other_max))

    # Create density heatmap
    plt.figure()
    plt.hexbin(df['Teff'], df['kmag_abs'], mincnt=1, cmap='YlGnBu')
    if other is not None:
        plt.scatter(other['Teff'], other['kmag_abs'], c='brown', alpha=0.4)

    # Plot highest density lines
    # fast_x = [bin.mid for bin, _ in fast_densities]
    # fast_y = [bin.mid for _, bin in fast_densities]
    slow_x = np.array([bin.mid for bin, _ in slow_densities])
    slow_y = np.array([bin.mid for _, bin in slow_densities])
    if other is not None:
        other_x = np.array([bin.mid for bin ,_ in other_densities])
        other_y = np.array([bin.mid for _, bin in other_densities])


    # plt.scatter(fast_x, fast_y, s=100, label='Fast Rotators')
    plt.scatter(slow_x, slow_y, s=100, c='r', label='Slow Rotators')
    plt.scatter(other_x, other_y, s=100, c='g')

    plt.xlabel('Teff')
    plt.ylabel('Kmag_abs')
    plt.title('Density of Rotation Periods')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()

    sort_idx = np.argsort(slow_x)
    slow_x = slow_x[sort_idx]
    slow_y = slow_y[sort_idx]
    mist_slow_vals = np.array(mist_slow_vals)[sort_idx]
    diff_mag = slow_y - mist_slow_vals
    coeff = np.polyfit(slow_x, diff_mag, deg=poly_ord)
    print("coeff : ", coeff)
    poly_x = np.poly1d(coeff)
    fit_y = poly_x(slow_x)

    # Plot kmag_diff with slow rotators and MIST comparisons
    plt.figure()
    plt.hexbin(df['Teff'], df['kmag_diff'], mincnt=1, cmap='YlGnBu')
    # plt.scatter(slow_x, diff_mag, c='r', s=100, label='Slow Rotators')
    plt.plot(slow_x, fit_y, c='g', label='Fit')
    plt.xlabel('Teff')
    plt.ylabel('Kmag_diff')
    plt.title('Slow Rotators vs MIST')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()

    return poly_x

def piecewise_boundary(x, poly, x_min=None, x_max=None):
    res = poly(x)
    if x_min is not None:
        res[x < x_min] = poly(x_min)
    if x_max is not None:
        res[x > x_max] = poly(x_max)

    return res

def piecewise_boundary_single(x, poly, x_min=None, x_max=None):
    if (x_max is not None) and (x > x_max):
        return poly(x_max)
    if (x_min is not None) and (x < x_min):
        return poly(x_min)
    return poly(x)


def density_mag_boundary(df, min_T_fit=3800, max_T_fit=5000, n_bins=25, buffer=0.1):
    # Create temperature and magnitude bins
    bins = np.linspace(min_T_fit, max_T_fit, n_bins)
    mag_bins = np.linspace(-2, 0, n_bins)

    # Bin the temperature values
    df['Teff_bin'] = pd.cut(df['Teff'], bins=bins)

    # Dictionary to store results
    boundary = []
    teffs = []

    # Process each temperature bin
    for teff in df['Teff_bin'].dropna().unique():
        # Get data for this temperature bin
        df_t = df[df['Teff_bin'] == teff]

        # Create magnitude bins
        df_t['mag_bin'] = pd.cut(df_t['kmag_diff'], bins=mag_bins)

        # Count points in each magnitude bin
        bin_counts = df_t['mag_bin'].value_counts()

        # Find the bin with maximum density
        max_density_bin = bin_counts.idxmax()
        boundary.append(max_density_bin.mid - buffer)
        teffs.append(teff.mid)

    # plt.hexbin(df['Teff'], df['kmag_diff'], mincnt=1, cmap='YlGnBu')
    # plt.scatter(teffs, boundary, s=200, c='black')
    # plt.gca().invert_xaxis()
    # plt.gca().invert_yaxis()
    # plt.show()
    return np.array(teffs), np.array(boundary)


def mag_boxplot(dfs, labels, mag_label='kmag_abs',
                bin_size=500, min_T_fit=3800,
                max_T_fit=5600,
                poly_order=1,
                quartile='med'):
    # Create temperature bins
    teff_min = dfs[0]['Teff'].min()
    teff_max = dfs[0]['Teff'].max()
    bins = np.arange(teff_min - (teff_min % bin_size), teff_max + bin_size, bin_size)

    # Define colors for each dataset
    colors = ['lightblue', 'wheat', 'lightgray', 'lightgreen']

    # Create empty lists to store boxes for legend
    boxes = []
    medians = []
    quartile_75 = []  # Changed to store 75th percentiles

    # Assign bins to temperatures
    for df, name, color in zip(dfs, labels, colors):
        df['teff_bin'] = pd.cut(df['Teff'], bins=bins, labels=bins[:-1])
        positions = bins[:-1]

        # Store the boxplot output
        bp = plt.boxplot([group[mag_label] for name, group in df.groupby('teff_bin')],
                         positions=positions, widths=bin_size / 2, patch_artist=True,
                         boxprops=dict(facecolor=color, alpha=0.5), labels=bins[:-1])
        boxes.append(bp["boxes"][0])

        # Calculate and store both medians and 75th percentiles
        group_stats = df.groupby('teff_bin')[mag_label].agg(['median', lambda x: x.quantile(0.75)])
        medians.append(group_stats['median'].values)
        quartile_75.append(group_stats['<lambda_0>'].values)

    plt.xlabel('Teff (K)', fontsize=22)
    plt.ylabel('$\Delta K_{iso}$', fontsize=22)
    plt.legend(boxes, labels, fontsize=18, loc='upper left')

    # Format x-axis labels
    plt.xticks(bins[:-1], [f'{int(x)}' for x in bins[:-1]], rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.savefig('imgs/mag_box_plot.png')
    plt.close()


    # Fit polynomial to medians
    if quartile == 'med':
        vals = medians[1]
    else:
        vals = quartile_75[1]
    # vals_poly = vals[np.logical_and(positions >= min_T_fit, positions <= max_T_fit)]
    # positions_poly = positions[np.logical_and(positions >= min_T_fit, positions <= max_T_fit)]
    # coeff = np.polyfit(positions_poly, vals_poly, deg=poly_order)
    # poly = np.poly1d(coeff)
    # x = np.linspace(min_T_fit, positions[-1], 200)
    # y = piecewise_boundary(x, poly, x_min=min_T_fit)

    # Plot medians
    # plt.scatter(positions, medians[1], label=f'{labels[1]} (median)', s=100)
    # plt.plot(x, y, c='brown', label='median fit')
    # plt.scatter(positions, medians[0], label=f'{labels[0]} (median)', s=100)

    # Plot 75th percentiles
    if quartile == 'med':
        plt.scatter(positions, medians[1], label=f'{labels[1]} ', s=100, marker='s', color='orange')
        plt.scatter(positions, medians[0], label=f'{labels[0]} ', s=100, marker='s', color='blue')
    else:
        plt.scatter(positions, quartile_75[1], label=f'{labels[1]}', s=100, color='orange', edgecolors='black')
        plt.scatter(positions, quartile_75[0], label=f'{labels[0]} ', s=100,  color='blue', edgecolors='black')

    plt.xticks(bins[:-1], [f'{int(x)}' for x in bins[:-1]], rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.xlabel('Teff (K)', fontsize=22)
    plt.ylabel('$\Delta K_{iso}$', fontsize=22)
    plt.legend()
    plt.savefig('imgs/mag_box_plot_median.png')
    plt.close()

    return positions, vals

def get_kmag_boundaries(df, known_binaries, t_thresh):
    t_boundary_cold, k_boundary_cold = density_mag_boundary(df,
                                                            min_T_fit=3800,
                                                            max_T_fit=7000,
                                                            buffer=0.03)
    t_b, k_b = mag_boxplot([df, known_binaries],
                           ['Kepler', 'gaia-nss+EBs'],
                           mag_label='kmag_diff', bin_size=200,
                           min_T_fit=3800,
                           max_T_fit=5600,
                           poly_order=4,
                           quartile='q75')

    t_b = t_b[~np.isnan(k_b)]
    k_b = k_b[~np.isnan(k_b)]

    t_combined = np.concatenate((t_boundary_cold[t_boundary_cold < t_thresh], t_b[t_b > t_thresh]))
    k_combined = np.concatenate((k_boundary_cold[t_boundary_cold < t_thresh], k_b[t_b > t_thresh]))
    coeffs = np.polyfit(t_combined, k_combined, deg=5)
    poly = np.poly1d(coeffs)
    x = np.linspace(t_combined.min(), t_combined.max(), 100)
    y = poly(x)
    plt.hexbin(df['Teff'], df['kmag_diff'], mincnt=1, cmap='YlGnBu')
    plt.scatter(t_boundary_cold, k_boundary_cold, s=100, c='black')
    plt.scatter(t_b, k_b, s=100, c='r')
    # plt.scatter(t_combined, k_combined, s=100, c='green')
    plt.plot(x, y, color='green')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.close()
    return poly
def magnitude_displacement(df, known_binaries, min_feh=-0.1, max_feh=0.1):
    df = df[(df['kmag_diff'].abs() < 2) & (df['kmag_diff'] < 0) & (df['FeH'] > min_feh) & (df['FeH'] < max_feh)]
    known_binaries = known_binaries[(known_binaries['kmag_diff'].abs() < 2) & (known_binaries['kmag_diff'] < 0)
                                    & (known_binaries['FeH'] > min_feh) & (known_binaries['FeH'] < max_feh)]
    print("after magnitude threshold: ", len(df))
    df_slow = df[df['predicted period'] >= 20]
    # plot_tables(df , name='clean')
    # boundary = plot_hr_main_squence_line(df, other=knwon_binaries, poly_ord=2)
    # mag_boxplot([df, knwon_binaries], ['Kepler', 'gaia+EBs'], bin_size=200)
    boundary_fn = get_kmag_boundaries(df, known_binaries, 7000)
    # boundary_fn = partial(piecewise_boundary, poly=poly, x_min=3800, x_max=5600)
    # boundary_fn_single = partial(piecewise_boundary_single, poly=poly, x_min=3800, x_max=5600)
    plot_density_boundary(df['Teff'], df['kmag_diff'], boundary_fn=boundary_fn)
    # exit()
    # mag_boxplot([df_fast, knwon_binaries], ['Kepler Fast', 'gaia+EBs'], mag_label='kmag_diff', bin_size=200)
    # mag_boxponlylot([df_slow, knwon_binaries], ['Kepler Slow', 'gaia+EBs'], mag_label='kmag_diff', bin_size=200)
    # mag_boxplot([df_slow, df_fast], ['Kepler Fasr', 'Kepler Slow'], mag_label='kmag_diff', bin_size=200)

    # boundary = find_density_boundary(df['Teff'].values, df['kmag_diff'].values,
    #                                  kmag_range=[-0.33, 0], teff_bins=20, teff_range=[4400, 6000])
    # plot_density_boundary(df['Teff'].values, df['kmag_diff'].values, boundary, smooth=True)
    df['position'] = df.apply(lambda x: is_binary(boundary_fn, x['Teff'], x['kmag_diff'],
                                                  teff_range=[3800, 7000]), axis=1)
    known_binaries['position'] = known_binaries.apply(lambda x: is_binary(boundary_fn, x['Teff'], x['kmag_diff']), axis=1)
    df['kmag_binary'] = df.apply(lambda x: x['position']=='binary', axis=1)
    known_binaries['kmag_binary'] = known_binaries.apply(lambda x: x['position']=='binary', axis=1)
    df.loc[df['position']=='boundary', 'binary'] = np.nan
    # df['binary'] = df.apply(lambda x: x['kmag_diff']< - 0.01, axis=1)
    # known_binaries['binary'] = known_binaries.apply(lambda x: x['kmag_diff'] < - 0.01, axis=1)



    # plot_feh_bins(df)

    df_fast = df[df['predicted period'] < 7]
    tab10 = get_cmap('tab10')
    tab10_colors = [tab10(i) for i in range(tab10.N)]
    # ax = plot_binaries_frac(known_binaries, c=tab10_colors[0], label='gaia+EBs')
    # ax = plot_binaries_frac(df_fast, c=tab10_colors[-1], ax=ax, label='Kepler Fast')
    # ax = plot_binaries_frac(df, ax=ax, c=tab10_colors[4], label='Kepler')
    # ax = plot_binaries_frac(short_b, ax=ax, c=tab10_colors[-1], label='Kepler Fast')
    # ax.legend()
    # ax.hlines(0.75, 4000, 6400, linestyles='dashed', colors='gray')
    # ax.set_xlabel('$T_{eff}$')
    # ax.set_ylabel('Fraction of Binaries (%)')
    # plt.savefig('imgs/binaries_fracs.png')
    # plt.show()
    # plot_tables(df,other=known_binaries, line_of_best_fit=boundary, hline=[0], vline=[6000], data_col='kmag_diff', name='all')
    # plot_tables(short_b, simonian=None, other=None, color_col=None, hline=[0], data_col='kmag_diff', name='all_fast')
    # plot_tables(known_binaries, simonian=None, other=None, color_col=None, hline=[0], data_col='kmag_diff', name='all_binaries')
    # sample_binaries()
    df = df.dropna(subset=['binary'])
    known_binaries = known_binaries.dropna(subset=['kmag_binary'])
    teffs = [4400, 5000, 5400, 5800, 6200, 6400, 6600]
    teffs_short = [5200, 5800,6200,6800]
    known_binaries['predicted period'] = known_binaries['period']
    ax, t, p, fit = plot_period_cutoffs(df, 50, p_name='predicted period', teffs=teffs, bin_size=2,
                                                   data_col='kmag_diff', method='g-r boundary', plot_every=1,
                                                   name='norm_long', fit=False)
    err_p = 2
    err_t = np.array([np.array(t) - np.array(teffs[:-1]), teffs[1:] - np.array(t)])
    show_period_cutoff(ax, 'predicted_period', name='all')
    # ax, _, _, _  = plot_period_cutoffs(known_binaries, 50, bin_size=6,  p_name='predicted period', teffs=teffs,
    #                                                data_col='kmag_diff', method='g-r boundary', plot_every=1,
    #                                                name='norm_long', fit=False)
    # show_period_cutoff(ax, 'predicted_period', name='binary')


    return df, t, p, err_t, err_p, boundary_fn


def plot_binaries_and_fit(df, binary_fit, Tmin=3800, Tmax=5800, density=True):
    df['binary_p'] = 0

    mask = (df['Teff'] >= Tmin) & (df['Teff'] <= Tmax) & (
            np.exp(binary_fit(np.log(df['Teff']))) > df['predicted period']
    )

    df.loc[mask, 'binary_p'] = 1

    x = df['Teff']
    y = df['predicted period']
    x_sorted = np.linspace(Tmin, Tmax, 100)
    y_fit = binary_fit(np.log(x_sorted))
    if density:
        plt.hexbin(x, y, mincnt=1, gridsize=100, cmap='viridis')
    else:
        plt.scatter(x, y, c=df['kmag_binary'])
    plt.plot(x_sorted, np.exp(y_fit), c='r')
    plt.colorbar()
    plt.show()
    return df

    # plot_kepler_cluster(clusters)
def clusters_binaries_fit(binary_fit):
    clusters_df = Table.read('tables/bouma2023_calibration.txt', encoding='utf-8', format='ascii.cds').to_pandas()
    classes = clusters_df['Cluster'].unique()
    for cluster in classes:
        df_c = clusters_df[clusters_df['Cluster'] == cluster]
        df_c = df_c.dropna(subset=['Teff'])
        df_c = df_c[(df_c['Teff'] <= 6000) & (df_c['Teff'] >= 4000)]
        p_cut = binary_fit(df_c['Teff'].values)
        print(cluster, p_cut.mean())
        plt.hist(p_cut)
        plt.title(cluster)
        plt.show()

def filter_main_sequence_with_logg(df, n_bins, buffer=0.3):
    """
    Filter dataframe to keep only rows where logg values fall between temperature-dependent bounds.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing 'Teff' and 'logg' columns
    n_bins : int
        Number of temperature bins to use for defining logg bounds

    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe containing only rows within the logg bounds
    """
    # Get the temperature-dependent logg bounds
    bins, logg_min, logg_max = get_t_logg_relations(n_bins)

    # Create a mask initialized as False
    mask = np.zeros(len(df), dtype=bool)

    # Use enumerate to get position index instead of DataFrame index
    for pos, (_, row) in enumerate(df.iterrows()):
        # Find which bin this temperature falls into
        bin_idx = np.digitize(row['Teff'], bins) - 1

        # Skip if temperature is outside our bin range
        if bin_idx < 0 or bin_idx >= len(logg_min):
            continue

        # Check if logg is within bounds for this temperature bin
        if (logg_min[bin_idx] - buffer) <= row['logg'] <= (logg_max[bin_idx] + buffer):
            mask[pos] = True

    # Return filtered dataframe
    return df[mask]

def binaries_probs(dfs, names):
    for df, name in zip(dfs, names):
        plt.hist(df['kmag_diff'], histtype='step', density=True, linewidth=4, label=name)
    plt.legend()
    plt.xlabel('$\Delta K_{iso}$')
    plt.ylabel('pdf')
    plt.savefig('imgs/kmag_pdfs.png')
    plt.show()

def binaries_scatter(dfs, names):
    markers = ['o', '^', 'v', 'D', 'p', '*', 'h', '+', 'x'][: len(dfs)]
    for df, name, marker in zip(dfs, names, markers):
        plt.scatter(df['Teff'], df['kmag_diff'], c=df['FeH'], marker=marker, label=name)
    plt.legend()
    plt.ylabel('$\Delta K_{iso}$')
    plt.xlabel('$T_{eff}$')
    plt.colorbar(label='FeH')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.savefig('imgs/kmag_scatter.png')
    plt.show()


def plot_feh_bins(df, gaia_nss, ebs):
    feh_bins = [-0.2, -0.1, 0, 0.1, 0.2]
    df['feh_bin'] = pd.cut(df['FeH'], feh_bins)
    gaia_nss['feh_bin'] = pd.cut(gaia_nss['FeH'], feh_bins)
    ebs['feh_bin'] = pd.cut(ebs['FeH'], feh_bins)
    for feh_bin in df['feh_bin'].dropna().unique():
        df_feh = df[df['feh_bin'] == feh_bin]
        gaia_feh = gaia_nss[gaia_nss['feh_bin'] == feh_bin]
        ebs_feh = ebs[ebs['feh_bin'] == feh_bin]
        binaries = pd.concat([gaia_feh, ebs_feh])
        print(feh_bin, len(df_feh))
        _, _, _, _ = magnitude_displacement(df_feh, gaia_feh, ebs_feh)


def plot_clusters_period(df, p_label='PRot'):
    cmap = ['lightskyblue', 'royalblue', 'gold', 'yellowgreen', 'tomato', 'lightpink', 'pink']  # Choose a colormap
    ps = []
    teffs = []
    young_cmap = ['gold']
    bins = np.linspace(3000, 6600, 20)
    df['Teff_bin'] = pd.cut(df['Teff'], bins=bins)
    df_young = df[(df['cluster_age'] <= 300) & (df['cluster_age'] > 120)]
    for age, color in zip(df_young.cluster_age.unique(), young_cmap):
        df_age = df_young[(df_young['cluster_age'] == age) & (df_young['Binary'] == False)]
        df_age = df_age[df_age[p_label] > 0]

        for teff in df_age['Teff_bin'].dropna().unique():
            # Get data for this temperature bin
            df_t = df_age[df_age['Teff_bin'] == teff]
            avg_p = np.median(df_t[p_label])
            ps.append(avg_p)
            teffs.append(teff.mid)
        # print(age, color, len(df_age))
        # plt.scatter(teffs, ps, label=age, s=100, c=color, edgecolors='black')
    teffs_high = np.array(teffs)
    ps_high = np.array(ps)[teffs_high > 3800]
    teffs_high = teffs_high[teffs_high > 3800]
    coeff = np.polyfit(teffs_high, ps_high, deg=5)
    poly = np.poly1d(coeff)
    x = np.linspace(teffs_high.min(), teffs_high.max(), 200)
    p_fit = poly(x)


    fig,axes = plt.subplots(1,2, figsize=(26,14))
    for age, color in zip(df.cluster_age.unique(), cmap):
        ps = []
        teffs = []
        df_age = df[df['cluster_age'] == age]
        df_age = df_age[df_age[p_label] > 0]

        df_age['too_fast'] = df_age.apply(lambda x: poly(x['Teff']) > x['PRot'], axis=1)
        potential_binaries = df_age[df_age['too_fast']]
        # Plot binaries (stars)
        df_binary = df_age[df_age['Binary'] == True]
        axes[0].scatter(df_binary['Teff'], df_binary[p_label], c=color, marker='*',edgecolors='black',
                     s=100)
        # Plot non-binaries (circles)
        df_nonbinary = df_age[df_age['Binary'] == False]
        axes[0].scatter(df_nonbinary['Teff'], df_nonbinary[p_label], c=color, edgecolors='black', marker='o',
                    label=f'{age} Myr', s=50)

        for teff in df_age['Teff_bin'].dropna().unique():
            # Get data for this temperature bin
            df_t = df_age[df_age['Teff_bin'] == teff]
            avg_p = np.median(df_t[p_label])
            ps.append(avg_p)
            teffs.append(teff.mid)
        print(age, color, len(df_age))
        axes[1].scatter(teffs, ps, s=100, c=color, edgecolors='black')

    axes[1].plot(x, p_fit, c='gray')
    fig.supxlabel('$T_{eff}$')
    fig.supylabel('$P_{rot}$')
    axes[0].legend(fontsize=18)
    axes[0].invert_xaxis()
    axes[1].invert_xaxis()
    plt.savefig('imgs/clusters_period.png')
    plt.show()


    return poly

def get_triple_catalog(short_p_cat):
    lightpred_df = pd.read_csv('tables/lightpred_cat.csv')
    lightpred_df['Prot_ref'] = 'lightpred'
    gaia_kepler = Table.read('tables/kepler_dr3_1arcsec.fits', format='fits').to_pandas()
    czava23 = Table.read('tables/czavalinga2023.fit', format='fits').to_pandas()
    czava23['ref'] = 'czava23'
    bashi24 = Table.read('tables/bashi2024.fit', format='fits').to_pandas()
    bashi24['ref'] = 'bashi24'
    borkovic16 = Table.read('tables/borkovic2016.fit', format='fits').to_pandas()
    borkovic16['ref'] = 'borkovic16'
    cat = gaia_kepler.merge(pd.concat([czava23, bashi24]), left_on='source_id', right_on='GaiaDR3')
    cat.rename(columns={'kepid':'KIC'}, inplace=True)
    cat = pd.concat([cat, borkovic16]).drop_duplicates('KIC').rename(columns={'KIC':'KID'})
    cat = cat.merge(lightpred_df[['KID', 'predicted period', 'mean_period_confidence', 'Prot_ref']],
                    on='KID', how='left')
    cat.rename(columns={'predicted_period':'Prot'}, inplace=True)
    cat = cat.merge(short_p_cat[['KID', 'Prot', 'Prot_ref']],
                    on='KID', how='left')
    cat = get_mag_data(cat, MIST_PATH)
    return cat

def get_catalogs_short_period():
    mazeh = pd.read_csv('tables/Table_1_periodic.txt')
    mazeh_short = mazeh[(mazeh['Prot'] < 3) & (mazeh['Prot'] > 0)]
    mazeh_short['Prot_ref'] = 'McQ14'
    santos = pd.read_csv('tables/santos_periods_19_21.csv')
    santos_short = santos[(santos['Prot'] < 3) & (santos['Prot'] > 0)]
    santos_short['Prot_ref'] = 'Santos19+Santos21'
    reinhold = pd.read_csv('tables/reinhold2023.csv')
    reinhold_short = reinhold[(reinhold['Prot'] < 3) & (reinhold['Prot'] > 0)]
    reinhold_short['Prot_ref'] = 'Reinhold23'
    short = pd.concat([mazeh_short, santos_short, reinhold_short]).drop_duplicates('KID')
    short = get_mag_data(short, mist_path=MIST_PATH)
    return short

def potential_triples(known_binaries, df, clusters, min_teff=3800,
                      max_teff=6800, n_bins=10, boundary_fn=None, kmag_thresh=-0.3):
    print("max teff: ", max_teff)
    # known_binaries = known_binaries[known_binaries['FeH'].abs() < 0.05]
    known_binaries.dropna(subset=['kmag_diff'], inplace=True)
    # df = df[df['FeH'].abs() < 0.05]
    df.dropna(subset=['kmag_diff'], inplace=True)
    short = get_catalogs_short_period()
    short = short[(short['Teff'] < max_teff) & (short['Teff'] > min_teff)]

    triples_cat = get_triple_catalog(short)

    short_in_triples = short.merge(triples_cat, on='KID')
    print('short in triples: ', len(short_in_triples))
    bouma24 = pd.read_csv('tables/bouma2024_planets.csv')
    bouma_fast = bouma24[(bouma24['li_median'] < 300) & (bouma24['gyro_median'] < 300)]
    short = short[~short['KID'].isin(bouma_fast['kepid'])]
    short = short[~short['KID'].isin(clusters['KID'])]
    # short = short[short['FeH'].abs() < 0.05]
    planets = pd.read_csv('tables/kois.csv')
    planets = planets[planets['koi_disposition']=='CONFIRMED']
    short_planets = short.merge(planets, on='KID')
    print("short planets :", len(short_planets), "all shorts: ", len(short))
    plt.scatter(short['Teff'], short['kmag_diff'])
    if boundary_fn is not None:
        short['non_single'] = short.apply(lambda x: boundary_fn(x['Teff']) > x['kmag_diff'], axis=1)
        known_binaries['non_single'] = known_binaries.apply(lambda x: boundary_fn(x['Teff']) > x['kmag_diff'], axis=1)
        x = np.linspace(short['Teff'].min(), short['Teff'].max(), 200)
        y = boundary_fn(x)
        plt.plot(x,y, c='r')
        print("non singles fraction fast rotators:", len(short[short['non_single']]) / len(short))
        print("non singles fraction binaries:", len(known_binaries[known_binaries['non_single']]) / len(known_binaries))

        short = short[short['kmag_diff'] < 0]

        print("number potential triple systems: ", len(short))

    triples_final = short[['KID', 'Prot', 'Teff', 'logg', 'FeH', 'kmag_diff',
                           'kmag_abs', 'Kmag_MIST', 'Prot_ref']]
    triples_final[['Prot', 'kmag_diff', 'kmag_abs', 'Kmag_MIST']] = \
        triples_final[['Prot', 'kmag_diff', 'kmag_abs', 'Kmag_MIST']].round(decimals=3)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.show()

    bins = np.linspace(3800, 6400, n_bins)
    known_binaries['teff_bin'] = pd.cut(known_binaries['Teff'], bins=bins)
    df['teff_bin'] = pd.cut(df['Teff'], bins=bins)
    short['teff_bin'] = pd.cut(short['Teff'], bins=bins)
    triples_cat['teff_bin'] = pd.cut(triples_cat['Teff'], bins=bins)
    unique_bins = known_binaries['teff_bin'].cat.categories

    teffs = []
    triples = []
    kmag_mid_long = []
    kmag_mid_all = []
    kmag_mid_b = []
    kmag_mid_short = []
    kmag_mid_middle = []
    b_prob = []
    b_middle_prob = []


    for teff_bin in unique_bins:
        teffs.append(teff_bin.mid)

        binaries_t = known_binaries[known_binaries['teff_bin'] == teff_bin]
        kmag_mid_b.append(np.median(binaries_t['kmag_diff']))
        df_t = df[df['teff_bin'] == teff_bin]
        df_long_t = df_t[(df_t['predicted period'] > 7)]
        df_middle_t = df_t[(df_t['predicted period'] < 7)]
        kmag_mid_all.append(np.median(df_t['kmag_diff']))
        kmag_mid_long.append(np.median(df_long_t['kmag_diff']))
        kmag_mid_middle.append(np.median(df_middle_t['kmag_diff']))
        short_t = short[short['teff_bin'] == teff_bin]
        kmag_mid_short.append(np.median(short_t['kmag_diff']))
        triples_t = triples_cat[triples_cat['teff_bin'] == teff_bin]
        triples.append(np.median(triples_t['kmag_diff']))

        low_k_b = len(binaries_t[binaries_t['kmag_diff'] < kmag_thresh])
        b_ids = binaries_t['KID'].unique()
        all_t_non_b = df_t[~df_t['KID'].isin(b_ids)]
        low_k_non_b = len(all_t_non_b[all_t_non_b['kmag_diff'] < kmag_thresh])
        b_prob.append(low_k_b / (low_k_b + low_k_non_b))

        binaries_t_middle = binaries_t[binaries_t['period'] < 7]
        low_k_b_middle = len(binaries_t_middle[binaries_t_middle['kmag_diff'] < kmag_thresh])
        middle_b_ids = binaries_t_middle['KID'].unique()
        middle_t_non_b = df_middle_t[~df_middle_t['KID'].isin(middle_b_ids)]
        low_k_non_b_middle = len(middle_t_non_b[middle_t_non_b['kmag_diff'] < kmag_thresh])
        b_middle_prob.append(low_k_b_middle / (low_k_b_middle + low_k_non_b_middle))


    print("all stars: ", len(df), ' slow: ', len(df[df['predicted period'] > 7]), ' middle: ',
          len(df[df['predicted period'] < 7]))
    # plt.plot(teffs, triples, c='r', label='known triples')
    plt.plot(teffs, kmag_mid_short, color='khaki', label='$P_{rot} < 3 $ days')
    plt.plot(teffs, kmag_mid_b, color='lightsalmon', label='all non singles')
    plt.plot(teffs, kmag_mid_all, color='peru', label='all stars')
    plt.plot(teffs, kmag_mid_long,  color='silver', label='$P_{rot} > 7 $ days')
    plt.plot(teffs, kmag_mid_middle,  color='plum', label='$3 < P_{rot} < 7 $ days')
    plt.xlabel('Teff (K)')
    plt.ylabel('$\Delta K_{iso}$')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.savefig('imgs/kmag_diff_median.png')
    plt.show()

    # plt.plot(teffs, b_prob, label='All')
    plt.plot(teffs, b_middle_prob)
    plt.xlabel('Teff (K)')
    plt.ylabel(r'$P(Binary \mid P_{rot} < 7,  \Delta K_{iso} < -0.3)$')
    plt.gca().invert_xaxis()
    plt.savefig('imgs/binary_kmag_prob.png')
    plt.show()

    return triples_final


def separation_fractions(known_binaries, df, separation_fn, n_bins=15):
    bins = np.linspace(3800, 6600, n_bins)
    known_binaries['teff_bin'] = pd.cut(known_binaries['Teff'], bins=bins)
    df['teff_bin'] = pd.cut(df['Teff'], bins=bins)

    # Apply separation function
    known_binaries['is_below_line'] = known_binaries.apply(lambda x: separation_fn(x['Teff']) > x['period'], axis=1)
    df['is_below_line'] = df.apply(lambda x: separation_fn(x['Teff']) > x['predicted period'], axis=1)

    # Filter datasets
    x = np.linspace(3800, 6600, 200)
    y = separation_fn(x)
    x_crit = x[y < 7][0]
    print("x crit: ", x_crit)
    df_long = df[df['predicted period'] > 7]
    binaries_short = known_binaries[known_binaries['period'] < 7]
    binaries_long = known_binaries[known_binaries['period'] > 7]

    orbital_b = []
    sb1_b = []
    eb_b = []
    eb_kepler_b = []
    eb_all_b = []
    astrospectro_b = []
    fractions_b = []
    fractions_b_short = []
    fractions_b_non_sync = []
    fractions_all = []
    fractions_all_long = []
    teffs = []
    b_probs = []

    # Get unique bin categories
    unique_bins = known_binaries['teff_bin'].cat.categories

    for teff_bin in unique_bins:
        teffs.append(teff_bin.mid)

        # Use .isin() method for categorical data or filter by bin name
        binaries_t = known_binaries[known_binaries['teff_bin'] == teff_bin]
        binaries_short_t = binaries_short[binaries_short['teff_bin'] == teff_bin]
        binaries_long_t = binaries_long[binaries_long['teff_bin'] == teff_bin]

        # Handle division by zero
        if len(binaries_t) > 0:
            num_below_b = binaries_t['is_below_line'].sum()
            fractions_b.append(num_below_b / len(binaries_t))
            fractions_b_short.append(binaries_short_t['is_below_line'].sum() / len(binaries_short_t))
            num_long_b = binaries_long_t['is_below_line'].sum()
            fractions_b_non_sync.append(num_long_b / num_below_b)

            orbital = binaries_t[binaries_t['nss_solution_type']=='Orbital']
            eb = binaries_t[binaries_t['nss_solution_type'] =='EclipsingBinary']
            eb_kepler = binaries_t[binaries_t['nss_solution_type']=='EclipsingBinaryKepler']
            sb1 = binaries_t[binaries_t['nss_solution_type']=='SB1']
            astrospectro = binaries_t[binaries_t['nss_solution_type']=='AstroSpectroSB1']
            orbital_b.append(orbital['is_below_line'].sum() / len(orbital))
            eb_b.append(eb['is_below_line'].sum() / len(eb))
            eb_kepler_b.append(eb_kepler['is_below_line'].sum() / len(eb_kepler))
            sb1_b.append(sb1['is_below_line'].sum() / len(sb1))
            astrospectro_b.append(astrospectro['is_below_line'].sum() / len(astrospectro))
            num_eb_below = eb['is_below_line'].sum() + eb_kepler['is_below_line'].sum()
            num_eb = len(eb) + len(eb_kepler)
            eb_all_b.append(num_eb_below / num_eb)

        else:
            fractions_b.append(np.nan)
            fractions_b_short.append(np.nan)
            fractions_b_non_sync.append(np.nan)

            orbital_b.append(np.nan)
            eb_b.append(np.nan)
            eb_kepler_b.append(np.nan)
            sb1_b.append(np.nan)
            astrospectro_b.append(np.nan)
            eb_all_b.append(np.nan)

        all_t = df[df['teff_bin'] == teff_bin]
        all_long_t = df_long[df_long['teff_bin'] == teff_bin]
        b_ids = binaries_t['KID'].unique()
        all_t_non_b = all_t[~all_t['KID'].isin(b_ids)]

        if len(all_t) > 0:
            num_all_below = all_t['is_below_line'].sum()
            num_all_below_non_b = all_t_non_b['is_below_line'].sum()
            fractions_all.append(num_all_below / len(all_t))
            fractions_all_long.append(all_long_t['is_below_line'].sum() / len(all_long_t))
            b_prob = num_below_b / (num_below_b + num_all_below_non_b)
            b_probs.append(b_prob)

        else:
            fractions_all.append(np.nan)
            fractions_all_long.append(np.nan)
            b_probs.append(np.nan)

    # Plot with NaN values filtered out
    # plt.plot(teffs, fractions_b_short, 's-', color='khaki', label='synchronized non singles')
    plt.plot(teffs, fractions_b, 'o-', color='lightsalmon', label='all non singles')
    plt.plot(teffs, eb_all_b, 'o-', color='lightgreen', label='EB')
    plt.plot(teffs, fractions_all, '^-', color='peru', label='all stars')
    plt.plot(teffs, fractions_all_long, 'v-', color='plum', label='slow rotators')
    # plt.vlines(x_crit, 0, 1, colors='gray', linestyles='--')
    plt.xlabel('Teff (K)')
    plt.ylabel('Fraction below separation')
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.savefig('imgs/separation_fractions.png')
    plt.show()

    plt.plot(teffs, orbital_b, 's-', color='khaki', label='Orbital')
    plt.plot(teffs, eb_b, 'o-', color='lightsalmon', label='Gaia Eclipsing Binary')
    plt.plot(teffs, sb1_b, '^-', color='peru', label='SB1')
    plt.plot(teffs, astrospectro_b, 'v-', color='plum', label='AstroSpectroSB1')
    plt.plot(teffs, eb_kepler_b, 'v-', color='lightgreen', label='Kepler Eclipsing Binary')

    # plt.vlines(x_crit, 0, 1, colors='gray', linestyles='--')
    plt.xlabel('Teff (K)')
    plt.ylabel('Fraction below separation')
    plt.legend(loc='best')
    plt.gca().invert_xaxis()
    plt.savefig('imgs/separation_fractions_binaries.png')
    plt.show()

    plt.plot(teffs, b_probs, 'o-')
    plt.xlabel('Teff (K)')
    plt.ylabel(r'$P(\text{Binary} \mid \text{below line})$')
    plt.gca().invert_xaxis()
    # plt.semilogy()
    plt.savefig('imgs/fractions_eb_prob.png')
    plt.show()



def get_p_k_probs(df, poly, p_name='predicted period', scale_factor=2):
    df['p_dist'] = df.apply(lambda x: x[p_name] - poly(x['Teff']), axis=1)
    scale = scale_factor * df['p_dist'].median()
    df['p_prob'] = np.clip(np.exp(-df['p_dist'] / scale), a_min=None, a_max=1)
    df['k_prob'] = (2 - df['kmag_diff']) / 4
    return df

def create_ages_dataset(df, poly):
    godoy = pd.read_csv('imgs/tables/GodoyRivera25_TableA1.csv')
    berger = pd.read_csv('tables/berger_catalog_full.csv')
    godoy = godoy.merge(berger[['KID', 'Teff', 'logg', 'FeH', 'Rstar']],
                        right_on='KID', left_on='KIC', how='left')
    godoy = godoy.merge(df[['KID', 'predicted period', 'kmag_diff']],
                        right_on='KID', left_on='KIC', how='left')
    dataset_seismo = read_machine_readable_table('tables/apokasc2025.txt').to_pandas()
    bouma24 = pd.read_csv('tables/bouma2024.csv')
    angus23 = pd.read_csv('tables/angus2023.txt', sep=';')
    bouma24_p = pd.read_csv('tables/bouma2024_planets.csv')


    dataset_seismo['final_age'] = np.nan
    dataset_seismo['age_error'] = np.nan
    dataset_seismo['RGBAge_error'] = (dataset_seismo['E_RGBAgeM'] - dataset_seismo['e_RGBAgeM']) / 2
    dataset_seismo['RCAge_error'] = (dataset_seismo['E_RCAgeM'] - dataset_seismo['e_RCAgeM']) / 2
    dataset_seismo.loc[dataset_seismo['ESA3'] == 2, 'final_age'] = dataset_seismo[dataset_seismo['ESA3']==2]['RCAgeM']
    dataset_seismo.loc[dataset_seismo['ESA3'] == 1, 'final_age'] = dataset_seismo[dataset_seismo['ESA3'] == 1]['RGBAgeM']
    dataset_seismo.loc[dataset_seismo['ESA3'] == 2, 'age_error'] = dataset_seismo[dataset_seismo['ESA3'] == 2]['RCAge_error']
    dataset_seismo.loc[dataset_seismo['ESA3'] == 1, 'age_error'] = dataset_seismo[dataset_seismo['ESA3'] == 1]['RGBAge_error']
    dataset_seismo['age_ref'] = 'asteroseismology'
    dataset_seismo.rename(columns={'KIC':'KID'}, inplace=True)

    merged_angus_bouma = angus23.merge(bouma24, left_on='KID', right_on='KIC')
    merged_angus_bouma['gyro_median'] = merged_angus_bouma['gyro_median'] / 1000
    merged_angus_bouma['age_diff'] = np.abs(merged_angus_bouma['age'] - merged_angus_bouma['gyro_median'])
    merged_angus_bouma['valid'] = merged_angus_bouma.apply(lambda x: x['age_diff'] < x['age'] * 0.2, axis=1)

    dataset_gyro_gyro = merged_angus_bouma[merged_angus_bouma['valid'] == True]
    dataset_gyro_gyro['final_age'] = (dataset_gyro_gyro['age'] + dataset_gyro_gyro['gyro_median']) / 2
    dataset_gyro_gyro['age_error'] = (dataset_gyro_gyro['E_Age'] + ((dataset_gyro_gyro['gyro_+1sigma'] + dataset_gyro_gyro['gyro_-1sigma']) / 2) / 1000) / 2
    dataset_gyro_gyro['age_ref'] = 'gyro_gyro'
    dataset_gyro_gyro.drop_duplicates('KID', inplace=True)
    dataset_gyro_gyro = dataset_gyro_gyro[dataset_gyro_gyro['final_age'] <= 2.7]
    # dataset_gyro_gyro.rename(columns={'KIC': 'KID'}, inplace=True)

    dataset_gyro = bouma24[bouma24['gyro_median'] < 2700]
    dataset_gyro['final_age'] = dataset_gyro['gyro_median'] / 1000
    dataset_gyro['age_error'] =(dataset_gyro_gyro['gyro_+1sigma'] + dataset_gyro_gyro['gyro_-1sigma']) / 2000
    dataset_gyro['age_ref'] = 'gyro_bouma'


    bouma24_p['final_age'] = bouma24_p['li_median'] / 1000
    dataset_li = bouma24_p[bouma24_p['li_median'].notna()]
    dataset_li['age_error'] = (dataset_li['li_+1sigma'] + dataset_li['li_-1sigma']) / 2000
    dataset_li.rename(columns={'kepid': 'KID'}, inplace=True)
    dataset_li['age_ref'] = 'lithium'
    dataset_li.drop_duplicates('KID', inplace=True)

    all_data = pd.concat([dataset_seismo, dataset_gyro_gyro, dataset_li])
    # all_data = dataset_gyro
    # Get the list of duplicated KIDs
    duplicated_kids = all_data[all_data.duplicated('KID', keep=False)]['KID'].unique()

    priority_map = {'lithium': 1, 'gyro_bouma': 3, 'asteroseismology': 2}
    all_data['priority'] = all_data['age_ref'].map(priority_map)

    # Sort by KID and priority (descending), then drop duplicates keeping highest priority
    all_data = all_data.sort_values(['KID', 'priority'], ascending=[True, False])

    # Remove duplicates - keep the record with highest priority for each KID
    final_dataset = all_data.drop_duplicates('KID', keep='first')

    # Drop the temporary priority column
    final_dataset = final_dataset.drop('priority', axis=1)

    final_dataset = final_dataset[final_dataset['final_age'] < 11]

    final_dataset = final_dataset.merge(godoy, left_on='KID', right_on='KIC')
    # final_dataset['flag_sync_binary'] = final_dataset.apply(lambda x: x['predicted period'] < poly(x['Teff']), axis=1)

    # final_dataset = final_dataset[final_dataset['flag_Binary_Union'] == False]
    final_dataset.to_csv('tables/ages_dataset.csv')
    print(len(final_dataset))


def create_nss_dataset(df, poly):
    godoy = pd.read_csv('imgs/tables/GodoyRivera25_TableA1.csv')
    berger = pd.read_csv('tables/berger_catalog_full.csv')
    godoy = godoy.merge(berger[['KID', 'Teff', 'logg', 'FeH', 'Rstar']],
                        right_on='KID', left_on='KIC', how='left')
    godoy = godoy.merge(df[['KID', 'predicted period', 'kmag_diff']],
                        right_on='KID', left_on='KIC', how='left')
    godoy['flag_sync_binary'] = godoy.apply(lambda x: x['predicted period'] < poly(x['Teff']), axis=1)
    godoy_fast = godoy[godoy['predicted period'] < 7]
    godoy_binary = godoy[godoy['flag_Binary_Union'] == True]
    godoy_non_binary = godoy[godoy['flag_Binary_Union'] == False]
    ruwe = godoy_binary[godoy_binary['flag_RUWE']]
    rv = godoy_binary[godoy_binary['flag_RVvariable']]
    nss = godoy_binary[godoy_binary['flag_NSS']]
    eb_kepler = godoy_binary[godoy_binary['flag_EB_Kepler']]
    eb_gaia = godoy_binary[godoy_binary['flag_EB_Gaia']]
    eb = pd.concat([eb_kepler, eb_gaia]).drop_duplicates('KIC')
    godoy['single'] = godoy.apply(lambda x: (x['flag_Binary_Union'] == False) and
                                            (x['flag_sync_binary'] == False) and
                                            (x['RUWE'] < 1) and
                                            (np.abs(x['kmag_diff']) < 0.3), axis=1)

    singles = godoy[godoy['single'] == True]
    singles['binarity_class'] = 0
    eb['binarity_class'] = 1
    rv['binarity_class'] = 2
    ruwe['binarity_class'] = 3
    nss['binarity_class'] = 4

    dataset = pd.concat([singles, eb, rv, ruwe, nss])

    dataset.to_csv('imgs/tables/nss_dataset.csv')

    return dataset


    # known_nss = known_nss[known_nss['Teff'] <= df['Teff'].max()]
    # df = df[df['kmag_diff'].abs() <= 2]
    # known_nss = known_nss[known_nss['kmag_diff'].abs() <= 2]
    # df = get_p_k_probs(df, poly, p_name='predicted period', scale_factor=3)
    # df['binary_prob'] = df['p_prob'] * df['k_prob']
    # known_nss['binary_prob'] = 1
    # known_nss['k_prob'] = (2 - df['kmag_diff']) / 4
    # dataset = pd.concat([known_nss, df])[['KID', 'binary_prob', 'p_prob', 'k_prob', 'kmag_diff',
    #                                       'p_dist', 'Teff', 'predicted period']]
    # binaries = dataset[dataset['binary_prob'] > 0.5]
    # print("number of binaries : ", len(binaries) / len(dataset))
    # plt.hist(dataset['binary_prob'], bins=40)
    # plt.show()
    # plt.hist(dataset['p_prob'], density=True, histtype='step', label='p')
    # plt.hist(dataset['k_prob'], density=True, histtype='step', label='k')
    # plt.legend()
    # plt.show()


def calculate_binary_probability(df, teff_bins=None, period_bins=None, kmag_diff_bins=None):
    """
    Calculate the probability of being a binary (flag_Binary_Union=True) for
    different bins of Teff, predicted period, and kmag_diff.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'Teff', 'predicted period', 'kmag_diff', and 'flag_Binary_Union' columns
    teff_bins : list or np.array, optional
        Bins for Teff. If None, 5 equally spaced bins will be created
    period_bins : list or np.array, optional
        Bins for predicted period. If None, 5 equally spaced bins will be created
    kmag_diff_bins : list or np.array, optional
        Bins for kmag_diff. If None, 5 equally spaced bins will be created

    Returns:
    --------
    result_df : pandas DataFrame
        DataFrame with probability of being a binary for each bin combination
    """
    # Create default bins if not provided
    if teff_bins is None:
        teff_bins = np.linspace(df['Teff'].min(), df['Teff'].max(), 10)
    if period_bins is None:
        period_bins = np.linspace(df['predicted period'].min(), df['predicted period'].max(), 10)
    if kmag_diff_bins is None:
        kmag_diff_bins = np.linspace(df['kmag_diff'].min(), df['kmag_diff'].max(), 10)

    # Create bin labels for better readability
    teff_labels = [f"{teff_bins[i]:.1f}-{teff_bins[i + 1]:.1f}" for i in range(len(teff_bins) - 1)]
    period_labels = [f"{period_bins[i]:.2f}-{period_bins[i + 1]:.2f}" for i in range(len(period_bins) - 1)]
    kmag_labels = [f"{kmag_diff_bins[i]:.2f}-{kmag_diff_bins[i + 1]:.2f}" for i in range(len(kmag_diff_bins) - 1)]

    # Bin the data
    df_copy = df.copy()
    df_copy['Teff_bin'] = pd.cut(df_copy['Teff'], bins=teff_bins, labels=teff_labels)
    df_copy['Period_bin'] = pd.cut(df_copy['predicted period'], bins=period_bins, labels=period_labels)
    df_copy['Kmag_bin'] = pd.cut(df_copy['kmag_diff'], bins=kmag_diff_bins, labels=kmag_labels)

    # Group by bins and calculate probability
    grouped = df_copy.groupby(['Teff_bin', 'Period_bin', 'Kmag_bin'])

    # Calculate probability and counts
    result = grouped.agg(
        binary_count=('flag_Binary_Union', 'sum'),
        total_count=('flag_Binary_Union', 'count'),
    )

    # Calculate probability
    result['probability'] = result['binary_count'] / result['total_count']

    # Reset index for easier viewing
    result_df = result.reset_index()

    return result_df


def plot_binary_probability(result_df):
    """
    Create plots to visualize binary probabilities across different parameter combinations.

    Parameters:
    -----------
    result_df : pandas DataFrame
        Output from calculate_binary_probability function

    Returns:
    --------
    None (displays plots)
    """
    # Create a figure with 3 subplots for each 2D visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Drop any rows with NaN values which could cause issues in visualization
    clean_df = result_df.dropna(subset=['Teff_bin', 'Period_bin', 'Kmag_bin', 'probability'])

    # 1. Teff vs Period heatmap
    pivot1 = pd.pivot_table(
        clean_df,
        values='probability',
        index='Teff_bin',
        columns='Period_bin',
        aggfunc='mean'
    )

    sns.heatmap(pivot1, annot=True, cmap='Blues', fmt='.2f', ax=axes[0])
    axes[0].set_title('Binary Probability: Teff vs Period')
    axes[0].set_ylabel('Effective Temperature (K)')
    axes[0].set_xlabel('Predicted Period')

    # 2. Teff vs Kmag heatmap
    pivot2 = pd.pivot_table(
        clean_df,
        values='probability',
        index='Teff_bin',
        columns='Kmag_bin',
        aggfunc='mean'
    )

    sns.heatmap(pivot2, annot=True, cmap='Blues', fmt='.2f', ax=axes[1])
    axes[1].set_title('Binary Probability: Teff vs Kmag Difference')
    axes[1].set_ylabel('Effective Temperature (K)')
    axes[1].set_xlabel('Kmag Difference')

    # 3. Period vs Kmag heatmap
    pivot3 = pd.pivot_table(
        clean_df,
        values='probability',
        index='Period_bin',
        columns='Kmag_bin',
        aggfunc='mean'
    )

    sns.heatmap(pivot3, annot=True, cmap='Blues', fmt='.2f', ax=axes[2])
    axes[2].set_title('Binary Probability: Period vs Kmag Difference')
    axes[2].set_ylabel('Predicted Period')
    axes[2].set_xlabel('Kmag Difference')

    plt.tight_layout()
    plt.show()

    # Create 3D scatter plot to visualize all dimensions at once
    fig = plt.figure(figsize=(30, 18))
    ax = fig.add_subplot(111, projection='3d')

    # Safe function to extract bin midpoints
    def safe_bin_midpoint(bin_str):
        try:
            if not bin_str or bin_str == '':
                return np.nan
            parts = bin_str.split('-')
            if len(parts) != 2:
                return np.nan
            return (float(parts[0]) + float(parts[1])) / 2
        except (ValueError, TypeError):
            return np.nan

    # Create midpoint mappings
    teff_midpoints = {bin_val: safe_bin_midpoint(bin_val) for bin_val in clean_df['Teff_bin'].unique()}
    period_midpoints = {bin_val: safe_bin_midpoint(bin_val) for bin_val in clean_df['Period_bin'].unique()}
    kmag_midpoints = {bin_val: safe_bin_midpoint(bin_val) for bin_val in clean_df['Kmag_bin'].unique()}

    # Convert bin labels to midpoints
    x = [teff_midpoints.get(bin_val, np.nan) for bin_val in clean_df['Teff_bin']]
    y = [period_midpoints.get(bin_val, np.nan) for bin_val in clean_df['Period_bin']]
    z = [kmag_midpoints.get(bin_val, np.nan) for bin_val in clean_df['Kmag_bin']]

    # Filter out any NaN values
    valid_points = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x_valid = np.array(x)[valid_points]
    y_valid = np.array(y)[valid_points]
    z_valid = np.array(z)[valid_points]
    probabilities_valid = clean_df['probability'].values[valid_points]

    if len(x_valid) > 0:  # Only plot if we have valid points
        # Plot points with color and size indicating probability
        sc = ax.scatter(x_valid, y_valid, z_valid,
                        c=probabilities_valid,
                        s=probabilities_valid * 500,  # Size by probability
                        cmap='Blues',
                        alpha=0.8)

        ax.set_xlabel('Effective Temperature (K)')
        ax.set_ylabel('Predicted Period')
        ax.set_zlabel('Kmag Difference')

        cbar = plt.colorbar(sc)
        cbar.set_label('Binary Probability')

        plt.title('3D Visualization of Binary Probability')
    else:
        plt.text(0.5, 0.5, "No valid points to display in 3D plot",
                 ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.show()
def visualize_binary_probability(result_df, pivot_columns=['Teff_bin', 'Period_bin']):
    """
    Create a pivot table visualization of binary probabilities.

    Parameters:
    -----------
    result_df : pandas DataFrame
        Output from calculate_binary_probability function
    pivot_columns : list, optional
        Two columns to use for pivot table index and columns

    Returns:
    --------
    pivot_table : pandas DataFrame
        Pivot table showing probabilities
    """
    # Make sure we have exactly 2 columns for pivoting
    if len(pivot_columns) != 2:
        raise ValueError("pivot_columns must contain exactly 2 column names")

    # Find the third column not in pivot_columns
    all_bins = ['Teff_bin', 'Period_bin', 'Kmag_bin']
    for col in all_bins:
        if col not in pivot_columns:
            third_col = col
            break

    # Group by the third column
    grouped_results = {}

    # For each value in the third column, create a pivot table
    for val in result_df[third_col].unique():
        subset = result_df[result_df[third_col] == val]
        pivot = pd.pivot_table(
            subset,
            values='probability',
            index=pivot_columns[0],
            columns=pivot_columns[1],
            fill_value=0
        )
        grouped_results[val] = pivot

    return grouped_results

def godoy_binaries(df, poly, n_bins=20):
    godoy = pd.read_csv('tables/GodoyRivera25_TableA1.csv')
    godoy = godoy.merge(df, right_on='KID', left_on='KIC', how='left')
    godoy = godoy[~godoy['predicted period'].isna()]

    godoy_k = godoy[godoy['kmag_diff'].abs() <= 2]

    # binary_probabilities = calculate_binary_probability(godoy_k)
    #
    # plot_binary_probability(binary_probabilities)
    #
    # # Print the results
    # print(binary_probabilities)
    #
    # # To visualize the results as pivot tables
    # pivot_tables = visualize_binary_probability(binary_probabilities)
    # for kmag_bin, pivot_table in pivot_tables.items():
    #     print(f"\nFor {kmag_bin} bin:")
    #     print(pivot_table)

    godoy['flag_sync_binary'] = godoy.apply(lambda x: x['predicted period'] < poly(x['Teff']), axis=1)
    godoy_fast = godoy[godoy['predicted period'] < 7]
    godoy_binary = godoy[godoy['flag_Binary_Union']==True]
    godoy_non_binary = godoy[godoy['flag_Binary_Union']==False]
    ruwe = godoy_binary[godoy_binary['flag_RUWE']]
    rv = godoy_binary[godoy_binary['flag_RVvariable']]
    nss = godoy_binary[godoy_binary['flag_NSS']]
    eb_kepler = godoy_binary[godoy_binary['flag_EB_Kepler']]

    dfs = [godoy_binary, ruwe, rv, nss, eb_kepler, godoy, godoy_fast]
    names = ['All Binaries', 'RUWE', 'RV', 'NSS', 'EB_Kepler', 'All Stars', 'Fast Rotators']
    for df, name in zip(dfs, names):
        if 'All' in name or 'Fast' in name:
            plt.hist(df['kmag_diff'], histtype='step', density=True, label=name)
    plt.legend()
    plt.xlabel('$\Delta K_{iso}$')
    plt.ylabel('PDF')
    plt.show()


    def sync_binary_frac(df):
        return df['flag_sync_binary'].sum() / len(df)
    fracs_fast = sync_binary_frac(godoy_fast)
    fracs_all = sync_binary_frac(godoy)

    print("fraction fast: ", fracs_fast)
    print("fraction all: ", fracs_all)

    plt.scatter(godoy_fast['Teff'], godoy_fast['predicted period'], c=godoy_fast['flag_sync_binary'])
    x = np.linspace(3500, 7000, 200)
    y = poly(x)
    plt.plot(x,y, c='gold')
    plt.show()

    p_bins = np.linspace(3, 30, n_bins + 1)
    t_bins = np.linspace(3500, 7000, 20)
    fracs_all = []
    fracs_fast = []
    fracs_b = []
    fracs_non_b = []
    fracs_ruwe = []
    fracs_rv = []
    fracs_nss = []
    fracs_eb_kepler = []
    # for p in p_bins:
    #     binaries_p = godoy_binary[godoy_binary['predicted period'] < p]
    #     non_binaries_p = godoy_non_binary[godoy_non_binary['predicted period'] < p]
    #     ruwe_p = ruwe[ruwe['predicted period'] < p]
    #     rv_p = rv[rv['predicted period'] < p]
    #     nss_p = nss[nss['predicted period'] < p]
    #     eb_kepler_p = eb_kepler[eb_kepler['predicted period'] < p]
    #     fracs_b.append(sync_binary_frac(binaries_p))
    #     fracs_non_b.append(sync_binary_frac(non_binaries_p))
    #     fracs_ruwe.append(sync_binary_frac(ruwe_p))
    #     fracs_rv.append(sync_binary_frac(rv_p))
    #     fracs_nss.append(sync_binary_frac(nss_p))
    #     fracs_eb_kepler.append(sync_binary_frac(eb_kepler_p))

    for t in t_bins:
        all_p = godoy[godoy['Teff'] < t]
        fast_p = godoy_fast[godoy_fast['Teff'] < t]
        binaries_p = godoy_binary[godoy_binary['Teff'] < t]
        non_binaries_p = godoy_non_binary[godoy_non_binary['Teff'] < t]
        ruwe_p = ruwe[ruwe['Teff'] < t]
        rv_p = rv[rv['Teff'] < t]
        nss_p = nss[nss['Teff'] < t]
        eb_kepler_p = eb_kepler[eb_kepler['Teff'] < t]
        fracs_all.append((sync_binary_frac(all_p)))
        fracs_fast.append(sync_binary_frac(fast_p))
        fracs_b.append(sync_binary_frac(binaries_p))
        fracs_non_b.append(sync_binary_frac(non_binaries_p))
        fracs_ruwe.append(sync_binary_frac(ruwe_p))
        fracs_rv.append(sync_binary_frac(rv_p))
        fracs_nss.append(sync_binary_frac(nss_p))
        fracs_eb_kepler.append(sync_binary_frac(eb_kepler_p))

    plt.plot(t_bins, np.array(fracs_fast), label='fast rotators')
    plt.plot(t_bins, np.array(fracs_all), label='all sample')
    plt.xlabel('upper $T_{eff}$')
    plt.ylabel('Fraction below boundary line')
    # plt.legend()
    # plt.gca().invert_xaxis()
    # plt.show()

    plt.plot(t_bins, np.array(fracs_ruwe), label='RUWE')
    plt.plot(t_bins, np.array(fracs_rv), label='RV')
    plt.plot(t_bins, np.array(fracs_nss), label='NSS')
    plt.plot(t_bins, np.array(fracs_eb_kepler), label='EB_Kepler')
    plt.plot(t_bins, np.array(fracs_all), label='All stars')
    # plt.xlabel('upper $T_{eff}$')
    # plt.ylabel('fraction')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig('imgs/godoy_binaries_t_vs_frac.png')
    plt.show()

    return godoy, godoy_binary

def get_binary_prob(df, probs_model, params_cols=['Teff', 'predicted period'], min_val=None, max_val=None):
    params = df[params_cols].values
    young_frac = probs_model.predict(params)
    # young_prob = 1 / (1 + np.exp(-young_frac))
    # Scale to [0,1] range
    min_val = min_val or np.min(young_frac)
    max_val = max_val or np.max(young_frac)

    print("using min and max vals: ", min_val, max_val)

    if max_val > min_val:  # Avoid division by zero
        young_prob = (young_frac - min_val) / (max_val - min_val)
    else:
        young_prob = np.ones_like(young_frac) * 0.5
    df['b_prob'] = 1 - young_prob
    df['b_prob'] = df['b_prob'].astype(np.float32).round(3)
    return df

def plot_binary_probs(binaries, t_binaries, poly, probs_model):

    print("average prob binaries: ", np.mean(binaries['b_prob']))
    print("average prob triples: ", np.mean(t_binaries['b_prob']))

    min_T = min(binaries['Teff'].min(), t_binaries['Teff'].min())
    max_T = max(binaries['Teff'].max(), t_binaries['Teff'].max())
    x = np.linspace(min_T, max_T, 200)
    y = poly(x)
    # binaries = binaries.merge(kinematic_df[['KID','sigma']], on='KID', how='left')
    # t_binaries = t_binaries.merge(kinematic_df[['KID', 'sigma']], on='KID', how='left')
    plt.scatter(binaries['Teff'], binaries['predicted period'], c=binaries['b_prob'])
    plt.scatter(t_binaries['Teff'], t_binaries['Prot'], c=t_binaries['b_prob'])
    plt.plot(x, y, color='salmon', alpha=0.5)

    plt.xlabel('Teff (K)')
    plt.ylabel('$P_{rot}$ (Days)')
    plt.colorbar(label='Probability')
    plt.gca().invert_xaxis()
    plt.savefig('imgs/teff_prot_b_prob_sigma.png')
    plt.show()

def potential_binaries(df, clusters):
    binaries = df[df['binary']]
    binaries['Prot_ref'] = 'Kamai24'
    binaries = binaries[['KID', 'predicted period', 'Teff', 'logg', 'FeH', 'kmag_diff',
                         'kmag_abs', 'Kmag_MIST', 'Prot_ref']]
    bouma24 = pd.read_csv('tables/bouma2024_planets.csv')
    bouma_fast = bouma24[(bouma24['li_median'] < 300) & (bouma24['gyro_median'] < 300)]
    binaries = binaries[~binaries['KID'].isin(bouma_fast['kepid'])]
    binaries_final = binaries[~binaries['KID'].isin(clusters['KID'])]
    return binaries_final

def short_binaries(df_path, p_thresh=7):
    (df_full, df_short, simonian, gaia_nss, gaia_wide, ebs,
     clusters, all_clusters) = get_all_tables(df_path, p_thresh, read_csv=True)
    df_full = df_full[df_full['predicted period'] > 3]

    ebs['nss_solution_type'] = 'EclipsingBinaryKepler'
    poly = plot_clusters_period(all_clusters)
    df_full['binary'] = df_full.apply(lambda x: poly(x['Teff']) > x['predicted period'], axis=1)
    df_full['t_conv'] = df_full.apply(lambda x: tau_circ_convective(x['predicted period'], x['Mstar']), axis=1)
    print("before filtering: ", len(df_full))
    df = filter_main_sequence_with_logg(df_full, n_bins=40)
    print("after filtering: ", len(df))
    godoy, godoy_binary = godoy_binaries(df, poly, 20)
    known_binaries = pd.concat([gaia_nss, ebs])
    known_binaries = pd.concat([known_binaries, pd.get_dummies(known_binaries['nss_solution_type'])])
    known_binaries['is_eb'] = known_binaries['nss_solution_type'].astype(str).apply(
        lambda val: 'EclipsingBinary' in val)
    kinematic_df, probs_model = kinematic_age(df_full, known_binaries, poly)
    # df_full = df_full.merge(kinematic_df[['KID', 'sigma']], on='KID')

    # separation_fractions(known_binaries, df, poly)
    gaia_nss_ms = filter_main_sequence_with_logg(gaia_nss, n_bins=40)
    ebs_ms = filter_main_sequence_with_logg(ebs, n_bins=40)
    known_binaries_ms = pd.concat([gaia_nss_ms, ebs_ms])
    #
    df_reduced, teff_bins, p_cutoffs, err_teff, err_p, kmag_boundary = magnitude_displacement(df, known_binaries,
                                                                        min_feh=-0.05, max_feh=0.05)
    binaries_final = potential_binaries(df, clusters)

    triples_final = potential_triples(known_binaries_ms, df, clusters, max_teff=6500, boundary_fn=kmag_boundary)

    triples_final['binary'] = triples_final.apply(lambda x: poly(x['Teff']) > x['Prot'], axis=1)
    t_binaries = triples_final[triples_final['binary']]
    t_binaries = get_binary_prob(t_binaries, probs_model, ['Teff', 'Prot'])
    binaries_final = get_binary_prob(binaries_final, probs_model, ['Teff', 'predicted period'])
    plot_binary_probs(binaries_final, t_binaries, poly, probs_model)

    binaries_final[['predicted period', 'kmag_diff', 'kmag_abs', 'Kmag_MIST', 'b_prob']] = \
        binaries_final[['predicted period', 'kmag_diff', 'kmag_abs', 'Kmag_MIST', 'b_prob']].round(decimals=3)
    print("final numer of binaries: ", len(binaries_final))
    print("final number of triples: ", len(triples_final))

    t_binaries.to_csv('tables/triples_catalog.csv', index=False)
    binaries_final.to_csv('tables/binaries_catalog.csv', index=False)

    planet_host(df_full, poly, probs_model, clusters)

    plot_binary_props(df_full, gaia_nss, 'gaia_nss')
    plot_binary_props(df_full, ebs, 'kepler_ebs', p_err_name='period_err', e_name='e')





def plot_p_cutoff_with_separation_line(err_teff, p_cutoffs, poly, teff_bins):
    x = np.linspace(np.min(teff_bins), np.max(teff_bins), 200)
    y = poly(x)
    plt.plot(x, y, c='gold')
    plt.scatter(teff_bins, p_cutoffs, c='black', s=100)
    plt.errorbar(teff_bins, p_cutoffs, color='black', xerr=err_teff, fmt='none', alpha=0.5)
    plt.gca().invert_xaxis()
    plt.xlabel("$T_{eff}$")
    plt.ylabel('$P_{rot}$')
    plt.savefig('imgs/p_cutoff_vs_initial_p.png')
    plt.show()


def plot_binaries_frac(df, n_bins=10, teff_min=3800, teff_max=7200, ax=None, c='r', label=''):
    if ax is None:
        fig, ax = plt.subplots()
    df = df.dropna(subset=['kmag_diff', 'position'])
    bins = np.linspace(teff_min, teff_max, n_bins + 1)  # Create n_bins evenly spaced bins
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Get bin centers for plotting
    binary_fracs = []
    errs = []
    for i in range(len(bins) - 1):
        # Select data in the current bin range
        df_t = df[(df['Teff'] >= bins[i]) & (df['Teff'] < bins[i + 1])]
        print(bin_centers[i], len(df_t))
        if len(df_t) > 2:
            n_total = len(df_t)
            n_binary = len(df_t[df_t['position'] == 'binary'])

            # Calculate binary fraction
            binary_frac = n_binary / n_total
            binary_fracs.append(binary_frac)
            err = binary_frac * np.sqrt(1 / n_binary + 1 / n_total) if n_binary > 0 else 0
            errs.append(err)
        else:
            binary_fracs.append(np.nan)
            errs.append(np.nan)

    binary_fracs = np.array(binary_fracs)
    errs = np.array(errs)

    # Plot the results with error bars
    ax.errorbar(bin_centers, binary_fracs, yerr=errs, fmt='-o',
                 c=c, markersize=10, capsize=5)
    # Plot the results
    ax.plot(bin_centers, binary_fracs,'-o', label=label, c=c, markersize=10)
    ax.set_xlabel('$T_{eff}$ (K)')
    ax.set_ylabel('Fraction of binaries (%)')
    print('avg: ', np.nanmean(binary_fracs))

    return ax

def sample_binaries(n_samples=int(1e5), n_iters=int(1e4)):
    qs = np.random.uniform(size=n_samples)
    binaries = np.random.choice(a=qs, size=n_iters, replace=False)
    luminosity = (1+binaries) ** 4
    plt.hist(luminosity, histtype='step', density=True, label='luminosity Excess')
    # plt.hist((1+binaries), histtype='step', density=True, label='mass Excess')
    plt.xlabel('q')
    plt.ylabel('PDF')
    plt.show()

def plot_period_cutoffs(df, p_thresh, p_min=4, bin_size=2, ax=None, linestyle='-',
                         p_thresh_long=50, cutoff_ref_p=8, p_name='predicted period',
                        teffs=None, data_col='kmag_diff', method='simple',
                        plot_every=2, fit=True, name='norm'):
    teffs = np.array(teffs) if teffs is not None else np.arange(4000, 7000, 200)
    avgs_dicts = dict()
    ps = np.arange(p_min, p_thresh, 1)
    ps_short = np.arange(p_min, 25, 1)
    p_ref_idx = np.where(ps==cutoff_ref_p)[0][0]
    ps_long = np.arange(p_thresh, p_thresh_long, 1)
    minima_points = []
    mid_teffs = []
    colors = get_cmap('tab20')
    if ax is None:
        fig, ax = plt.subplots(1,1)
    for i, teff in enumerate(teffs[:-1]):
        if i < len(teffs) - 1:
            df_t = df[(df['Teff'] >= teff) & (df['Teff'] <= teffs[i + 1])]
            label = r'%s $ < T_{eff} \leq$ %s' % (teff, teffs[i + 1])
            mid_teff = (teff + teffs[i+1]) // 2

        else:
            df_t = df[df['Teff'] >= teff]
            label = r'$T_{eff} \geq $ %s' % (teff)

        # fracs, errors, avgs = get_over_lum_fracs(df_t, data_col, ps, p_name=p_name,
        #                                    method=method, upper_env=g_r_upper, lower_env=g_r_lower)
        # fracs_b, errors_b, avgs_b = get_over_lum_bin(df_t, data_col, ps_short, p_name=p_name,
        #                                    method=method, upper_env=g_r_upper, lower_env=g_r_lower, bin_size=bin_size)
        fracs, bins, errors, avgs = get_binary_frac(df_t, ps, data_col, cumulative=True)
        fracs_b, bins_b, errors_b, avgs_b = get_binary_frac(df_t, ps, data_col, bin_size=bin_size)

        if len(fracs):
            fracs = fracs / fracs[-1]
            thresh_idx = np.where(fracs > 1.06)[0]
            if len(thresh_idx):
                minima_points.append(bins[thresh_idx[-1]])
            else:
                minima_points.append(np.nan)
        else:
            minima_points.append(np.nan)
        mid_teffs.append(mid_teff)


        # if i % plot_every == 0:

            # if len(ps_valid_b) > 2:
            #     ax[0].plot(ps_valid_b, fracs_b, color=colors(i), linestyle=linestyle)
            #     if linestyle == '-':
            #         ax[0].errorbar(ps_valid_b, fracs_b, yerr=errors_b,
            #                       color=colors(i), alpha=0.2, barsabove=True,
            #                      fmt='none', capsize=10, markersize=4)
        ax.plot(bins, fracs, color=colors(i), label=label, linestyle=linestyle)
        if linestyle == '-':
            ax.errorbar(bins, fracs, yerr=errors,
                          color=colors(i), alpha=0.2, barsabove=True,
                           fmt='none', capsize=10, markersize=4)

    if fit:
        x = np.array(teffs[:-1])[~np.isnan(minima_points)] + (dt / 2)
        y = np.array(minima_points)[~np.isnan(minima_points)]
        # popt, _ = curve_fit(power_cuvre, x, y, p0=[0, -0.01])
        # max_idx = np.argmax(y)
        # x_down = x[max_idx:]
        # y_down = y[max_idx:]
        coeff = np.polyfit(np.log(x[:-1]),np.log(y[:-1]),1)
        print("coeffs: ", coeff)
        poly = np.poly1d(coeff)
        # if max_idx > 0:
        #     x_up = x[:max_idx]
        #     y_up = y[:max_idx]
        #     coeff_up = np.polyfit(np.log(x_up),np.log(y_up),1)
        #     poly_up = np.poly1d(coeff_up)
        # else:
        #     poly_up = None
        return ax, x, y,  poly
    else:
        return ax, mid_teffs, minima_points, None


def compare_cum_period_cutoffs(df, df2, p_thresh, p_min=4, bin_size=2, ax=None, linestyle='-',
                         p_thresh_long=50, cutoff_ref_p=8, p_name='predicted period',
                        teffs=None, data_col='kmag_diff', method='simple',
                        plot_every=2, fit=True, name='norm'):
    teffs = teffs if teffs is not None else np.arange(4000, 7000, 200)
    avgs_dicts = dict()
    ps = np.arange(p_min, p_thresh, 1)
    ps_short = np.arange(p_min, 25, 1)
    p_ref_idx = np.where(ps==cutoff_ref_p)[0][0]
    ps_long = np.arange(p_thresh, p_thresh_long, 1)
    minima_points = []
    colors = get_cmap('tab20')
    g_r_upper = np.load('imgs/tables/g-r_upper_envelope.npy')
    g_r_lower = np.load('imgs/tables/g-r_lower_envelope.npy')
    if ax is None:
        fig, ax = plt.subplots(figsize=[25,12])
    for i, teff in enumerate(teffs):
        if i < len(teffs) - 1:
            df_t = df[(df['Teff'] >= teff) & (df['Teff'] <= teffs[i + 1])]
            df2_t = df2[(df2['Teff'] >= teff) & (df2['Teff'] <= teffs[i + 1])]
            label = r'%s $\leq T_{eff} \leq$ %s' % (teff, teffs[i + 1])

        else:
            df_t = df[df['Teff'] >= teff]
            df2_t = df2[(df2['Teff'] >= teff)]
            label = r'$T_{eff} \geq $ %s' % (teff)

        fracs, errors, avgs = get_over_lum_fracs(df_t, data_col, ps, p_name=p_name,
                                           method=method, upper_env=g_r_upper, lower_env=g_r_lower)
        fracs2, errors2, avgs2 = get_over_lum_fracs(df2_t, data_col, ps, p_name=p_name,
                                           method=method, upper_env=g_r_upper, lower_env=g_r_lower)
        avgs_diff = avgs - avgs2
        errors_diff = np.sqrt(errors**2 + errors2**2)
        valids = np.where(~np.isnan(fracs))[0]
        frac_diff = fracs[valids]
        errors = errors[valids]
        avgs = avgs[valids]
        ps_valid = ps[valids]
        valids2 = np.where(~np.isnan(fracs2))[0]
        fracs2 = fracs2[valids2]
        errors2 = errors2[valids2]
        avgs2 = avgs2[valids2]
        ps_valid2 = ps[valids2]

        if 'norm' in name:
            fracs = fracs / fracs[-1]
            fracs2 = fracs2 / fracs[-1]
            avgs = avgs - avgs[-1]
            avgs2 = avgs2 - avgs2[-1]
            avgs_diff = avgs_diff - avgs_diff[-1]
        if 'avg' in name:
            fracs = avgs
            fracs2 = avgs2
        fracs = savgol(fracs, window_length=3, polyorder=2)
        fracs2 = savgol(fracs2, window_length=3, polyorder=2)
        if i % plot_every == 0:
            ax.plot(ps, avgs_diff, color=colors(i), label=label)
            # ax.plot(ps_valid2, fracs2, color=colors(i), label=label, linestyle=linestyle)
            ax.errorbar(ps, avgs_diff, yerr=errors_diff,
                          color=colors(i), alpha=0.2, barsabove=True,
                         fmt='none', capsize=10, markersize=4)
            if 'avg' in name:
                ax.invert_yaxis()
                ax.set_xlim(p_min,10)
                # ax[1].set_xlim(p_min,20)
                ax.set_xlabel(r'$P_{rot}$ (Days)')
                ax.set_ylabel(r'$avg_{\Delta K_{env}}(\Delta K_{iso})$ diff.')
    ax.legend()
    plt.savefig(f'imgs/{data_col}_vs_p_thresh_{name}.png')
    plt.show()

def show_period_cutoff(ax, p_name, data_col='kmag_diff', name='all'):
    ax.set_xlabel(r'$P_{cutoff}$ (Days)')
    ax.set_ylabel(r'Binaries Fraction (%)')
    # ax[1].set_ylabel(r'Binaries Fraction (%)')

    ax.legend()
    # ax.hlines(0.75, 0, 50, linestyle='dashed', colors='gray')
    # ax[1].set_ylim(0.9, 2)
    plt.tight_layout()
    plt.savefig(f'imgs/{data_col}_vs_p_thresh_{name}.png')
    plt.show()


def fit_period_cutoffs(teffs, minima_points, p_name, cutoff_t):
    plt.scatter(teffs, minima_points, s=400)
    plt.xlabel(r'$T_{\mathrm{eff}}$')
    plt.ylabel(r'$P_{cut}$')
    plt.legend(fontsize=10, loc="upper left", handlelength=2, frameon=False)
    plt.savefig(f'imgs/local_minima_points_{p_name}.png')
    plt.show()
    teffs = teffs[:len(minima_points)]
    coefficients = np.polyfit(teffs, minima_points, 3)
    print('coefs: ', coefficients)
    poly = np.poly1d(coefficients)
    # Generate fitted values
    x_fit = np.linspace(min(teffs), max(teffs), 1000)
    y_fit = poly(x_fit)

    a3, a2, a1, a0 = coefficients
    label = (
        r"$y = {:.2g}x^3 + {:.2g}x^2 + {:.2g}x + {:.2g}$"
        .format(a3, a2, a1, a0)
    )
    label = label.replace("e", r"\times 10^{").replace("+", "") + "}"
    print(label)
    print(a3, a2, a1, a0)


    np.save(f'imgs/tables/{p_name}_cutoff_best_fits.npy', coefficients)

    cutoff_t_idx = np.where(np.logical_and(cutoff_t[1] > teffs, teffs > cutoff_t[0]))[0]
    plt.scatter(np.log(teffs), np.log(minima_points), s=400)
    minima_points = np.array(minima_points)

    log_fit = np.polyfit(np.log(teffs[cutoff_t_idx]), np.log(minima_points[cutoff_t_idx]), 1)
    poly_log = np.poly1d(log_fit)
    print(log_fit)
    plt.xlabel(r'$\log{T_{\mathrm{eff}}}$')
    plt.ylabel(r'$\log{P_{cut}}$')
    plt.plot(np.log(teffs[cutoff_t_idx]), poly_log(np.log(teffs[cutoff_t_idx])), c='r')
    plt.savefig(f'imgs/local_minima_points_{p_name}_log.png')
    plt.show()
    np.save(f'imgs/tables/{p_name}_cutoff_best_fit_log.npy', coefficients)


def plot_avgs_mag(df, data_col, teffs, p_thresh, p_min=3, method='g-r boundary', ):
    ps = np.arange(p_min, p_thresh, 1)
    g_r_upper = np.load('imgs/tables/g-r_upper_envelope.npy')
    g_r_lower = np.load('imgs/tables/g-r_lower_envelope.npy')
    for i, teff in enumerate(teffs):
        if i < len(teffs) - 1:
            df_t = df[(df['Teff'] >= teff) & (df['Teff'] <= teffs[i + 1])]
            label = r'%s $\leq T_{eff} \leq$ %s' % (teff, teffs[i + 1])
        else:
            df_t = df[df['Teff'] >= teff]
            label = r'$T_{eff} \geq $ %s' % (teff)
        fracs, errors, avgs = get_over_lum_fracs(df_t, data_col, ps, p_name='predicted period',
                                                 method=method, upper_env=g_r_upper, lower_env=g_r_lower)
        plt.plot(ps , avgs, label=label)
    plt.xlabel('$P_{rot}$')
    plt.ylabel('avg magnitude')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
def get_over_lum_bin(df, data_col, ps, bin_size=1, p_name='predicted period',
                       method='simple', mag_name='kmag_diff_detrended',
                       upper_env=None, lower_env=None):
    over_lum = []
    errors = []
    avgs = []
    for p in ps:
        df_b = df[(df[p_name] <= (p + bin_size / 2)) & (df[p_name] > (p - bin_size / 2))]
        if not len(df_b):
            over_lum.append(np.nan)
            errors.append(np.nan)
            avgs.append(np.nan)
            continue
        if method == 'simple':
            over_lum_bin = (df_b[data_col] < 0).values
        elif 'boundary' in method:

            points_b = np.column_stack([
                df_b['Teff'].values,
                df_b[data_col].values
            ])
            over_lum_bin = is_inside_shape(points_b)
            if 'g-r' in method:

                df_b['upper_env'] = df_b['GRCOLOR_abs'] * upper_env[0] + upper_env[1]
                df_b['lower_env'] = df_b['GRCOLOR_abs'] * lower_env[0] + lower_env[1]
                df_b['in_env'] = (df_b['RMAG_abs'] < df_b['upper_env']) & (df_b['RMAG_abs'] > df_b['lower_env'])
                over_lum_bin = np.logical_and(over_lum_bin, df_b['in_env'].values)
        N = over_lum_bin.sum()
        total = len(df_b)
        frac = N / total
        avg = df_b.loc[over_lum_bin, data_col].mean()
        # Poisson error calculation
        error = np.sqrt(N) / total
        over_lum.append(frac)
        errors.append(error)
        avgs.append(avg)
    return np.array(over_lum)  , np.array(errors), np.array(avgs)


def get_over_lum_fracs(df, data_col, ps, p_name='predicted period',
                       method='simple', mag_name='kmag_diff_detrended',
                       upper_env=None, lower_env=None):
    over_lum_fracs = []
    errors = []
    avgs = []
    for p in ps:
        df_p = df[(df[p_name] <= p)]
        # print(len(df_p), end=' ')
        if method == 'simple':
            over_lum = (df_p[data_col] < 0).values
        elif 'boundary' in method:
            points = np.column_stack([
                df_p['Teff'].values,
                df_p[data_col].values
            ])
            over_lum = is_inside_shape(points)
            if 'g-r' in method:
                df_p['upper_env'] = df_p['GRCOLOR_abs'] * upper_env[0] + upper_env[1]
                df_p['lower_env'] = df_p['GRCOLOR_abs'] * lower_env[0] + lower_env[1]
                df_p['in_env'] = (df_p['RMAG_abs'] < df_p['upper_env']) & (df_p['RMAG_abs'] > df_p['lower_env'])
                over_lum = np.logical_and(over_lum, df_p['in_env'].values)

        N = over_lum.sum()
        total = len(df_p)
        frac = N / total
        avg = df_p.loc[over_lum, data_col].mean()
        # Poisson error calculation
        error = np.sqrt(N) / total
        over_lum_fracs.append(frac)
        errors.append(error)
        avgs.append(avg)
    return np.array(over_lum_fracs), np.array(errors), np.array(avgs)


def get_binary_frac(df, ps, data_col, p_name='predicted period', bin_size=2, cumulative=False):
    if cumulative:
        binaries_frac = []
        errors = []
        avgs = []
        bins_mid = []
        for p in ps:
            df_b = df[df[p_name] <= p]
            if len(df_b) > 2:
                n_total = len(df_b)
                n_binary = len(df_b[df_b['binary']])
                # n_binary = len(df_b[df_b[data_col] < -0.1])
                binary_frac = n_binary / n_total or np.nan

                # Proper error propagation for binary fraction
                error = binary_frac * np.sqrt(1 / n_binary + 1 / n_total) if n_binary > 0 else 0

                binaries_frac.append(binary_frac)
                errors.append(error)
                avgs.append(df_b[df_b['binary']][data_col].mean())
                bins_mid.append(p)
    else:
        min_p = min(ps)
        max_p = max(ps)
        bins = np.arange(min_p, max_p + bin_size, bin_size)

        df['bin'] = pd.cut(df[p_name], bins=bins, right=False)
        bin_counts = df.groupby('bin').size()

        binaries_frac = []
        bins_mid = []
        errors = []
        avgs = []

        all_bins = pd.IntervalIndex.from_breaks(bins, closed='left')
        for bin_interval in all_bins:
            if bin_interval in bin_counts.index and bin_counts[bin_interval] >= 5:
                df_b = df[df['bin'] == bin_interval]
                n_total = len(df_b)
                n_binary = len(df_b[df_b['binary']])
                binary_frac = n_binary / n_total

                error = binary_frac * np.sqrt(1 / n_binary + 1 / n_total) if n_binary > 0 else 0

                binaries_frac.append(binary_frac)
                errors.append(error)
                avgs.append(df_b[df_b['binary']][data_col].mean())
                bins_mid.append(bin_interval.mid)

        df.drop('bin', axis=1, inplace=True)

    return np.array(binaries_frac), np.array(bins_mid), np.array(errors), np.array(avgs)


def compare_mcq14(df_mcq14):
    plt.scatter(df_mcq14['Prot_mcq14'], df_mcq14['predicted period'], c=df_mcq14['mean_period_confidence'])
    acc20p_idx = np.abs(df_mcq14['predicted period'] - df_mcq14['Prot_mcq14']) < (df_mcq14['Prot_mcq14'] * 0.2)
    acc20p = len(df_mcq14[acc20p_idx]) / len(df_mcq14)
    acc1_idx = np.abs(df_mcq14['predicted period'] - df_mcq14['Prot_mcq14']) < 1
    acc1 = len(df_mcq14[acc1_idx]) / len(df_mcq14)
    acc2_idx = np.abs(df_mcq14['predicted period'] - df_mcq14['Prot_mcq14']) < 2
    acc2 = len(df_mcq14[acc2_idx]) / len(df_mcq14)
    plt.title(f"acc20%: {acc20p:.2f}, acc2 {acc2:.2f}, acc1 {acc1:.2f}")
    plt.xlim(0, 20)
    plt.close()


def get_mazeh_planets(df, disposition='CONFIRMED'):
    kois = pd.read_csv('tables/kois.csv')
    mazeh_planets = (pd.read_csv('tables/mazeh_planets2013.txt')
                     .rename(columns={'keplerid': 'KID'}))
    mazeh_planets = mazeh_planets.merge(kois, on='KID')
    mazeh_planets['planet_Prot'] = (mazeh_planets['planet_Prot'].combine_first(mazeh_planets['P_orb'])
                                    .drop(columns=['P_orb']))
    mazeh_planets['planet_count'] = mazeh_planets.groupby('KID')['KID'].transform('size')
    mazeh_planets = (mazeh_planets[(mazeh_planets['koi_disposition'] == disposition) &
                                   (mazeh_planets['P_rot'] < 7)]
                     .merge(pd.read_csv('tables/berger_catalog.csv')
                            [['KID', 'Mstar', 'FeH', 'Dist', 'Rstar', 'E_Rstar', 'e_Rstar']]
                            , on='KID', how='left')
                     .merge(pd.read_csv('tables/r_var.csv'),
                            on='KID', how='left')
                     .rename(columns={'teff':'Teff'}))
    mazeh_planets = get_mag_data(mazeh_planets, mist_path=MIST_PATH)
    mazeh_planets['P_source'] = 'Mazeh13'

    return mazeh_planets

def segment_acf(lc, segment_len=720*48, segment_step=90*48):
    p_full, lags, acf, peaks, lph = analyze_lc(lc)
    ps = []
    num_segs = (len(lc) - segment_len) // segment_step
    print("lc len: ", len(lc), "number of segments: ", num_segs)
    start = 0
    for i in range(7):
        seg = lc[start:start + segment_len]
        p_acf, lags, acf, peaks, lph = analyze_lc(seg)
        ps.append(p_acf)
        start += segment_step
    return ps, p_full

def validate_cbp(cbp_candidates):
    godoy = pd.read_csv('tables/GodoyRivera25_TableA1.csv')
    godoy_binaries = godoy[godoy['flag_Binary_Union']==True]
    plt.hist(godoy_binaries['RUWE'], bins=100, histtype='step', density=True, label='Binaries')
    plt.hist(godoy['RUWE'], bins=100, histtype='step', density=True, label='All')
    plt.xlim(0,10)
    plt.legend()
    plt.show()

    # kinematic_table = read_machine_readable_table('tables/kinematic_v.txt').to_pandas()
    # kinematic_table['sigma'] = np.sqrt((kinematic_table['U'] ** 2 + kinematic_table['V'] ** 2 + kinematic_table['W'] ** 2) / 3)
    # cbp_candidates = cbp_candidates.merge(kinematic_table[['KIC', 'sigma']], left_on='KID', right_on='KIC', how='left')
    for i, cand in cbp_candidates.iterrows():
        kid = cand['KID']
        ruwe = godoy[godoy['KIC']==kid]['RUWE']
        prob = cand['b_prob']
        kmag_diff = cand['kmag_diff']
        kepler_name = cand['kepler_name']
        print('name: ', kepler_name, 'RUWE: ', ruwe, 'delta K: ', kmag_diff, 'b_prob: ', prob)
        lc = np.load(rf'C:\Users\Ilay\projects\kepler\data\lightPred\data/npy/{kid}.npy')
        lc = savgol(lc, 48, 1, mode='mirror')
        t = np.linspace(0, lc.shape[0]//48, lc.shape[0])
        acf_segment_preds, acf_full_pred = segment_acf(lc)

        pred_cols = [c for c in cand.keys() if 'predicted period_' in c]
        conf_cols = [c for c in cand.keys() if 'period confidence' in c]
        pred_ps = cand[pred_cols].values
        pred_confs = cand[conf_cols].values
        fig, axes = plt.subplots(3,1, figsize=(30,16))
        axes[0].plot(t, lc)
        axes[1].plot(t[5000:7500], lc[5000:7500])
        sc = axes[2].scatter(np.arange(len(pred_cols)), pred_ps.squeeze(), c=pred_confs, s=100)
        axes[2].scatter(np.arange(len(acf_segment_preds)), acf_segment_preds, s=100, c='r', label='ACF')
        axes[2].hlines(pred_ps.squeeze().mean(), 0, len(pred_cols), linestyles='dashed', colors='gray', label='lightPred Avg')
        axes[2].hlines(np.median(pred_ps.squeeze()), 0, len(pred_cols),  linestyles='dashed', colors='black', label='lightPred Median')
        axes[2].set_xlabel('# Segment')
        axes[2].set_ylabel('$P_{rot}$')
        cbar = fig.colorbar(sc, ax=axes[2])
        cbar.set_label('Confidence')
        axes[2].legend()
        fig.suptitle(f"{cand['kepler_name']}, KID - {cand['KID']}")
        plt.tight_layout()
        plt.savefig(f'imgs/cbp_validation_{kid}.png')
        plt.show()






def planet_host(df, boundary, probs_model, clusters, disposition='CONFIRMED'):
    df_high_conf, short_planets_all = get_potential_circumb(df, boundary, disposition, 0.86)

    short_planets_all.dropna(subset=['binary'], inplace=True)

    df_tight = df[df['kmag_diff'].abs() < 2]
    xval = 'kepler_name' if disposition == 'CONFIRMED' else 'kepoi_name'
    planet_binaries = short_planets_all[(short_planets_all['binary']==True) &
                                        (short_planets_all['Teff'] > 3800)]
    planet_binaries = get_binary_prob(planet_binaries, probs_model,
                                      ['Teff', 'predicted period'],
                                      min_val=-5.3998, max_val=15.906)
    print("average b_prob for planet hosts: ", planet_binaries['b_prob'].mean())
    circumbinary_candidates, fp_candidates = plot_planet_stability(planet_binaries, xval=xval, name=disposition)
    # circumbinary_candidates = circumbinary_candidates.iloc[1:3] # only 1521 and 1184
    validate_cbp(circumbinary_candidates)

    planet_clusters = planet_binaries.merge(clusters[['KID', 'Cluster', 'cluster_age', 'PRot']], on='KID')
    x = np.linspace(planet_binaries['Teff'].min(), planet_binaries['Teff'].max(), 200)
    y = boundary(x)

    T_err = np.vstack((planet_binaries['E_Teff'], -planet_binaries['e_Teff']))
    T_err[np.isnan(T_err)] = 100
    p_err = planet_binaries['predicted period'] * (planet_binaries['total error'] / 2)
    p_err_mazeh = planet_binaries['Prot_err']
    p_err[np.isnan(p_err)] = p_err_mazeh[np.isnan(p_err)]
    plt.scatter(planet_binaries['Teff'], planet_binaries['predicted period'], s=100, c=planet_binaries['b_prob'])
    plt.errorbar(planet_binaries['Teff'], planet_binaries['predicted period'], yerr=p_err, ecolor='gray',
                 xerr=T_err, fmt='none', capsize=5, alpha=0.5)
    plt.plot(x, y, c='salmon')
    plt.colorbar(label='probability')

    # Add text labels
    for i, txt in enumerate(planet_binaries['kepler_name']):
        plt.text(planet_binaries.iloc[i]['Teff'], planet_binaries.iloc[i]['predicted period'] - 0.3,
                 txt, fontsize=16, ha='right', va='bottom')
    plt.xlabel('$T_{eff}$')
    plt.ylabel('$P_{rot}$')
    plt.gca().invert_xaxis()
    plt.savefig('imgs/planet_host_scatter')
    plt.show()

    x = np.linspace(circumbinary_candidates['Teff'].min(), circumbinary_candidates['Teff'].max() + 200, 200)
    y = boundary(x)
    for i, row in circumbinary_candidates.iterrows():
        name = row['kepler_name']
        p_err = row['predicted period'] * (row['total error'] / 2)
        p_err = p_err or row['Prot_err']
        teff_m = MORTON_TEFF[name]
        plt.scatter(row['Teff'], row['predicted period'],s=100, c='black')
        plt.scatter(teff_m, row['predicted period'],s=100, c='r')
        plt.errorbar(row['Teff'], row['predicted period'], yerr=p_err, fmt='none', capsize=5, c='black', alpha=0.5)
        plt.errorbar(teff_m, row['predicted period'], yerr=p_err, fmt='none', capsize=5, c='r', alpha=0.5)
        plt.plot([row['Teff'], teff_m], [row['predicted period'], row['predicted period']], c='gray')
        plt.text(row['Teff'], row['predicted period'] - 0.3,
                 name, fontsize=16, ha='right', va='bottom')
    plt.plot(x,y,c='gold')
    plt.xlabel('$T_{eff}$')
    plt.ylabel('$P_{rot}$')
    plt.gca().invert_xaxis()
    plt.savefig('imgs/circum_candidates')

    plt.show()




    # test_planet_fp(fp_candidates)
    # circumbinary_candidates.to_csv(f'tables/circum_candidates_{disposition}.csv')
    # df_filtered_fast = df_filtered[df_filtered['predicted period'] < 5]
    # plot_scatter(df_filtered_fast, circumbinary_candidates, x_val='predicted period')
    # plot_scatter(df_filtered_fast, circumbinary_candidates, x_val='Teff')
    # plot_scatter(df_filtered_fast, circumbinary_candidates, x_val='Teff', y_val='Mstar')
    # plot_scatter(df_filtered, circumbinary_candidates, x_val='Teff', y_val='Mstar')

    # plot_lightcurves(circumbinary_candidates)

def test_planet_fp(fp_candidates):
    p_orb = []
    p_rot_acf = []
    p_rot_model = []
    for i, row in fp_candidates.iterrows():
        kid = row['KID']
        print(i, kid)
        p_orb.append(row['planet_Prot'])
        res = lk.search_lightcurve(f'KIC {kid}', cadence='long', author='kepler')
        lc = res.download_all().stitch()
        lc = lc.flatten(window_length=401)
        flux = fill_nan_np(np.array(lc.flux), interpolate=True)
        time = np.linspace(0, len(lc) / 48, len(lc))
        p_acf = analyze_lc(flux)[0]
        p_rot_acf.append(p_acf)
        # p_rot_model.append(row['predicted period'])
    plt.scatter(p_orb, p_rot_acf, label='acf', s=100)
    plt.plot(p_rot_acf, p_rot_acf, c='r')
    # plt.scatter(p_orb, p_rot_acf, label='lightPred', alpha=0.5)
    plt.xlabel('$P_{orb}$')
    plt.ylabel('$P_{rot}$')
    plt.legend()
    plt.show()



def get_potential_circumb(df_all, boundary, disposition, conf=0.95):
    df_high_conf = df_all[df_all['mean_period_confidence'] > conf]
    df = df_high_conf[(df_high_conf['koi_disposition'] == disposition)]
    df['period_ratio'] = df['planet_Prot'] / df['predicted period']
    df['P_source'] = 'Kamai24'
    mazeh_planets = get_mazeh_planets(df_all, disposition)
    mazeh_planets['period_ratio'] = mazeh_planets['planet_Prot'] / mazeh_planets['P_rot']
    short_planets_all = pd.concat([df, mazeh_planets], axis=0)
    short_planets_all = short_planets_all[~short_planets_all['KID'].duplicated()]

    short_planets_all = get_semi_major_axis(short_planets_all,
                                            'planet_Prot',
                                            comp_mass=1/1000,
                                            p_err_name='Prot_err')
    short_planets_all['a/R'] = short_planets_all['a'] * cgs.au / short_planets_all['Rstar'] / cgs.R_sun
    short_planets_all['R_err'] = (short_planets_all['E_Rstar'] - short_planets_all['e_Rstar']) / 2
    short_planets_all['a/R_err'] = (short_planets_all.apply
                                    (lambda x: fractional_error(x['a'] * cgs.au, x['Rstar'] * cgs.R_sun,
                                            x['a_err'] * cgs.au, x['R_err'] * cgs.R_sun),
                                            axis=1))
    short_planets_all['predicted period'] = (short_planets_all['predicted period'].fillna
                                             (short_planets_all['P_rot']))
    short_planets_all['P_err'] = short_planets_all['total error'] * short_planets_all['predicted period'] / 2
    short_planets_all['P_err'] = (short_planets_all['P_err'].fillna
                                             (short_planets_all['P_rot_err']))
    short_planets_all['period_ratio_err'] = (short_planets_all.apply
                                             (lambda x: fractional_error(x['planet_Prot'],
                                                                         x['predicted period'],
                                                    x['Prot_err'], x['P_err']), axis=1))
    short_planets_all['binary'] = short_planets_all.apply(lambda x:
                                                          boundary(x['Teff'])
                                                          > x['predicted period'] + x['predicted period'] * 0.1
                                                          , axis=1)
    return df_high_conf, short_planets_all


def get_float(value):
    if (isinstance(value, str)) and (value.lower() == 'false'):
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan

    return float(value)

def cluster_gyro_ratio(calib_path, gyro_path, teff_thresh=5450):
    df_calib = Table.read(calib_path, encoding='utf-8', format='ascii.cds').to_pandas()
    df_gyro = Table.read(gyro_path, encoding='utf-8', format='ascii.cds').to_pandas()
    df_calib = (df_calib.merge(df_gyro, right_on='Gaia', left_on='DR3', how='left', suffixes=('', '_2024'))
                .dropna(subset=['tGyro']))
    df_calib['Prot'] = np.where(df_calib['Prot'].notna(), df_calib['Prot'], df_calib['PRot'])
    df_calib.drop(columns=['PRot'], inplace=True)
    new_clusters = get_all_clusters()
    long_clusters = Table.read('imgs/tables/long2023.fit', format='fits').to_pandas()
    long_clusters = process_long_clusters(gaia_cross_match(long_clusters, merge_on='Gaia', dr=3))
    all_clusters = pd.concat([df_calib, new_clusters])
    clusters_names = all_clusters['Cluster'].unique()
    poly_coeff = np.load('imgs/tables/clusters_separation.npy')
    poly = np.poly1d(poly_coeff)
    all_clusters['binary_p'] = all_clusters.apply(lambda x : poly(x['Teff']) > x['Prot']
                                                    if x['Teff'] is not None else False, axis=1)
    all_clusters['binary_sep'] = all_clusters.apply(lambda x: poly(x['Teff']) - x['Prot'], axis=1)
    ages = []
    fig, axs = plt.subplots()
    for cluster_name in clusters_names:
        age = CLUSTERS[cluster_name]
        # if age > 150:
        #     continue
        ages.append(age)
        print(cluster_name, age)
        data = all_clusters[all_clusters['Cluster'] == cluster_name]
        # axs.scatter(np.ones(len(data)) * age, data['tGyro'] / age, s=100, c=data['binary_p'])
        axs.scatter(data['binary_sep'], data['tGyro'] / age, label=age)
        # scatter_colors = ['red' if hot else 'blue' for hot in data['hot']]
        # plt.scatter(np.ones(len(data)) * age, data['tGyro'] / age, c=scatter_colors)
    # axs.hlines(1, 0 ,max(ages), linestyles='dashed', color='black')
    # plt.xticks(ages, clusters_names, rotation='vertical')
    axs.semilogy()
    # plt.legend(handles=[hot_patch, cool_patch], loc='best')
    # plt.colorbar(label='teff')
    fig.supylabel('Age ratio (Myr)')
    fig.supxlabel(r'Distance to boundary (days)')
    # fig.supxlabel('Cluster Age (Myr)')
    # fig.supylabel(r'$\frac{t_{Gyro}}{t_{Cluster}}$')
    plt.savefig('imgs/all_clusters_compare.png')
    plt.legend()
    plt.show()

    cat = pd.read_csv('tables/berger_catalog.csv')
    all_clusters = all_clusters.merge(cat[['KID', 'Rstar', 'E_Rstar', 'e_Rstar']], left_on='KIC', right_on='KID')
    all_clusters = all_clusters[all_clusters['Teff'] > 4800]
    all_clusters = get_vsini(all_clusters).dropna(subset=['vsini', 'Teff'])
    xs = all_clusters['Teff'].sort_values()
    scatter = plt.scatter(all_clusters['Teff'],
                          all_clusters['vsini'],
                          c=all_clusters['Cluster'].astype('category').cat.codes,  # Convert to numerical categories
                          cmap='tab10')  # Use a colormap suitable for categorical data

    # Add the polynomial fit line
    plt.plot(xs, poly(xs), color='red', label='Polynomial fit')

    # Add legend using the scatter plot handles
    legend1 = plt.legend(scatter.legend_elements()[0],
                         all_clusters['Cluster'].unique(),
                         title="Clusters",
                         loc="upper left")
    plt.gca().add_artist(legend1)

    # Customize the plot
    plt.xlabel('Teff')
    plt.ylabel('vsini')

    plt.show()

def age_tests(calib_path, gyro_path):
    df_calib = Table.read(calib_path, encoding='utf-8', format='ascii.cds').to_pandas()
    df_calib.rename(columns={'PRot': 'Prot'}, inplace=True)
    df_calib = df_calib[df_calib['Prot'] > 0]
    berger_cat = pd.read_csv('tables/berger_catalog.csv')
    df_gyro = Table.read(gyro_path, encoding='utf-8', format='ascii.cds').to_pandas()
    new_clusters = get_all_clusters()
    df_combined = df_calib.merge(df_gyro, right_on='Gaia', left_on='DR3', how='left', suffixes=('', '_2024'))
    df_gyro = df_gyro.dropna(subset=['tGyro'])
    df_gyro = df_gyro.merge(berger_cat[['KID', 'Rstar', 'E_Rstar',
                                                'e_Rstar', 'Mstar', 'E_Mstar',
                                                'e_Mstar', 'Lstar', 'E_Lstar', 'e_Lstar']], left_on='KIC', right_on='KID')
    # df_gyro = get_vsini(df_gyro)
    scatter_p_cut_age(df_gyro)
    exit()
    # hot_patch = mpatches.Patch(color='red', label=r'$T_{eff} > 5450 K$')
    # cool_patch = mpatches.Patch(color='blue', label=r'$T_{eff} < 5450 K$')
    teff_min = df_combined['Teff'].min()
    teff_max = df_combined['Teff'].max()
    norm = mcolors.Normalize(vmin=teff_min, vmax=teff_max)


    # df = convert_cols_to_float(df, cols=['Teff', 'r_Prot'])
    df_calib = df_calib.dropna(subset=['Teff', 'Prot', 'Cluster'])
    # df_calib = get_vsini(df_calib)

    cluster_ages = {'80': 1, '120': 3, '300': 2, '670': 1, '1000': 1, '2500': 1, '2700': 1}
    clusters = df_calib['Cluster'].unique()

    scatter_by_cluster(df_calib, clusters, cluster_ages, 'Prot')

    scatter_avg_std_by_cluster(df_combined, cluster_ages, clusters, 'Prot')

def scatter_p_cut_age(df):
    # df = df[df['Teff'] >= 4800]
    coeff = np.load('imgs/tables/predicted period_cutoff_best_fits.npy')
    poly = np.poly1d(coeff)
    df['gyro_valid'] = df.apply(lambda row: cutoff_threshold_pandas(row, poly, 'Prot'), axis=1)
    xs = df['Teff'].sort_values()
    ys = poly(xs)
    ys[xs < 4800] = np.nan

    mean_xerr = np.mean(df['e_Teff'])
    mean_yerr = np.mean(df['e_Prot'])

    # Example position for the average error bar
    example_x = df['Teff'].min() + (df['Teff'].max() - df['Teff'].min()) * 0.1
    example_y = df['Prot'].max() + (df['Prot'].max() - df['Prot'].min()) * 0.1

    plt.scatter(df['Teff'], df['Prot'], c=df['tGyro'])
    # plt.errorbar(df['Teff'], df['vsini'], yerr=df['e_vsini'], xerr=df['e_Teff'], fmt='none', ecolor='dimgray', alpha=0.5)
    plt.plot(xs, ys, color='red')
    plt.semilogy()
    plt.colorbar(label='age (Myr)')
    plt.errorbar(
        x=example_x,
        y=example_y,
        xerr=mean_xerr,
        yerr=mean_yerr,
        fmt='o',
        color='black',
        ecolor='black',
        elinewidth=2,
        capsize=3,
        label='Mean Error Bar'
    )
    plt.xlabel('$T_{eff}$')
    plt.ylabel("$P_{rot}$ (Days)")
    plt.savefig('imgs/prot_age_kepler.png')
    plt.show()

    plt.scatter(df['Teff'], df['Prot'], c=df['gyro_valid'])
    plt.semilogy()
    plt.show()


    valid_fracs = []
    ages = np.arange(500, 4000, 500)
    for age in ages:
        df_young = df[df['tGyro'] <= age]
        frac = df_young['gyro_valid'].sum() / len(df_young)
        valid_fracs.append(frac)

    plt.scatter(ages, valid_fracs, s=100)
    plt.xlabel('age (Myr)')
    plt.ylabel('$v\sin{i} < v\sin{i}_{sat}$ (%)')
    plt.savefig('imgs/gyro_efficiency')
    plt.show()

def scatter_avg_std_by_cluster(df, cluster_ages, clusters, y_val='Prot'):
    bins = np.linspace(df['Teff'].min(), df['Teff'].max(), 20)  # Adjust the number of bins as needed
    df['Teff_bin'] = pd.cut(df['Teff'], bins)
    grouped = df.groupby(['Cluster', 'Teff_bin'])
    # Compute the mean and std for each bin and cluster
    stats = grouped[y_val].agg(['mean', 'std']).reset_index()
    # Extract the center of each bin for plotting
    stats['Teff_center'] = stats['Teff_bin'].apply(lambda x: x.mid)
    cmap = ['lightskyblue', 'royalblue', 'gold', 'yellowgreen', 'tomato', 'lightpink', 'pink']  # Choose a colormap
    age_to_color = {age: cmap[i] for i, age in enumerate(cluster_ages)}
    fig, axs = plt.subplots(1, 2)
    averaged_stats = []
    start_idx = 0
    for age, idx in cluster_ages.items():
        clusters_in_age = [clusters[i] for i in range(start_idx, start_idx + idx)]
        age_stats = stats[stats['Cluster'].isin(clusters_in_age)]
        avg_std = age_stats.groupby('Teff_center')['std'].mean().reset_index()
        avg_avg = age_stats.groupby('Teff_center')['mean'].mean().reset_index()
        averaged_stats.append((age, avg_std))
        axs[1].plot(avg_std['Teff_center'], avg_std['std'], label=f'Age {age} Myr', color=age_to_color[age])
        axs[0].plot(avg_avg['Teff_center'], avg_avg['mean'], linestyle='--', color=age_to_color[age])
        start_idx = start_idx + idx
    fig.supxlabel('Effective Temperature [K]')
    axs[1].set_ylabel(r'Average $Std (P_{rot})$ (days)')
    axs[1].set_yticks(np.arange(0, 25, 5))
    axs[1].set_yticklabels([str(x) for x in range(0, 25, 5)])
    axs[1].legend(title='Cluster Age', loc='best')
    axs[1].grid(True)
    axs[0].set_ylabel(r'Average $P_rot$ (days)')
    axs[0].set_yticks(range(0, 25, 5))
    axs[0].set_yticklabels([str(x) for x in range(0, 25, 5)])
    axs[0].legend(title='Cluster Age', loc='best')
    axs[0].grid(True)
    fig.savefig('imgs/age_tests_avg_std.png')
    plt.show()


def scatter_by_cluster(df, clusters, cluster_ages, y_val='Prot'):
    bins = np.linspace(df['Teff'].min(), df['Teff'].max(), 20)  # Adjust the number of bins as needed
    df['Teff_bin'] = pd.cut(df['Teff'], bins)
    # Group by cluster and Teff_bin
    grouped = df.groupby(['Cluster', 'Teff_bin'])
    # Compute the mean and std for each bin and cluster
    stats = grouped[y_val].agg(['mean', 'std']).reset_index()
    # Extract the center of each bin for plotting
    stats['Teff_center'] = stats['Teff_bin'].apply(lambda x: x.mid)
    # Plot the results
    # plt.figure(figsize=(10, 6))
    cmap = ['lightskyblue', 'royalblue', 'gold', 'yellowgreen', 'tomato', 'lightpink', 'pink']  # Choose a colormap
    age_to_color = {age: cmap[i] for i, age in enumerate(cluster_ages)}
    start_idx = 0
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for age, idx in cluster_ages.items():
        print(age, idx)
        color = age_to_color[age]
        for i in range(start_idx, start_idx + idx):
            cluster_data = stats[stats['Cluster'] == clusters[i]]
            # cluster_gyro = df_combined[df_combined['Cluster'] == clusters[i]]
            df_reduced = df[df['Cluster'] == clusters[i]]
            ax1.scatter(cluster_data['Teff_center'], cluster_data['std'],
                        label=f'{clusters[i]}', s=100, color=color)
            ax2.scatter(df_reduced['Teff'], df_reduced['Prot'], color=color,
                        s=100, edgecolors='black', linewidths=1.5)

        start_idx = start_idx + idx
    ax1.set_xlabel('Effective Temperature [K]')
    ax1.set_ylabel(r'Std ($P_{rot})$ [days]')
    ax1.legend(title='Cluster', loc='best')
    ax1.grid(True)
    fig1.savefig('imgs/age_tests.png')
    plt.show()

    ax2.set_xlabel(r'$T_{eff}$ (K)')
    ax2.set_ylabel(r'$P_{rot}$ (Days)')
    ax2.invert_xaxis()
    # ax2.grid(True)
    fig2.savefig('imgs/age_tests_scatter.png')
    plt.show()


def get_all_clusters():
    kepler_gaia = Table.read('tables/kepler_dr2_1arcsec.fits', format='fits').to_pandas()
    df_gyro = Table.read("tables/bouma2024.txt", encoding='utf-8', format='ascii.cds').to_pandas()
    berger_cat = pd.read_csv('tables/berger_catalog.csv')
    lightpred_gyro = pd.read_csv('tables/gyrointerp_lightpred_doubles.csv')
    clustres_paths = [p for p in os.listdir('imgs/clusters') if 'fit' in p]
    dfs = []
    for cluster_path in clustres_paths:
        cluster_name = cluster_path.split('.')[0]
        df = Table.read(f'imgs/clusters/{cluster_path}',format='fits').to_pandas()
        if 'KIC' not in df.columns:
            df = df.merge(kepler_gaia, how='left', left_on='Gaia', right_on='source_id')
            df['KIC'] = df['kepid']
        df['Cluster'] = cluster_name
        df.rename(columns={'teff': 'Teff'}, inplace=True)
        if 'Teff' not in df.columns:
            df = df.merge(berger_cat, left_on='KIC', right_on='KID')
        print(cluster_name, df['Teff'].isna().sum())
        dfs.append(df)
    total_df = pd.concat(dfs)
    total_df = total_df.dropna(subset=['KIC'])
    # total_df['KIC'] = total_df['KIC'].astype(str).str.strip()
    # lightpred_gyro['KID'] = lightpred_gyro['KID'].astype(str).str.strip()
    # df_gyro['KIC'] = df_gyro['KIC'].astype(str).str.strip()

    # Merge with lightpred_gyro
    merged_lightpred = total_df.merge(lightpred_gyro, left_on='KIC', right_on='KID', suffixes=['', '_lightpred'])
    merged_lightpred['prot_source'] = 'lightpred'
    merged_lightpred.rename(columns={'age':'tGyro'}, inplace=True)

    # Merge with df_gyro
    merged_bouma = total_df.merge(df_gyro, on='KIC', how='inner', suffixes=['', '_bouma'])
    merged_bouma['prot_source'] = 'bouma'

    # Align columns for concatenation
    all_columns = set(merged_lightpred.columns).union(set(merged_bouma.columns))
    for col in all_columns:
        if col not in merged_lightpred.columns:
            merged_lightpred[col] = None
        if col not in merged_bouma.columns:
            merged_bouma[col] = None

    # Concatenate merged dataframes vertically
    merged = pd.concat([merged_lightpred, merged_bouma], axis=0, ignore_index=True)

    # Drop duplicates based on 'KIC'
    merged = merged.drop_duplicates(subset=['KIC'])
    return merged

def process_long_clusters(df):
    # lightpred = create_catalog('tables/kepler_model_pred_exp45.csv', conf=0.86, low_p=3)
    lightpred = pd.read_csv('tables/gyrointerp_lightPred_doubles.csv')
    merged_df = df.merge(lightpred[['KID', 'age']],
                         left_on='kepid', right_on='KID', how='left')
    # merged_df.loc[merged_df['PRot']==0, 'PRot'] = merged_df.loc[merged_df['PRot']==0, 'predicted period']
    merged_df = merged_df[merged_df['age'] > 0].rename(columns={'age':'tGyro'})
    merged_df['Cluster'] = merged_df['Cl'].apply(lambda x: re.sub(r'([a-zA-Z])(\d)',
                                                r'\1-\2', x.decode('utf-8').strip()))
    return merged_df
def gaia_cross_match(df, merge_on='Gaia', dr=3):
    gaia_kepler = Table.read(f'tables/kepler_dr{dr}_1arcsec.fits', format='fits').to_pandas()
    merged_df = df.merge(gaia_kepler, how='left', left_on=merge_on, right_on='source_id')
    merged_df.rename(columns={merge_on:f'DR{dr}'}, inplace=True)
    return merged_df


def test_mist_effects(group_col, y_col='Kmag_MIST', ymin=3, ymax=6, xmin=5000, xmax=6000):
    mist_df = pd.read_csv('imgs/tables/mist_age_effect.csv')
    mist_df = mist_df[(mist_df['Teff_MIST'] > xmin) & (mist_df['Teff_MIST'] < xmax)]
    mist_df = mist_df[(mist_df[y_col] > ymin) & (mist_df[y_col] < ymax)]

    vals = mist_df[group_col].unique()

    # Create marker and color maps
    feh_values = sorted(mist_df['feh'].unique())
    print(feh_values)
    markers = ['o', '^', 'v', 'D', 'p', '*', 'h', '+', 'x'][:len(feh_values)]
    feh_marker_map = dict(zip(feh_values, markers))

    # Use a color cycle for different ages
    colors = plt.cm.rainbow(np.linspace(0, 1, len(vals)))

    fig, axis = plt.subplots(1, 1)
    for i, val in enumerate(vals):
        df_reduced = mist_df[mist_df[group_col] == val]
        df_reduced = df_reduced.dropna(subset=['Teff_MIST'])
        df_reduced = df_reduced[df_reduced['Teff_MIST'] > 0].sort_values('M')

        if val >= 1:
            power = int(np.log10(val))
            const = int(val // 10 ** power)
            if const > 1:
                label = f'${const}*10^{{{int(np.log10(val))}}}$ yr'
            else:
                label = f'$10^{{{int(np.log10(val))}}}$ yr'
        else:
            label = f'[FeH]={val}'

        # Plot with consistent color for age and markers for feh
        for feh in [0]:
            feh_group = df_reduced[df_reduced['feh'] == feh]
            axis.plot(feh_group['Teff_MIST'], feh_group[y_col],
                            color=colors[i], label=label
                            )
            # axis[1].scatter(feh_group['Teff_MIST'], feh_group[y_col],
            #                 color=colors[i],
            #                 label=label)

    axis.invert_yaxis()
    axis.invert_xaxis()
    axis.legend(fontsize=18)

    # axis[1].invert_yaxis()
    # axis[1].invert_xaxis()
    # axis[1.set_xlim(6000, 4000)
    # axis[1].set_ylim(ymax, ymin)
    # axis[1].legend(fontsize=14)

    fig.supxlabel('Teff(K)')
    fig.supylabel('$K_{MIST}$ (mag)')
    plt.tight_layout()
    plt.savefig(f'imgs/mist_effects_{group_col}_{y_col}.png')
    plt.show()


def get_t_logg_relations(n_bins, min_age=1e7, max_age=2e9):
    mist_df = pd.read_csv('tables/mist_age_effect.csv')
    ages = mist_df['age'].unique()
    mist_df = mist_df[(mist_df['age'] > min_age) & (mist_df['age'] < max_age)]

    # Bin the dataframe based on 'Teff_MIST'
    bins = np.linspace(mist_df['Teff_MIST'].min(), mist_df['Teff_MIST'].max(), n_bins + 1)
    mist_df['Teff_bin'] = pd.cut(mist_df['Teff_MIST'], bins)

    # Calculate the min and max 'logg_MIST' for each bin
    logg_min = mist_df.groupby('Teff_bin')['logg_MIST'].min().values
    logg_max = mist_df.groupby('Teff_bin')['logg_MIST'].max().values
    logg_mean = mist_df.groupby('Teff_bin')['logg_MIST'].mean().values

    plt.plot(bins[:-1], logg_min)
    plt.plot(bins[:-1], logg_max)

    plt.xlabel('Teff (K)')
    plt.ylabel('logg')

    plt.legend(title='Age')
    plt.gca().invert_xaxis()
    plt.close()

    return bins, logg_min, logg_max


def kinematic_age(df, known_binaries, poly):
    url = 'https://content.cld.iop.org/journals/1538-3881/162/3/100/revision1/ajac0f08t3_mrt.txt'
    download_machine_readable_table(url, 'tables/kinematic_v')
    table = read_machine_readable_table('tables/kinematic_v.txt').to_pandas()
    table['sigma'] = np.sqrt((table['U'] ** 2 + table['V'] ** 2 + table['W'] ** 2) / 3)
    table = table[(table['Teff'] < df['Teff'].max()) & (table['Teff'] > df['Teff'].min())]
    table['teff_bin'] = pd.cut(table['Teff'], bins=10)
    short = get_catalogs_short_period().rename(columns={'Prot':'predicted period'})
    # df = pd.concat([df, short])
    kinematic_df = table.merge(df, left_on='KIC', right_on='KID')

    bouma24 = pd.read_csv('tables/bouma2024.csv')
    bouma_ags = bouma24[bouma24['gyro_median'].notna()]
    bouma_ags = bouma_ags.merge(table[['KIC', 'sigma']], on='KIC', how='left')

    kinematic_df['is_below'] = kinematic_df.apply(lambda x: poly(x['Teff_x']) > x['predicted period'], axis=1)
    kinematic_df['dist_below'] = kinematic_df.apply(lambda x: poly(x['Teff_x']) - x['predicted period'] if x['is_below'] else 0, axis=1)
    kinematic_df['dist_below_bin'] = pd.cut(kinematic_df['dist_below'], bins=10)
    kinematic_df['sigma_bin'] = pd.cut(kinematic_df['sigma'], bins=10)

    kinematic_df_short = kinematic_df[kinematic_df['is_below'] == True]
    kinematic_df_short['predicted_period_bin'] = pd.cut(kinematic_df_short['predicted period'], bins=10)


    kinematic_binaries = table.merge(known_binaries, left_on='KIC', right_on='KID')

    print("kinematic df: ", len(kinematic_df), "kinematic binaries: ", len(kinematic_binaries), "kinematic_df_short: ",
          len(kinematic_df_short))
    plt.hist(kinematic_df['sigma'], histtype='step', linewidth=3,
             color='slategray', density=True, bins=20, label='All')
    plt.hist(kinematic_df_short['sigma'], histtype='step', linewidth=3,
             color='saddlebrown', density=True, bins=20, label='below line')
    plt.hist(kinematic_binaries['sigma'], histtype='step', linewidth=3, density=True,
             color='lightsalmon', bins=20, label='All binaries')
    plt.legend()
    plt.xlabel(r'$\sigma(km \cdot s^{-1})$')
    plt.ylabel('pdf')
    plt.savefig('imgs/sigma_dists.png')
    plt.show()

    ks_test_binaries = ks_2samp(kinematic_df['sigma'], kinematic_binaries['sigma'])
    ks_test_short = ks_2samp(kinematic_df['sigma'], kinematic_df_short['sigma'])
    print("ks test binaries: ", ks_test_binaries.pvalue)
    print("ks test short period: ", ks_test_short.pvalue)
    young_probs = []
    err_probs = []
    sigmas_threshs = np.arange(10, 35, 2)
    for thresh in sigmas_threshs:
        excess_res = calculate_young_star_fraction(kinematic_df['sigma'],
                                                   kinematic_df_short['sigma'], young_threshold=thresh)
        young_probs.append(excess_res['young_star_fraction'])
        err_probs.append(excess_res['error'])
    plt.plot(sigmas_threshs, young_probs)
    plt.errorbar(sigmas_threshs, young_probs, yerr=err_probs, fmt='none', color='gray')
    plt.xlabel(r'$\sigma_{thresh}(km \cdot s^{-1})$')
    plt.ylabel('distribution excess')
    plt.savefig('imgs/young_probs_sigma.png')
    plt.show()

    excess_res = calculate_young_star_fraction(kinematic_df['sigma'],
                                               kinematic_df_short['sigma'], young_threshold=12)

    print("excess of entire temperatures: ", excess_res['young_star_fraction'], " pm ", excess_res['error'])

    unique_bins = table['teff_bin'].cat.categories
    teffs = []
    probs = []
    errs = []
    for teff_bin in unique_bins:
        kinematic_t = kinematic_df[kinematic_df['teff_bin'] == teff_bin]
        kinematic_short_t = kinematic_df_short[kinematic_df_short['teff_bin'] == teff_bin]
        # print(teff_bin, len(kinematic_short_t))
        if (len(kinematic_t) and len(kinematic_short_t)):
            teffs.append(teff_bin.mid)
            excess_res = calculate_young_star_fraction(kinematic_t['sigma'],
                                                       kinematic_short_t['sigma'], young_threshold=12)
            probs.append(excess_res['young_star_fraction'])
            errs.append(excess_res['error'])
    print("average excess: ", np.mean(probs), " pm ", np.mean(errs))
    plt.plot(teffs, probs)
    plt.errorbar(teffs, probs, yerr=errs, fmt='none', color='gray')
    plt.xlabel('Teff (K)')
    plt.ylabel('distribution excess')
    # plt.ylim((-0.05, 1.5))
    plt.gca().invert_xaxis()
    plt.savefig('imgs/young_probs_teff.png')
    plt.show()

    kinematic_short_young = kinematic_df_short[kinematic_df_short['sigma'] < 12]
    kinematic_short_old = kinematic_df_short[kinematic_df_short['sigma'] >= 12]
    # print("number of youngs: ", len(kinematic_short_young))
    # print("number of binaries non young: ", len(kinematic_short_old))
    plt.hist(kinematic_short_young['predicted period'], histtype='step', linewidth=3,
             density=True, bins=20, color='slategray',
             label=r'$\sigma < 12$ $(km \cdot s^{-1})$')
    plt.hist(kinematic_short_old['predicted period'], histtype='step', linewidth=3,
             density=True, bins=20, color='lightsalmon',
             label=r'$\sigma \geq 12$ $(km \cdot s^{-1})$')
    plt.legend()
    plt.xlabel('$P_{rot}$')
    plt.ylabel('pdf')
    plt.savefig('imgs/young_old_prot_dist.png')
    plt.show()

    sigmas = np.arange(5,45, 10)
    for sigma_bin in sigmas:
        kinematic_bin = kinematic_df_short[kinematic_df_short['sigma'] > sigma_bin]
        plt.hist(kinematic_bin['predicted period'], histtype='step', linewidth=3,
                 density=True, bins=20,
                 label=fr'$\sigma$ > {sigma_bin}')
    plt.xlabel('$P_{rot}$ (Days)')
    plt.ylabel('PDF')
    plt.legend()
    plt.show()

    plt.hist(kinematic_short_young['kmag_diff'], histtype='step', linewidth=3,
             density=True, bins=20, color='slategray',
             label=r'$\sigma < 12$ $(km \cdot s^{-1})$')
    plt.hist(kinematic_short_old['kmag_diff'], histtype='step', linewidth=3,
             density=True, bins=20, color='lightsalmon',
             label=r'$\sigma \geq 12$ $(km \cdot s^{-1})$')
    plt.legend()
    plt.xlabel(r'$\Delta K_{iso}$')
    plt.ylabel('pdf')
    plt.savefig('imgs/young_old_kmag_dist.png')
    plt.show()

    # Get the categories from the full dataframe
    teff_bins = sorted(kinematic_df['teff_bin'].cat.categories)
    period_bins = sorted(kinematic_df_short['predicted_period_bin'].cat.categories)

    # Initialize data structures for the 3D plot
    teff_grid = []
    period_grid = []
    prob_values = []
    sigma_values = []

    # Calculate probabilities for each 2D bin combination
    for teff_bin in teff_bins:
        for period_bin in period_bins:
            # Filter data for this 2D bin
            kinematic_bin = kinematic_df[kinematic_df['teff_bin'] == teff_bin]
            kinematic_short_bin = kinematic_df_short[(kinematic_df_short['teff_bin'] == teff_bin) &
                                                     (kinematic_df_short['predicted_period_bin'] == period_bin)]
            # Only proceed if we have enough data points
            if len(kinematic_short_bin) > 10:
                # print(teff_bin, period_bin, len(kinematic_bin), len(kinematic_short_bin))
                excess_res = calculate_young_star_fraction(kinematic_bin['sigma'],
                                                           kinematic_short_bin['sigma'],
                                                           young_threshold=12)
                if excess_res['young_star_fraction'] > 0:
                    # Store the midpoints of bins and the probability
                    teff_grid.append(teff_bin.mid)
                    period_grid.append(period_bin.mid)
                    prob_values.append(excess_res['young_star_fraction'])

    teff_array = np.array(teff_grid)
    period_array = np.array(period_grid)
    prob_array = np.array(prob_values)

    data_df = pd.DataFrame({
        'teff': teff_array,
        'period': period_array,
        'probability': prob_array
    })

    X = data_df[['teff', 'period']].values

    y = data_df['probability'].values
    model, best_degree = fit_polynomial_surface(X, y, max_degree=3)

    print("number of points in X: ", len(X))

    plot_polynomial_surface(model, best_degree, X, teff_array, period_array, prob_array)

    # Save and show the plot
    plt.savefig('imgs/young_probs_teff_period_3d.png')
    plt.show()

    bouma_ags['is_below_line'] = bouma_ags.apply(lambda x: poly(x['adopted_Teff']) > x['Prot'], axis=1)
    bouma_below = bouma_ags[bouma_ags['is_below_line']]
    bouma_params = bouma_below[['adopted_Teff', 'Prot']].values
    bouma_probs = model.predict(bouma_params)
    bouma_probs[np.logical_or(bouma_probs < 0, bouma_probs > 1)] = np.nan
    bouma_below['young_prob'] = bouma_probs

    plt.scatter(bouma_below['gyro_median'], bouma_below['young_prob'])
    plt.xlabel('Age (Myr)')
    plt.ylabel('$f(T,P)$')
    plt.savefig('imgs/bouma_age_prob_scatter.png')
    plt.show()

    bouma_below['age_bin'] = pd.cut(bouma_below['gyro_median'], bins=15)
    ages = []
    probs = []
    teffs = []
    sigmas = []
    for age_bin in bouma_below['age_bin'].unique():
        bouma_bin = bouma_below[bouma_below['age_bin']==age_bin]
        ages.append(age_bin.mid)
        probs.append(bouma_bin['young_prob'].median())
        sigmas.append(np.nanmedian(bouma_bin['sigma']))
        teffs.append(bouma_bin['adopted_Teff'].median())

    plt.scatter(ages, probs, s=100, c=teffs)
    plt.xlabel('Age (Myr)')
    plt.ylabel('median ($f(T.P)$)')
    plt.colorbar(label=r'$T_{eff}$ (K)')
    plt.savefig('imgs/bouma_age_prob_median.png')
    plt.show()
    return kinematic_df, model

def calculate_young_star_fraction(full_sample, subset_sample, young_threshold=20, bins=40):
    # Focus on stars below the threshold
    full_young = full_sample[full_sample < young_threshold]
    subset_young = subset_sample[subset_sample < young_threshold]

    # Get actual counts in each sample
    N_full = len(full_sample)
    N_subset = len(subset_sample)
    N_full_young = len(full_young)
    N_subset_young = len(subset_young)

    # Expected count of young stars in subset if distribution matched full sample
    expected_young_in_subset = N_subset * (N_full_young / N_full)

    # Excess count (observed - expected)
    excess_count = N_subset_young - expected_young_in_subset

    # Young star fraction
    young_star_fraction = excess_count / N_subset if N_subset > 0 else 0

    # Poissonian error calculation
    # For count data, error = sqrt(count)
    error_N_full_young = np.sqrt(N_full_young)
    error_N_subset_young = np.sqrt(N_subset_young)

    # Error propagation for the expected count
    error_expected = expected_young_in_subset * np.sqrt(
        (error_N_full_young / N_full_young) ** 2 + (np.sqrt(N_full) / N_full) ** 2)

    # Error propagation for the excess count (quadrature sum of errors)
    error_excess = np.sqrt(error_N_subset_young ** 2 + error_expected ** 2)

    # Error on the fraction
    error_fraction = error_excess / N_subset if N_subset > 0 else 0

    return {
        "young_star_fraction": young_star_fraction,
        "error": error_fraction,
        "excess_count": excess_count,
        "N_subset": N_subset,
        "N_subset_young": N_subset_young,
        "expected_young": expected_young_in_subset
    }

    # kinemtaic_df = kinemtaic_df[kinemtaic_df['is_below']==True]
    # kinematic_binaries['sigma'] = np.sqrt((kinematic_binaries['U'] ** 2 + kinematic_binaries['V'] ** 2 + kinematic_binaries['W'] ** 2) / 3)
    # kinematic_binaries['is_below'] = kinematic_binaries.apply(lambda x: poly(x['Teff_x']) > x['period'], axis=1)
    # kinematic_binaries = kinematic_binaries[kinematic_binaries['period'] < 10]
    # plt.scatter(kinemtaic_df['predicted period'], kinemtaic_df['sigma'], label='all')
    # plt.scatter(kinematic_binaries['period'], kinematic_binaries['sigma'], label='binaries')
    #
    # plt.ylabel('simga ($km s^-1$)')
    # plt.xlabel('period (days)')
    # plt.show()


def fit_polynomial_surface(X, y, max_degree=3):
    best_score = -np.inf
    best_model = None
    best_degree = None

    for degree in range(1, max_degree + 1):
        # Create polynomial features
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )

        # Fit model
        model.fit(X, y)

        # Predict and calculate score
        y_pred = model.predict(X)
        score = r2_score(y, y_pred)

        print(f"Degree {degree} R² score: {score:.4f}")
        if score > 0.98:
            best_score = score
            best_model = model
            best_degree = degree
            break

        if score > best_score:
            best_score = score
            best_model = model
            best_degree = degree

    print(f"\nBest polynomial degree: {best_degree} (R² = {best_score:.4f})")
    return best_model, best_degree

def plot_polynomial_surface(model, best_degree, X, teff_array, period_array, prob_array):
    teff_min, teff_max = min(teff_array), max(teff_array)
    period_min, period_max = min(period_array), max(period_array)

    teff_range = np.linspace(teff_min, teff_max, 50)
    period_range = np.linspace(period_min, period_max, 50)
    teff_grid_mesh, period_grid_mesh = np.meshgrid(teff_range, period_range)

    # Prepare grid points for prediction
    grid_points = np.column_stack([teff_grid_mesh.ravel(), period_grid_mesh.ravel()])
    predicted_values = model.predict(grid_points)

    # Reshape predicted values to grid
    probability_mesh = predicted_values.reshape(teff_grid_mesh.shape)

    # Visualization
    fig = plt.figure(figsize=(32, 20))

    # Plot 1: 3D Scatter of original data with fitted surface
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(teff_array, period_array, prob_array,
                          c=prob_array, cmap=cm.viridis,
                          s=100, alpha=1.0, label='Original data')
    surface = ax1.plot_surface(teff_grid_mesh, period_grid_mesh, probability_mesh,
                               cmap=cm.viridis, alpha=0.7,
                               linewidth=0, antialiased=True)
    # ax1.set_xlabel('Effective Temperature (K)')
    # ax1.set_ylabel('Period (days)')
    # ax1.set_zlabel('Young Star Fraction')
    # ax1.set_title(f'Polynomial Surface Fit (degree {best_degree})')

    # Plot 2: Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(teff_grid_mesh, period_grid_mesh, probability_mesh,
                           levels=20, cmap=cm.viridis)
    ax2.scatter(teff_array, period_array, c=prob_array,
                cmap=cm.viridis, s=100, edgecolors='k')
    ax2.set_xlabel('Effective Temperature (K)')
    ax2.set_ylabel('Period (days)')
    ax2.set_title('Contour Map of Young Star Fraction')
    plt.colorbar(contour, ax=ax2, label='Young Star Fraction')

    plt.tight_layout()
    plt.savefig('imgs/prob_teff_p_polynomial_plot.png')
    plt.show()

    fig = plt.figure()

    # Plot 3: Original data vs predicted data
    ax3 = fig.add_subplot(121)
    predicted_original = model.predict(X)
    ax3.scatter(prob_array, predicted_original, s=100)
    ax3.plot([0, max(prob_array)], [0, max(prob_array)], 'r--')
    ax3.set_xlabel('Actual Young Star Fraction')
    ax3.set_ylabel('$f(T,P)$')
    ax3.set_title(f'R² = {r2_score(prob_array, predicted_original):.4f}')

    # Plot 4: Residuals
    ax4 = fig.add_subplot(122)
    residuals = prob_array - predicted_original
    ax4.scatter(predicted_original, residuals, s=100)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('$f(T,P)$')
    ax4.set_ylabel('Residuals')
    # ax4.set_title('Residual Plot')

    plt.tight_layout()
    plt.savefig('imgs/prob_teff_p_polynomial_fit.png')
    plt.show()

    # Print model coefficients
    poly_features = model.named_steps['polynomialfeatures']
    linear_regression = model.named_steps['linearregression']
    feature_names = poly_features.get_feature_names_out(['teff', 'period'])
    coefficients = linear_regression.coef_

    print("\nPolynomial Coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature}: {coef:.6f}")


if __name__ == '__main__':
    # lithium_ages()
    # cluster_gyro_ratio('tables/bouma2023_calibration.txt', 'tables/bouma2024.txt')
    # age_tests('tables/bouma2023_calibration.txt', 'tables/bouma2024.txt')
    # berger = pd.read_csv('tables/berger_catalog.csv')
    # hr_diagram(berger)
    # hr_diagram(berger, cols=['Teff', 'logg', 'Mstar'])
    # hr_diagram(berger, cols=['Teff', 'Mstar'])
    # get_t_logg_relations(40)
    cat = short_binaries(r'C:\Users\Ilay\projects\kepler\data\lightPred\tables/kepler_model_pred_exp45.csv', p_thresh=7)
    # kinematic_age()
    # test_mist_effects('age')
    # test_mist_effects('age', y_col='logg_MIST', ymin=4, ymax=5)
