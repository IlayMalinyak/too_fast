import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
import matplotlib.cm as cm
# from short_binaries import create_catalog
import pandas as pd
from astropy.table import Table, join, vstack

day2sec = 24*60*60
R_sun_km = 696340
M_sun_gr = 1.989e33
R_sun_cm = R_sun_km * 1e5
AU_cm = 1.496e13
G = 6.674e-8

def angular_velocity(row):
    return np.sqrt(row['mass'] * M_sun_gr * G / (row['sep_AU'] * AU_cm) ** 3)


def get_kepler_gaia_wide_binary_samples():
    # Read the input tables
    all_gaia_binaries = Table.read('tables/gaia_binaries.fits', format='fits')
    kepler_gaia_table = Table.read('tables/kepler_dr3_4arcsec.fits', format='fits')

    # Create a dictionary mapping source_id to kepid for efficient lookup
    source_id_to_kepid = {row['source_id']: row['kepid']
                          for row in kepler_gaia_table}

    # Find binary systems where either component matches a Kepler source ID
    matched_indices = np.where(
        (np.isin(all_gaia_binaries['source_id1'], list(source_id_to_kepid.keys()))) |
        (np.isin(all_gaia_binaries['source_id2'], list(source_id_to_kepid.keys())))
    )[0]

    # Select matched binary systems
    matched_binaries = all_gaia_binaries[matched_indices]

    # Determine which source has a Kepler ID
    matched_binaries['KID'] = np.where(
        [source_id_to_kepid.get(sid, np.nan) for sid in matched_binaries['source_id1']] != np.nan,
        [source_id_to_kepid.get(sid, np.nan) for sid in matched_binaries['source_id1']],
        [source_id_to_kepid.get(sid, np.nan) for sid in matched_binaries['source_id2']]
    )

    # Convert to DataFrame, dropping kepid1 and kepid2 if they exist
    matched_binaries_df = matched_binaries.to_pandas()

    # Remove any kepid1 or kepid2 columns if they exist
    for col in ['kepid1', 'kepid2']:
        if col in matched_binaries_df.columns:
            matched_binaries_df = matched_binaries_df.drop(columns=[col])

    return matched_binaries_df

def get_kepler_gaia_primary_binaries():
    all_gaia_binaries = Table.read('tables/gaia_binaries.fits', format='fits')
    kepler_gaia_table = Table.read('tables/kepler_dr3_4arcsec.fits', format='fits')
    all_gaia_binaries.rename_column('source_id1', 'source_id')
    kepler_gaia_binaries = join(all_gaia_binaries, kepler_gaia_table, keys='source_id')
    kepler_gaia_binaries = kepler_gaia_binaries.to_pandas()
    kepler_gaia_binaries = kepler_gaia_binaries[~kepler_gaia_binaries['kepid'].duplicated()]
    return kepler_gaia_binaries


def astrometric_binaries(kepler_path):
    kepler_period_cat = create_catalog(kepler_path, conf=0.95, low_p=0)
    kepler_period_cat = kepler_period_cat.loc[:, ~kepler_period_cat.columns.str.contains(r'diff|\d|acc')]
    kepler_gaia_table = Table.read('tables/kepler_dr3_4arcsec.fits', format='fits')
    astro_binaries = Table.read("tables/AMRF_table1.fits", format='fits')
    kepler_gaia_astro_binaries = join(astro_binaries, kepler_gaia_table, keys='source_id').to_pandas()
    merged_cat = kepler_period_cat.merge(kepler_gaia_astro_binaries, right_on='kepid', left_on='KID')
    kepler_wide_binaries = get_kepler_gaia_binaries()
    merged_cat = merged_cat.merge(kepler_wide_binaries, left_on='KID', right_on='kepid', how='left')
    plt.scatter(merged_cat['m1']/merged_cat['AMRF_m2'], merged_cat['predicted period'],
                c=merged_cat['mean_period_confidence'])
    plt.colorbar()
    plt.show()

def gaia_wide(kepler_path):
    kepler_period_cat = create_catalog(kepler_path, conf=0.95, low_p=0)
    kepler_period_cat = kepler_period_cat.loc[:, ~kepler_period_cat.columns.str.contains(r'diff|\d|acc')]
    print("entire period catalog: ", len(kepler_period_cat))
    kepler_period_cat_short = kepler_period_cat[kepler_period_cat['predicted period'] < 7]
    print("short period catalog: ", len(kepler_period_cat_short))
    kepler_gaia_binaries = get_kepler_gaia_binaries()
    print("entire gaia kepler binaries: ", len(kepler_gaia_binaries))
    kepler_gaia_binaries = kepler_gaia_binaries.merge(kepler_period_cat, left_on='kepid', right_on='KID')
    print("merged catalog and gaia kepler binaries: ", len(kepler_gaia_binaries))
    kepler_gaia_binaries_short = kepler_gaia_binaries[kepler_gaia_binaries['predicted period'] < 7]
    print("merged short period catalog and gaia kepler binaries: ", len(kepler_gaia_binaries_short))
    print("short period fraction from entire period catalog: ", len(kepler_period_cat_short) / len(kepler_period_cat))
    print("short period binaries fraction from binaries catalog: ",
          len(kepler_gaia_binaries_short) / len(kepler_gaia_binaries))

    kepler_gaia_binaries_short['sini'] = ((kepler_gaia_binaries_short['radial_velocity']
                                           * kepler_gaia_binaries_short['predicted period']) * day2sec /
                                          (kepler_gaia_binaries_short['radius'] * R_sun_km * 2 * np.pi))
    kepler_gaia_binaries_short['i'] = np.rad2deg(np.arcsin(kepler_gaia_binaries_short['sini']))
    kepler_gaia_binaries_short['w'] = kepler_gaia_binaries_short.apply(angular_velocity, axis=1)
    kepler_gaia_binaries_short['orbital period'] = 2 * np.pi / kepler_gaia_binaries_short['w'] / day2sec

    # Convert 'binary_type' to a numeric code and get unique colors for each type
    binary_types = kepler_gaia_binaries_short['binary_type'].unique()
    color_map = dict(zip(binary_types, plt.cm.viridis(np.linspace(0, 1, len(binary_types)))))
    kepler_gaia_binaries_short['binary_type_code'] = kepler_gaia_binaries_short['binary_type'].map(color_map)

    # Plot scatter with colors based on the mapped binary types
    plt.scatter(kepler_gaia_binaries_short['sep_AU'], kepler_gaia_binaries_short['predicted period'],
                c=kepler_gaia_binaries_short['binary_type_code'], s=100)

    only_ebs = kepler_gaia_binaries_short[kepler_gaia_binaries_short['eb']]
    plt.scatter(only_ebs['sep_AU'], only_ebs['predicted period'],
                c=only_ebs['binary_type_code'], marker='*', s=100)

    # Create a custom legend using the original string labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=bt,
                          markersize=10, markerfacecolor=color_map[bt]) for bt in binary_types]
    plt.legend(handles=handles, title="Binary Type")

    plt.semilogx()
    plt.xlabel('Binary Separation (AU)')
    plt.ylabel('Primary Stellar Period - ACF (Days)')
    plt.show()

def lamost(kepler_path):
    lamost = pd.read_csv('tables/lamost_dr10_gaia_dr3_kepler_AFGK.csv')
    kepler_period_cat = create_catalog(kepler_path)
    kepler_period_cat = kepler_period_cat.loc[:, ~kepler_period_cat.columns.str.contains(r'diff|\d|acc')]
    lamost_kepler_cat = lamost.merge(kepler_period_cat, right_on='KID', left_on='kepid')
    lamost_kepler_cat['sini'] = ((lamost_kepler_cat['vsini_lasp']
                                     * kepler_period_cat['predicted period']) * day2sec /
                                    (lamost_kepler_cat['radius'] * R_sun_km * 2 * np.pi))
    lamost_kepler_cat['i'] = np.rad2deg(np.arcsin(lamost_kepler_cat['sini']))
    lamost_kepler_cat = lamost_kepler_cat[lamost_kepler_cat['i'].notna()]
    plt.hist(lamost_kepler_cat['i'])
    plt.show()



if __name__ == "__main__":
    # gaia_wide('tables/kepler_model_pred_exp45.csv' )
    # astrometric_binaries('tables/kepler_model_pred_exp45.csv')
    gaia_kep = get_kepler_gaia_seperate_binaries()
    print(gaia_kep.head())
