
import pandas as pd
import numpy as np
import re
import ast

def selective_merge_dfs(df_1, df_2, columns_to_update, key_column='KID'):
    """
    Update values in df_2 with values from df_1 where the key_column matches.

    Parameters:
    df_1 (pd.DataFrame): The dataframe containing the values to prioritize.
    df_2 (pd.DataFrame): The dataframe to be updated.
    key_column (str): The column name to join on (e.g., 'KID').
    columns_to_update (list): List of columns to update.

    Returns:
    pd.DataFrame: Updated dataframe.
    """
    # Merge the dataframes on the key_column with a left join
    merged_df = pd.merge(df_2, df_1, on=key_column, how='left', suffixes=('_df2', '_df1'))

    # Update the columns
    for col in columns_to_update:
        if col in df_2.columns:
            # If the column is in df_2, update it with values from df_1 where available
            merged_df[col] = merged_df[f'{col}_df1'].combine_first(merged_df[f'{col}_df2'])
            # Drop the temporary columns
            merged_df = merged_df.drop([f'{col}_df2', f'{col}_df1'], axis=1)
    return merged_df

def convert_cols_to_float(df, cols):
    for c in cols:
        if df[c].dtype == 'int64':  # Skip integer columns
            continue
        # try:
            # Convert column values to floats
            df[c] = df[c].apply(lambda x: float(x.strip().split(',')[0]) if x.lower() != 'false' else float('nan'))
        # except ValueError:
        #     df[c] = np.nan  # Set invalid values to NaN
    return df

def string_to_list(string_array):
    if string_array is np.nan:
        return None
    try:
        # Remove any non-printable characters, including U+2009
        clean_string = re.sub(r'[^\x20-\x7E]+', '', string_array)
        return ast.literal_eval(clean_string)
    except (ValueError, SyntaxError) as e:
        return None  # or any other appropriate action

def extract_values(lst):
    if isinstance(lst, list) and len(lst) == 3:
        return pd.Series([lst[1], lst[0], lst[2]], index=['value', 'error_low', 'error_high'])
    else:
        return pd.Series([None, None, None], index=['value', 'error_low', 'error_high'])

def extract_values_and_errors_from_list(df):
    for col in ['i', 'dnu']:
        df[col] = df[col].apply(string_to_list)
    for col in ['i', 'dnu']:
        df[[f"{col}_value", f"{col}_error_low", f"{col}_error_high"]] = df[col].apply(extract_values)
    df = df.drop(columns=['i', 'dnu'])
    return df

def merge_all_cats(kepler_inference):
    acf_df = pd.read_csv('tables/kepler_acf_pred_exp14.csv')
    acf_df_7 = pd.read_csv('tables/kepler_acf_pred_exp7.csv')
    acf_df_7.rename(columns={'predicted acf_p': 'predicted acf_p no doubles', 'valid': 'valid acf no doubles'}, inplace=True)
    eb_df = pd.read_csv('tables/kepler_eb.txt')
    eb_df.rename(columns={'period': 'eb_orbital_period'}, inplace=True)
    acf_df.rename(columns={'double_peaked': 'second_peak'}, inplace=True)
    mcq2014 = pd.read_csv('tables/Table_1_Periodic.txt')
    mcq2014.rename(columns={'Prot':'Prot_mcq14'}, inplace=True)
    berger_cat = pd.read_csv('tables/berger_catalog.csv')
    berger_cat.rename(columns={'Age':'Age_Berger20'}, inplace=True)
    berger2018 = pd.read_csv('tables/berger2018.txt', sep='\t')
    santos2019_p = convert_cols_to_float(pd.read_csv(
                                    'tables/santos2019_period.txt', sep=';'),
                                    cols=['Fl1'])
    santos2019_p.rename(columns={'Fl1': 'Fl1_p',  'Prot': 'Prot_santos2019'}, inplace=True)
    santos2019_no_p = convert_cols_to_float(pd.read_csv
                                         ('tables/santos2019_no_period.txt', sep=';'),
                                          cols=['Fl1'])
    santos2019_no_p.rename(columns={'Fl1': 'Fl1_no_p', 'Prot': 'Prot_santos19'}, inplace=True)

    simonian2019 = pd.read_csv('tables/simonian2019.txt', sep='\t')

    santos2021 = pd.read_csv('tables/santos2021_full.txt', sep=';')
    flags_cols = [col for col in santos2021.columns if 'flag' in col]
    santos2021 = convert_cols_to_float(santos2021, cols=flags_cols)
    santos2021_p = convert_cols_to_float(pd.read_csv('tables/santos2021_p.txt', sep=';'),
                                         cols=['flag1'])
    santos2021_p.rename(columns={'flag1': 'flag1_p', 'Prot': 'Prot_santos21'}, inplace=True)
    reinhold23 = pd.read_csv('tables/reinhold2023.csv')
    reinhold23.rename(columns={'Prot':'Prot_reinhold23'}, inplace=True)

    kamiaka18 = extract_values_and_errors_from_list(pd.read_csv('tables/kamiaka2018.txt', sep='\t'))
    kamiaka18_planets = extract_values_and_errors_from_list(pd.read_csv('tables/kamiaka2018_planets.txt', sep='\t'))
    kamiaka18 = pd.concat([kamiaka18, kamiaka18_planets], axis=0)
    # kamiaka18 = kamiaka18.astype(np.float16)
    r_var = pd.read_csv('tables/r_var.csv')
    s_ph = pd.read_csv('tables/s_ph.csv')

    gyrointerp_lightpred = pd.read_csv('tables/gyrointerp_lightPred_doubles.csv')
    gyrointerp_lightpred.rename(columns={'age': 'age_gyrointerp_model', 'e_age_up':'e_age_up_model',
                                         'e_age_low': 'e_age_low_model'}, inplace=True)
    angus2023 = pd.read_csv('tables/angus2023.txt', sep=';')
    angus2023['age'] = angus2023['age'] * 1000
    angus2023.rename(columns={'age':'age_angus23'}, inplace=True)
    bouma2024 = pd.read_csv('tables/bouma2024.csv')
    bouma2024.rename(columns={'gyro_median':'age_bouma24_gyro', 'KIC':'KID'}, inplace=True)
    bouma2024_planets = pd.read_csv('tables/bouma2024_planets.csv')
    bouma2024_planets.rename(columns={'gyro_median':'age_bouma24_gyro',
                                      'li_median':'age_bouma24_li', 'kepid':'KID'}, inplace=True)
    bouma2024 = pd.concat([bouma2024, bouma2024_planets], axis=0)
    kois = pd.read_csv('tables/kois.csv')


    kepler_inference = selective_merge_dfs(berger_cat, kepler_inference, columns_to_update=['Teff',
                                                                                            'logg',
                                                                                            'Dist',
                                                                                            'Lstar',
                                                                                            'FeH',
                                                                                            'Mstar',
                                                                                            'Age_Berger18'])
    kepler_inference = selective_merge_dfs(acf_df[['KID', 'predicted acf_p', 'second_peak']], kepler_inference,
                                           columns_to_update=['predicted acf_p', 'second_peak'])
    kepler_inference = selective_merge_dfs(acf_df_7[['KID', 'predicted acf_p no doubles', 'valid acf no doubles']], kepler_inference,
                                           columns_to_update=['predicted acf_p no doulbes', 'valid acf no doubles'])
    kepler_inference = selective_merge_dfs(berger2018[['KID', 'Bin']], kepler_inference,
                                           columns_to_update=['Bin'])
    kepler_inference = selective_merge_dfs(santos2019_p[['KID', 'Fl1_p','Prot_santos2019']], kepler_inference,
                                           columns_to_update=['Fl1', 'Prot_santos2019']
                                           )
    kepler_inference = selective_merge_dfs(santos2019_no_p[['KID', 'Fl1_no_p']], kepler_inference,
                                           columns_to_update=['Fl1_no_p']
                                           )

    kepler_inference = selective_merge_dfs(simonian2019[['KID', 'dK']], kepler_inference,
                                           columns_to_update=['dK'])

    kepler_inference = selective_merge_dfs(santos2021[flags_cols + ['KID']], kepler_inference,
                                           columns_to_update=flags_cols)

    kepler_inference = selective_merge_dfs(santos2021_p[['KID', 'flag1_p', 'Prot_santos21']], kepler_inference,
                                           columns_to_update=['flag1_p', 'Prot_santos21'])

    kepler_inference = selective_merge_dfs(r_var[['r_var', 'kmag', 'KID']], kepler_inference,
                                           columns_to_update=['r_var', 'kmag'])

    kepler_inference = selective_merge_dfs(s_ph[['KID', 's_ph']], kepler_inference,
                                           columns_to_update=['s_ph'])

    kepler_inference = selective_merge_dfs(mcq2014[['KID', 'w', 'Prot_mcq14']], kepler_inference,
                                           columns_to_update=['w', 'Prot_mcq14'])
    kepler_inference = selective_merge_dfs(reinhold23[['KID', 'Prot_reinhold23']], kepler_inference,
                                           columns_to_update=['Prot_reinhold23'])

    kepler_inference = selective_merge_dfs(eb_df[['KID', 'eb_orbital_period']], kepler_inference,
                                           columns_to_update=['eb_orbital_period'])

    kepler_inference = selective_merge_dfs(gyrointerp_lightpred[['KID', 'age_gyrointerp_model',
                                                                 'e_age_up_model', 'e_age_low_model']],
                                                                kepler_inference,
                                           columns_to_update=['age_gyrointerp_model',
                                                                 'e_age_up_model', 'e_age_low_model'])

    kepler_inference = selective_merge_dfs(angus2023[['KID', 'age_angus23',
                                                                 ]],
                                           kepler_inference,
                                           columns_to_update=['age_angus23',
                                                              ])

    kepler_inference = selective_merge_dfs(bouma2024[['KID', 'age_bouma24_gyro',
                                                      'age_bouma24_li'
                                                      ]],
                                           kepler_inference,
                                           columns_to_update=['age_bouma24_gyro',
                                                      'age_bouma24_li'
                                                              ])

    kepler_inference = selective_merge_dfs(kois[['KID', 'kepoi_name',
                                                 'kepler_name',
                                                 'koi_disposition',
                                                 'planet_Prot',
                                                 'koi_prad']],
                                           kepler_inference,
                                           columns_to_update=['kepoi_name','kepler_name','koi_disposition',
                                                              'planet_Prot', 'koi_prad'])

    kepler_inference = selective_merge_dfs(kamiaka18, kepler_inference, columns_to_update=['i_value', 'i_error_low',
                                                                                           'i_error_high','dnu_value',
                                                                                           'dnu_error_low', 'dnu_error_high'
                                                                                           ])

    kepler_inference = kepler_inference[~kepler_inference['KID'].duplicated()]

    return kepler_inference


def aggregate_results(df, target_att='predicted period', selected_numbers=None):
    df.rename(columns={f'{target_att}': f'{target_att}_0'}, inplace=True)
    # If selected_numbers is provided, add columns with specified suffixes
    if selected_numbers:
        # Build regex pattern to match exact `target_att` with specific numbered suffixes
        number_pattern = "|".join(map(str, selected_numbers))
        pattern = f'^{target_att}_(?:{number_pattern})$'  # e.g., '^predicted period_(1|3|5)$'
        selected_columns = df.filter(regex=pattern).columns
    else:
        # Otherwise, select all columns matching the target attribute
        selected_columns = df.filter(regex=f'^{target_att}(_[0-9]+)?$').columns

    # Aggregate the selected columns by calculating the mean across rows
    df[target_att] = df[selected_columns].median(axis=1)
    return df


def get_doubles_from_acf(df, p_label):
    # df['period_diff'] = np.abs(df['predicted period']*2 - df['Prot'])
    df['doubles'] = np.abs(df['predicted period'] * 2 - df[p_label]) < df[p_label] * 0.2
    cond1 = (df['doubles'] == True) & (df['second_peak']==True)
    # cond2 = (df['doubles'] == True) & (np.abs(df['predicted acf_p'] - df[p_label]) < 2)
    print("number of double period mistakes: ", len(df[cond1]))
    df.loc[cond1 , 'predicted period'] =  df.loc[cond1, p_label]
    # df.loc[cond2 , 'predicted period'] =  df.loc[cond2, p_label]

    return df

def apply_constraints(df, teff=None, contaminants=True, doubles=True, conf=None, low_p=None, error=None):
    print("****number of samples****")
    print("before constraints: ", len(df))
    if teff is not None:
        df = df[df['Teff'] < teff]
        print("after removing Teff > 7000 from Berger: ", len(df))
    if contaminants:
        df = df[df['flag1'] != 6]
        df = df[df['flag1'] != 5]
        df = df[df['flag1'] != 4]
        print("after removing contaminants: ", len(df))
    if doubles:
        df = get_doubles_from_acf(df, p_label='Prot_mcq14')
        print("after applying acf second peak: ", len(df))
    if low_p is not None:
        print('number of low acf - ', len(df[(df['predicted acf_p no doubles'] < low_p) &
                                            (df['predicted acf_p no doubles'] > 0)]))
        df = df[~((df['predicted acf_p no doubles'] > 0) & (df['predicted acf_p no doubles'] <= low_p))]
        print("after removing fast rotators: ", len(df))
        df = df[~((df['eb_orbital_period'] < low_p) & (df['eb_orbital_period'] > 0))]
        print("after removing eb fast rotators: ", len(df))
        df = df[~((df['Prot_mcq14'] > 0) & (df['Prot_mcq14'] <= 3))]
        print("after removing mcq14 fast rotators: ", len(df))

    if conf is not None:
        df = df[df['mean_period_confidence'] > conf]
        print("after applying confidence threshold: ", len(df ))
    if error is not None:
        df = df[df['total error'] < error]
        print("after applying error threshold: ", len(df))
    return df


def fill_nan_np(x, interpolate=True):

    # Find indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(x))[0]

    # Find indices of NaN values
    nan_indices = np.where(np.isnan(x))[0]
    if interpolate:
        # Interpolate NaN values using linear interpolation
        interpolated_values = np.interp(nan_indices, non_nan_indices, x[non_nan_indices])

        # Replace NaNs with interpolated values
        x[nan_indices] = interpolated_values
    else:
        x[nan_indices] = 0
    return x