import os
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def regression_munich_vij(regenerate_data):
    data = pd.read_csv('../Munich/munich_hpm_sample.csv')
    od = pd.read_csv('../Munich/OD_matrix_with_info.csv')
    current_dis = od.loc[od['work_ID']==od['home_ID']][['home_ID','work_ID','distance']]
    data = data.merge(current_dis[['work_ID','home_ID','distance']], on =['work_ID', 'home_ID'], how='left')
    data = data.rename(columns={'distance': 'commute_dis'})
    # p75_dis = np.percentile(data['commute_dis'], 75)
    # p50_dis = np.percentile(data['commute_dis'], 50)
    # p25_dis = np.percentile(data['commute_dis'], 25)
    # data['if_commute_dis_greater_p75'] = 0
    # data.loc[data['commute_dis']>=p75_dis, 'if_commute_dis_greater_p75'] = 1
    # data['if_commute_dis_p50_75'] = 0
    # data.loc[(data['commute_dis']<p75_dis) & (data['commute_dis']>=p50_dis), 'if_commute_dis_p50_75'] = 1
    # data['if_commute_dis_p25_50'] = 0
    # data.loc[(data['commute_dis']<p50_dis) & (data['commute_dis']>=p25_dis), 'if_commute_dis_p25_50'] = 1
    data['commute_dis'] /= 1000
    # data['if_commute_dis_greater_17km'] = 0
    # data.loc[data['commute_dis'] >= 17, 'if_commute_dis_greater_17km'] = 1
    # data['commute_dis_log'] = np.log(data['commute_dis'] + 1)

    data['if_income_leq_3'] = 0
    data.loc[data['income']<=3, 'if_income_leq_3'] = 1
    data['if_income_geq_7'] = 0
    data.loc[data['income'] >= 7, 'if_income_geq_7'] = 1

    ind_category_var = ['edu', 'hhsize','income']
    home_category_var = ['buildtype','room']
    y_col = 'log_hcost'
    ind_continue_var = ['commute_dis']
    data['log_dis_cbd'] = np.log(data['cbd'])
    data['log_dis_lake'] = np.log(data['dis_lake'])
    # data['poi'] = np.sqrt(data['poi'])
    data['if_cbd_geq_5000'] = 0
    data.loc[data['cbd']>=5000, 'if_cbd_geq_5000'] = 1
    #
    # data['if_cbd_2000_5000'] = 0
    # data.loc[(data['cbd']<5000) & (data['cbd']>2000), 'if_cbd_2000_5000'] = 1

    # data['if_cbd_leq_500'] = 0
    # data.loc[data['cbd']<=200, 'if_cbd_leq_500'] = 1
    home_continue_var = ['subway', 'poi', 'if_cbd_geq_5000', 'aveincome', 'dis_lake'] #,



    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        data[col] = data[col].astype(int)
        return data

    data = fill_by_most_freq(data, 'edu')
    data = fill_by_most_freq(data, 'income')
    data = fill_by_most_freq(data, 'hhsize')
    data = fill_by_most_freq(data, 'buildtype')
    data.loc[data['hcost']<=0, 'hcost'] = np.mean(data.loc[data['hcost']>0, 'hcost'])

    for col in ind_category_var + home_category_var:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
    for col in ind_continue_var + home_continue_var:
        # fill na by mean
        data[col] = data[col].fillna(np.mean(data[col]))

    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, np.int64)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list


    # Create dummy variables for categorical variables

    data, ind_category_var_dummy = add_dummy(data, ind_category_var)
    data, home_category_var_dummy = add_dummy(data, home_category_var)

    data_used = data.loc[data['hcost'] > 0].copy()
    data_used['log_hcost'] = np.log(data_used['hcost'])

    ############

    X_columns = ind_continue_var + home_continue_var + ind_category_var_dummy + home_category_var_dummy
    X = data_used[X_columns]

    # Add a constant term for regression
    X = sm.add_constant(X)

    # Define the dependent variable (Y)
    Y = data_used[y_col]

    # Fit the regression model
    model = sm.OLS(Y, X).fit()

    # Save the regression results to a CSV

    # Get the model summary
    summary = model.summary()

    # Save the summary as a CSV file
    with open('regression_res/mu_housing_value_res_vij_summary.csv', 'w') as f:
        f.write(summary.as_csv())
    print(summary)
    results_summary = model.summary2().tables[1]
    results_summary.to_csv('regression_res/mu_housing_value_res_vij.csv')

    # results_summary = model.summary().tables[0].to_csv('regression_res/mu_housing_value_res_vij_summary.csv')
    # results_summary.to_csv('regression_res/mu_housing_value_res_vij_summary.csv',index=False,header=False)

    # Print the regression results
    # print(model.summary())
    return model, data, ind_continue_var+ind_category_var_dummy, home_continue_var+home_category_var_dummy, X_columns

def predict_vij_munich(model, data, ind_col, home_col, X_columns):
    ind_data = data[['work_ID'] + ind_col]
    home_data = data[['home_ID'] + home_col]
    data_pair = ind_data.merge(home_data, how='cross')
    data_pair['commute_dis'] = 0 # when inference, we calculate the willingness to pay for amentity only
    n_samples = 100
    params_mean = model.params
    params_std = model.bse

    np.random.seed(100)
    sampled_params = np.random.normal(loc=params_mean.values, scale=params_std.values,
                                      size=(n_samples, len(params_mean)))
    X = data_pair[X_columns]
    # Add a constant term for regression
    X = sm.add_constant(X)
    predictions = np.dot(X.values, sampled_params.T)
    std_predictions = predictions.std(axis=1)
    # Fit the regression model
    print('std prop')
    log_hcost = model.predict(X)
    log_hcost_high = log_hcost + 0.5 * std_predictions
    log_hcost_low = log_hcost - 0.5 * std_predictions
    data_pair['inferred_v_ij'] = np.exp(log_hcost)
    data_pair['inferred_v_ij_high'] = np.exp(log_hcost_high)
    data_pair['inferred_v_ij_low'] = np.exp(log_hcost_low)
    print(f"high %: {np.mean(data_pair['inferred_v_ij_high']) / np.mean(data_pair['inferred_v_ij'])}, low %: {np.mean(data_pair['inferred_v_ij_low']) / np.mean(data_pair['inferred_v_ij'])}")
    round_col = ['inferred_v_ij','inferred_v_ij_high', 'inferred_v_ij_low']
    for col in round_col:
        data_pair[col] = np.round(data_pair[col], 4)
    data_pair[['work_ID','home_ID','inferred_v_ij','inferred_v_ij_high', 'inferred_v_ij_low']].to_parquet('../Munich/munich_hpm_paired_results_vij.parquet', index=False)


def regression_munich_pi():
    data = pd.read_csv('../Munich/munich_hpm_sample.csv')


    home_category_var = ['buildtype']
    y_col = 'log_hcost'
    home_continue_var = ['room', 'subway', 'poi', 'cbd', 'aveincome', 'dis_lake']

    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        data[col] = data[col].astype(int)
        return data

    data = fill_by_most_freq(data, 'edu')
    data = fill_by_most_freq(data, 'income')
    data = fill_by_most_freq(data, 'hhsize')
    data = fill_by_most_freq(data, 'buildtype')
    data.loc[data['hcost']<=0, 'hcost'] = np.mean(data.loc[data['hcost']>0, 'hcost'])

    for col in home_category_var:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
    for col in home_continue_var:
        # fill na by mean
        data[col] = data[col].fillna(np.mean(data[col]))

    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, np.int64)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list


    # Create dummy variables for categorical variables

    data, home_category_var_dummy = add_dummy(data, home_category_var)

    data_used = data.loc[data['hcost'] > 0].copy()
    data_used['log_hcost'] = np.log(data_used['hcost'])

    ############

    X_columns = home_continue_var + home_category_var_dummy
    X = data_used[X_columns]

    # Add a constant term for regression
    X = sm.add_constant(X)

    # Define the dependent variable (Y)
    Y = data_used[y_col]

    # Fit the regression model
    model = sm.OLS(Y, X).fit()

    # Save the regression results to a CSV
    results_summary = model.summary2().tables[1]
    results_summary.to_csv('regression_res/mu_housing_value_res_pi.csv')

    with open('regression_res/mu_housing_value_res_pi_summary.csv', 'w') as f:
        f.write(model.summary().as_csv())

    # Print the regression results
    print(model.summary())
    return model, data, home_continue_var+home_category_var_dummy, X_columns

def predict_pi_munich(model, data, home_col, X_columns):
    home_data = data[['home_ID'] + home_col].copy()

    X = home_data[X_columns]
    # Add a constant term for regression
    X = sm.add_constant(X)
    # Fit the regression model
    log_hcost = model.predict(X)
    home_data['inferred_pi'] = np.exp(log_hcost)
    data['build_type_room'] = data['buildtype'].astype('str') + '_' + (data['room']).astype('int').astype('str')
    home_data = home_data.merge(data[['home_ID','build_type_room']],on=['home_ID'])
    home_data[
        ['home_ID','inferred_pi','build_type_room',
         'aveincome','subway','cbd','poi','dis_lake']].rename(
        columns={"aveincome":'ave_income', 'subway':'dis_to_subway', 'cbd': 'dis_to_cbd', 'poi':'num_poi','dis_lake':'dis_to_lake',
                 'build_type_room': 'build_type'
                 }).to_csv('../Munich/munich_hpm_paired_results_pi.csv', index=False)

def regression_sg_vij(regenerate_data):
    data = pd.read_csv('../Singapore/hpm_house.csv')
    data = data.drop(columns=['hhsize']) # included in ind data
    home_ID_used = pd.read_csv('../Singapore/homeID_homePcode.csv')
    work_ID_used = pd.read_csv('../Singapore/workID_workPcode.csv')
    all_sg_ind = pd.read_csv('../Singapore/all_sg_individuals.csv')[['H1_HHID','HP_ID','Home_pcode','Work_pcode']]
    all_sg_ind = all_sg_ind.merge(work_ID_used[['Work_pcode','work_ID']], on =['Work_pcode'])
    data = data.merge(home_ID_used[['home_ID']], on =['home_ID'])

    # print(len(data), len(pd.unique(data['HHID'])))
    ind_data = pd.read_csv('../Singapore/ind_attributes.csv')

    ### filter ind to keep only earners
    ind_data = ind_data.loc[ind_data['Employ'].isin(['Self-employed', 'Employed Full-time'])].rename(columns={'H1_HHID': 'HHID'})
    ind_data = ind_data.merge(all_sg_ind[['HP_ID','Work_pcode','work_ID']], on =['HP_ID'])
    data = data.merge(ind_data, on=['HHID'])

    age_mapping = {
        '15-19 yrs old': '15-34 yrs',
        '20-24 yrs old': '15-34 yrs',
        '25-29 yrs old': '15-34 yrs',
        '30-34 yrs old': '15-34 yrs',
        '35-39 yrs old': '35-54 yrs',
        '40-44 yrs old': '35-54 yrs',
        '45-49 yrs old': '35-54 yrs',
        '50-54 yrs old': '35-54 yrs',
        '55-59 yrs old': '55-74 yrs',
        '60-64 yrs old': '55-74 yrs',
        '65-69 yrs old': '55-74 yrs',
        '70-74 yrs old': '55-74 yrs',
        '75-79 yrs old': '75 yrs & above',
        '80-84 yrs old': '75 yrs & above',
        '85 yrs & above': '75 yrs & above'
    }
    data['Age'] = data['Age'].apply(lambda x: age_mapping[x])


    od = pd.read_csv('../Singapore/final_singapore_ODmatrix.csv')
    data = data.merge(od[['home_ID','work_ID','distance']], on =['work_ID','home_ID'])
    data = data.rename(columns={'distance': 'commute_dis'})
    data['commute_dis'] /= 1000

    all_households = data[['HHID']].drop_duplicates()
    all_individual = data[['HHID', 'HP_ID','home_ID', 'Work_pcode','work_ID']].drop_duplicates()

    print(f'num individuals {len(all_individual)}')
    print(f'num household: {len(all_households)}')

    print(f'num unique home ID {len(pd.unique(data["home_ID"]))}')
    print(f'num unique work ID: {len(pd.unique(data["work_ID"]))}')

    ## merge OD to get commute distance

    ind_category_var = ['Ethnic', 'Age', 'Citizen', 'Gender', 'Occup', 'income']
    home_category_var = ['DwellingType']
    y_col = 'log_hcost'
    ind_continue_var = ['commute_dis']
    home_continue_var = ['dis_to_subway', 'dis_to_CBD', 'POI', 'area', 'aveIncome','dis_to_wetlands']

    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data
    #
    for key in ind_category_var + home_category_var:
        data = fill_by_most_freq(data, key)
    data.loc[data['tot_price']<=0, 'tot_price'] = np.mean(data.loc[data['tot_price']>0, 'tot_price'])

    # for col in ind_category_var + home_category_var:
    #     data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
    for col in ind_continue_var + home_continue_var:
        # fill na by mean
        data[col] = data[col].fillna(np.mean(data[col]))

    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, np.int64) or isinstance(key, str)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list


    # Create dummy variables for categorical variables

    data, ind_category_var_dummy = add_dummy(data, ind_category_var)
    data, home_category_var_dummy = add_dummy(data, home_category_var)

    data_used = data.loc[data['tot_price'] > 0].copy()
    data_used['log_hcost'] = np.log(data_used['tot_price'])

    ############

    X_columns = ind_continue_var + home_continue_var + ind_category_var_dummy + home_category_var_dummy
    X = data_used[X_columns]

    # Add a constant term for regression
    X = sm.add_constant(X)

    # Define the dependent variable (Y)
    Y = data_used[y_col]

    # Fit the regression model
    model = sm.OLS(Y, X).fit()


    # Save the summary as a CSV file
    with open('regression_res/sg_housing_value_res_vij_summary.csv', 'w') as f:
        f.write(model.summary().as_csv())

    # Save the regression results to a CSV
    results_summary = model.summary2().tables[1]
    results_summary.to_csv('regression_res/sg_housing_value_res_vij.csv')

    # Print the regression results
    print(model.summary())
    return model, data, ind_continue_var+ind_category_var_dummy, home_continue_var+home_category_var_dummy, X_columns


def predict_vij_sg(model, data, ind_col, home_col, X_columns):
    ind_data = data[['HP_ID','work_ID'] + ind_col].reset_index(drop=True)
    home_data = data[['HHID','home_ID'] + home_col].drop_duplicates().reset_index(drop=True)
    print(f'num ind: {len(ind_data)}')
    print(f'num household: {len(home_data)}')
    # Split ind_data into manageable chunks
    chunk_size = 1600  # Adjust chunk size based on available memory
    num_chunk = len(ind_data) / chunk_size
    print(f'num chunks {num_chunk}')
    chunks = [ind_data[i:i + chunk_size] for i in range(0, len(ind_data), chunk_size)]
    # print(f'num var {len(X_columns)}')
    # Process each chunk
    results = []
    chunk_id = 0
    for chunk in chunks:
        chunk_id += 1
        print(f'current chunk {chunk_id}, total {num_chunk}')
        data_pair_chunk = chunk.merge(home_data, how='cross')
        data_pair_chunk['commute_dis'] = 0.0
        X_chunk = data_pair_chunk[X_columns].copy()
        X_chunk = sm.add_constant(X_chunk, has_constant='add')
        log_hcost_chunk = model.predict(X_chunk)
        data_pair_chunk['inferred_v_ij'] = np.exp(log_hcost_chunk)
        results.append(data_pair_chunk[['HP_ID', 'work_ID', 'HHID', 'home_ID', 'inferred_v_ij']])

    # Combine all chunks and save results
    final_result = pd.concat(results, ignore_index=True)
    # final_result.to_csv('../Singapore/sg_hpm_paired_results_vij.csv', index=False)
    final_result.to_parquet('../Singapore/sg_hpm_paired_results_vij.parquet', index=False)


    # data_pair = ind_data.merge(home_data, how='cross')
    # X = data_pair[X_columns]
    # # Add a constant term for regression
    # X = sm.add_constant(X)
    # # Fit the regression model
    # log_hcost = model.predict(X)
    # data_pair['inferred_v_ij'] = np.exp(log_hcost)
    # data_pair[['work_ID','home_ID','inferred_v_ij']].to_csv('../Singapore/sg_hpm_paired_results_vij.csv', index=False)



def regression_sg_pi():
    data = pd.read_csv('../Singapore/hpm_house.csv')
    data = data.drop(columns=['hhsize']) # included in ind data
    home_ID_used = pd.read_csv('../Singapore/homeID_homePcode.csv')
    work_ID_used = pd.read_csv('../Singapore/workID_workPcode.csv')
    all_sg_ind = pd.read_csv('../Singapore/all_sg_individuals.csv')[['H1_HHID','HP_ID','Home_pcode','Work_pcode']]
    all_sg_ind = all_sg_ind.merge(work_ID_used[['Work_pcode','work_ID']], on =['Work_pcode'])
    data = data.merge(home_ID_used[['home_ID']], on =['home_ID'])

    # print(len(data), len(pd.unique(data['HHID'])))
    ind_data = pd.read_csv('../Singapore/ind_attributes.csv')
    ### filter ind to keep only earners
    ind_data = ind_data.loc[ind_data['Employ'].isin(['Self-employed', 'Employed Full-time'])].rename(columns={'H1_HHID': 'HHID'})
    ind_data = ind_data.merge(all_sg_ind[['HP_ID','Work_pcode','work_ID']], on =['HP_ID'])
    data = data.merge(ind_data, on=['HHID'])
    all_households = data[['HHID']].drop_duplicates()
    all_individual = data[['HHID', 'HP_ID','home_ID', 'Work_pcode','work_ID']].drop_duplicates()
    all_individual.to_csv('../Singapore/final_used_individuals.csv',index=False)

    print(f'num individuals {len(all_individual)}')
    print(f'num household: {len(all_households)}')

    print(f'num unique home ID {len(pd.unique(data["home_ID"]))}')
    print(f'num unique work ID: {len(pd.unique(data["work_ID"]))}')


    home_category_var = ['DwellingType']
    y_col = 'log_hcost'

    home_continue_var = ['dis_to_subway', 'dis_to_CBD', 'POI', 'area', 'aveIncome','dis_to_wetlands']

    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data
    #
    for key in home_category_var:
        data = fill_by_most_freq(data, key)

    data.loc[data['tot_price']<=0, 'tot_price'] = np.mean(data.loc[data['tot_price']>0, 'tot_price'])

    # for col in ind_category_var + home_category_var:
    #     data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
    for col in home_continue_var:
        # fill na by mean
        data[col] = data[col].fillna(np.mean(data[col]))

    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, np.int64) or isinstance(key, str)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list


    # Create dummy variables for categorical variables


    data, home_category_var_dummy = add_dummy(data, home_category_var)

    data_used = data.loc[data['tot_price'] > 0].copy()
    data_used['log_hcost'] = np.log(data_used['tot_price'])


    X_columns = home_continue_var + home_category_var_dummy
    data_used = data_used[['log_hcost','HHID']+X_columns].drop_duplicates()
    ############


    X = data_used[X_columns]
    print(f"num samples {len(X)}")

    # Add a constant term for regression
    X = sm.add_constant(X)

    # Define the dependent variable (Y)
    Y = data_used[y_col]

    # Fit the regression model
    model = sm.OLS(Y, X).fit()

    with open('regression_res/sg_housing_value_res_pi_summary.csv', 'w') as f:
        f.write(model.summary().as_csv())

    # Save the regression results to a CSV
    results_summary = model.summary2().tables[1]
    results_summary.to_csv('regression_res/sg_housing_value_res_pi.csv')

    # Print the regression results
    print(model.summary())
    return model, data, home_continue_var+home_category_var_dummy, X_columns


def predict_pi_sg(model, data, home_col, X_columns):
    home_data = data[['HHID', 'home_ID'] + home_col].drop_duplicates()

    X = home_data[X_columns]
    # Add a constant term for regression
    X = sm.add_constant(X)
    # Fit the regression model
    log_hcost = model.predict(X)
    home_data['inferred_pi'] = np.exp(log_hcost)
    used_col = ['DwellingType','dis_to_CBD','dis_to_wetlands','dis_to_subway','POI','aveIncome']
    home_data = home_data.merge(data[['HHID', 'home_ID', 'DwellingType']].drop_duplicates(),on=['HHID','home_ID'])
    print(f'num houses: {len(home_data)}')
    home_data[['HHID', 'home_ID','inferred_pi'] + used_col].rename(columns={
        'DwellingType': 'build_type','aveIncome':'ave_income',
        'POI':'num_poi', 'dis_to_wetlands':'dis_to_lake','dis_to_CBD':'dis_to_cbd'}).to_csv('../Singapore/sg_hpm_paired_results_pi.csv', index=False)




def regression_bj_vij(regenerate_data):
    data = pd.read_csv('../Beijing/ind_home_sample_final.csv')
    data = data.rename(columns={'homeID':'home_ID'})
    od = pd.read_csv("../Beijing/Beijing_ODmatrix.csv")
    current_dis = od.loc[od['home_ID']==od['work_ID']]
    data = data.merge(current_dis[['home_ID', 'work_ID', 'distance']], on =['home_ID'])
    data = data.rename(columns={'distance':'commute_dis','house_price':'hcost'})
    data['commute_dis'] /= 1000
    # make it consistent with mu
    data['work_ID'] -= 1
    data['home_ID'] -= 1
    od['work_ID'] -= 1
    od['home_ID'] -= 1

    ### process_chinese
    data['age'] = data['age'].str.replace('岁','')
    # print(data['hhtype'].unique())
    edu_map= {
    "本科（大专）": "Bachelor",
    "硕士": "Master",
    "高中及以下": "High School and Below",
    "博士及以上": "Doctorate and Above"
    }
    data['edu'] = data['edu'].apply(lambda x: edu_map[x])
    gender_map= {
    "女": "female",
    "男": "male",
    }
    data['gender'] = data['gender'].apply(lambda x: gender_map[x])

    living_arrangements = {
        "独居": "Living Alone",
        "伴侣携子女": "With Partner and Children",
        "三代及以上同住": "Three Generations or More Living Together",
        "其他": "Other",
        "伴侣同住": "Living with Partner",
        "与父母同住": "Living with Parents"
    }
    data['hhtype'] = data['hhtype'].apply(lambda x: living_arrangements[x])


    if regenerate_data or (not os.path.exists('../Beijing/processed_od_matrix.csv')):
        od.to_csv('../Beijing/processed_od_matrix.csv',index=False)
    if regenerate_data or (not os.path.exists('../Beijing/processed_individual_data.csv')):
        data.to_csv('../Beijing/processed_individual_data.csv',index=False)
    ind_category_var = ['edu', 'income','age','gender']
    home_category_var = ['hhtype']
    y_col = 'log_hcost'
    ind_continue_var = ['commute_dis']
    # data['POI'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
    # plt.show()
    # data['if_POI_leq_50'] = 0
    # data.loc[data['POI'] <= 50, 'if_POI_leq_50'] = 1
    # data['if_POI_geq_500'] = 0
    # data.loc[data['POI'] >= 500, 'if_POI_geq_500'] = 1
    # # data['log_poi'] = np.log(data['POI']+1)
    # # data['poi_sq'] = data['POI']**2
    home_continue_var = ['dis_to_subway', 'newPOI','dis_to_cbd', 'AveIncome', 'dis_to_Kunminghu']#'if_POI_leq_50','if_POI_geq_500'

    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data




    data = fill_by_most_freq(data, 'edu')
    data = fill_by_most_freq(data, 'income')
    data = fill_by_most_freq(data, 'age')
    data = fill_by_most_freq(data, 'gender')
    data = fill_by_most_freq(data, 'hhtype')
    data.loc[data['hcost']<=0, 'hcost'] = np.mean(data.loc[data['hcost']>0, 'hcost'])

    # for col in ind_category_var + home_category_var:
    #     data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')


    for col in ind_continue_var + home_continue_var:
        # fill na by mean
        data[col] = data[col].fillna(np.mean(data[col]))

    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, np.int64) or isinstance(key, str)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list


    # Create dummy variables for categorical variables

    data, ind_category_var_dummy = add_dummy(data, ind_category_var)
    data, home_category_var_dummy = add_dummy(data, home_category_var)

    data_used = data.loc[data['hcost'] > 0].copy()
    data_used['log_hcost'] = np.log(data_used['hcost'])

    ############

    X_columns = ind_continue_var + home_continue_var + ind_category_var_dummy + home_category_var_dummy
    X = data_used[X_columns]

    # Add a constant term for regression
    X = sm.add_constant(X)

    # Define the dependent variable (Y)
    Y = data_used[y_col]

    # Fit the regression model
    model = sm.OLS(Y, X).fit()

    # Save the summary as a CSV file
    with open('regression_res/bj_housing_value_res_vij_summary.csv', 'w') as f:
        f.write(model.summary().as_csv())

    # Save the regression results to a CSV
    results_summary = model.summary2().tables[1]
    results_summary.to_csv('regression_res/bj_housing_value_res_vij.csv')

    # Print the regression results
    print(model.summary())
    return model, data, ind_continue_var+ind_category_var_dummy, home_continue_var+home_category_var_dummy, X_columns


def predict_vij_bj(model, data, ind_col, home_col, X_columns):
    ind_data = data[['work_ID'] + ind_col]
    home_data = data[['home_ID'] + home_col]
    data_pair = ind_data.merge(home_data, how='cross')
    data_pair['commute_dis'] = 0.0

    n_samples = 100
    params_mean = model.params
    params_std = model.bse

    np.random.seed(100)
    sampled_params = np.random.normal(loc=params_mean.values, scale=params_std.values,
                                      size=(n_samples, len(params_mean)))
    X = data_pair[X_columns]
    # Add a constant term for regression
    X = sm.add_constant(X)
    predictions = np.dot(X.values, sampled_params.T)
    std_predictions = predictions.std(axis=1)
    # Fit the regression model
    print('std prop')
    log_hcost = model.predict(X)
    log_hcost_high = log_hcost + 1.1 * std_predictions
    log_hcost_low = log_hcost - 1.1 * std_predictions
    data_pair['inferred_v_ij'] = np.exp(log_hcost)
    data_pair['inferred_v_ij_high'] = np.exp(log_hcost_high)
    data_pair['inferred_v_ij_low'] = np.exp(log_hcost_low)
    print(f"high %: {np.mean(data_pair['inferred_v_ij_high']) / np.mean(data_pair['inferred_v_ij'])}, low %: {np.mean(data_pair['inferred_v_ij_low']) / np.mean(data_pair['inferred_v_ij']) - 1}")
    round_col = ['inferred_v_ij','inferred_v_ij_high', 'inferred_v_ij_low']
    for col in round_col:
        data_pair[col] = np.round(data_pair[col], 4)
    data_pair[['work_ID','home_ID','inferred_v_ij','inferred_v_ij_high', 'inferred_v_ij_low']].to_parquet('../Beijing/bj_hpm_paired_results_vij.parquet', index=False)


def regression_bj_pi():
    data = pd.read_csv('../Beijing/processed_individual_data.csv')


    home_category_var = ['hhtype']
    y_col = 'log_hcost'
    home_continue_var = ['dis_to_subway', 'newPOI','dis_to_cbd', 'AveIncome', 'dis_to_Kunminghu']


    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data


    data = fill_by_most_freq(data, 'hhtype')
    data.loc[data['hcost']<=0, 'hcost'] = np.mean(data.loc[data['hcost']>0, 'hcost'])


    for col in home_continue_var:
        # fill na by mean
        data[col] = data[col].fillna(np.mean(data[col]))

    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, np.int64) or isinstance(key, str)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list


    # Create dummy variables for categorical variables

    data, home_category_var_dummy = add_dummy(data, home_category_var)

    data_used = data.loc[data['hcost'] > 0].copy()
    data_used['log_hcost'] = np.log(data_used['hcost'])

    ############

    X_columns = home_continue_var + home_category_var_dummy
    X = data_used[X_columns]

    # Add a constant term for regression
    X = sm.add_constant(X)

    # Define the dependent variable (Y)
    Y = data_used[y_col]

    # Fit the regression model
    model = sm.OLS(Y, X).fit()


    with open('regression_res/bj_housing_value_res_pi_summary.csv', 'w') as f:
        f.write(model.summary().as_csv())

    # Save the regression results to a CSV
    results_summary = model.summary2().tables[1]
    results_summary.to_csv('regression_res/bj_housing_value_res_pi.csv')

    # Print the regression results
    print(model.summary())
    return model, data, home_continue_var+home_category_var_dummy, X_columns

def predict_pi_bj(model, data, home_col, X_columns):
    home_data = data[['home_ID'] + home_col].copy()

    X = home_data[X_columns]
    # Add a constant term for regression
    X = sm.add_constant(X)
    # Fit the regression model
    log_hcost = model.predict(X)
    home_data['inferred_pi'] = np.exp(log_hcost)
    home_data = home_data.merge(data[['home_ID', 'hhtype']],on=['home_ID'])

    home_data[['home_ID','inferred_pi','hhtype','AveIncome','newPOI','dis_to_Kunminghu','dis_to_cbd','dis_to_subway']].rename(
        columns={"AveIncome":'ave_income', 'newPOI':'num_poi','dis_to_Kunminghu':'dis_to_lake','hhtype': 'build_type'
                 }).to_csv('../Beijing/bj_hpm_paired_results_pi.csv', index=False)




def regression_sg_ind_vij(regenerate_data):
    data = pd.read_csv('../Singapore_ind/hpm_house.csv')
    data = data.drop(columns=['hhsize'])  # included in ind data
    home_ID_used = pd.read_csv('../Singapore_ind/homeID_homePcode.csv')
    work_ID_used = pd.read_csv('../Singapore_ind/workID_workPcode.csv')
    all_sg_ind = pd.read_csv('../Singapore_ind/all_sg_individuals.csv')[['H1_HHID', 'HP_ID', 'Home_pcode', 'Work_pcode']]
    all_sg_ind = all_sg_ind.merge(work_ID_used[['Work_pcode', 'work_ID']], on=['Work_pcode'])
    data = data.merge(home_ID_used[['home_ID']], on=['home_ID'])

    # print(len(data), len(pd.unique(data['HHID'])))
    ind_data = pd.read_csv('../Singapore_ind/ind_attributes.csv')


    ### filter ind to keep only earners
    ind_data = ind_data.loc[ind_data['Employ'].isin(['Self-employed', 'Employed Full-time'])].rename(
        columns={'H1_HHID': 'HHID'})
    ind_data = ind_data.merge(all_sg_ind[['HP_ID', 'Work_pcode', 'work_ID']], on=['HP_ID'])
    data = data.merge(ind_data, on=['HHID'])
    age_mapping = {
        '15-19 yrs old': '15-34 yrs',
        '20-24 yrs old': '15-34 yrs',
        '25-29 yrs old': '15-34 yrs',
        '30-34 yrs old': '15-34 yrs',
        '35-39 yrs old': '35-54 yrs',
        '40-44 yrs old': '35-54 yrs',
        '45-49 yrs old': '35-54 yrs',
        '50-54 yrs old': '35-54 yrs',
        '55-59 yrs old': '55-74 yrs',
        '60-64 yrs old': '55-74 yrs',
        '65-69 yrs old': '55-74 yrs',
        '70-74 yrs old': '55-74 yrs',
        '75-79 yrs old': '75 yrs & above',
        '80-84 yrs old': '75 yrs & above',
        '85 yrs & above': '75 yrs & above'
    }
    data['Age'] = data['Age'].apply(lambda x: age_mapping[x])
    od = pd.read_csv('../Singapore_ind/final_singapore_ODmatrix.csv')
    data = data.merge(od[['home_ID', 'work_ID', 'distance']], on=['work_ID', 'home_ID'])
    data = data.rename(columns={'distance': 'commute_dis'})
    data['commute_dis'] /= 1000

    #### keep highest income ind for each household
    data = data.sort_values(['income'], ascending=False)
    data = data.groupby(['HHID']).first().reset_index()


    all_households = data[['HHID']].drop_duplicates()
    all_individual = data[['HHID', 'HP_ID', 'home_ID', 'Work_pcode', 'work_ID']].drop_duplicates()


    od = od.merge(home_ID_used[['home_ID','Home_pcode']], on=['home_ID'])
    od = od.merge(work_ID_used[['work_ID', 'Work_pcode']], on=['work_ID'])

    # assign new home & work ID
    data = data.sort_values(['HP_ID'])
    data['home_ID'] = np.arange(len(data))
    data['work_ID'] = np.arange(len(data))
    data = data.rename(columns={'postcode':'Home_pcode'})
    s_time = time.time()
    print('start saving new data')
    if regenerate_data or (not os.path.exists('../Singapore_ind/processed_od_matrix.parquet')):
        # generate new od
        od = od[['Home_pcode','Work_pcode','distance']].drop_duplicates()
        od_new = data[['home_ID','Home_pcode']].merge(data[['work_ID','Work_pcode']], how='cross')
        od_new = od_new.merge(od,on=['Home_pcode','Work_pcode'])
        del od
        ########### save processed data
        od_new.to_parquet('../Singapore_ind/processed_od_matrix.parquet',index=False)
    if regenerate_data or (not os.path.exists('../Singapore_ind/processed_individual_data.csv')):
        data.to_csv('../Singapore_ind/processed_individual_data.csv',index=False)
    print(f'finish saving data, time: {time.time()-s_time}')
    print(f'num individuals {len(all_individual)}')
    print(f'num household: {len(all_households)}')

    print(f'num unique home ID {len(pd.unique(data["home_ID"]))}')
    print(f'num unique work ID: {len(pd.unique(data["work_ID"]))}')

    ## merge OD to get commute distance

    ind_category_var = ['Ethnic', 'Age', 'Citizen', 'Gender', 'Occup', 'income']
    home_category_var = ['DwellingType']
    y_col = 'log_hcost'
    ind_continue_var = ['commute_dis']
    home_continue_var = ['dis_to_subway', 'dis_to_CBD', 'POI', 'area', 'aveIncome','dis_to_wetlands']

    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data

    #
    for key in ind_category_var + home_category_var:
        data = fill_by_most_freq(data, key)
    data.loc[data['tot_price'] <= 0, 'tot_price'] = np.mean(data.loc[data['tot_price'] > 0, 'tot_price'])

    # for col in ind_category_var + home_category_var:
    #     data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
    for col in ind_continue_var + home_continue_var:
        # fill na by mean
        data[col] = data[col].fillna(np.mean(data[col]))

    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, np.int64) or isinstance(key, str)]
            all_values = sorted(all_values)
            for val in all_values[:-1]:  # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col] == val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list

    # Create dummy variables for categorical variables

    data, ind_category_var_dummy = add_dummy(data, ind_category_var)
    data, home_category_var_dummy = add_dummy(data, home_category_var)

    data_used = data.loc[data['tot_price'] > 0].copy()
    data_used['log_hcost'] = np.log(data_used['tot_price'])

    ############

    X_columns = ind_continue_var + home_continue_var + ind_category_var_dummy + home_category_var_dummy
    X = data_used[X_columns]

    # Add a constant term for regression
    X = sm.add_constant(X)

    # Define the dependent variable (Y)
    Y = data_used[y_col]

    # Fit the regression model
    model = sm.OLS(Y, X).fit()

    # Save the summary as a CSV file
    with open('regression_res/sg_ind_housing_value_res_vij_summary.csv', 'w') as f:
        f.write(model.summary().as_csv())

    # Save the regression results to a CSV
    results_summary = model.summary2().tables[1]
    results_summary.to_csv('regression_res/sg_ind_housing_value_res_vij.csv')

    # Print the regression results
    print(model.summary())
    return model, data, ind_continue_var + ind_category_var_dummy, home_continue_var + home_category_var_dummy, X_columns



def predict_vij_sg_ind(model, data, ind_col, home_col, X_columns):
    ind_data = data[['work_ID'] + ind_col]
    home_data = data[['home_ID'] + home_col]

    chunk_size = 1600  # Adjust chunk size based on available memory
    num_chunk = len(ind_data) / chunk_size
    print(f'num chunks {num_chunk}')
    chunks = [ind_data[i:i + chunk_size] for i in range(0, len(ind_data), chunk_size)]
    # print(f'num var {len(X_columns)}')
    # Process each chunk
    results = []
    chunk_id = 0
    for chunk in chunks:
        chunk_id += 1
        print(f'current chunk {chunk_id}, total {num_chunk}')
        data_pair_chunk = chunk.merge(home_data, how='cross')
        data_pair_chunk['commute_dis'] = 0.0
        X_chunk = data_pair_chunk[X_columns].copy()
        X_chunk = sm.add_constant(X_chunk, has_constant='add')

        n_samples = 100
        params_mean = model.params
        params_std = model.bse

        np.random.seed(100)
        sampled_params = np.random.normal(loc=params_mean.values, scale=params_std.values,
                                          size=(n_samples, len(params_mean)))

        predictions = np.dot(X_chunk.values, sampled_params.T)
        std_predictions = predictions.std(axis=1)
        # Fit the regression model
        print('std prop')
        log_hcost = model.predict(X_chunk)
        log_hcost_high = log_hcost + 3.4 * std_predictions
        log_hcost_low = log_hcost - 3.4 * std_predictions
        data_pair_chunk['inferred_v_ij'] = np.exp(log_hcost)
        data_pair_chunk['inferred_v_ij_high'] = np.exp(log_hcost_high)
        data_pair_chunk['inferred_v_ij_low'] = np.exp(log_hcost_low)

        results.append(data_pair_chunk[['work_ID', 'home_ID', 'inferred_v_ij','inferred_v_ij_high','inferred_v_ij_low']])

    # Combine all chunks and save results
    final_result = pd.concat(results, ignore_index=True)
    round_col = ['inferred_v_ij','inferred_v_ij_high', 'inferred_v_ij_low']
    for col in round_col:
        final_result[col] = np.round(final_result[col], 4)
    final_result.to_parquet('../Singapore_ind/sg_ind_hpm_paired_results_vij.parquet', index=False)



def regression_sg_ind_pi():
    data = pd.read_csv('../Singapore_ind/processed_individual_data.csv')


    home_category_var = ['DwellingType']
    y_col = 'log_hcost'
    home_continue_var = ['dis_to_subway', 'dis_to_CBD', 'POI', 'area', 'aveIncome','dis_to_wetlands']


    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data
    #
    for key in home_category_var:
        data = fill_by_most_freq(data, key)

    data.loc[data['tot_price']<=0, 'tot_price'] = np.mean(data.loc[data['tot_price']>0, 'tot_price'])

    # for col in ind_category_var + home_category_var:
    #     data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')
    for col in home_continue_var:
        # fill na by mean
        data[col] = data[col].fillna(np.mean(data[col]))

    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, np.int64) or isinstance(key, str)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list


    # Create dummy variables for categorical variables


    data, home_category_var_dummy = add_dummy(data, home_category_var)

    data_used = data.loc[data['tot_price'] > 0].copy()
    data_used['log_hcost'] = np.log(data_used['tot_price'])


    X_columns = home_continue_var + home_category_var_dummy
    data_used = data_used[['log_hcost','work_ID']+X_columns].drop_duplicates()
    ############


    X = data_used[X_columns]
    print(f"num samples {len(X)}")

    # Add a constant term for regression
    X = sm.add_constant(X)

    # Define the dependent variable (Y)
    Y = data_used[y_col]

    # Fit the regression model
    model = sm.OLS(Y, X).fit()


    with open('regression_res/sg_ind_housing_value_res_pi_summary.csv', 'w') as f:
        f.write(model.summary().as_csv())

    # Save the regression results to a CSV
    results_summary = model.summary2().tables[1]
    results_summary.to_csv('regression_res/sg_ind_housing_value_res_pi.csv')

    # Print the regression results
    print(model.summary())
    return model, data, home_continue_var+home_category_var_dummy, X_columns

def predict_pi_sg_ind(model, data, home_col, X_columns):
    home_data = data[['home_ID'] + home_col].copy()

    X = home_data[X_columns]
    # Add a constant term for regression
    X = sm.add_constant(X)
    # Fit the regression model
    log_hcost = model.predict(X)
    home_data['inferred_pi'] = np.exp(log_hcost)
    home_data = home_data.merge(data[['home_ID','DwellingType']],on=['home_ID'])

    home_data[['home_ID','inferred_pi','DwellingType','aveIncome','POI','dis_to_wetlands','dis_to_CBD','dis_to_subway']].rename(
        columns={"aveIncome":'ave_income', 'POI':'num_poi','dis_to_wetlands':'dis_to_lake','dis_to_CBD':'dis_to_cbd','DwellingType': 'build_type'
                 }).to_csv('../Singapore_ind/sg_ind_hpm_paired_results_pi.csv', index=False)




if __name__ == '__main__':
    ################# Munich
    # model, data, ind_col, home_col, X_columns = regression_munich_vij(regenerate_data=True)
    # predict_vij_munich(model, data, ind_col, home_col, X_columns)
    # # # #
    # model, data, home_col, X_columns = regression_munich_pi()
    # predict_pi_munich(model, data, home_col, X_columns)
    #
    # # ################# Singapore
    # #
    # model, data, ind_col, home_col, X_columns = regression_sg_vij(regenerate_data=True)
    # predict_vij_sg(model, data, ind_col, home_col, X_columns)
    #
    model, data, home_col, X_columns = regression_sg_pi()
    predict_pi_sg(model, data, home_col, X_columns)
    # # #
    # # #
    # # # # ################# Beijing
    # model, data, ind_col, home_col, X_columns = regression_bj_vij(regenerate_data=False)
    # predict_vij_bj(model, data, ind_col, home_col, X_columns)
    # #
    # model, data, home_col, X_columns = regression_bj_pi()
    # predict_pi_bj(model, data, home_col, X_columns)
    # #
    # #
    # # # ################# Singapore_ind
    # model, data, ind_col, home_col, X_columns = regression_sg_ind_vij(regenerate_data=False)
    # predict_vij_sg_ind(model, data, ind_col, home_col, X_columns)
    # data_pair = pd.read_parquet('../Singapore_ind/sg_ind_hpm_paired_results_vij.parquet')
    # print(f"high %: {np.mean(data_pair['inferred_v_ij_high']) / np.mean(data_pair['inferred_v_ij'])}, low %: {np.mean(data_pair['inferred_v_ij_low']) / np.mean(data_pair['inferred_v_ij']) - 1}")
    # # # #
    # model, data, home_col, X_columns = regression_sg_ind_pi()
    # predict_pi_sg_ind(model, data, home_col, X_columns)