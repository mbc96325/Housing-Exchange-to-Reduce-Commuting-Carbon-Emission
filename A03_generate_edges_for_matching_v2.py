import os

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import time
import _constants
#
#
# def get_munich_config(scenario_name):
#     config = {
#         'scenario_name': scenario_name,
#         'avg_tenure_at_one_work': 5,
#         'free_income_prop': _constants.free_income_prop['Munich'],
#         'annual_income_growth_rate': 0.026,
#         'transaction_cost_percent': _constants.transaction_cost_percent['Munich'],
#         'v_ij_change': -0.10,
#         'vot_std': 0.0,
#         'avg_house_increase_per_year': np.power(1.543, 1 / 20) - 1,  # 20 years increase 54%
#         'house_price_and_increase_rate': 'Direct',  # Inverse
#         'v_ij_col': 'inferred_v_ij',
#         'future_gain_weight': 1.0,
#         'budget_income_years': 1,
#         'utility_constraints': True,
#         'v_ij_constraints': True,
#         'same_build_type': True,
#         'similar_house_attributes': False
#     }
#     if scenario_name == '1_0_ideal':
#         config['same_build_type'] = False
#         config['future_gain_weight'] = 0.0
#         config['transaction_cost_weight'] = 0.0
#         config['residential_benefit_weight'] = 0.0
#
#
#     if scenario_name == '2_0_in_between':
#         config['future_gain_weight'] = 0.0
#
#     if scenario_name == '2_1_in_between_high_vij':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_col'] = 'inferred_v_ij_high'
#
#     if scenario_name == '2_2_in_between_low_vij':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_col'] = 'inferred_v_ij_low'
#
#     if scenario_name == '2_3_in_between_high_vot':
#         config['future_gain_weight'] = 0.0
#         config['vot_std'] = 2.0
#
#     if scenario_name == '2_4_in_between_low_vot':
#         config['future_gain_weight'] = 0.0
#         config['vot_std'] = -2.0
#
#     if scenario_name == '2_5_in_between_vij_20_percent':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_change'] = -0.20
#
#     if scenario_name == '2_6_in_between_vij_30_percent':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_change'] = -0.30
#
#     if scenario_name == '3_0_realistic':
#         pass
#
#     if scenario_name == '3_1_inverse_future_gain':
#         config['house_price_and_increase_rate'] = 'Inverse'
#
#     if scenario_name == '3_2_time_span_3_years':
#         config['avg_tenure_at_one_work'] = 3
#
#     if scenario_name == '3_3_time_span_8_years':
#         config['avg_tenure_at_one_work'] = 8
#
#     if scenario_name == '3_4_time_span_10_years':
#         config['avg_tenure_at_one_work'] = 10
#
#     if scenario_name == '5_1_similar_attributes':
#         config['similar_house_attributes'] = True
#         config['house_attributes_percent'] = 0.1
#
#     if scenario_name == '5_1_similar_attributes_wo_vij':
#         config['similar_house_attributes'] = True
#         config['house_attributes_percent'] = 0.1
#
#
#     if 'random' in scenario_name:
#         sample_percent = int(scenario_name.split('random_')[1].split('_percent')[0])
#         config['sample_percent'] = sample_percent
#         config['sample_seed'] = int(scenario_name.split('seed_')[1])
#
#     return config
#
#
# def generate_edges_for_munich(config, all_data, skip_existing=False):
#     if skip_existing and os.path.exists(f'../Munich/final_pair_for_matching_{config["scenario_name"]}.csv'):
#         print(f'{config["scenario_name"]} exists, skip...')
#         return all_data
#
#     s_time = time.time()
#     if 'od_matrix' in all_data:
#         data = all_data['od_matrix'].copy()
#     else:
#         data = pd.read_csv('../Munich/OD_matrix_with_info.csv')
#         all_data['od_matrix'] = data.copy()
#     print(f"load od matrix time: {time.time() - s_time}")
#
#     if 'sample_percent' in config:
#         all_ids = np.sort(pd.unique(data['work_ID'])) #note work ID = home ID for single ind case
#         N = len(all_ids)
#         # Set the random seed for reproducibility
#         # Calculate the number of samples to draw (10% of the array)
#         num_samples = int(config['sample_percent'] / 100 * N)
#         np.random.seed(config['sample_seed'])
#         sampled_indices = np.random.choice(N, num_samples, replace=False)
#         sampled_values = all_ids[sampled_indices]
#         data = data.merge(pd.DataFrame({'work_ID': sampled_values}), on =['work_ID'])
#         data = data.merge(pd.DataFrame({'home_ID': sampled_values}), on =['home_ID'])
#
#     print(f'num samples check point 1: {len(pd.unique(data["work_ID"]))}')
#
#     if 'v_ij' in all_data:
#         house_value = all_data['v_ij'].copy()
#     else:
#         house_value = pd.read_parquet('../Munich/munich_hpm_paired_results_vij.parquet')
#         all_data['v_ij'] = house_value.copy()
#
#     house_value['inferred_v_ij'] = house_value[config['v_ij_col']].drop(columns=['inferred_v_ij_high','inferred_v_ij_low'])
#
#     if 'p_i' in all_data:
#         price_value = all_data['p_i'].copy()
#     else:
#         price_value = pd.read_csv('../Munich/munich_hpm_paired_results_pi.csv')
#         all_data['p_i'] = price_value.copy()
#
#     ind_to_work_id = pd.read_csv('../Munich/munich_hpm_sample.csv')
#     age = pd.read_csv('../Munich/Individual_Age.csv')
#     work_id_age = ind_to_work_id[['LfdNr','work_ID']].merge(age[['LfdNr','age']], on =['LfdNr'])
#     work_id_age.loc[work_id_age['age']<20, 'age'] = np.nan
#     data = data.merge(house_value,on=['home_ID','work_ID'], how='left')
#
#     print(f'num samples check point 2: {len(pd.unique(data["work_ID"]))}')
#
#     max_house_price = max(price_value['inferred_pi'])
#     min_house_price = min(price_value['inferred_pi'])
#     mean_house_price = np.mean(price_value['inferred_pi'])
#     min_change_rate = config['avg_house_increase_per_year'] * 0.7
#     if config['house_price_and_increase_rate'].lower() == 'direct':
#         price_value['price_change_rate'] = (price_value['inferred_pi'] - min_house_price) / (mean_house_price - min_house_price) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
#     else:
#         # inverse
#         price_value['price_change_rate'] = (max_house_price - price_value['inferred_pi']) / (
#                     max_house_price - mean_house_price) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
#     print(f"rate input {config['avg_house_increase_per_year']}, rate_cal {np.mean(price_value['price_change_rate'])}")
#
#     data = data.merge(price_value.rename(columns={'inferred_pi':'inferred_p_j','price_change_rate':'price_change_rate_j','build_type_room':'build_type_j'}), on=['home_ID'], how='left')
#     data = data.merge(work_id_age[['work_ID','age']], on=['work_ID'], how='left')
#     data['age'].fillna(np.mean(data['age']), inplace=True)
#     data['age'] = np.round(data['age'])
#     # data['age'].hist()
#     # plt.show()
#     data['inferred_v_ij'].fillna(np.mean(data['inferred_v_ij']),inplace=True)
#     data['inferred_p_j'].fillna(np.mean(data['inferred_p_j']), inplace=True)
#     data_status_quo = data.loc[data['work_ID']==data['home_ID']].rename(columns={'t_ij':'t_ii', 'c_ij':'c_ii',
#                                                                                  'inferred_v_ij':'inferred_v_ii','inferred_p_j':'inferred_p_i',
#                                                                                  'price_change_rate_j': 'price_change_rate_i','build_type_j': 'build_type_i'})
#
#     data = data.merge(data_status_quo[['work_ID','t_ii','c_ii','inferred_p_i','inferred_v_ii','price_change_rate_i','build_type_i']], on=['work_ID'])
#
#     print(f'num samples check point 3: {len(pd.unique(data["work_ID"]))}')
#
#     ############ first filter: same build type
#     ori_len = len(data)
#     if config['same_build_type']:
#         data = data.loc[data['build_type_i']==data['build_type_j']]
#         print(f'num samples check point 4: {len(pd.unique(data["work_ID"]))}')
#         print(f'remaining_prop after same build type: {len(data) / ori_len}')
#
#     if config['similar_house_attributes']:
#         attributes = ['ave_income', 'num_poi', 'dis_to_lake', 'dis_to_cbd', 'dis_to_subway']
#         for att in attributes:
#             avg_att = np.mean(data_status_quo[f'{att}_i'])
#             data = data.loc[np.abs(data[f'{att}_i'] - data[f'{att}_j']) <= avg_att * config['house_attributes_percent']]
#         print(f'num samples check point 5: {len(pd.unique(data["work_ID"]))}')
#         print(f'remaining_prop after similar_house_attributes: {len(data) / ori_len}')
#
#     #####################
#
#     # data['price_diff'] = (data['inferred_p_j'] - data['inferred_p_i']) * config['residential_benefit_weight']
#     data['price_diff'] = (data['inferred_v_ij'] - data['inferred_v_ii'])  * config['residential_benefit_weight']
#     data['year_until_change_location_or_retire'] = config['avg_tenure_at_one_work']# np.minimum(np.maximum(config['retire_age'] - data['age'], 0), config['avg_tenure_at_one_work']) # assuming people stay there for 15 years
#     # test
#     # data['year_until_retirement'] = 10
#     print(f"avg year staying: {np.mean(data['year_until_change_location_or_retire'])}")
#
#
#     data['future_gain_j'] = data['inferred_v_ij'] * (np.power(1+data['price_change_rate_j'], data['year_until_change_location_or_retire']) - 1.0)
#     data['future_gain_i'] = data['inferred_v_ii'] * (np.power(1+data['price_change_rate_i'], data['year_until_change_location_or_retire']) - 1.0)
#
#     # relocation cost = price diff + transaction fee + future gain
#     data['transaction_cost'] = data['inferred_v_ij'] * (data['home_ID']!=data['work_ID']) * config['transaction_cost_percent']
#     # need to pay for the price diff
#     data['r_ij'] = data['transaction_cost']
#     data['future_gain_diff'] = (data['future_gain_j'] - data['future_gain_i']) * config['future_gain_weight']
#
#
#     para = pd.read_csv('regression_res/dcm_mode_choice_munich.csv', index_col=0)
#     B_TIME = para.loc[f'B_TIME']['Value']
#     B_COST = para.loc[f'B_COST']['Value']
#     std_B_TIME = abs(para.loc[f'B_COST']['Std err'])
#     std_B_COST = abs(para.loc[f'B_TIME']['Std err'])
#     ## approximate vot mean and std
#     N = 10000
#     np.random.seed(100)
#     X_samples = np.random.normal(loc=B_TIME, scale=std_B_TIME, size=N)
#     Y_samples = np.random.normal(loc=B_COST, scale=std_B_COST, size=N)
#     # Ensure no division by zero (filter out small Y values)
#     Y_samples = np.where(Y_samples == 0, np.finfo(float).eps, Y_samples)
#     Z_samples = X_samples / Y_samples
#     # Compute mean and standard deviation
#     std_of_vot = np.std(Z_samples)
#     vot_no_income = B_TIME / B_COST
#     vot_factor = (1 + config['vot_std'] * std_of_vot / vot_no_income)
#     print(f'VOT Factor: {vot_factor}')
#     print(f'****VOT income prop****: {vot_factor * B_TIME / B_COST}')
#
#     data['vot_i_long_term'] =0.5 * (
#             (data['vot_i'] * vot_factor * np.power(1+config['annual_income_growth_rate'],
#                                                    data['year_until_change_location_or_retire'])) + data['vot_i']) # assume linear increase, mean will be 1/2(begin+end)
#
#     data['commute_benefit'] = (data['c_ii'] - data['c_ij'] + data['vot_i_long_term'] * (
#             data['t_ii'] - data['t_ij'])) * data['year_until_change_location_or_retire'] * 2 * _constants.num_work_days_per_year #260 workdays per year, 2 trips per day
#     data['residential_benefit'] = (data['inferred_v_ij'] - data['inferred_v_ii']) + data['future_gain_diff']
#
#     data['w_ij'] = data['commute_benefit'] + data['residential_benefit'] - data['r_ij']
#
#
#     num_month_per_year = 12 #  12 month per year
#     data['budget_i'] = config['free_income_prop'] * data['income_after_tax'] * num_month_per_year * config['budget_income_years'] #
#     ####
#
#
#
#     ### save_to_csv
#     data_used = data.loc[data['w_ij']>=0].copy()
#
#     data_used['v_ij_change_percent'] = (data_used['inferred_v_ij'] - data_used['inferred_v_ii']) / data_used['inferred_v_ii']
#     data_used = data_used.loc[data_used['v_ij_change_percent'] * config['residential_benefit_weight'] >= config['v_ij_change']]
#     data_used = data_used.loc[data_used['transaction_cost'] + data_used['price_diff'] < data_used['budget_i']]
#     print(f'remaining edge prop: {len(data_used) / ori_len }')
#
#     print(f'num samples check point 5: {len(pd.unique(data_used["work_ID"]))}')
#
#     saved_col = ['commute_benefit', 'residential_benefit','distance',
#                  'r_ij','t_ii','t_ij','c_ii','c_ij', 'inferred_v_ii', 'inferred_v_ij', 'inferred_p_i', 'inferred_p_j', 'year_until_change_location_or_retire']
#     prob_col = [key for key in data_used.columns if 'prob_' in key]
#     saved_col += prob_col
#
#     pollutant_col = ['CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij', 'SO2_ij', 'PM25_ij']
#     data_used['CO2_ij'] = 0
#     data_used['NOx_ij'] = 0
#     data_used['VOC_ij'] = 0
#     data_used['CO_ij'] = 0
#     data_used['SO2_ij'] = 0
#     data_used['PM25_ij'] = 0
#
#     ev_ratio = 0.05
#     for prob in prob_col:
#         if 'car' in prob:
#             data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO2'] * (1-ev_ratio))
#             data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['NOx'] * (1-ev_ratio))
#             data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['VOC'] * (1-ev_ratio))
#             data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO'] * (1-ev_ratio))
#             data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['SO2'] * (1-ev_ratio))
#             data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['PM25'] * (1-ev_ratio))
#         elif 'pt' in prob:
#             data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO2']
#             data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['NOx']
#             data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['VOC']
#             data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO']
#             data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['SO2']
#             data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['PM25']
#
#     saved_col += pollutant_col
#     for key in saved_col:
#         data_used[key] = np.round(data_used[key],4)
#
#     # weighted avg emissions
#
#     data_used[['home_ID','work_ID']+saved_col].to_csv(f'../Munich/final_pair_for_matching_{config["scenario_name"]}.csv',index=False)
#     return all_data
#
#
# def get_bj_config(scenario_name):
#
#     config={
#         'scenario_name': scenario_name,
#         'avg_tenure_at_one_work': 5,
#         'free_income_prop': _constants.free_income_prop['Beijing'],
#         'annual_income_growth_rate': 0.081,
#         'transaction_cost_percent': _constants.transaction_cost_percent['Beijing'],
#         'v_ij_change': -0.10,
#         'vot_std': 0.0,
#         'avg_house_increase_per_year': _constants.avg_house_increase_per_year['Beijing'],
#         'house_price_and_increase_rate': 'Direct', # Inverse
#         'v_ij_col': 'inferred_v_ij',
#         'future_gain_weight': 1.0,
#         'transaction_cost_weight': 1.0,
#         'residential_benefit_weight': 1.0,
#         'budget_income_years': 1,
#     }
#
#     if scenario_name == '1_0_ideal':
#         config['future_gain_weight'] = 0.0
#         config['transaction_cost_weight'] = 0.0
#         config['residential_benefit_weight'] = 0.0
#
#     if scenario_name == '2_0_in_between':
#         config['future_gain_weight'] = 0.0
#
#     if scenario_name == '2_1_in_between_high_vij':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_col'] = 'inferred_v_ij_high'
#
#     if scenario_name == '2_2_in_between_low_vij':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_col'] = 'inferred_v_ij_low'
#
#     if scenario_name == '2_3_in_between_high_vot':
#         config['future_gain_weight'] = 0.0
#         config['vot_std'] = 1.0
#
#     if scenario_name == '2_4_in_between_low_vot':
#         config['future_gain_weight'] = 0.0
#         config['vot_std'] = -1.0
#
#     if scenario_name == '2_5_in_between_vij_20_percent':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_change'] = -0.20
#
#     if scenario_name == '2_6_in_between_vij_30_percent':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_change'] = -0.30
#
#     if scenario_name == '3_0_realistic':
#         pass
#
#     if scenario_name == '3_1_inverse_future_gain':
#         config['house_price_and_increase_rate'] = 'Inverse'
#
#     if scenario_name == '3_2_time_span_3_years':
#         config['avg_tenure_at_one_work'] = 3
#
#     if scenario_name == '3_3_time_span_8_years':
#         config['avg_tenure_at_one_work'] = 8
#
#     if scenario_name == '3_4_time_span_10_years':
#         config['avg_tenure_at_one_work'] = 10
#
#     if 'random' in scenario_name:
#         sample_percent = int(scenario_name.split('random_')[1].split('_percent')[0])
#         config['sample_percent'] = sample_percent
#         config['sample_seed'] = int(scenario_name.split('seed_')[1])
#
#     return config
#
#
# def generate_edges_for_bj(config, all_data, skip_existing=False):
#     if skip_existing and os.path.exists(f'../Beijing/final_pair_for_matching_{config["scenario_name"]}.csv'):
#         print(f'{config["scenario_name"]} exists, skip...')
#         return all_data
#
#     s_time = time.time()
#     if 'od_matrix' in all_data:
#         data = all_data['od_matrix'].copy()
#     else:
#         data = pd.read_parquet('../Beijing/OD_matrix_with_info.parquet')
#         all_data['od_matrix'] = data.copy()
#     print(f"load od matrix time: {time.time() - s_time}")
#
#     if 'sample_percent' in config:
#         all_ids = np.sort(pd.unique(data['work_ID'])) #note work ID = home ID for single ind case
#         N = len(all_ids)
#         # Set the random seed for reproducibility
#         # Calculate the number of samples to draw (10% of the array)
#         num_samples = int(config['sample_percent'] / 100 * N)
#         np.random.seed(config['sample_seed'])
#         sampled_indices = np.random.choice(N, num_samples, replace=False)
#         sampled_values = all_ids[sampled_indices]
#         data = data.merge(pd.DataFrame({'work_ID': sampled_values}), on =['work_ID'])
#         data = data.merge(pd.DataFrame({'home_ID': sampled_values}), on =['home_ID'])
#
#     print(f'num samples check point 1: {len(pd.unique(data["work_ID"]))}')
#
#     if 'v_ij' in all_data:
#         house_value = all_data['v_ij'].copy()
#     else:
#         house_value = pd.read_parquet('../Beijing/bj_hpm_paired_results_vij.parquet')
#         all_data['v_ij'] = house_value.copy()
#
#     house_value['inferred_v_ij'] = house_value[config['v_ij_col']].drop(columns=['inferred_v_ij_high','inferred_v_ij_low'])
#
#     if 'p_i' in all_data:
#         price_value = all_data['p_i'].copy()
#     else:
#         price_value = pd.read_csv('../Beijing/bj_hpm_paired_results_pi.csv')
#         all_data['p_i'] = price_value.copy()
#
#     data = data.merge(house_value,on=['home_ID','work_ID'], how='left')
#
#     print(f'num samples check point 1: {len(pd.unique(data["work_ID"]))}')
#
#     max_house_price = max(price_value['inferred_pi'])
#     min_house_price = min(price_value['inferred_pi'])
#     mean_house_price = np.mean(price_value['inferred_pi'])
#     min_change_rate = config['avg_house_increase_per_year'] * 0.7
#     if config['house_price_and_increase_rate'].lower() == 'direct':
#         price_value['price_change_rate'] = (price_value['inferred_pi'] - min_house_price) / (mean_house_price - min_house_price) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
#     else:
#         # inverse
#         price_value['price_change_rate'] = (max_house_price - price_value['inferred_pi']) / (
#                     max_house_price - mean_house_price) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
#     print(f"rate input {config['avg_house_increase_per_year']}, rate_cal {np.mean(price_value['price_change_rate'])}")
#
#     data = data.merge(price_value.rename(columns={'inferred_pi':'inferred_p_j','price_change_rate':'price_change_rate_j','hhtype':'build_type_j'}), on=['home_ID'], how='left')
#
#     # data['age'].hist()
#     # plt.show()
#     data['inferred_v_ij'].fillna(np.mean(data['inferred_v_ij']),inplace=True)
#     data['inferred_p_j'].fillna(np.mean(data['inferred_p_j']), inplace=True)
#     data_status_quo = data.loc[data['work_ID']==data['home_ID']].rename(columns={'t_ij':'t_ii', 'c_ij':'c_ii',
#                                                                                  'inferred_v_ij':'inferred_v_ii','inferred_p_j':'inferred_p_i',
#                                                                                  'price_change_rate_j': 'price_change_rate_i','build_type_j':'build_type_i'})
#     data = data.merge(data_status_quo[['work_ID','t_ii','c_ii','inferred_p_i','inferred_v_ii','price_change_rate_i','build_type_i']], on=['work_ID'])
#
#     print(f'num samples check point 1: {len(pd.unique(data["work_ID"]))}')
#
#     ############ first filter: same build type
#     ori_len = len(data)
#     if '1_0_ideal' not in config['scenario_name']:
#         data = data.loc[data['build_type_i']==data['build_type_j']]
#         print(f'num samples check point 4: {len(pd.unique(data["work_ID"]))}')
#         print(f'remaining_prop after same build type: {len(data) / ori_len}')
#
#     ###############
#
#
#     data['price_diff'] = (data['inferred_v_ij'] - data['inferred_v_ii']) * config['residential_benefit_weight']
#     data['year_until_change_location_or_retire'] = config['avg_tenure_at_one_work']# np.minimum(np.maximum(config['retire_age'] - data['age'], 0), config['avg_tenure_at_one_work']) # assuming people stay there for 15 years
#     # test
#     # data['year_until_retirement'] = 10
#     print(f"avg year staying: {np.mean(data['year_until_change_location_or_retire'])}")
#
#
#     data['future_gain_j'] = data['inferred_v_ij'] * (np.power(1+data['price_change_rate_j'], data['year_until_change_location_or_retire']) - 1.0)
#     data['future_gain_i'] = data['inferred_v_ii'] * (np.power(1+data['price_change_rate_i'], data['year_until_change_location_or_retire']) - 1.0)
#
#     # relocation cost = price diff + transaction fee + future gain
#     data['transaction_cost'] = data['inferred_v_ij'] * (data['home_ID']!=data['work_ID']) * config['transaction_cost_percent'] * config['transaction_cost_weight'] #
#     # need to pay for the price diff
#     data['r_ij'] = data['transaction_cost']
#     data['future_gain_loss'] = (data['future_gain_i'] - data['future_gain_j']) * config['future_gain_weight']
#     data['r_ij'] += data['future_gain_loss']
#
#     para = pd.read_csv('regression_res/dcm_mode_choice_bj.csv', index_col=0)
#     B_TIME = para.loc[f'B_TIME']['Value']
#     B_COST = para.loc[f'B_COST']['Value']
#     std_B_TIME = abs(para.loc[f'B_COST']['Std err'])
#     std_B_COST = abs(para.loc[f'B_TIME']['Std err'])
#     ## approximate vot mean and std
#     N = 10000
#     np.random.seed(100)
#     X_samples = np.random.normal(loc=B_TIME, scale=std_B_TIME, size=N)
#     Y_samples = np.random.normal(loc=B_COST, scale=std_B_COST, size=N)
#     # Ensure no division by zero (filter out small Y values)
#     Y_samples = np.where(Y_samples == 0, np.finfo(float).eps, Y_samples)
#     Z_samples = X_samples / Y_samples
#     # Compute mean and standard deviation
#     std_of_vot = np.std(Z_samples)
#     vot_no_income = B_TIME / B_COST
#     vot_factor = (1 + config['vot_std'] * std_of_vot / vot_no_income)
#     print(f'VOT Factor: {vot_factor}')
#     print(f'VOT income prop: {vot_factor * B_TIME / B_COST}')
#
#     data['vot_i_long_term'] =0.5 * ((data['vot_i'] * vot_factor * np.power(1+config['annual_income_growth_rate'], data['year_until_change_location_or_retire'])) + data['vot_i']) # assume linear increase, mean will be 1/2(begin+end)
#
#     data['commute_benefit'] = (data['c_ii'] - data['c_ij'] + data['vot_i_long_term'] * (data['t_ii'] - data['t_ij'])) * data['year_until_change_location_or_retire'] * 2 * 250 #250 workdays per year, 2 trips per day
#     data['residential_benefit'] = (data['inferred_v_ij'] - data['inferred_v_ii']) * config['residential_benefit_weight']
#
#     data['w_ij'] = data['commute_benefit'] + data['residential_benefit'] - data['r_ij']
#
#     data['budget_i'] = config['free_income_prop'] * data['income_after_tax'] * 12 * config['budget_income_years']
#     ####
#
#
#
#     ### save_to_csv
#     data_used = data.loc[data['w_ij']>=0].copy()
#     data_used['v_ij_change_percent'] = (data_used['inferred_v_ij'] - data_used['inferred_v_ii']) / data_used['inferred_v_ii']
#     data_used = data_used.loc[data_used['v_ij_change_percent'] * config['residential_benefit_weight'] >= config['v_ij_change']]
#     data_used = data_used.loc[data_used['transaction_cost'] + data_used['price_diff'] < data_used['budget_i']]
#     print(f'remaining edge prop: {len(data_used) / ori_len }')
#
#     saved_col = ['commute_benefit', 'residential_benefit','distance',
#                  'r_ij','t_ii','t_ij','c_ii','c_ij', 'inferred_v_ii', 'inferred_v_ij', 'inferred_p_i', 'inferred_p_j', 'year_until_change_location_or_retire']
#     prob_col = [key for key in data_used.columns if 'prob_' in key]
#     saved_col += prob_col
#
#     pollutant_col = ['CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij', 'SO2_ij', 'PM25_ij']
#     data_used['CO2_ij'] = 0
#     data_used['NOx_ij'] = 0
#     data_used['VOC_ij'] = 0
#     data_used['CO_ij'] = 0
#     data_used['SO2_ij'] = 0
#     data_used['PM25_ij'] = 0
#
#
#     ev_ratio = 0.10
#     for prob in prob_col:
#         if 'car' in prob:
#             data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO2'] * (1-ev_ratio))
#             data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['NOx'] * (1-ev_ratio))
#             data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['VOC'] * (1-ev_ratio))
#             data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO'] * (1-ev_ratio))
#             data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['SO2'] * (1-ev_ratio))
#             data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['PM25'] * (1-ev_ratio))
#         elif 'pt' in prob:
#             data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO2']
#             data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['NOx']
#             data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['VOC']
#             data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO']
#             data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['SO2']
#             data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['PM25']
#
#     saved_col += pollutant_col
#     for key in saved_col:
#         data_used[key] = np.round(data_used[key],4)
#
#     # weighted avg emissions
#
#     data_used[['home_ID','work_ID']+saved_col].to_csv(f'../Beijing/final_pair_for_matching_{config["scenario_name"]}.csv',index=False)
#     return all_data
#
#
#
# def get_sg_ind_config(scenario_name):
#
#
#     config={
#         'scenario_name': scenario_name,
#         'avg_tenure_at_one_work': 5,
#         'free_income_prop': _constants.free_income_prop['Singapore'],
#         'annual_income_growth_rate': 0.022,
#         'transaction_cost_percent': _constants.transaction_cost_percent['Singapore'],
#         'v_ij_change': -0.10,
#         'vot_std': 0.0,
#         'avg_house_increase_per_year': np.power(1 + 1.1668, 1/15) - 1, #
#         'house_price_and_increase_rate': 'Direct', # Inverse
#         'v_ij_col': 'inferred_v_ij',
#         'future_gain_weight': 1.0,
#         'transaction_cost_weight': 1.0,
#         'residential_benefit_weight': 1.0,
#         'budget_income_years': 1,
#     }
#
#
#     if scenario_name == '1_0_ideal':
#         config['future_gain_weight'] = 0.0
#         config['transaction_cost_weight'] = 0.0
#         config['residential_benefit_weight'] = 0.0
#
#     if scenario_name == '2_0_in_between':
#         config['future_gain_weight'] = 0.0
#
#     if scenario_name == '2_1_in_between_high_vij':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_col'] = 'inferred_v_ij_high'
#
#     if scenario_name == '2_2_in_between_low_vij':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_col'] = 'inferred_v_ij_low'
#
#     if scenario_name == '2_3_in_between_high_vot':
#         config['future_gain_weight'] = 0.0
#         config['vot_std'] = 0.5
#
#     if scenario_name == '2_4_in_between_low_vot':
#         config['future_gain_weight'] = 0.0
#         config['vot_std'] = -0.5
#
#     if scenario_name == '2_5_in_between_vij_20_percent':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_change'] = -0.20
#
#     if scenario_name == '2_6_in_between_vij_30_percent':
#         config['future_gain_weight'] = 0.0
#         config['v_ij_change'] = -0.30
#
#     if scenario_name == '3_0_realistic':
#         pass
#
#     if scenario_name == '3_1_inverse_future_gain':
#         config['house_price_and_increase_rate'] = 'Inverse'
#
#     if scenario_name == '3_2_time_span_3_years':
#         config['avg_tenure_at_one_work'] = 3
#
#     if scenario_name == '3_3_time_span_8_years':
#         config['avg_tenure_at_one_work'] = 8
#
#     if scenario_name == '3_4_time_span_10_years':
#         config['avg_tenure_at_one_work'] = 10
#
#     if 'random' in scenario_name:
#         sample_percent = int(scenario_name.split('random_')[1].split('_percent')[0])
#         config['sample_percent'] = sample_percent
#         config['sample_seed'] = int(scenario_name.split('seed_')[1])
#
#     return config
#
#
#
# def generate_edges_for_sg_ind(config, all_data, skip_existing=False):
#     if skip_existing and os.path.exists(f'../Singapore_ind/final_pair_for_matching_{config["scenario_name"]}.parquet'):
#         print(f'{config["scenario_name"]} exists, skip...')
#         return all_data
#
#     s_time = time.time()
#     if 'od_matrix' in all_data:
#         data = all_data['od_matrix'].copy()
#     else:
#         data = pd.read_parquet('../Singapore_ind/OD_matrix_with_info.parquet')
#         all_data['od_matrix'] = data.copy()
#     print(f"load od matrix time: {time.time() - s_time}")
#
#     if 'sample_percent' in config:
#         all_ids = np.sort(pd.unique(data['work_ID'])) #note work ID = home ID for single ind case
#         N = len(all_ids)
#         # Set the random seed for reproducibility
#         # Calculate the number of samples to draw (10% of the array)
#         num_samples = int(config['sample_percent'] / 100 * N)
#         np.random.seed(config['sample_seed'])
#         sampled_indices = np.random.choice(N, num_samples, replace=False)
#         sampled_values = all_ids[sampled_indices]
#         data = data.merge(pd.DataFrame({'work_ID': sampled_values}), on =['work_ID'])
#         data = data.merge(pd.DataFrame({'home_ID': sampled_values}), on =['home_ID'])
#
#     if 'v_ij' in all_data:
#         house_value = all_data['v_ij'].copy()
#     else:
#         house_value = pd.read_parquet('../Singapore_ind/sg_ind_hpm_paired_results_vij.parquet')
#         all_data['v_ij'] = house_value.copy()
#
#     house_value['inferred_v_ij'] = house_value[config['v_ij_col']].drop(columns=['inferred_v_ij_high','inferred_v_ij_low'])
#
#     if 'p_i' in all_data:
#         price_value = all_data['p_i'].copy()
#     else:
#         price_value = pd.read_csv('../Singapore_ind/sg_ind_hpm_paired_results_pi.csv')
#         all_data['p_i'] = price_value.copy()
#
#     data = data.merge(house_value,on=['home_ID','work_ID'], how='left')
#
#     max_house_price = max(price_value['inferred_pi'])
#     min_house_price = min(price_value['inferred_pi'])
#     mean_house_price = np.mean(price_value['inferred_pi'])
#     min_change_rate = config['avg_house_increase_per_year'] * 0.7
#     if config['house_price_and_increase_rate'].lower() == 'direct':
#         price_value['price_change_rate'] = (price_value['inferred_pi'] - min_house_price) / (mean_house_price - min_house_price) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
#     else:
#         # inverse
#         price_value['price_change_rate'] = (max_house_price - price_value['inferred_pi']) / (
#                     max_house_price - mean_house_price) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
#     print(f"rate input {config['avg_house_increase_per_year']}, rate_cal {np.mean(price_value['price_change_rate'])}")
#
#     data = data.merge(price_value.rename(columns={'inferred_pi':'inferred_p_j','price_change_rate':'price_change_rate_j','DwellingType':'build_type_j'}), on=['home_ID'], how='left')
#
#     # data['age'].hist()
#     # plt.show()
#     data['inferred_v_ij'].fillna(np.mean(data['inferred_v_ij']),inplace=True)
#     data['inferred_p_j'].fillna(np.mean(data['inferred_p_j']), inplace=True)
#     data_status_quo = data.loc[data['work_ID']==data['home_ID']].rename(columns={'t_ij':'t_ii', 'c_ij':'c_ii',
#                                                                                  'inferred_v_ij':'inferred_v_ii','inferred_p_j':'inferred_p_i',
#                                                                                  'price_change_rate_j': 'price_change_rate_i','build_type_j':'build_type_i'})
#     data = data.merge(data_status_quo[['work_ID','t_ii','c_ii','inferred_p_i','inferred_v_ii','price_change_rate_i','build_type_i']], on=['work_ID'])
#
#     ############ first filter: same build type
#     ori_len = len(data)
#     if '1_0_ideal' not in config['scenario_name']:
#         data = data.loc[data['build_type_i']==data['build_type_j']]
#         print(f'num samples check point 4: {len(pd.unique(data["work_ID"]))}')
#         print(f'remaining_prop after same build type: {len(data) / ori_len}')
#
#     ###############
#
#
#     data['price_diff'] = (data['inferred_v_ij'] - data['inferred_v_ii']) * config['residential_benefit_weight']
#     data['year_until_change_location_or_retire'] = config['avg_tenure_at_one_work']# np.minimum(np.maximum(config['retire_age'] - data['age'], 0), config['avg_tenure_at_one_work']) # assuming people stay there for 15 years
#     # test
#     # data['year_until_retirement'] = 10
#     print(f"avg year staying: {np.mean(data['year_until_change_location_or_retire'])}")
#
#
#     data['future_gain_j'] = data['inferred_v_ij'] * (np.power(1+data['price_change_rate_j'], data['year_until_change_location_or_retire']) - 1.0)
#     data['future_gain_i'] = data['inferred_v_ii'] * (np.power(1+data['price_change_rate_i'], data['year_until_change_location_or_retire']) - 1.0)
#
#     # relocation cost = price diff + transaction fee + future gain
#     data['transaction_cost'] = data['inferred_p_j'] * (data['home_ID']!=data['work_ID']) * config['transaction_cost_percent'] * config['transaction_cost_weight'] #
#     # need to pay for the price diff
#     data['r_ij'] = data['transaction_cost']
#     data['future_gain_loss'] = (data['future_gain_i'] - data['future_gain_j']) * config['future_gain_weight']
#     data['r_ij'] += data['future_gain_loss']
#
#     para = pd.read_csv('regression_res/dcm_mode_choice_sg_ind.csv', index_col=0)
#     B_TIME = para.loc[f'B_TIME']['Value']
#     B_COST = para.loc[f'B_COST']['Value']
#     std_B_TIME = abs(para.loc[f'B_COST']['Std err'])
#     std_B_COST = abs(para.loc[f'B_TIME']['Std err'])
#     ## approximate vot mean and std
#     N = 10000
#     np.random.seed(100)
#     X_samples = np.random.normal(loc=B_TIME, scale=std_B_TIME, size=N)
#     Y_samples = np.random.normal(loc=B_COST, scale=std_B_COST, size=N)
#     # Ensure no division by zero (filter out small Y values)
#     Y_samples = np.where(Y_samples == 0, np.finfo(float).eps, Y_samples)
#     Z_samples = X_samples / Y_samples
#     # Compute mean and standard deviation
#     std_of_vot = np.std(Z_samples)
#     vot_no_income = B_TIME / B_COST
#     vot_factor = (1 + config['vot_std'] * std_of_vot / vot_no_income)
#     print(f'VOT Factor: {vot_factor}')
#     print(f'VOT income prop: {vot_factor * B_TIME / B_COST}')
#
#     data['vot_i_long_term'] =0.5 * ((data['vot_i'] * vot_factor * np.power(1+config['annual_income_growth_rate'], data['year_until_change_location_or_retire'])) + data['vot_i']) # assume linear increase, mean will be 1/2(begin+end)
#
#     data['commute_benefit'] = (data['c_ii'] - data['c_ij'] + data['vot_i_long_term'] * (data['t_ii'] - data['t_ij'])) * data['year_until_change_location_or_retire'] * 2 * 250 #250 workdays per year, 2 trips per day
#     data['residential_benefit'] = (data['inferred_v_ij'] - data['inferred_v_ii']) * config['residential_benefit_weight']
#
#     data['w_ij'] = data['commute_benefit'] + data['residential_benefit'] - data['r_ij']
#
#     data['budget_i'] = config['free_income_prop'] * data['income_after_tax'] * 12 * config['budget_income_years']
#     ####
#
#
#
#     ### save_to_csv
#     data_used = data.loc[data['w_ij']>=0].copy()
#     data_used['v_ij_change_percent'] = (data_used['inferred_v_ij'] - data_used['inferred_v_ii']) / data_used['inferred_v_ii']
#     data_used = data_used.loc[data_used['v_ij_change_percent']*config['residential_benefit_weight'] >= config['v_ij_change']]
#     data_used = data_used.loc[data_used['transaction_cost'] + data_used['price_diff'] < data_used['budget_i']]
#     print(f'remaining edge prop: {len(data_used) / ori_len}')
#
#     saved_col = ['commute_benefit', 'residential_benefit','distance',
#                  'r_ij','t_ii','t_ij','c_ii','c_ij', 'inferred_v_ii', 'inferred_v_ij', 'inferred_p_i', 'inferred_p_j', 'year_until_change_location_or_retire']
#     prob_col = [key for key in data_used.columns if 'prob_' in key]
#     saved_col += prob_col
#
#     pollutant_col = ['CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij', 'SO2_ij', 'PM25_ij']
#     data_used['CO2_ij'] = 0
#     data_used['NOx_ij'] = 0
#     data_used['VOC_ij'] = 0
#     data_used['CO_ij'] = 0
#     data_used['SO2_ij'] = 0
#     data_used['PM25_ij'] = 0
#
#
#
#     ev_ratio = 0.027
#     for prob in prob_col:
#         if 'car' in prob:
#             data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO2'] * (1-ev_ratio))
#             data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['NOx'] * (1-ev_ratio))
#             data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['VOC'] * (1-ev_ratio))
#             data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO'] * (1-ev_ratio))
#             data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['SO2'] * (1-ev_ratio))
#             data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['PM25'] * (1-ev_ratio))
#         elif 'pt' in prob:
#             data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO2']
#             data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['NOx']
#             data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['VOC']
#             data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO']
#             data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['SO2']
#             data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['PM25']
#
#     saved_col += pollutant_col
#     for key in saved_col:
#         data_used[key] = np.round(data_used[key],4)
#
#     # weighted avg emissions
#
#     data_used[['home_ID','work_ID']+saved_col].to_parquet(f'../Singapore_ind/final_pair_for_matching_{config["scenario_name"]}.parquet',index=False)
#     return all_data
#
#
#
#
#
#





def get_all_config(scenario_name, city_name):
    config = {
        'scenario_name': scenario_name,
        'avg_tenure_at_one_work': 5,
        'free_income_prop': _constants.free_income_prop[city_name],
        'annual_income_growth_rate': _constants.annual_income_growth_rate[city_name],
        'transaction_cost_percent': _constants.transaction_cost_percent[city_name],
        'v_ij_change': -0.10,
        'vot_std': 0.0,
        'ev_ratio': _constants.ev_ratio[city_name],
        'avg_house_increase_per_year': _constants.avg_house_increase_per_year[city_name],
        'house_price_and_increase_rate': 'Direct',  # Inverse
        'v_ij_col': 'inferred_v_ij',
        'future_gain_weight': 1.0,
        'residential_benefit_weight': 1.0,
        'budget_income_years': 2,
        'utility_constraints': True,
        'v_ij_constraints': True,
        'same_build_type': True,
        'similar_house_attributes': False
    }
    if scenario_name == '1_0_ideal':
        config['residential_benefit_weight'] = 0.0
        config['future_gain_weight'] = 0.0
        config['same_build_type'] = False
        config['v_ij_constraints'] = False

    if scenario_name == '2_0_in_between':
        config['future_gain_weight'] = 0.0

    if scenario_name == '2_1_in_between_high_vij':
        config['future_gain_weight'] = 0.0
        config['v_ij_col'] = 'inferred_v_ij_high'

    if scenario_name == '2_2_in_between_low_vij':
        config['future_gain_weight'] = 0.0
        config['v_ij_col'] = 'inferred_v_ij_low'

    if scenario_name == '2_3_in_between_high_vot':
        config['future_gain_weight'] = 0.0
        config['vot_std'] = _constants.vot_std[city_name]

    if scenario_name == '2_4_in_between_low_vot':
        config['future_gain_weight'] = 0.0
        config['vot_std'] = -(_constants.vot_std[city_name])

    if scenario_name == '2_5_in_between_vij_20_percent':
        config['future_gain_weight'] = 0.0
        config['v_ij_change'] = -0.20

    if scenario_name == '2_6_in_between_vij_30_percent':
        config['future_gain_weight'] = 0.0
        config['v_ij_change'] = -0.30

    if scenario_name == '3_0_realistic':
        pass

    if scenario_name == '3_1_inverse_future_gain':
        config['house_price_and_increase_rate'] = 'Inverse'

    if scenario_name == '3_2_time_span_3_years':
        config['avg_tenure_at_one_work'] = 3

    if scenario_name == '3_3_time_span_8_years':
        config['avg_tenure_at_one_work'] = 8

    if scenario_name == '3_4_time_span_10_years':
        config['avg_tenure_at_one_work'] = 10

    if scenario_name == '5_1A_similar_attributes':
        config['attributes_list'] = ['ave_income', 'dis_to_cbd', 'dis_to_subway']
        config['similar_house_attributes'] = True
        config['house_attributes_percent'] = 0.3

    if scenario_name == '5_1B_similar_attributes':
        config['attributes_list'] = ['ave_income', 'dis_to_cbd', 'num_poi']
        config['similar_house_attributes'] = True
        config['house_attributes_percent'] = 0.3

    if scenario_name == '5_1C_similar_attributes':
        config['attributes_list'] = ['ave_income', 'dis_to_cbd', 'dis_to_subway']
        config['similar_house_attributes'] = True
        config['house_attributes_percent'] = 0.1

    if scenario_name == '5_1D_similar_attributes':
        config['attributes_list'] = ['ave_income', 'dis_to_cbd', 'num_poi']
        config['similar_house_attributes'] = True
        config['house_attributes_percent'] = 0.1

    if scenario_name == '5_2A_similar_attributes':
        config['attributes_list'] = ['ave_income', 'dis_to_cbd', 'dis_to_subway']
        config['similar_house_attributes'] = True
        config['v_ij_constraints'] = False
        config['house_attributes_percent'] = 0.3

    if scenario_name == '5_2B_similar_attributes':
        config['attributes_list'] = ['ave_income', 'dis_to_cbd', 'num_poi']
        config['similar_house_attributes'] = True
        config['v_ij_constraints'] = False
        config['house_attributes_percent'] = 0.3

    if scenario_name == '5_2C_similar_attributes':
        config['attributes_list'] = ['ave_income', 'dis_to_cbd', 'dis_to_subway']
        config['similar_house_attributes'] = True
        config['v_ij_constraints'] = False
        config['house_attributes_percent'] = 0.1

    if scenario_name == '5_2D_similar_attributes':
        config['attributes_list'] = ['ave_income', 'dis_to_cbd', 'num_poi']
        config['similar_house_attributes'] = True
        config['v_ij_constraints'] = False
        config['house_attributes_percent'] = 0.1
    # if scenario_name == '5_2_similar_attributes_wo_vij':
    #     config['similar_house_attributes'] = True
    #     config['v_ij_constraints'] = False
    #     config['house_attributes_percent'] = 0.3


    if 'random' in scenario_name:
        sample_percent = int(scenario_name.split('random_')[1].split('_percent')[0])
        config['sample_percent'] = sample_percent
        config['sample_seed'] = int(scenario_name.split('seed_')[1])

    return config




def generate_edges_for_all(config, city_folder, model_res_name, all_data, skip_existing=False):
    if skip_existing and os.path.exists(f'../{city_folder}/final_pair_for_matching_{config["scenario_name"]}.csv'):
        print(f'{config["scenario_name"]} exists, skip...')
        return all_data


    if city_folder == 'Singapore':
        ind_id = 'HP_ID'
    else:
        ind_id = 'work_ID'
    s_time = time.time()
    if 'od_matrix' in all_data:
        data = all_data['od_matrix'].copy()
    else:
        if os.path.exists(f'../{city_folder}/OD_matrix_with_info.parquet'):
            data = pd.read_parquet(f'../{city_folder}/OD_matrix_with_info.parquet')
            all_data['od_matrix'] = data.copy()
        else:
            data = pd.read_csv(f'../{city_folder}/OD_matrix_with_info.csv')
            all_data['od_matrix'] = data.copy()

    print(f"load od matrix time: {time.time() - s_time}")

    if 'sample_percent' in config:
        if city_folder == 'Singapore':
            print('sample percent not supported for HH case')
            exit()
        all_ids = np.sort(pd.unique(data['work_ID'])) #note work ID = home ID for single ind case
        N = len(all_ids)
        # Set the random seed for reproducibility
        # Calculate the number of samples to draw (10% of the array)
        num_samples = int(config['sample_percent'] / 100 * N)
        np.random.seed(config['sample_seed'])
        sampled_indices = np.random.choice(N, num_samples, replace=False)
        sampled_values = all_ids[sampled_indices]
        data = data.merge(pd.DataFrame({'work_ID': sampled_values}), on =['work_ID'])
        data = data.merge(pd.DataFrame({'home_ID': sampled_values}), on =['home_ID'])

    print(f'num samples check point 1: {len(pd.unique(data[ind_id]))}')

    if 'v_ij' in all_data:
        house_value = all_data['v_ij'].copy()
    else:
        house_value = pd.read_parquet(f'../{city_folder}/{model_res_name}_hpm_paired_results_vij.parquet')
        all_data['v_ij'] = house_value.copy()

    house_value['inferred_v_ij'] = house_value[config['v_ij_col']].drop(columns=['inferred_v_ij_high','inferred_v_ij_low'])

    if 'p_i' in all_data:
        price_value = all_data['p_i'].copy()
    else:
        price_value = pd.read_csv(f'../{city_folder}/{model_res_name}_hpm_paired_results_pi.csv')
        all_data['p_i'] = price_value.copy()

    price_value['build_type'] = pd.factorize(price_value['build_type'])[0]  # map to int to save memory

    if city_folder == 'Singapore':
        # houseold
        data = data.drop(columns=['work_ID'])
        data = data.rename(columns={'HHID': 'HHID_original'})
        data = house_value.merge(data,
                                 on=['HP_ID', 'home_ID'])  # HP_ID is current individual, home ID is potential house
    else:
        data = data.merge(house_value,on=['home_ID','work_ID'], how='left')

    print(f'num samples check point 2: {len(pd.unique(data["work_ID"]))}')

    max_value = max(price_value['dis_to_cbd'])
    min_value = min(price_value['dis_to_cbd'])
    mean_value = np.mean(price_value['dis_to_cbd'])
    min_change_rate = config['avg_house_increase_per_year'] * 0.7
    if config['house_price_and_increase_rate'].lower() == 'direct':
        # closer to cbd high increase
        price_value['price_change_rate'] = (max_value - price_value['dis_to_cbd']) / (
                    max_value - mean_value) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
    else:
        # inverse
        # further high increase
        price_value['price_change_rate'] = (price_value['dis_to_cbd'] - min_value) / (
                mean_value - min_value) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
    print(f"rate input {config['avg_house_increase_per_year']}, rate_cal {np.mean(price_value['price_change_rate'])}")

    price_value = price_value.rename(columns={'inferred_pi':'inferred_p_j'})
    add_j_col = ['price_change_rate', 'build_type','num_poi','ave_income','dis_to_subway','dis_to_lake','dis_to_cbd']
    for col in add_j_col:
        price_value = price_value.rename(columns={col:f'{col}_j'})

    if city_folder == 'Singapore':
        data = data.merge(price_value, on=['HHID'], how='left')
    else:
        data = data.merge(price_value, on=['home_ID'], how='left')
    print('finish join price_value')
    # data['age'].hist()
    # plt.show()
    data['inferred_v_ij'].fillna(np.mean(data['inferred_v_ij']),inplace=True)
    data['inferred_p_j'].fillna(np.mean(data['inferred_p_j']), inplace=True)

    if city_folder == 'Singapore':
        data_status_quo = data.loc[data['HHID_original'] == data['HHID']].copy()
    else:
        data_status_quo = data.loc[data['work_ID'] == data['home_ID']].copy()

    data_status_quo = data_status_quo.rename(columns={'t_ij':'t_ii',
                                                      'c_ij':'c_ii',
                                                      'inferred_v_ij':'inferred_v_ii',
                                                      'inferred_p_j':'inferred_p_i'})

    used_col_status_quo = ['t_ii', 'c_ii', 'inferred_p_i', 'inferred_v_ii']

    used_col_status_quo = [ind_id] + used_col_status_quo
    for col in add_j_col:
        data_status_quo = data_status_quo.rename(columns={f'{col}_j': f'{col}_i'})
        used_col_status_quo.append(f'{col}_i')

    data = data.merge(data_status_quo[used_col_status_quo], on=[ind_id])

    print(f'num samples check point 3: {len(pd.unique(data[ind_id]))}')

    ############ first filter: same build type
    ori_len = len(data)
    if config['same_build_type']:
        data = data.loc[data['build_type_i']==data['build_type_j']]
        print(f'num samples check point 4: {len(pd.unique(data[ind_id]))}')
        print(f'remaining_prop after same build type: {len(data) / ori_len}')

    if config['similar_house_attributes']:
        attributes = config['attributes_list']#, 'dis_to_subway','num_poi']
        for att in attributes:
            # avg_att = np.mean(data_status_quo[f'{att}_i'])
            # data['temp_diff'] = np.abs(data[f'{att}_i'] - data[f'{att}_j'])
            # data = data.loc[data['temp_diff'] <= avg_att * config['house_attributes_percent']]
            data = data.loc[
                np.abs(data[f'{att}_i'] - data[f'{att}_j']) <= data[f'{att}_i'] * config['house_attributes_percent']]
            print(f'remaining_prop after similar {att}: {len(data) / ori_len}')
            # data = data.loc[np.abs(data[f'{att}_i'] - data[f'{att}_j']) <= data[f'{att}_i']*config['house_attributes_percent']]
        print(f'num samples check point 5: {len(pd.unique(data[ind_id]))}')
        print(f'remaining_prop after similar_house_attributes: {len(data) / ori_len}')

    #####################

    # data['price_diff'] = (data['inferred_p_j'] - data['inferred_p_i']) * config['residential_benefit_weight']
    data['price_diff'] = (data['inferred_v_ij'] - data['inferred_v_ii'])
    data['year_until_change_location_or_retire'] = config['avg_tenure_at_one_work']# np.minimum(np.maximum(config['retire_age'] - data['age'], 0), config['avg_tenure_at_one_work']) # assuming people stay there for 15 years
    # test
    # data['year_until_retirement'] = 10
    print(f"avg year staying: {np.mean(data['year_until_change_location_or_retire'])}")


    data['future_gain_j'] = data['inferred_v_ij'] * (np.power(1+data['price_change_rate_j'], data['year_until_change_location_or_retire']) - 1.0)
    data['future_gain_i'] = data['inferred_v_ii'] * (np.power(1+data['price_change_rate_i'], data['year_until_change_location_or_retire']) - 1.0)

    # relocation cost = price diff + transaction fee + future gain
    if city_folder == 'Singapore':
        data['transaction_cost'] = data['inferred_v_ij'] * (data['HHID'] != data['HHID_original']) * config[
            'transaction_cost_percent']  #
    else:
        data['transaction_cost'] = data['inferred_v_ij'] * (data['home_ID']!=data['work_ID']) * config['transaction_cost_percent']
    data['r_ij'] = data['transaction_cost']
    # need to pay for the price diff
    data['future_gain_diff'] = (data['future_gain_j'] - data['future_gain_i'])


    para = pd.read_csv(f'regression_res/dcm_mode_choice_{model_res_name}.csv', index_col=0)
    B_TIME = para.loc[f'B_TIME']['Value']
    B_COST = para.loc[f'B_COST']['Value']
    std_B_TIME = abs(para.loc[f'B_COST']['Std err'])
    std_B_COST = abs(para.loc[f'B_TIME']['Std err'])
    ## approximate vot mean and std
    N = 10000
    np.random.seed(100)
    X_samples = np.random.normal(loc=B_TIME, scale=std_B_TIME, size=N)
    Y_samples = np.random.normal(loc=B_COST, scale=std_B_COST, size=N)
    # Ensure no division by zero (filter out small Y values)
    Y_samples = np.where(Y_samples == 0, np.finfo(float).eps, Y_samples)
    Z_samples = X_samples / Y_samples
    # Compute mean and standard deviation
    std_of_vot = np.std(Z_samples)
    vot_no_income = B_TIME / B_COST
    vot_factor = (1 + config['vot_std'] * std_of_vot / vot_no_income)
    print(f'VOT Factor: {vot_factor}')
    print(f'* VOT income prop: {vot_factor * B_TIME / B_COST} *')

    data['vot_i_long_term'] = 0.5 * (
            (data['vot_i'] * vot_factor * np.power(1+config['annual_income_growth_rate'],
                                                   data['year_until_change_location_or_retire'])) + data['vot_i']) # assume linear increase, mean will be 1/2(begin+end)

    data['commute_benefit'] = (data['c_ii'] - data['c_ij'] + data['vot_i_long_term'] * (
            data['t_ii'] - data['t_ij'])) * data['year_until_change_location_or_retire'] * 2 * _constants.num_work_days_per_year #250 workdays per year, 2 trips per day
    data['residential_benefit'] = (data['inferred_v_ij'] - data['inferred_v_ii']) + data['future_gain_diff']
    num_month_per_year = 12  # 12 month per year
    data['budget_i'] = config['free_income_prop'] * data['income_after_tax'] * num_month_per_year * config['budget_income_years']
    ####



    ### save_to_csv
    if config['utility_constraints']:
        data = data.loc[
            data['residential_benefit'] * config['residential_benefit_weight']
            + data['commute_benefit']
            + data['future_gain_diff'] * config['future_gain_weight']
            >= data['r_ij']
        ]
        print(f'remaining edge prop after utility const: {len(data) / ori_len}')

    if config['v_ij_constraints']:
        data = data.loc[(data['inferred_v_ij'] - data['inferred_v_ii']) >= config['v_ij_change'] * data['inferred_v_ii']]
        print(f'remaining edge prop after v_ij not too low: {len(data) / ori_len}')
        data = data.loc[data['r_ij'] + data['price_diff'] < data['budget_i']]
        print(f'remaining edge prop after v_ij budget const: {len(data) / ori_len}')

    print(f'remaining edge prop: {len(data) / ori_len }')
    print(f'num samples check point 5: {len(pd.unique(data[ind_id]))}')


    if city_folder == 'Singapore':
        ##### delete those not all ind are available
        all_ind = pd.read_csv(f'../{city_folder}/mode_choice_sample_updated.csv')[
            ['HHID', 'HP_ID', 'home_ID', 'work_ID', 'Age']]
        hh_ind_num = all_ind[['HHID', 'HP_ID']].groupby(['HHID'])['HP_ID'].count().reset_index().rename(
            columns={'HP_ID': 'num_inds_old', 'HHID': 'HHID_original'})
        data['num_inds'] = data[['HHID_original', 'HHID', 'HP_ID']].groupby(['HHID_original', 'HHID'])[
            'HP_ID'].transform('count')
        data = data.merge(hh_ind_num[['HHID_original', 'num_inds_old']], on=['HHID_original'])
        data = data.loc[data['num_inds'] == data['num_inds_old']]
        print(f'remaining edge prop after family same house: {len(data) / ori_len}')

    saved_col = ['commute_benefit', 'residential_benefit', 'future_gain_diff','distance',
                 'r_ij','t_ii','t_ij','c_ii','c_ij', 'inferred_v_ii', 'inferred_v_ij', 'inferred_p_i', 'inferred_p_j', 'year_until_change_location_or_retire']
    prob_col = [key for key in data.columns if 'prob_' in key]
    saved_col += prob_col

    data_used = data.copy()
    del data

    pollutant_col = ['CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij', 'SO2_ij', 'PM25_ij']
    data_used['CO2_ij'] = 0
    data_used['NOx_ij'] = 0
    data_used['VOC_ij'] = 0
    data_used['CO_ij'] = 0
    data_used['SO2_ij'] = 0
    data_used['PM25_ij'] = 0

    ev_ratio = config['ev_ratio']
    for prob in prob_col:
        if 'car' in prob:
            data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO2'] * (1-ev_ratio))
            data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['NOx'] * (1-ev_ratio))
            data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['VOC'] * (1-ev_ratio))
            data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO'] * (1-ev_ratio))
            data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['SO2'] * (1-ev_ratio))
            data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['PM25'] * (1-ev_ratio))
        elif 'pt' in prob:
            data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO2']
            data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['NOx']
            data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['VOC']
            data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO']
            data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['SO2']
            data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['PM25']

    saved_col += pollutant_col
    for key in saved_col:
        data_used[key] = np.round(data_used[key],4)

    # weighted avg emissions

    if city_folder == 'Singapore':
        data_used[['HP_ID', 'HHID_original', 'HHID'] + saved_col].to_parquet(
            f'../Singapore/final_pair_for_matching_{config["scenario_name"]}.parquet', index=False)
    else:
        data_used[['home_ID','work_ID']+saved_col].to_parquet(f'../{city_folder}/final_pair_for_matching_{config["scenario_name"]}.parquet',index=False)
    return all_data



# def get_sg_HH_config(scenario):
#     config={
#         'scenario_name': scenario,
#         'avg_tenure_at_one_work': 5,
#         'free_income_prop': _constants.free_income_prop['Singapore'],
#         'annual_income_growth_rate': 0.022,
#         'transaction_cost_percent': _constants.transaction_cost_percent['Singapore'],
#         'v_ij_change': -0.10,
#         'vot_std': 0.0,
#         'avg_house_increase_per_year': np.power(1 + 1.1668, 1/15) - 1, # 15 years increase 116.68%
#         'house_price_and_increase_rate': 'Direct', # Inverse
#         'residential_benefit_weight': 1.0,
#         'future_gain_weight': 1.0,
#         'v_ij_constraints': True,
#         'future_gain_weight': 1.0,
#         'residential_benefit_weight': 1.0,
#         'budget_income_years': 2,
#         'utility_constraints': True,
#         'v_ij_constraints': True,
#         'same_build_type': True,
#         'similar_house_attributes': False
#     }
#     if scenario == '1_0_ideal':
#         config['residential_benefit_weight'] = 0.0
#         config['future_gain_weight'] = 0.0
#         config['v_ij_constraints'] = False
#
#     return config

#
# def generate_edges_for_sg_HH(config):
#     s_time = time.time()
#     data = pd.read_parquet('../Singapore/OD_matrix_with_info.parquet')
#     print(f"load od matrix time: {time.time() - s_time}")
#     s_time = time.time()
#     house_value = pd.read_parquet('../Singapore/sg_hpm_paired_results_vij.parquet')
#     print(f"load vij time: {time.time() - s_time}")
#     price_value = pd.read_csv('../Singapore/sg_hpm_paired_results_pi.csv')
#
#     all_ind = pd.read_csv('../Singapore/mode_choice_sample_updated.csv')[['HHID','HP_ID','home_ID','work_ID','Age']]
#
#     # all_ind['age'] = np.mean(all_ind['Age'].str.extract(r'(\d+)-(\d+)').astype(float, errors='ignore'),axis=1)
#     # all_ind.loc[all_ind['Age']=='85 yrs & above', 'age'] = 85
#     # assert sum(all_ind['age'].isna()) == 0
#
#     s_time = time.time()
#     data = data.drop(columns=['work_ID'])
#     data = data.rename(columns={'HHID': 'HHID_original'})
#     data = house_value.merge(data,on=['HP_ID','home_ID']) #HP_ID is current individual, home ID is potential house
#     print(f"data merge vij time: {time.time() - s_time}")
#
#     ###########
#     max_value = max(price_value['dis_to_cbd'])
#     min_value = min(price_value['dis_to_cbd'])
#     mean_value = np.mean(price_value['dis_to_cbd'])
#     min_change_rate = config['avg_house_increase_per_year'] * 0.7
#     if config['house_price_and_increase_rate'].lower() == 'direct':
#         # closer to cbd high increase
#         price_value['price_change_rate'] = (max_value - price_value['dis_to_cbd']) / (
#                     max_value - mean_value) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
#     else:
#         # inverse
#         # further high increase
#         price_value['price_change_rate'] = (price_value['dis_to_cbd'] - min_value) / (
#                 mean_value - min_value) * (config['avg_house_increase_per_year'] - min_change_rate) + min_change_rate
#
#     print(f"rate input {config['avg_house_increase_per_year']}, rate_cal {np.mean(price_value['price_change_rate'])}")
#
#
#     price_value['build_type'] = pd.factorize(price_value['build_type'])[0] # map to int to save memory
#     price_value = price_value.rename(columns={
#         'inferred_pi': 'inferred_p_j', 'price_change_rate': 'price_change_rate_j', 'build_type': 'build_type_j'})[
#         ['HHID', 'inferred_p_j', 'price_change_rate_j', 'build_type_j']]
#     s_time = time.time()
#     data = data.merge(price_value, on=['HHID'], how='left')
#
#     print(f"data merge pi time: {time.time() - s_time}")
#
#     data_status_quo = data.loc[data['HHID_original']==data['HHID']].rename(columns={'t_ij':'t_ii', 'c_ij':'c_ii',
#                                                                                  'inferred_v_ij':'inferred_v_ii','inferred_p_j':'inferred_p_i',
#                                                                                  'price_change_rate_j': 'price_change_rate_i','build_type_j':'build_type_i'})
#     s_time = time.time()
#     data = data.merge(data_status_quo[['HP_ID','t_ii','c_ii','inferred_p_i','inferred_v_ii','price_change_rate_i','build_type_i']], on=['HP_ID'])
#
#     ############ first filter: same build type
#     ori_len = len(data)
#     if '1_0_ideal' not in config['scenario_name']:
#         data = data.loc[data['build_type_i']==data['build_type_j']]
#         print(f'num samples check point 4: {len(pd.unique(data["work_ID"]))}')
#         print(f'remaining_prop after same build type: {len(data) / ori_len}')
#     #####################
#
#     data['price_diff'] = data['inferred_v_ij'] - data['inferred_v_ii']
#     data['year_until_change_location_or_retire'] = config['avg_tenure_at_one_work'] # np.minimum(np.maximum(config['retire_age'] - data['age'], 0), config['avg_tenure_at_one_work']) # assuming people stay there for 15 years
#     # test
#
#     data['future_gain_j'] = data['inferred_v_ij'] * (np.power(1+data['price_change_rate_j'], data['year_until_change_location_or_retire']) - 1.0)
#     data['future_gain_i'] = data['inferred_v_ii'] * (np.power(1+data['price_change_rate_i'], data['year_until_change_location_or_retire']) - 1.0)
#
#     # relocation cost = price diff + transaction fee + future gain
#     data['transaction_cost'] = data['inferred_v_ij'] * (data['HHID']!=data['HHID_original']) * config['transaction_cost_percent']  #
#     # need to pay for the price diff
#     data['r_ij'] = data['transaction_cost']
#     data['future_gain_loss'] = data['future_gain_i'] - data['future_gain_j']
#     data['r_ij'] += data['future_gain_loss']
#     print(f'finish cal rij, {time.time() - s_time}')
#
#     para = pd.read_csv('regression_res/dcm_mode_choice_sg.csv', index_col=0)
#     B_TIME = para.loc[f'B_TIME']['Value']
#     B_COST = para.loc[f'B_COST']['Value']
#     std_B_TIME = abs(para.loc[f'B_COST']['Std err'])
#     std_B_COST = abs(para.loc[f'B_TIME']['Std err'])
#     ## approximate vot mean and std
#     N = 10000
#     np.random.seed(100)
#     X_samples = np.random.normal(loc=B_TIME, scale=std_B_TIME, size=N)
#     Y_samples = np.random.normal(loc=B_COST, scale=std_B_COST, size=N)
#     # Ensure no division by zero (filter out small Y values)
#     Y_samples = np.where(Y_samples == 0, np.finfo(float).eps, Y_samples)
#     Z_samples = X_samples / Y_samples
#     # Compute mean and standard deviation
#     std_of_vot = np.std(Z_samples)
#     vot_no_income = B_TIME / B_COST
#     vot_factor = (1 + config['vot_std'] * std_of_vot / vot_no_income)
#     print(f'VOT Factor: {vot_factor}')
#     print(f'VOT income prop: {vot_factor * B_TIME / B_COST}')
#
#     data['vot_i_long_term'] =0.5 * ((data['vot_i'] * vot_factor * np.power(1+config['annual_income_growth_rate'], data['year_until_change_location_or_retire'])) + data['vot_i']) # assume linear increase, mean will be 1/2(begin+end)
#
#     data['commute_benefit'] = (
#                                       data['c_ii'] - data['c_ij'] + data['vot_i_long_term'] * (
#                                       data['t_ii'] - data['t_ij'])
#                               ) * data['year_until_change_location_or_retire'] * 2 * _constants.num_work_days_per_year #workdays per year, 2 trips per day
#     data['residential_benefit'] = data['inferred_v_ij'] - data['inferred_v_ii']
#
#     data['w_ij'] = data['commute_benefit'] + data['residential_benefit'] - data['r_ij']
#
#     data['budget_i'] = config['free_income_prop'] * data['income_after_tax'] * 12 * config['budget_income_years']
#     ####
#
#
#
#     ### save_to_csv
#
#     data = data.loc[data['w_ij']>=0].copy()
#     data['v_ij_change_percent'] = (data['inferred_v_ij'] - data['inferred_v_ii']) / data['inferred_v_ii']
#     data = data.loc[data['v_ij_change_percent'] >= config['v_ij_change']]
#     data = data.loc[data['transaction_cost'] + data['price_diff'] < data['budget_i']]
#
#
#     ##### delete those not all ind are available
#     hh_ind_num = all_ind[['HHID','HP_ID']].groupby(['HHID'])['HP_ID'].count().reset_index().rename(columns={'HP_ID':'num_inds_old', 'HHID':'HHID_original'})
#     data['num_inds'] = data[['HHID_original', 'HHID','HP_ID']].groupby(['HHID_original', 'HHID'])['HP_ID'].transform('count')
#     data = data.merge(hh_ind_num[['HHID_original', 'num_inds_old']],on=['HHID_original'])
#     data = data.loc[data['num_inds'] == data['num_inds_old']]
#     print(f'remaining edge prop: {len(data) / ori_len }')
#     data_used = data
#     saved_col = ['commute_benefit', 'residential_benefit','distance',
#                  'r_ij','t_ii','t_ij','c_ii','c_ij', 'inferred_v_ii', 'inferred_v_ij', 'inferred_p_i', 'inferred_p_j', 'year_until_change_location_or_retire']
#     prob_col = [key for key in data.columns if 'prob_' in key]
#     saved_col += prob_col
#
#     pollutant_col = ['CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij', 'SO2_ij', 'PM25_ij']
#     data_used['CO2_ij'] = 0
#     data_used['NOx_ij'] = 0
#     data_used['VOC_ij'] = 0
#     data_used['CO_ij'] = 0
#     data_used['SO2_ij'] = 0
#     data_used['PM25_ij'] = 0
#
#     ev_ratio = _constants.ev_ratio['Singapore']
#     for prob in prob_col:
#         if 'car' in prob:
#             data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO2'] * (1-ev_ratio))
#             data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['NOx'] * (1-ev_ratio))
#             data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['VOC'] * (1-ev_ratio))
#             data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['CO'] * (1-ev_ratio))
#             data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['SO2'] * (1-ev_ratio))
#             data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * (_constants.emissions_car['PM25'] * (1-ev_ratio))
#         elif 'pt' in prob:
#             data_used['CO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO2']
#             data_used['NOx_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['NOx']
#             data_used['VOC_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['VOC']
#             data_used['CO_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['CO']
#             data_used['SO2_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['SO2']
#             data_used['PM25_ij'] += data_used[prob] * data_used['distance'] / 1000.0 * _constants.emissions_pt['PM25']
#
#     saved_col += pollutant_col
#     for key in saved_col:
#         data_used[key] = np.round(data_used[key],4)
#
#     # weighted avg emissions
#
#     data_used[['HP_ID','HHID_original', 'HHID']+saved_col].to_csv(f'../Singapore/final_pair_for_matching_{config["scenario_name"]}.csv',index=False)
#


if __name__ == '__main__':

    scenario_list = [
        '1_0_ideal',
        '2_0_in_between',
        '2_1_in_between_high_vij',
        '2_2_in_between_low_vij',
        '2_3_in_between_high_vot',
        '2_4_in_between_low_vot',
        '2_5_in_between_vij_20_percent',
        '2_6_in_between_vij_30_percent',
        '3_0_realistic',
        '3_1_inverse_future_gain',
        '3_2_time_span_3_years',
        '3_3_time_span_8_years',
        '3_4_time_span_10_years',
        '5_1A_similar_attributes',
        # '5_1B_similar_attributes',
        # '5_1C_similar_attributes',
        # '5_1D_similar_attributes',
        '5_2A_similar_attributes',
        # '5_2B_similar_attributes',
        # '5_2C_similar_attributes',
        # '5_2D_similar_attributes',
    ]
    random_percent = list(range(10,100,10))
    num_seed = 5
    random_select_scenario = []
    for percent in random_percent:
        for i in range(num_seed):
            random_select_scenario.append(f'4_0_random_{percent}_percent_seed_{i}')
    scenario_list += random_select_scenario

    skip_existing = False
    ########## Munich
    # # scenario_list =['1_0_ideal', '2_0_in_between', '3_0_realistic']
    # city_folder = 'Munich'
    # model_res_name = 'munich'
    # city_name = 'Munich'
    # all_data = {}
    # for scenario in scenario_list:
    #     print(f'#########Munich scenario {scenario}###########')
    #     config = get_all_config(scenario, city_name)
    #     all_data = generate_edges_for_all(config, city_folder, model_res_name, all_data, skip_existing=skip_existing)
    #
    #
    # # ########## Beijing
    # # scenario_list =['1_0_ideal', '2_0_in_between', '3_0_realistic']
    # city_folder = 'Beijing'
    # model_res_name = 'bj'
    # city_name = 'Beijing'
    # all_data = {}
    # for scenario in scenario_list:
    #     print(f'#########Beijing scenario {scenario}###########')
    #     config = get_all_config(scenario, city_name)
    #     all_data = generate_edges_for_all(config, city_folder, model_res_name, all_data, skip_existing=skip_existing)


    # # ########## Singapore_ind
    # scenario_list = ['1_0_ideal', '2_0_in_between', '3_0_realistic']
    # city_folder = 'Singapore_ind'
    # model_res_name = 'sg_ind'
    # city_name = 'Singapore'
    # all_data = {}
    # for scenario in scenario_list:
    #     print(f'#########Singapore_ind scenario {scenario}###########')
    #     config = get_all_config(scenario, city_name)
    #     all_data = generate_edges_for_all(config, city_folder, model_res_name, all_data, skip_existing=skip_existing)


    # ########## Singapore HH
    # #### only consider refer scenario
    scenario_list = ['1_0_ideal', '2_0_in_between', '3_0_realistic']
    city_folder = 'Singapore'
    model_res_name = 'sg'
    city_name = 'Singapore'
    all_data = {}
    for scenario in scenario_list:
        print(f'#########Singapore_HH scenario {scenario}###########')
        config = get_all_config(scenario, city_name)
        all_data = generate_edges_for_all(config, city_folder, model_res_name, all_data, skip_existing=skip_existing)

