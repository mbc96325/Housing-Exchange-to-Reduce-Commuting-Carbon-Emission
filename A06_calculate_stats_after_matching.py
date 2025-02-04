import os
import numpy as np
import pandas as pd
import time
import re

#
# def calculate_res_munich(scenario,refer_scenario_name):
#     if os.path.exists(f'../Munich/matched_res_{scenario}.csv'):
#         matched_res = pd.read_csv(f'../Munich/matched_res_{scenario}.csv')
#     else:
#         print(f'../Munich/matched_res_{scenario}.csv does not exists')
#         return pd.DataFrame()
#     if 'random' in scenario or 'optimal' in scenario:
#         pair_info = pd.read_parquet(f'../Munich/final_pair_for_matching_{refer_scenario_name}.parquet')
#     else:
#         pair_info = pd.read_parquet(f'../Munich/final_pair_for_matching_{scenario}.parquet')
#     original_info = pair_info.loc[pair_info['home_ID']==pair_info['work_ID']].copy()
#     matched_info = pair_info.merge(matched_res,on=['home_ID','work_ID'])
#
#     if 'random' in scenario or 'optimal' in scenario:
#         no_included_sample = original_info.merge(matched_info[['work_ID']], on='work_ID', how='left', indicator=True)
#         no_included_sample = no_included_sample[no_included_sample['_merge'] == 'left_only'].drop(columns=['_merge'])
#         matched_info = pd.concat([matched_info, no_included_sample])
#
#     assert len(original_info) == len(matched_info)
#
#
#     pre_cal_col = ['prop_relocation', 'prop_better_commute', 'prop_better_residential', 'prop_worse_commute', 'prop_worse_residential']
#     for key in pre_cal_col:
#         original_info[key] = 0
#     original_info['num_inds'] = len(original_info)
#     matched_info['num_inds'] = len(matched_info)
#     matched_info['prop_relocation'] = (matched_info['home_ID'] != matched_info['work_ID']).astype(int) / len(matched_info)
#     matched_info['prop_better_commute'] = (matched_info['commute_benefit'] > 0).astype(int) / len(matched_info)
#     matched_info['prop_better_residential'] = (matched_info['residential_benefit'] > 0).astype(int) / len(matched_info)
#     matched_info['prop_worse_commute'] = (matched_info['commute_benefit'] < 0).astype(int) / len(matched_info)
#     matched_info['prop_worse_residential'] = (matched_info['residential_benefit'] < 0).astype(int) / len(matched_info)
#
#     # test = matched_info.loc[(matched_info['commute_benefit'] < 0) & (matched_info['residential_benefit'] < 0)]
#     # print('both worse', len(test))
#
#     stats_col = ['distance', 'CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij',
#                  'SO2_ij', 'PM25_ij','prob_car','prob_pt','prob_bike','prob_walk'] + ['num_inds'] + pre_cal_col
#
#     compare = matched_info[['work_ID','home_ID'] + stats_col].merge(original_info[['work_ID','home_ID'] + stats_col], on =['work_ID'])
#
#     res = {'scenario': [], 'attributes':[], 'old':[], 'new':[]}
#     for key in stats_col:
#         res['scenario'].append(scenario)
#         res['attributes'].append(key)
#         if 'prob' in key or key == 'num_inds':
#             old = np.mean(compare[f'{key}_y'])
#             new = np.mean(compare[f'{key}_x'])
#         else:
#             old = sum(compare[f'{key}_y'])
#             new = sum(compare[f'{key}_x'])
#         res['old'].append(old)
#         res['new'].append(new)
#     res_df = pd.DataFrame(res)
#     # res_df['diff_percent'] = (res_df['new'] - res_df['old']) / res_df['old']
#     # res_df.to_csv(f'../Munich/final_matched_res_stats.csv',index=False)
#     return res_df



def calculate_res_sg(scenario, city_folder):
    matched_res = pd.read_csv(f'../Singapore/matched_res_{scenario}.csv')
    pair_info = pd.read_parquet(f'../Singapore/HH_paired_info_{scenario}.parquet')
    all_ind = pd.read_csv('../Singapore/mode_choice_sample_updated.csv')
    original_info = pair_info.loc[pair_info['home_ID'] == pair_info['work_ID']].copy()
    matched_info = pair_info.merge(matched_res,on=['home_ID','work_ID'])
    print('len should be same',len(original_info), len(matched_info))


    pre_cal_col = ['prop_relocation', 'prop_better_commute', 'prop_better_residential', 'prop_worse_commute', 'prop_worse_residential']
    for key in pre_cal_col:
        original_info[key] = 0
    original_info['num_households'] = len(original_info)
    matched_info['num_households'] = len(matched_info)
    original_info['num_inds'] = len(all_ind)
    matched_info['num_inds'] = len(all_ind)
    matched_info['prop_relocation'] = (matched_info['home_ID'] != matched_info['work_ID']).astype(int) / len(matched_info)
    matched_info['prop_better_commute'] = (matched_info['commute_benefit'] > 0).astype(int) / len(matched_info)
    matched_info['prop_better_residential'] = (matched_info['residential_benefit'] > 0).astype(int) / len(matched_info)
    matched_info['prop_worse_commute'] = (matched_info['commute_benefit'] < 0).astype(int) / len(matched_info)
    matched_info['prop_worse_residential'] = (matched_info['residential_benefit'] < 0).astype(int) / len(matched_info)

    # test = matched_info.loc[(matched_info['commute_benefit'] < 0) & (matched_info['residential_benefit'] < 0)]
    # print('both worse', len(test))

    stats_col = ['distance', 'CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij',
                 'SO2_ij', 'PM25_ij','prob_car','prob_pt','prob_bike','prob_walk'] + ['num_inds','num_households'] + pre_cal_col

    compare = matched_info[['work_ID','home_ID'] + stats_col].merge(original_info[['work_ID','home_ID'] + stats_col], on =['work_ID'])

    res = {'city':[], 'scenario':[], 'attributes':[], 'old':[], 'new':[]}
    for key in stats_col:
        res['city'].append(city_folder)
        res['scenario'].append(scenario)
        res['attributes'].append(key)
        if 'prob' in key or key == 'num_inds' or key == 'num_households':
            old = np.mean(compare[f'{key}_y'])
            new = np.mean(compare[f'{key}_x'])
        else:
            old = sum(compare[f'{key}_y'])
            new = sum(compare[f'{key}_x'])
        res['old'].append(old)
        res['new'].append(new)
    res_df = pd.DataFrame(res)
    res_df['diff_percent'] = (res_df['new'] - res_df['old']) / res_df['old']
    return res_df
    # res_df.to_csv(f'../Singapore/final_matched_res_stats.csv',index=False)



# def calculate_res_bj(scenario, refer_scenario_name):
#     if os.path.exists(f'../Beijing/matched_res_{scenario}.csv'):
#         matched_res = pd.read_csv(f'../Beijing/matched_res_{scenario}.csv')
#     else:
#         print(f'../Beijing/matched_res_{scenario}.csv does not exists')
#         return pd.DataFrame()
#
#     if 'random' in scenario or 'optimal' in scenario:
#         pair_info = pd.read_parquet(f'../Beijing/final_pair_for_matching_{refer_scenario_name}.parquet')
#     else:
#         pair_info = pd.read_parquet(f'../Beijing/final_pair_for_matching_{scenario}.parquet')
#     original_info = pair_info.loc[pair_info['home_ID']==pair_info['work_ID']].copy()
#     matched_info = pair_info.merge(matched_res,on=['home_ID','work_ID'])
#
#     if 'random' in scenario or 'optimal' in scenario:
#         no_included_sample = original_info.merge(matched_info[['work_ID']], on='work_ID', how='left', indicator=True)
#         no_included_sample = no_included_sample[no_included_sample['_merge'] == 'left_only'].drop(columns=['_merge'])
#         matched_info = pd.concat([matched_info, no_included_sample])
#
#     assert len(original_info) == len(matched_info)
#
#
#     pre_cal_col = ['prop_relocation', 'prop_better_commute', 'prop_better_residential', 'prop_worse_commute', 'prop_worse_residential']
#     for key in pre_cal_col:
#         original_info[key] = 0
#
#     original_info['num_inds'] = len(original_info)
#     matched_info['num_inds'] = len(matched_info)
#     matched_info['prop_relocation'] = (matched_info['home_ID'] != matched_info['work_ID']).astype(int) / len(matched_info)
#     matched_info['prop_better_commute'] = (matched_info['commute_benefit'] > 0).astype(int) / len(matched_info)
#     matched_info['prop_better_residential'] = (matched_info['residential_benefit'] > 0).astype(int) / len(matched_info)
#     matched_info['prop_worse_commute'] = (matched_info['commute_benefit'] < 0).astype(int) / len(matched_info)
#     matched_info['prop_worse_residential'] = (matched_info['residential_benefit'] < 0).astype(int) / len(matched_info)
#
#     # test = matched_info.loc[(matched_info['commute_benefit'] < 0) & (matched_info['residential_benefit'] < 0)]
#     # print('both worse', len(test))
#
#     stats_col = ['distance', 'CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij',
#                  'SO2_ij', 'PM25_ij','prob_car','prob_pt','prob_bike','prob_walk'] + ['num_inds'] + pre_cal_col
#
#     compare = matched_info[['work_ID','home_ID'] + stats_col].merge(original_info[['work_ID','home_ID'] + stats_col], on =['work_ID'])
#
#     res = {'scenario': [], 'attributes':[], 'old':[], 'new':[]}
#     for key in stats_col:
#         res['scenario'].append(scenario)
#         res['attributes'].append(key)
#         if 'prob' in key or key == 'num_inds':
#             old = np.mean(compare[f'{key}_y'])
#             new = np.mean(compare[f'{key}_x'])
#         else:
#             old = sum(compare[f'{key}_y'])
#             new = sum(compare[f'{key}_x'])
#         res['old'].append(old)
#         res['new'].append(new)
#     res_df = pd.DataFrame(res)
#     return res_df
#


#
# def calculate_res_sg_ind(scenario, refer_scenario_name):
#
#     if os.path.exists(f'../Singapore_ind/matched_res_{scenario}.csv'):
#         matched_res = pd.read_csv(f'../Singapore_ind/matched_res_{scenario}.csv')
#     else:
#         print(f'../Singapore_ind/matched_res_{scenario}.csv does not exists')
#         return pd.DataFrame()
#
#     if 'random' in scenario or 'optimal' in scenario:
#         pair_info = pd.read_parquet(f'../Singapore_ind/final_pair_for_matching_{refer_scenario_name}.parquet')
#     else:
#         pair_info = pd.read_parquet(f'../Singapore_ind/final_pair_for_matching_{scenario}.parquet')
#     original_info = pair_info.loc[pair_info['home_ID']==pair_info['work_ID']].copy()
#     matched_info = pair_info.merge(matched_res,on=['home_ID','work_ID'])
#
#     if 'random' in scenario or 'optimal' in scenario:
#         no_included_sample = original_info.merge(matched_info[['work_ID']], on='work_ID', how='left', indicator=True)
#         no_included_sample = no_included_sample[no_included_sample['_merge'] == 'left_only'].drop(columns=['_merge'])
#         matched_info = pd.concat([matched_info, no_included_sample])
#
#     assert len(original_info) == len(matched_info)
#
#
#     pre_cal_col = ['prop_relocation', 'prop_better_commute', 'prop_better_residential', 'prop_worse_commute', 'prop_worse_residential']
#     for key in pre_cal_col:
#         original_info[key] = 0
#     original_info['num_inds'] = len(original_info)
#     matched_info['num_inds'] = len(matched_info)
#     matched_info['prop_relocation'] = (matched_info['home_ID'] != matched_info['work_ID']).astype(int) / len(matched_info)
#     matched_info['prop_better_commute'] = (matched_info['commute_benefit'] > 0).astype(int) / len(matched_info)
#     matched_info['prop_better_residential'] = (matched_info['residential_benefit'] > 0).astype(int) / len(matched_info)
#     matched_info['prop_worse_commute'] = (matched_info['commute_benefit'] < 0).astype(int) / len(matched_info)
#     matched_info['prop_worse_residential'] = (matched_info['residential_benefit'] < 0).astype(int) / len(matched_info)
#
#     # test = matched_info.loc[(matched_info['commute_benefit'] < 0) & (matched_info['residential_benefit'] < 0)]
#     # print('both worse', len(test))
#
#     stats_col = ['distance', 'CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij',
#                  'SO2_ij', 'PM25_ij','prob_car','prob_pt','prob_bike','prob_walk'] + ['num_inds'] + pre_cal_col
#
#
#     compare = matched_info[['work_ID','home_ID'] + stats_col].merge(original_info[['work_ID','home_ID'] + stats_col], on =['work_ID'])
#
#     res = {'scenario': [], 'attributes':[], 'old':[], 'new':[]}
#     for key in stats_col:
#         res['scenario'].append(scenario)
#         res['attributes'].append(key)
#         if 'prob' in key or key == 'num_inds':
#             old = np.mean(compare[f'{key}_y'])
#             new = np.mean(compare[f'{key}_x'])
#         else:
#             old = sum(compare[f'{key}_y'])
#             new = sum(compare[f'{key}_x'])
#         res['old'].append(old)
#         res['new'].append(new)
#     res_df = pd.DataFrame(res)
#     return res_df
#
#




def calculate_res_all(scenario, refer_scenario_name, city_folder):
    if os.path.exists(f'../{city_folder}/matched_res_{scenario}.csv'):
        matched_res = pd.read_csv(f'../{city_folder}/matched_res_{scenario}.csv')
    else:
        print(f'../{city_folder}/matched_res_{scenario}.csv does not exists')
        return pd.DataFrame()
    if 'random' in scenario or 'optimal' in scenario:
        pair_info = pd.read_parquet(f'../{city_folder}/final_pair_for_matching_{refer_scenario_name}.parquet')
    else:
        pair_info = pd.read_parquet(f'../{city_folder}/final_pair_for_matching_{scenario}.parquet')
    original_info = pair_info.loc[pair_info['home_ID']==pair_info['work_ID']].copy()
    matched_info = pair_info.merge(matched_res,on=['home_ID','work_ID'])

    if 'random' in scenario or 'optimal' in scenario:
        no_included_sample = original_info.merge(matched_info[['work_ID']], on='work_ID', how='left', indicator=True)
        no_included_sample = no_included_sample[no_included_sample['_merge'] == 'left_only'].drop(columns=['_merge'])
        matched_info = pd.concat([matched_info, no_included_sample])

    assert len(original_info) == len(matched_info)


    pre_cal_col = ['prop_relocation', 'prop_better_commute', 'prop_better_residential', 'prop_worse_commute', 'prop_worse_residential']
    for key in pre_cal_col:
        original_info[key] = 0
    original_info['num_inds'] = len(original_info)
    matched_info['num_inds'] = len(matched_info)
    matched_info['prop_relocation'] = (matched_info['home_ID'] != matched_info['work_ID']).astype(int) / len(matched_info)
    matched_info['prop_better_commute'] = (matched_info['commute_benefit'] > 0).astype(int) / len(matched_info)
    matched_info['prop_better_residential'] = (matched_info['residential_benefit'] > 0).astype(int) / len(matched_info)
    matched_info['prop_worse_commute'] = (matched_info['commute_benefit'] < 0).astype(int) / len(matched_info)
    matched_info['prop_worse_residential'] = (matched_info['residential_benefit'] < 0).astype(int) / len(matched_info)

    # test = matched_info.loc[(matched_info['commute_benefit'] < 0) & (matched_info['residential_benefit'] < 0)]
    # print('both worse', len(test))

    stats_col = ['distance', 'CO2_ij', 'NOx_ij', 'VOC_ij', 'CO_ij',
                 'SO2_ij', 'PM25_ij','prob_car','prob_pt','prob_bike','prob_walk'] + ['num_inds'] + pre_cal_col

    compare = matched_info[['work_ID','home_ID'] + stats_col].merge(original_info[['work_ID','home_ID'] + stats_col], on =['work_ID'])

    res = {'city': [], 'scenario': [], 'attributes':[], 'old':[], 'new':[]}
    for key in stats_col:
        res['city'].append(city_folder)
        res['scenario'].append(scenario)
        res['attributes'].append(key)
        if 'prob' in key or key == 'num_inds':
            old = np.mean(compare[f'{key}_y'])
            new = np.mean(compare[f'{key}_x'])
        else:
            old = sum(compare[f'{key}_y'])
            new = sum(compare[f'{key}_x'])
        res['old'].append(old)
        res['new'].append(new)
    res_df = pd.DataFrame(res)
    # res_df['diff_percent'] = (res_df['new'] - res_df['old']) / res_df['old']
    # res_df.to_csv(f'../Munich/final_matched_res_stats.csv',index=False)
    return res_df


if __name__ == '__main__':

    # scenario_list = [
    #     'refer',
    #     'high_vot', 'low_vot',
    #     'vij_20_percent', 'vij_30_percent',
    #     'time_span_3', 'time_span_8', 'time_span_10',
    #     'inverse_future_gain',
    #     'high_vij', 'low_vij',
    #     'no_future_gain_zero_residential',
    # ]

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
    ]

    random_percent = list(range(10,100,10))
    num_seed = 5
    random_select_scenario = []
    for percent in random_percent:
        for i in range(num_seed):
            random_select_scenario.append(f'4_0_random_{percent}_percent_seed_{i}')
    scenario_list += random_select_scenario


    step = 2
    optimal_percent = list(range(1, 17+step, step))
    optimal_select_scenario = []
    for percent in optimal_percent:
        optimal_select_scenario.append(f'4_0_optimal_{percent}_percent')

    scenario_list += optimal_select_scenario




    # scenario_list = ['1_0_ideal', '2_0_in_between', '3_0_realistic']
    city_folder_list = ['Munich', 'Beijing', 'Singapore_ind']#'Munich', 'Beijing',
    for city_folder in city_folder_list:
        res_all = []
        for scenario in scenario_list:
            print(f'{city_folder} scenario: {scenario}')
            refer_scenario_name = '3_0_realistic'  # refer
            s_time = time.time()
            res_df = calculate_res_all(scenario, refer_scenario_name, city_folder)
            res_all.append(res_df)
        res_all_df = pd.concat(res_all)
        res_all_df['scenario_new'] = res_all_df['scenario'].apply(lambda x: re.sub(r"_seed_.*", "", x))
        res_all_df = res_all_df.groupby(['city', 'scenario_new','attributes'],sort=False)[['old', 'new']].mean().reset_index()
        res_all_df = res_all_df.rename(columns={'scenario_new': 'scenario'})
        res_all_df['diff_percent'] = (res_all_df['new'] - res_all_df['old']) / res_all_df['old']
        res_all_df.to_csv(f'../{city_folder}/final_matched_res_stats.csv', index=False)




    # # # ############### Singapore HH
    scenario_list_sg_hh = ['1_0_ideal', '2_0_in_between', '3_0_realistic']
    city_folder = 'Singapore'
    res_all = []
    for scenario in scenario_list_sg_hh:
        print(f'Singapore HH scenario: {scenario}')
        s_time = time.time()
        print(f'finish read data, time: {time.time() - s_time}')
        res_df = calculate_res_sg(scenario, city_folder)
        res_all.append(res_df)
    res_all_df = pd.concat(res_all)
    res_all_df['scenario_new'] = res_all_df['scenario'].apply(lambda x: re.sub(r"_seed_.*", "", x))
    res_all_df = res_all_df.groupby(['city', 'scenario_new','attributes'],sort=False)[['old', 'new']].mean().reset_index()
    res_all_df = res_all_df.rename(columns={'scenario_new': 'scenario'})
    res_all_df['diff_percent'] = (res_all_df['new'] - res_all_df['old']) / res_all_df['old']
    res_all_df.to_csv(f'../{city_folder}/final_matched_res_stats.csv', index=False)
