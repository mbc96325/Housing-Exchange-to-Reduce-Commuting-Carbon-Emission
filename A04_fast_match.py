
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import time
from scipy.sparse import coo_matrix

import os


def match_munich(data, scenario, skip_existing=False):
    if skip_existing and os.path.exists(f'../Munich/matched_res_{scenario}.csv'):
        print(f'{scenario} exist, skip...')
        return
    data['obj_weight'] = data['distance']
    # Map work_ID and home_ID to contiguous integers
    data['work_ID_map'] = pd.factorize(data['work_ID'])[0]
    data['home_ID_map'] = pd.factorize(data['home_ID'])[0]

    # Get the number of unique IDs
    num_work = data['work_ID_map'].max() + 1
    num_home = data['home_ID_map'].max() + 1

    # Determine the size of the square matrix
    N = max(num_work, num_home)

    # Create the sparse matrix
    s_time = time.time()
    sparse_matrix = coo_matrix(
        (data['obj_weight'], (data['work_ID_map'], data['home_ID_map'])),
        shape=(N, N)
    )
    print(f'Finish cost matrix, time: {time.time() - s_time}')

    # Convert the sparse matrix to a dense matrix
    s_time = time.time()
    dense_matrix = np.full((N, N), np.inf)
    dense_matrix[sparse_matrix.row, sparse_matrix.col] = sparse_matrix.data

    print(f'Finish process to dense matrix, time: {time.time() - s_time}')

    # Solve the assignment problem
    s_time = time.time()
    row_indices, col_indices = linear_sum_assignment(dense_matrix)
    print(f'Finish solve, time: {time.time() - s_time}')

    # Map back to original IDs
    matched_res = pd.DataFrame({
        'work_ID': pd.Categorical.from_codes(row_indices, pd.factorize(data['work_ID'])[1]),
        'home_ID': pd.Categorical.from_codes(col_indices, pd.factorize(data['home_ID'])[1])
    })
    matched_data = data.merge(matched_res, on =['work_ID','home_ID'])
    matched_data['w_ij'] = matched_data["commute_benefit"] + matched_data["residential_benefit"] - matched_data["r_ij"]
    print(f'commute_benefit change: {sum(matched_data["commute_benefit"])}')
    print(f'residential_benefit change: {sum(matched_data["residential_benefit"])}')
    print(f'total relocation cost: {sum(matched_data["r_ij"])}')
    print(f'total social welfare: {sum(matched_data["w_ij"])}')
    print(f'prop_relocated_people: {sum(matched_res["work_ID"] != matched_res["home_ID"]) / len(matched_res)}')
    matched_res.to_csv(f'../Munich/matched_res_{scenario}.csv',index=False)
    return


def match_sg(data, scenario):
    data = data.groupby(['HHID_original', 'HHID']).agg({
        'commute_benefit': 'sum',
        'distance': 'sum',
        'CO2_ij': 'sum',
        'NOx_ij': 'sum',
        'VOC_ij': 'sum',
        'CO_ij': 'sum',
        'SO2_ij': 'sum',
        'PM25_ij': 'sum',
        'prob_car': 'mean',
        'prob_pt': 'mean',
        'prob_bike': 'mean',
        'prob_walk': 'mean',
        'residential_benefit': 'mean', # related to house, take mean
        'r_ij': 'mean' # related to house, take mean
    }).reset_index()
    all_HH = data[['HHID']].drop_duplicates().sort_values(['HHID'])
    all_HH['HHID_original'] = all_HH['HHID']
    all_HH['work_ID'] = np.arange(len(all_HH))
    all_HH['home_ID'] = all_HH['work_ID']
    data = data.merge(all_HH[['HHID', 'home_ID']], on = ['HHID'])
    data = data.merge(all_HH[['HHID_original', 'work_ID']], on = ['HHID_original'])
    data.to_parquet(f"../Singapore/HH_paired_info_{scenario}.parquet",index=False)
    print(f"num work id: {len(pd.unique(data['work_ID']))}")
    print(f"num home id: {len(pd.unique(data['home_ID']))}")
    data['obj_weight'] = data['commute_benefit'] #(1-gamma) * (data['residential_benefit'] - data['r_ij'])
    # data = data.sort_values(['work_ID','home_ID'])
    s_time = time.time()
    N = data[['work_ID', 'home_ID']].max().max() + 1
    sparse_matrix = coo_matrix(
    (data['obj_weight'], (data['work_ID'], data['home_ID'])),
    shape=(N, N))
    print(f'finish cost matrix, time: {time.time() - s_time}')

    s_time = time.time()
    dense_matrix = np.full((N, N), -np.inf)

    # Fill the dense matrix with the sparse matrix values
    dense_matrix[sparse_matrix.row, sparse_matrix.col] = sparse_matrix.data

    # Negate the matrix for the maximum weight matching
    neg_matrix = -dense_matrix


    print(f'finish process to dense matrix, time: {time.time() - s_time}')

    s_time = time.time()
    # Solve the assignment problem using the Hungarian (Kuhn-Munkres) algorithm
    row_indices, col_indices = linear_sum_assignment(neg_matrix)
    print(f'finish solve, time: {time.time() - s_time}')
    matched_res = pd.DataFrame(
        {
        'work_ID': row_indices,
        'home_ID': col_indices}
    )
    matched_data = data.merge(matched_res, on =['work_ID','home_ID'])
    matched_data['w_ij'] = matched_data["commute_benefit"] + matched_data["residential_benefit"] - matched_data["r_ij"]
    print(f'commute_benefit change: {sum(matched_data["commute_benefit"])}')
    print(f'residential_benefit change: {sum(matched_data["residential_benefit"])}')
    print(f'total relocation cost: {sum(matched_data["r_ij"])}')
    print(f'total social welfare: {sum(matched_data["w_ij"])}')
    print(f'prop_relocated_people: {sum(matched_res["work_ID"] != matched_res["home_ID"]) / len(matched_res)}')
    matched_res.to_csv(f'../Singapore/matched_res_{scenario}.csv',index=False)




def match_bj(data, scenario, skip_existing=False):
    if skip_existing and os.path.exists(f'../Beijing/matched_res_{scenario}.csv'):
        print(f'{scenario} exist, skip...')
        return
    data['obj_weight'] = data['distance'] # minimize distance
    # Map work_ID and home_ID to contiguous integers
    data['work_ID_map'] = pd.factorize(data['work_ID'])[0]
    data['home_ID_map'] = pd.factorize(data['home_ID'])[0]

    # Get the number of unique IDs
    num_work = data['work_ID_map'].max() + 1
    num_home = data['home_ID_map'].max() + 1

    # Determine the size of the square matrix
    N = max(num_work, num_home)

    # Create the sparse matrix
    s_time = time.time()
    sparse_matrix = coo_matrix(
        (data['obj_weight'], (data['work_ID_map'], data['home_ID_map'])),
        shape=(N, N)
    )
    print(f'Finish cost matrix, time: {time.time() - s_time}')

    # Convert the sparse matrix to a dense matrix
    s_time = time.time()
    dense_matrix = np.full((N, N), np.inf)
    dense_matrix[sparse_matrix.row, sparse_matrix.col] = sparse_matrix.data

    print(f'Finish process to dense matrix, time: {time.time() - s_time}')

    # Solve the assignment problem
    s_time = time.time()
    row_indices, col_indices = linear_sum_assignment(dense_matrix)
    print(f'Finish solve, time: {time.time() - s_time}')

    # Map back to original IDs
    matched_res = pd.DataFrame({
        'work_ID': pd.Categorical.from_codes(row_indices, pd.factorize(data['work_ID'])[1]),
        'home_ID': pd.Categorical.from_codes(col_indices, pd.factorize(data['home_ID'])[1])
    })
    matched_data = data.merge(matched_res, on =['work_ID','home_ID'])
    matched_data['w_ij'] = matched_data["commute_benefit"] + matched_data["residential_benefit"] - matched_data["r_ij"]
    print(f'commute_benefit change: {sum(matched_data["commute_benefit"])}')
    print(f'residential_benefit change: {sum(matched_data["residential_benefit"])}')
    print(f'total relocation cost: {sum(matched_data["r_ij"])}')
    print(f'total social welfare: {sum(matched_data["w_ij"])}')
    print(f'prop_relocated_people: {sum(matched_res["work_ID"] != matched_res["home_ID"]) / len(matched_res)}')
    matched_res.to_csv(f'../Beijing/matched_res_{scenario}.csv',index=False)
    return



def match_sg_ind(data, scenario, skip_existing=False):
    if skip_existing and os.path.exists(f'../Singapore_ind/matched_res_{scenario}.csv'):
        print(f'{scenario} exist, skip...')
        return
    data['obj_weight'] = data['distance'] # minimize distance
    # Map work_ID and home_ID to contiguous integers
    data['work_ID_map'] = pd.factorize(data['work_ID'])[0]
    data['home_ID_map'] = pd.factorize(data['home_ID'])[0]

    # Get the number of unique IDs
    num_work = data['work_ID_map'].max() + 1
    num_home = data['home_ID_map'].max() + 1

    # Determine the size of the square matrix
    N = max(num_work, num_home)

    # Create the sparse matrix
    s_time = time.time()
    sparse_matrix = coo_matrix(
        (data['obj_weight'], (data['work_ID_map'], data['home_ID_map'])),
        shape=(N, N)
    )
    print(f'Finish cost matrix, time: {time.time() - s_time}')

    # Convert the sparse matrix to a dense matrix
    s_time = time.time()
    dense_matrix = np.full((N, N), np.inf)
    dense_matrix[sparse_matrix.row, sparse_matrix.col] = sparse_matrix.data

    print(f'Finish process to dense matrix, time: {time.time() - s_time}')

    # Solve the assignment problem
    s_time = time.time()
    row_indices, col_indices = linear_sum_assignment(dense_matrix)
    print(f'Finish solve, time: {time.time() - s_time}')

    # Map back to original IDs
    matched_res = pd.DataFrame({
        'work_ID': pd.Categorical.from_codes(row_indices, pd.factorize(data['work_ID'])[1]),
        'home_ID': pd.Categorical.from_codes(col_indices, pd.factorize(data['home_ID'])[1])
    })
    matched_data = data.merge(matched_res, on =['work_ID','home_ID'])
    matched_data['w_ij'] = matched_data["commute_benefit"] + matched_data["residential_benefit"] - matched_data["r_ij"]
    print(f'commute_benefit change: {sum(matched_data["commute_benefit"])}')
    print(f'residential_benefit change: {sum(matched_data["residential_benefit"])}')
    print(f'total relocation cost: {sum(matched_data["r_ij"])}')
    print(f'total social welfare: {sum(matched_data["w_ij"])}')
    print(f'prop_relocated_people: {sum(matched_res["work_ID"] != matched_res["home_ID"]) / len(matched_res)}')
    matched_res.to_csv(f'../Singapore_ind/matched_res_{scenario}.csv',index=False)
    return


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
    # random_percent = list(range(10,100,10))
    # num_seed = 5
    # random_select_scenario = []
    # for percent in random_percent:
    #     for i in range(num_seed):
    #         random_select_scenario.append(f'random_{percent}_percent_seed_{i}')
    # scenario_list += random_select_scenario

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
        '5_2A_similar_attributes',
    ]
    random_percent = list(range(10,100,10))
    num_seed = 5
    random_select_scenario = []
    for percent in random_percent:
        for i in range(num_seed):
            random_select_scenario.append(f'4_0_random_{percent}_percent_seed_{i}')
    scenario_list += random_select_scenario


    Skip_existing = False
    # ########## Munich
    # # scenario_list = ['1_0_ideal', '2_0_in_between', '3_0_realistic']
    # for scenario in scenario_list:
    #     print(f"###############Munich Scenario {scenario}###############")
    #     s_time = time.time()
    #     data = pd.read_parquet(f'../Munich/final_pair_for_matching_{scenario}.parquet')
    #     print(f'finish read data, time: {time.time() - s_time}')
    #     match_munich(data, scenario, skip_existing=Skip_existing)
    #     print('\n')
    # #
    #
    # # # # ########## Beijing
    # # scenario_list = ['1_0_ideal', '2_0_in_between', '3_0_realistic']
    # for scenario in scenario_list:
    #     print(f"###############Beijing Scenario {scenario}###############")
    #     s_time = time.time()
    #     data = pd.read_parquet(f'../Beijing/final_pair_for_matching_{scenario}.parquet')
    #     print(f'finish read data, time: {time.time() - s_time}')
    #     match_bj(data, scenario, skip_existing=Skip_existing)
    #     print('\n')
    #
    # # # #
    # # # # ######### Singapore_ind
    # # scenario_list = ['1_0_ideal', '2_0_in_between', '3_0_realistic']
    # for scenario in scenario_list:
    #     print(f"###############Singapore_ind Scenario {scenario}###############")
    #     s_time = time.time()
    #     data = pd.read_parquet(f'../Singapore_ind/final_pair_for_matching_{scenario}.parquet')
    #     print(f'finish read data, time: {time.time() - s_time}')
    #     match_sg_ind(data, scenario, skip_existing=Skip_existing)
    #     print('\n')

    #
    # ######### Singapore HH
    print(f"###############SG Scenario refer_HH###############")
    scenario_list = ['1_0_ideal', '2_0_in_between', '3_0_realistic']
    for scenario in scenario_list:
        s_time = time.time()
        data = pd.read_parquet(f'../Singapore/final_pair_for_matching_{scenario}.parquet')
        print(f'finish read data, time: {time.time() - s_time}')
        match_sg(data, scenario)
        print('\n')
