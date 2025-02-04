import pandas as pd

def output_munich(scenario):
    res = pd.read_csv(f"../Munich/matched_res_{scenario}.csv")
    res = res.rename(columns = {'home_ID': 'new_home_ID'})
    res['old_home_ID'] = res['work_ID']
    res.to_csv(f'../Munich/new_home_info_{scenario}.csv', index=False)


def output_beijing(scenario):
    res = pd.read_csv(f"../Beijing/matched_res_{scenario}.csv")
    res['home_ID'] += 1
    res['work_ID'] += 1
    res = res.rename(columns = {'home_ID': 'new_home_ID'})
    res['old_home_ID'] = res['work_ID']
    res.to_csv(f'../Beijing/new_home_info_{scenario}.csv', index=False)


def output_singapore_ind(scenario):
    res = pd.read_csv(f"../Singapore_ind/matched_res_{scenario}.csv")
    id_map = pd.read_csv('../Singapore_ind/processed_individual_data.csv')
    res = res.merge(id_map[['work_ID','HP_ID','Home_pcode', 'Work_pcode']], on =['work_ID'])
    res = res.rename(columns={'Home_pcode': 'old_Home_pcode'})
    res = res.merge(id_map[['home_ID','Home_pcode']], on =['home_ID'])
    res = res.rename(columns={'Home_pcode': 'new_Home_pcode'})

    res[['HP_ID', 'Work_pcode', 'old_Home_pcode', 'new_Home_pcode']].to_csv(f'../Singapore_ind/new_home_info_{scenario}.csv', index=False)


if __name__ == '__main__':
    # scenario_list_output = ['refer', 'no_future_gain_zero_residential', 'optimal_10_percent']
    scenario_list_output = ['1_0_ideal', '2_0_in_between', '3_0_realistic', '4_0_optimal_5_percent']
    ###### Munich
    for scenario in scenario_list_output:
        print('Munich')
        output_munich(scenario)

    ####### Beijing
    for scenario in scenario_list_output:
        print('Beijing')
        output_beijing(scenario)


    ####### Singapore ind
    for scenario in scenario_list_output:
        print('Singapore ind')
        output_singapore_ind(scenario)


