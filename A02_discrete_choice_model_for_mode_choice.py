import pandas as pd
import biogeme.biogeme as bio
from biogeme import models
from biogeme.database import Database
from biogeme.expressions import Variable, Beta
import numpy as np

tax_df_bj = pd.DataFrame({
    'min_income':[0,3000,12000,25000,35000,55000,80000],
    'max_income':[3000,12000,25000,35000,55000,80000, np.inf],
    'tax_rate':[0.03,0.1,0.2,0.25,0.3,0.35,0.45]
})
tax_df_mu = pd.DataFrame({
    'min_income':[0,10908/12,20000/12,30000/12,40000/12,50000/12, 61973/12,277825/12],
    'max_income':[10908/12,20000/12,30000/12,40000/12,50000/12, 61973/12,277825/12, np.inf],
    'tax_rate':[0.0,0.14, 0.19, 0.245,0.300,0.354, 0.42,0.45]
})
tax_df_sg = pd.DataFrame({
    'min_income':[0,20000/12,30000/12,40000/12,80000/12,120000/12,160000/12,280000/12,500000/12],
    'max_income':[20000/12,30000/12,40000/12,80000/12,120000/12,160000/12,280000/12,500000/12, np.inf],
    'tax_rate':[0.0,0.02,0.035,0.07,0.115,0.15,0.18,0.20,0.24]
})

def process_tax(tax_df):
    cum_tax_value = 0
    cum_tax = []
    for min_income, max_income, tax_rate in zip(tax_df['min_income'], tax_df['max_income'], tax_df['tax_rate']):
        cum_tax.append(cum_tax_value)
        tax_this_bucket = (max_income - min_income) * tax_rate
        cum_tax_value += tax_this_bucket
    tax_df['cum_tax'] = np.array(cum_tax)
    return tax_df
tax_df_bj = process_tax(tax_df_bj)
tax_df_mu = process_tax(tax_df_mu)
tax_df_sg = process_tax(tax_df_sg)




def process_data_munich():
    data = pd.read_csv('../Munich/mode_choice_sample.csv')
    updated_pt_cost = pd.read_csv('../Munich/update_pt_cost_v2.csv')
    updated_pt_cost = updated_pt_cost.rename(columns = {'ptcostv2': 'ptcost_new'})


    updated_pt_time = pd.read_csv('../Munich/updated_pt_time.csv')
    updated_pt_time = updated_pt_time.rename(columns = {'cost_time(s)': 'pttime_new'})
    updated_pt_time['pttime_new'] /= 3600

    car_avail = pd.read_csv('../Munich/Ind_car_availability.csv')
    car_avail['car_avail'] = 0
    car_avail.loc[car_avail['v26'].isin([1,2,3]), 'car_avail'] = 1


    data = data.merge(updated_pt_cost[['LfdNr','ptcost_new']], on =['LfdNr'], how = 'left')
    data.loc[~data['ptcost_new'].isna(), 'pt_cost'] = data.loc[~data['ptcost_new'].isna(), 'ptcost_new']

    data = data.merge(updated_pt_time[['home_ID','pttime_new']], on =['home_ID'], how = 'left')
    data.loc[~data['pttime_new'].isna(), 'pt_time'] = data.loc[~data['pttime_new'].isna(), 'pttime_new']

    #1：car 2：pt 3： bike 4： walk 5： park+ride 6： bike+ride
    data['choice'] = data['commute_mode']
    data.loc[data['choice'].isin([5,6]), 'choice'] = 2
    ##


    most_frequent = data['choice'].mode()[0]
    data.loc[~data['choice'].isin([1,2,3,4]), 'choice'] = most_frequent # fill by most frequent
    data['choice'] = data['choice'].astype(int)

    for mode in range(1,5):
        print(f'{mode} prob: {sum(data["choice"]==mode)/len(data)}')

    data = data.merge(car_avail[['LfdNr', 'car_avail']], on = ['LfdNr'], how = 'left')
    # fix inconsistency
    data.loc[data['choice']==1, 'car_avail'] = 1
    data['car_avail'].fillna(0, inplace=True)


    all_choices = set(data['choice'])

    def fill_by_most_freq(data, col, fill_cols):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isin(fill_cols), col] = most_frequent  # fill by most frequent
        data[col] = data[col].astype(int)
        return data

    data = fill_by_most_freq(data, 'gender', [888])
    data = fill_by_most_freq(data, 'age', [888])
    data = fill_by_most_freq(data, 'income_cat', [888])
    data = fill_by_most_freq(data, 'hhsize', [888])
    data = fill_by_most_freq(data, 'edu', [888])
    # data.loc[data['car_avail']==1, 'car_cost'] = data.loc[data['car_avail']==1, 'distance'] / 1000 * 0.447
    data['car_cost'] = data['distance'] / 1000 * 0.447 / 1.3 #0.447 is cost per km for car, 1.3 is avg person per car trip
    # data.loc[data['car_avail'] == 0, 'car_cost'] *= 1.4/0.447
    # max_speed = 50
    # min_speed = 18
    data_len = len(data)
    data = data.merge(tax_df_mu, how = 'cross')
    data = data.loc[(data['income_cont']>data['min_income']) & (data['income_cont']<=data['max_income'])]
    assert len(data) == data_len
    data['income_after_tax'] = data['income_cont'] - (data['income_cont'] - data['min_income']) * data['tax_rate'] - data['cum_tax']
    data['income_cont'] /= 160 # assume 160 hr per month

    # data['car_speed'] = (data['distance'] - data['distance'].min()) / (data['distance'].max() - data['distance'].min()) * (max_speed - min_speed) + min_speed # proportional to distance
    # data['car_time'] = data['distance'] / 1000 / data['car_speed']  #
    # data['bike_time'] = data['distance'] / 1000 / 20  # 12km / hr
    # data['walk_time'] = data['distance'] / 1000 / 4  # 4km / hr
    # data['pt_time'] *= 3 #(0.1081 ** data['distance']+1119.7) / 3600
    data.to_csv('../Munich/mode_choice_sample_updated.csv', index=False)




def estimate_dcm_for_munich():
    data = pd.read_csv('../Munich/mode_choice_sample_updated.csv')
    ### change units to hr
    category_var = ['gender','age'] #['gender','age','hhsize','edu']#['gender','age','income_cat','hhsize']
    continuous_var = ['car_cost', 'car_time', 'pt_cost', 'pt_time', 'bike_cost', 'bike_time', 'walk_cost', 'walk_time']
    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, int)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list


    # Create dummy variables for categorical variables
    data, category_var_dummy = add_dummy(data, category_var)

    data_used = data.copy()
    # data_used = data.loc[data['distance'] > 0].copy()
    cost_var = ['car_cost', 'pt_cost', 'walk_cost', 'bike_cost']


    avg_hr_income = np.mean(data_used["income_cont"])
    print(f'avg_hr_income: {avg_hr_income}')
    for key in cost_var:
        data_used[key] /= data_used['income_cont']


    av_var = ['car_avail']
    # data_used['car_cost2'] = (data_used['distance'] / 1000) * 1.4 #0.52 is cost per km for car
    # Create a Biogeme database
    database = Database('CommuteMode', data_used[category_var_dummy+continuous_var+av_var+['choice']])

    # Define the choice column
    CHOICE = database.variables['choice']

    # Define social demographic variables (transform categorical variables into dummies)
    var_dict = {}
    for var in category_var_dummy:
        var_dict[var] = database.variables[var]



    # Define cost and time variables
    car_cost = database.variables['car_cost']
    car_time = database.variables['car_time']
    pt_cost = database.variables['pt_cost']
    pt_time = database.variables['pt_time']
    bike_cost = database.variables['bike_cost']
    bike_time = database.variables['bike_time']
    walk_cost = database.variables['walk_cost']
    walk_time = database.variables['walk_time']

    AV_Car = database.variables['car_avail']

    # Define parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_PT = Beta('ASC_PT', 0, None, None, 0)
    ASC_BIKE = Beta('ASC_BIKE', 0, None, None, 0)
    ASC_WALK = Beta('ASC_WALK', 0, None, None, 1)  # Base mode (reference)

    B_COST = Beta('B_COST', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    # B_CAR_COST = Beta('B_CAR_COST', 0, None, None, 0)
    # B_CAR_TIME = Beta('B_CAR_TIME', 0, None, None, 0)
    #
    # B_PT_COST = Beta('B_PT_COST', 0, None, None, 0)
    # B_PT_TIME = Beta('B_PT_TIME', 0, None, None, 0)
    #
    # B_WALK_COST = Beta('B_WALK_COST', 0, None, None, 0)
    # B_WALK_TIME = Beta('B_WALK_TIME', 0, None, None, 0)

    mode_list = ['car', 'pt', 'bike']

    beta_dict = {}
    for var in category_var_dummy:
        beta_dict[var] = {}
        for mode in mode_list:
            beta_dict[var][mode] = Beta(f'B_{mode.upper()}_{var.upper()}', 0, None, None, 0)

    # Define utility functions
    V_CAR = (
            ASC_CAR
            + B_COST * car_cost
            + B_TIME * car_time
    )
    for var in category_var_dummy:
        V_CAR += beta_dict[var]['car'] * var_dict[var]

    V_PT = (
            ASC_PT
            + B_COST * pt_cost
            + B_TIME * pt_time
    )
    for var in category_var_dummy:
        V_PT += beta_dict[var]['pt'] * var_dict[var]

    V_BIKE = (
            ASC_BIKE
            + B_COST * bike_cost
            + B_TIME * bike_time
    )
    for var in category_var_dummy:
        V_BIKE += beta_dict[var]['bike'] * var_dict[var]

    # base mode, not include social demo
    V_WALK = (
            ASC_WALK
            + B_COST * walk_cost
            + B_TIME * walk_time
    )

    # Associate utility functions with choice options
    V = {
        1: V_CAR,
        2: V_PT,
        3: V_BIKE,
        4: V_WALK,
    }

    # Availability of each mode (assume all are available)
    avail = {1: AV_Car, 2: 1, 3: 1, 4: 1}

    # Logit model
    logprob = models.loglogit(V, avail, CHOICE)

    # Define the Biogeme object

    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'CommuteModeModel'
    biogeme.generateHtml = False
    biogeme.generatePickle = False
    biogeme.saveIterations = False
    # Estimate the parameters
    results = biogeme.estimate()

    # Display results
    print(results.shortSummary())
    est_para = results.getEstimatedParameters()
    print(est_para)
    b_time = est_para.loc['B_TIME']['Value']
    b_cost_div_income = est_para.loc['B_COST']['Value']
    val_to_time = b_time / b_cost_div_income * data_used['income_cont'] # Euro per hour
    print(f'avg value per hr {np.mean(val_to_time)}')
    print(f'max value per hr {np.max(val_to_time)}')
    print(f'min value per hr {np.min(val_to_time)}')
    print(f'avg % vot divided by income {round(np.mean(val_to_time / data_used["income_cont"].values) * 100, 2)}%')
    print(f'max % vot divided by income {round(np.max(val_to_time / data_used["income_cont"].values) * 100, 2)}%')
    print(f'min % vot divided by income {round(np.min(val_to_time / data_used["income_cont"].values) * 100, 2)}%')
    # a=1
    est_para.to_csv('regression_res/dcm_mode_choice_munich.csv',index=True)


    summary_stats = {
        'Log-likelihood': results.data.logLike,
        'Initial Log-likelihood': results.data.initLogLike,
        'Null log-likelihood': results.data.nullLogLike,
        'Likelihood ratio test statistic': results.data.likelihoodRatioTest,
        'Rho-square (R^2)': results.data.rhoSquare,
        'Rho-bar-square': results.data.rhoBarSquare,
        'Number of observations': results.data.numberOfObservations,
        'Number of parameters': results.data.nparam,
        'Akaike Information Criterion (AIC)': results.data.akaike,
        'Bayesian Information Criterion (BIC)': results.data.bayesian
    }

    # Convert summary statistics to a DataFrame
    summary_stats_df = pd.DataFrame(summary_stats.items(), columns=['Metric', 'Value'])
    # Save summary statistics to a CSV file
    summary_stats_df.to_csv('regression_res/dcm_mode_choice_munich_summary.csv', index=True)

def calculate_new_travel_time_and_mode_munich():
    od_matrix = pd.read_csv('../Munich/ODmatrix_mode_choice.csv')
    od_matrix['car_cost'] = od_matrix['distance'] / 1000 * 0.447 / 1.3
    ind_data = pd.read_csv('../Munich/mode_choice_sample_updated.csv')
    od_matrix = od_matrix.merge(ind_data[['work_ID', 'gender', 'age','car_avail','income_cont','income_after_tax']], on = ['work_ID'])
    para = pd.read_csv('regression_res/dcm_mode_choice_munich.csv',index_col=0)
    category_var = ['gender','age'] #['gender','age','hhsize','edu']#['gender','age','income_cat','hhsize']
    def add_dummy(data, var_list):
        save_dummy_list = []
        for col in var_list:
            all_values = list(set(data[col]))
            all_values = [key for key in all_values if isinstance(key, int)]
            all_values = sorted(all_values)
            for val in all_values[:-1]: # drop last
                col_name = f'if_{col}_{val}'
                data[col_name] = 0
                data.loc[data[col]==val, col_name] = 1
                save_dummy_list.append(col_name)
        return data, save_dummy_list
    od_matrix, category_var_dummy = add_dummy(od_matrix, category_var)
    all_modes = ['car','pt','bike','walk']
    base_mode = 'walk'
    for mode in all_modes:
        if mode == base_mode:
            od_matrix[f'V_{mode}'] = 0.0
        else:
            od_matrix[f'V_{mode}'] = para.loc[f'ASC_{mode.upper()}']['Value']
        for key in category_var_dummy:
            if mode != base_mode:
                od_matrix[f'V_{mode}'] += para.loc[f'B_{mode.upper()}_{key.upper()}']['Value'] * od_matrix[key]
        od_matrix[f'V_{mode}'] += para.loc[f'B_TIME']['Value'] * od_matrix[f'{mode}_time']
        od_matrix[f'V_{mode}'] += para.loc[f'B_COST']['Value'] * (od_matrix[f'{mode}_cost'] / od_matrix['income_cont'])

    od_matrix['exp_sum'] = 0.0
    for mode in all_modes:
        if mode == 'car':
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}']) * od_matrix[f'car_avail']
        else:
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}'])

    for mode in all_modes:
        if mode == 'car':
            od_matrix[f'prob_{mode}'] = np.exp(od_matrix[f'V_{mode}']) * od_matrix[f'car_avail'] / od_matrix['exp_sum']
        else:
            od_matrix[f'prob_{mode}'] = np.exp(od_matrix[f'V_{mode}']) / od_matrix['exp_sum']

    for mode in all_modes:
        print(f'{mode}: avg prob: {np.mean(od_matrix[f"prob_{mode}"])}')

    # weighted avg travel time and cost
    od_matrix['t_ij'] = 0
    for mode in all_modes:
        od_matrix['t_ij'] += od_matrix[f'prob_{mode}'] * od_matrix[f'{mode}_time']

    od_matrix['vot_i'] = para.loc[f'B_TIME']['Value'] / para.loc[f'B_COST']['Value'] * od_matrix['income_cont']
    od_matrix['c_ij'] = 0


    for mode in all_modes:
        od_matrix['c_ij'] += od_matrix[f'prob_{mode}'] * od_matrix[f'{mode}_cost']


    # save data
    prob_col = []
    for mode in all_modes:
        prob_col.append(f'prob_{mode}')
        od_matrix[prob_col] = np.round(od_matrix[prob_col],4)
    saved_col = ['t_ij','c_ij','vot_i','income_after_tax']
    for col in saved_col:
        od_matrix[col] = np.round(od_matrix[col],4)
    od_matrix[['work_ID', 'home_ID', 'distance']+prob_col+saved_col].to_csv('../Munich/OD_matrix_with_info.csv',index=False)


def process_data_sg():
    ind_data = pd.read_csv('../Singapore/final_used_individuals.csv')
    survey_data = pd.read_csv("../Singapore/HITS 2012 Database.csv",encoding='latin1')
    data = ind_data.merge(survey_data, on=['HP_ID'])
    data['car_avail'] = 0
    data['bike_avail'] = 0
    data.loc[data['H5_VehAvailable']=='Yes', 'car_avail'] = 1
    data.loc[data['H7_Bike'] == 'Yes', 'bike_avail'] = 1
    data = data.loc[data['T6_Purpose'].isin(['Working for paid employment', 'Work Related Trip', 'Work-related (meetings, sales etc)'])]
    data['travel_mode'] = data['T10_Mode']
    data.loc[data['travel_mode'].isna(), 'travel_mode'] = data.loc[data['travel_mode'].isna(), 'T6cTripMode']

    all_modes = list(set(data['travel_mode']))
    mode_map = {1: ['Van / Lorry driver', 'Taxi', 'Motorcycle rider', 'Car driver', 'Car passenger','Motorcycle passenger','Van / Lorry passenger'],
     2: ['Public bus','MRT','Company bus','LRT', 'School bus','Shuttle bus'],
     3: ['Cycle'],
     4: ['Walk Only']}
    data['choice'] = -1
    for mode_id, mode_list in mode_map.items():
        data.loc[data['travel_mode'].isin(mode_list), 'choice'] = mode_id
    # test = data.loc[data['choice'] == -1]
    most_frequent = data['choice'].mode()[0]
    data.loc[~data['choice'].isin([1,2,3,4]), 'choice'] = most_frequent # fill by most frequent
    data['choice'] = data['choice'].astype(int)
    # data['walk_only_time'] = data['']
    data['trip_time'] = (
            data['T10a_WalkTime'].fillna(0) + data['T13_WaitTime'].fillna(0) + data['T14_InVehTime'].fillna(0) + data['T22_LastWalkTime'].fillna(0))
    data = data[['HP_ID','car_avail','bike_avail','choice','trip_time']].drop_duplicates()
    data['num_trips'] = data.groupby(['HP_ID'])['choice'].transform('count')
    data['total_trip_time'] = data.groupby(['HP_ID'])['trip_time'].transform('sum')
    data['total_trip_time'] /= 60 # to hr
    mode_priority = [2,1,3,4] #multi trips, select by the above sequence
    data['priority'] = 999
    for idx, mode in enumerate(mode_priority):
        data.loc[data['choice'] == mode, 'priority'] = idx
    data = data.sort_values(['priority'])
    data = data.groupby(['HP_ID']).first().reset_index()
    data = data.drop(columns=['num_trips', 'priority','trip_time'])
    print(len(data))
    data = ind_data.merge(data, on=['HP_ID'], how = 'left')
    print(len(data))
    miss_info = data.loc[data['choice'].isna()]
    print(f'% missing: {len(miss_info) / len(data)}')

    most_frequent = data['choice'].mode()[0]
    data.loc[~data['choice'].isin([1,2,3,4]), 'choice'] = most_frequent # fill by most frequent
    data['choice'] = data['choice'].astype(int)

    for mode in range(1,5):
        print(f'{mode} prob: {sum(data["choice"]==mode)/len(data)}')

    # fix inconsistency
    data.loc[data['choice']==1, 'car_avail'] = 1
    data['car_avail'].fillna(0, inplace=True)
    data.loc[data['choice']==3, 'bike_avail'] = 1
    data['bike_avail'].fillna(0, inplace=True)


    ind_attributes = pd.read_csv('../Singapore/ind_attributes.csv')
    data = data.merge(ind_attributes, on = ['HP_ID'])


    ind_category_var = ['Ethnic', 'Age', 'Citizen', 'Gender', 'Occup', 'income']
    all_choices = set(data['choice'])

    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data
    #
    for key in ind_category_var:
        data = fill_by_most_freq(data, key)


    time_correction_factor = {}

    od_matrix = pd.read_csv('../Singapore/final_singapore_ODmatrix.csv')
    data = data.merge(od_matrix[['work_ID','home_ID','distance']], on = ['work_ID','home_ID'])
    data['car_cost'] = data['distance'] / 1000 * 0.93 / 1.75 #1.13 is cost per km for car, 1.75 is avg person per car trip
    data['car_time'] = data['distance'] / 1000 / 30  # 45 km/h
    correction_factor = sum(data.loc[
        (data['choice']==1) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'total_trip_time']) / sum(data.loc[
        (data['choice']==1) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'car_time'])
    time_correction_factor['car'] = correction_factor

    data.loc[
        (data['choice']==1) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'car_time'] = data.loc[
        (data['choice']==1) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'total_trip_time']
    #

    data['pt_time'] = (0.1898 * data['distance'] + 775.91) / 3600 # obtained by regression R² = 0.8875
    correction_factor = sum(data.loc[
        (data['choice']==2) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'total_trip_time']) / sum(data.loc[
        (data['choice']==2) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'pt_time'])
    data['pt_time'] *= correction_factor
    time_correction_factor['pt'] = correction_factor

    data.loc[
        (data['choice']==2) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'pt_time'] = data.loc[
        (data['choice']==2) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'total_trip_time']
    data['pt_cost'] = -1e-09 * data['distance']**2 + 7e-05 * data['distance'] + 0.8546  # (R² = 0.9842)

    data['bike_time'] = data['distance'] / 1000 / 20 # 20 km/h

    correction_factor = sum(data.loc[
        (data['choice']==3) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'total_trip_time']) / sum(data.loc[
        (data['choice']==3) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'bike_time'])
    data['bike_time'] *= correction_factor
    time_correction_factor['bike'] = correction_factor

    data.loc[
        (data['choice']==3) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'bike_time'] = data.loc[
        (data['choice']==3) & (~data['total_trip_time'].isna()) & (data['total_trip_time']>0), 'total_trip_time']
    data['bike_cost'] = 0.0

    data['walk_time'] = data['distance'] / 1000 / 4.43 #

    # no samples for walk time
    correction_factor = 1.0
    # data['walk_time'] *= correction_factor

    time_correction_factor['walk'] = correction_factor


    data['walk_cost'] = 0.0

    data_len = len(data)
    data = data.merge(tax_df_sg, how = 'cross')
    data = data.loc[(data['income']>data['min_income']) & (data['income']<=data['max_income'])]
    assert len(data) == data_len
    data['income_after_tax'] = data['income'] - (data['income'] - data['min_income']) * data['tax_rate'] - data['cum_tax']



    data['income'] /= 160  # assume 160  hr per month

    print(f"num individuals {len(data)}")

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

    data.to_csv('../Singapore/mode_choice_sample_updated.csv', index=False)
    correction_factor_df = pd.DataFrame(time_correction_factor.items(),columns=['mode', 'time_correction_factor'])
    correction_factor_df.to_csv('../Singapore/time_correction_factor.csv', index=False)



def estimate_dcm_for_sg():
    data = pd.read_csv('../Singapore/mode_choice_sample_updated.csv')
    ### change units to hr
    # category_var = ['Ethnic', 'Age', 'Citizen', 'Gender', 'Occup']
    category_var = ['Age','Gender']#['Ethnic','Age'] #['gender','age','hhsize','edu']#['gender','age','income_cat','hhsize']
    continuous_var = ['car_cost', 'car_time', 'pt_cost', 'pt_time', 'bike_cost', 'bike_time', 'walk_cost', 'walk_time']
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
    data, category_var_dummy = add_dummy(data, category_var)

    data_used = data.copy()
    # data_used = data.loc[data['distance'] > 0].copy()
    # data_used['pt_time'] #*= 1.5
    cost_var = ['car_cost', 'pt_cost', 'walk_cost', 'bike_cost']


    avg_hr_income = np.mean(data_used["income"])
    print(f'avg_hr_income: {avg_hr_income}')
    for key in cost_var:
        data_used[key] /= data_used['income']


    av_var = ['car_avail','bike_avail']
    # data_used['car_cost2'] = (data_used['distance'] / 1000) * 1.4 #0.52 is cost per km for car
    # Create a Biogeme database
    database = Database('CommuteMode', data_used[category_var_dummy+continuous_var+av_var+['choice']])

    # Define the choice column
    CHOICE = database.variables['choice']

    # Define social demographic variables (transform categorical variables into dummies)
    var_dict = {}
    for var in category_var_dummy:
        var_dict[var] = database.variables[var]



    # Define cost and time variables
    car_cost = database.variables['car_cost']
    car_time = database.variables['car_time']
    pt_cost = database.variables['pt_cost']
    pt_time = database.variables['pt_time']
    bike_cost = database.variables['bike_cost']
    bike_time = database.variables['bike_time']
    walk_cost = database.variables['walk_cost']
    walk_time = database.variables['walk_time']

    AV_Car = database.variables['car_avail']
    AV_Bike = database.variables['bike_avail']

    # Define parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_PT = Beta('ASC_PT', 0, None, None, 0)
    ASC_BIKE = Beta('ASC_BIKE', 0, None, None, 0)
    ASC_WALK = Beta('ASC_WALK', 0, None, None, 1)  # Base mode (reference)

    B_COST = Beta('B_COST', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    # B_CAR_COST = Beta('B_CAR_COST', 0, None, None, 0)
    # B_CAR_TIME = Beta('B_CAR_TIME', 0, None, None, 0)
    #
    # B_PT_COST = Beta('B_PT_COST', 0, None, None, 0)
    # B_PT_TIME = Beta('B_PT_TIME', 0, None, None, 0)
    #
    # B_WALK_COST = Beta('B_WALK_COST', 0, None, None, 0)
    # B_WALK_TIME = Beta('B_WALK_TIME', 0, None, None, 0)

    mode_list = ['car', 'pt', 'bike']

    beta_dict = {}
    for var in category_var_dummy:
        beta_dict[var] = {}
        for mode in mode_list:
            beta_dict[var][mode] = Beta(f'B_{mode.upper()}_{var.upper()}', 0, None, None, 0)

    # Define utility functions
    V_CAR = (
            ASC_CAR
            + B_COST * car_cost
            + B_TIME * car_time
    )
    for var in category_var_dummy:
        V_CAR += beta_dict[var]['car'] * var_dict[var]

    V_PT = (
            ASC_PT
            + B_COST * pt_cost
            + B_TIME * pt_time
    )
    for var in category_var_dummy:
        V_PT += beta_dict[var]['pt'] * var_dict[var]

    V_BIKE = (
            ASC_BIKE
            + B_COST * bike_cost
            + B_TIME * bike_time
    )
    for var in category_var_dummy:
        V_BIKE += beta_dict[var]['bike'] * var_dict[var]

    # base mode, not include social demo
    V_WALK = (
            ASC_WALK
            + B_COST * walk_cost
            + B_TIME * walk_time
    )

    # Associate utility functions with choice options
    V = {
        1: V_CAR,
        2: V_PT,
        3: V_BIKE,
        4: V_WALK,
    }

    # Availability of each mode (assume all are available)
    avail = {1: AV_Car, 2: 1, 3: AV_Bike, 4: 1}

    # Logit model
    logprob = models.loglogit(V, avail, CHOICE)

    # Define the Biogeme object

    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'CommuteModeModel'
    biogeme.generateHtml = False
    biogeme.generatePickle = False
    biogeme.saveIterations = False
    # Estimate the parameters
    results = biogeme.estimate()

    # Display results
    print(results.shortSummary())
    est_para = results.getEstimatedParameters()
    print(est_para)
    b_time = est_para.loc['B_TIME']['Value']
    b_cost_div_income = est_para.loc['B_COST']['Value']
    val_to_time = b_time / b_cost_div_income * data_used['income'] # Euro per hour
    print(f'avg value per hr {np.mean(val_to_time)}')
    print(f'max value per hr {np.max(val_to_time)}')
    print(f'min value per hr {np.min(val_to_time)}')
    print(f'avg % vot divided by income {round(np.mean(val_to_time / data_used["income"].values) * 100, 2)}%')
    print(f'max % vot divided by income {round(np.max(val_to_time / data_used["income"].values) * 100, 2)}%')
    print(f'min % vot divided by income {round(np.min(val_to_time / data_used["income"].values) * 100, 2)}%')
    # a=1
    est_para.to_csv('regression_res/dcm_mode_choice_sg.csv',index=True)

    summary_stats = {
        'Log-likelihood': results.data.logLike,
        'Initial Log-likelihood': results.data.initLogLike,
        'Null log-likelihood': results.data.nullLogLike,
        'Likelihood ratio test statistic': results.data.likelihoodRatioTest,
        'Rho-square (R^2)': results.data.rhoSquare,
        'Rho-bar-square': results.data.rhoBarSquare,
        'Number of observations': results.data.numberOfObservations,
        'Number of parameters': results.data.nparam,
        'Akaike Information Criterion (AIC)': results.data.akaike,
        'Bayesian Information Criterion (BIC)': results.data.bayesian
    }

    # Convert summary statistics to a DataFrame
    summary_stats_df = pd.DataFrame(summary_stats.items(), columns=['Metric', 'Value'])
    # Save summary statistics to a CSV file
    summary_stats_df.to_csv('regression_res/dcm_mode_choice_sg_summary.csv', index=True)

def calculate_new_travel_time_and_mode_sg():
    od_matrix = pd.read_csv('../Singapore/final_singapore_ODmatrix.csv')[['home_ID','work_ID','distance']]
    correction_factor_df = pd.read_csv('../Singapore/time_correction_factor.csv')

    od_matrix['car_cost'] = od_matrix[
                           'distance'] / 1000 * 0.93 / 1.75  # 1.13 is cost per km for car, 1.3 is avg person per car trip
    od_matrix['car_time'] = od_matrix['distance'] / 1000 / 30  # 45 km/h
    correction_factor = correction_factor_df.loc[correction_factor_df['mode']=='car']['time_correction_factor'].iloc[0]
    od_matrix['car_time'] *= correction_factor

    od_matrix['pt_time'] = (0.1898 * od_matrix['distance'] + 775.91) / 3600  #
    correction_factor = correction_factor_df.loc[correction_factor_df['mode']=='pt']['time_correction_factor'].iloc[0]
    od_matrix['pt_time'] *= correction_factor
    od_matrix['pt_cost'] = -1e-09 * od_matrix['distance'] ** 2 + 7e-05 * od_matrix['distance'] + 0.8546

    od_matrix['bike_time'] = od_matrix['distance'] / 1000 / 20  # 20 km/h
    correction_factor = correction_factor_df.loc[correction_factor_df['mode']=='bike']['time_correction_factor'].iloc[0]
    od_matrix['bike_time'] *= correction_factor
    od_matrix['bike_cost'] = 0.0

    od_matrix['walk_time'] = od_matrix['distance'] / 1000 / 4.43  #
    correction_factor = correction_factor_df.loc[correction_factor_df['mode']=='walk']['time_correction_factor'].iloc[0]
    od_matrix['walk_time'] *= correction_factor
    od_matrix['walk_cost'] = 0.0

    print('finish calculate time cost...')
    ind_data = pd.read_csv('../Singapore/mode_choice_sample_updated.csv')
    od_matrix = od_matrix.merge(
        ind_data[['HP_ID', 'HHID', 'work_ID', 'Gender', 'Age','car_avail', 'bike_avail','income','income_after_tax']], on = ['work_ID'])

    print('finish join home with individual...')

    para = pd.read_csv('regression_res/dcm_mode_choice_sg.csv', index_col=0)
    category_var = ['Gender','Age'] #['gender','age','hhsize','edu']#['gender','age','income_cat','hhsize']

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

    od_matrix, category_var_dummy = add_dummy(od_matrix, category_var)
    all_modes = ['car','pt','bike','walk']
    base_mode = 'walk'
    for mode in all_modes:
        if mode == base_mode:
            od_matrix[f'V_{mode}'] = 0.0
        else:
            od_matrix[f'V_{mode}'] = para.loc[f'ASC_{mode.upper()}']['Value']
        for key in category_var_dummy:
            if mode != base_mode:
                od_matrix[f'V_{mode}'] += para.loc[f'B_{mode.upper()}_{key.upper()}']['Value'] * od_matrix[key]
        od_matrix[f'V_{mode}'] += para.loc[f'B_TIME']['Value'] * od_matrix[f'{mode}_time']
        od_matrix[f'V_{mode}'] += para.loc[f'B_COST']['Value'] * (od_matrix[f'{mode}_cost'] / od_matrix['income'])

    od_matrix['exp_sum'] = 0.0
    for mode in all_modes:
        if mode == 'car':
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}']) * od_matrix[f'car_avail']
        elif mode == 'bike':
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}']) * od_matrix[f'bike_avail']
        else:
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}'])

    for mode in all_modes:
        if mode == 'car':
            od_matrix[f'prob_{mode}'] = np.exp(od_matrix[f'V_{mode}']) * od_matrix[f'car_avail'] / od_matrix['exp_sum']
        else:
            od_matrix[f'prob_{mode}'] = np.exp(od_matrix[f'V_{mode}']) / od_matrix['exp_sum']

    for mode in all_modes:
        print(f'{mode}: avg prob: {np.mean(od_matrix[f"prob_{mode}"])}')

    # weighted avg travel time and cost
    od_matrix['t_ij'] = 0
    for mode in all_modes:
        od_matrix['t_ij'] += od_matrix[f'prob_{mode}'] * od_matrix[f'{mode}_time']

    od_matrix['vot_i'] = para.loc[f'B_TIME']['Value'] / para.loc[f'B_COST']['Value'] * od_matrix['income']
    od_matrix['c_ij'] = 0


    for mode in all_modes:
        od_matrix['c_ij'] += od_matrix[f'prob_{mode}'] * od_matrix[f'{mode}_cost']


    # save data
    prob_col = []
    for mode in all_modes:
        prob_col.append(f'prob_{mode}')
        od_matrix[prob_col] = np.round(od_matrix[prob_col],4)
    saved_col = ['t_ij','c_ij','vot_i','income_after_tax']
    for col in saved_col:
        od_matrix[col] = np.round(od_matrix[col],4)
    od_matrix[['HP_ID', 'HHID', 'work_ID', 'home_ID', 'distance']+prob_col+saved_col].to_parquet('../Singapore/OD_matrix_with_info.parquet', index=False)



def process_data_bj():
    data = pd.read_csv('../Beijing/processed_individual_data.csv')


    all_modes = list(set(data['mode']))
    print(all_modes)
    mode_map = {
        1: ['出租车/网约车', '私家车'],
        2: ['地铁', '公交车', '单位班车/配车'],
        3: ['自行车/电动自行车'],
        4: ['步行']}
    data['choice'] = -1
    for mode_id, mode_list in mode_map.items():
        data.loc[data['mode'].isin(mode_list), 'choice'] = mode_id
    # test = data.loc[data['choice'] == -1]
    most_frequent = data['choice'].mode()[0]
    data.loc[~data['choice'].isin([1, 2, 3, 4]), 'choice'] = most_frequent  # fill by most frequent
    data['choice'] = data['choice'].astype(int)


    for mode in range(1, 5):
        print(f'{mode} prob: {sum(data["choice"] == mode) / len(data)}')




    ind_category_var = ['edu', 'income','age','gender','income']


    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data


    #
    for key in ind_category_var:
        data = fill_by_most_freq(data, key)
    data['distance'] = data['commute_dis'] * 1000


    data['car_cost'] = data['distance'] / 1000 * 1.6 / 1.3  # 1.7 is cost per km for car, 1.3 is avg person per car trip
    data['car_time'] = data['distance'] / 1000 / 30  # 30 km/h


    data['pt_time'] = (2.1931 * data['distance'] / 1000 + 23.607) / 60  # obtained by regression R² = 0.8875


    data['pt_cost'] = 3.0
    data.loc[(data['distance']/1000 > 6) & (data['distance']/1000 <= 12), 'pt_cost'] = 4.0
    data.loc[(data['distance'] / 1000 > 6) & (data['distance'] / 1000 <= 12), 'pt_cost'] = 4.0
    data.loc[(data['distance'] / 1000 > 12) & (data['distance'] / 1000 <= 22), 'pt_cost'] = 5.0
    data.loc[(data['distance'] / 1000 > 22) & (data['distance'] / 1000 <= 32), 'pt_cost'] = 6.0
    data.loc[(data['distance'] / 1000 > 32), 'pt_cost'] = 6 + np.ceil((data.loc[(data['distance'] / 1000 > 32), 'distance'] / 1000 - 32) * 1.0)


    data['bike_time'] = data['distance'] / 1000 / 10  # 10 km/h
    data['bike_cost'] = 0.0
    data['walk_time'] = data['distance'] / 1000 / 4.43  #
    data['walk_cost'] = 0.0


    data_len = len(data)
    data = data.merge(tax_df_bj, how = 'cross')
    data = data.loc[(data['income']>data['min_income']) & (data['income']<=data['max_income'])]
    assert len(data) == data_len
    data['income_after_tax'] = data['income'] - (data['income'] - data['min_income']) * data['tax_rate'] - data['cum_tax']


    data['income'] /= 160  # assume 160  hr per month


    print(f"num individuals {len(data)}")
    data.to_csv('../Beijing/mode_choice_sample_updated.csv', index=False)








def estimate_dcm_for_bj():
    data = pd.read_csv('../Beijing/mode_choice_sample_updated.csv')
    ### change units to hr
    # category_var = ['Ethnic', 'Age', 'Citizen', 'Gender', 'Occup']
    category_var = ['age','gender']#['Ethnic','Age'] #['gender','age','hhsize','edu']#['gender','age','income_cat','hhsize']
    continuous_var = ['car_cost', 'car_time', 'pt_cost', 'pt_time', 'bike_cost', 'bike_time', 'walk_cost', 'walk_time']


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
    data, category_var_dummy = add_dummy(data, category_var)


    data_used = data.copy()
    # data_used = data.loc[data['distance'] > 0].copy()
    # data_used['pt_time'] #*= 1.5
    cost_var = ['car_cost', 'pt_cost', 'walk_cost', 'bike_cost']




    avg_hr_income = np.mean(data_used["income"])
    print(f'avg_hr_income: {avg_hr_income}')
    for key in cost_var:
        data_used[key] /= data_used['income']




    av_var = []
    # data_used['car_cost2'] = (data_used['distance'] / 1000) * 1.4 #0.52 is cost per km for car
    # Create a Biogeme database
    database = Database('CommuteMode', data_used[category_var_dummy+continuous_var+av_var+['choice']])


    # Define the choice column
    CHOICE = database.variables['choice']


    # Define social demographic variables (transform categorical variables into dummies)
    var_dict = {}
    for var in category_var_dummy:
        var_dict[var] = database.variables[var]






    # Define cost and time variables
    car_cost = database.variables['car_cost']
    car_time = database.variables['car_time']
    pt_cost = database.variables['pt_cost']
    pt_time = database.variables['pt_time']
    bike_cost = database.variables['bike_cost']
    bike_time = database.variables['bike_time']
    walk_cost = database.variables['walk_cost']
    walk_time = database.variables['walk_time']


    # AV_Car = database.variables['car_avail']
    # AV_Bike = database.variables['bike_avail']


    # Define parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_PT = Beta('ASC_PT', 0, None, None, 0)
    ASC_BIKE = Beta('ASC_BIKE', 0, None, None, 0)
    ASC_WALK = Beta('ASC_WALK', 0, None, None, 1)  # Base mode (reference)


    B_COST = Beta('B_COST', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    # B_CAR_COST = Beta('B_CAR_COST', 0, None, None, 0)
    # B_CAR_TIME = Beta('B_CAR_TIME', 0, None, None, 0)
    #
    # B_PT_COST = Beta('B_PT_COST', 0, None, None, 0)
    # B_PT_TIME = Beta('B_PT_TIME', 0, None, None, 0)
    #
    # B_WALK_COST = Beta('B_WALK_COST', 0, None, None, 0)
    # B_WALK_TIME = Beta('B_WALK_TIME', 0, None, None, 0)

    mode_list = ['car', 'pt', 'bike']

    beta_dict = {}
    for var in category_var_dummy:
        beta_dict[var] = {}
        for mode in mode_list:
            beta_dict[var][mode] = Beta(f'B_{mode.upper()}_{var.upper()}', 0, None, None, 0)

    # Define utility functions
    V_CAR = (
            ASC_CAR
            + B_COST * car_cost
            + B_TIME * car_time
    )
    for var in category_var_dummy:
        V_CAR += beta_dict[var]['car'] * var_dict[var]

    V_PT = (
            ASC_PT
            + B_COST * pt_cost
            + B_TIME * pt_time
    )
    for var in category_var_dummy:
        V_PT += beta_dict[var]['pt'] * var_dict[var]

    V_BIKE = (
            ASC_BIKE
            + B_COST * bike_cost
            + B_TIME * bike_time
    )
    for var in category_var_dummy:
        V_BIKE += beta_dict[var]['bike'] * var_dict[var]

    # base mode, not include social demo
    V_WALK = (
            ASC_WALK
            + B_COST * walk_cost
            + B_TIME * walk_time
    )


    # Associate utility functions with choice options
    V = {
        1: V_CAR,
        2: V_PT,
        3: V_BIKE,
        4: V_WALK,
    }


    # Availability of each mode (assume all are available)
    avail = {1: 1, 2: 1, 3: 1, 4: 1}


    # Logit model
    logprob = models.loglogit(V, avail, CHOICE)


    # Define the Biogeme object


    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'CommuteModeModel'
    biogeme.generateHtml = False
    biogeme.generatePickle = False
    biogeme.saveIterations = False
    # Estimate the parameters
    results = biogeme.estimate()


    # Display results
    print(results.shortSummary())
    est_para = results.getEstimatedParameters()
    print(est_para)
    b_time = est_para.loc['B_TIME']['Value']
    b_cost_div_income = est_para.loc['B_COST']['Value']
    val_to_time = b_time / b_cost_div_income * data_used['income'] # Euro per hour
    print(f'avg value per hr {np.mean(val_to_time)}')
    print(f'max value per hr {np.max(val_to_time)}')
    print(f'min value per hr {np.min(val_to_time)}')
    print(f'avg % vot divided by income {round(np.mean(val_to_time / data_used["income"].values) * 100, 2)}%')
    print(f'max % vot divided by income {round(np.max(val_to_time / data_used["income"].values) * 100, 2)}%')
    print(f'min % vot divided by income {round(np.min(val_to_time / data_used["income"].values) * 100, 2)}%')
    # a=1
    est_para.to_csv('regression_res/dcm_mode_choice_bj.csv',index=True)


    summary_stats = {
        'Log-likelihood': results.data.logLike,
        'Initial Log-likelihood': results.data.initLogLike,
        'Null log-likelihood': results.data.nullLogLike,
        'Likelihood ratio test statistic': results.data.likelihoodRatioTest,
        'Rho-square (R^2)': results.data.rhoSquare,
        'Rho-bar-square': results.data.rhoBarSquare,
        'Number of observations': results.data.numberOfObservations,
        'Number of parameters': results.data.nparam,
        'Akaike Information Criterion (AIC)': results.data.akaike,
        'Bayesian Information Criterion (BIC)': results.data.bayesian
    }

    # Convert summary statistics to a DataFrame
    summary_stats_df = pd.DataFrame(summary_stats.items(), columns=['Metric', 'Value'])
    # Save summary statistics to a CSV file
    summary_stats_df.to_csv('regression_res/dcm_mode_choice_bj_summary.csv', index=True)



def calculate_new_travel_time_and_mode_bj():
    od_matrix = pd.read_csv('../Beijing/processed_od_matrix.csv')
    ind_data = pd.read_csv('../Beijing/mode_choice_sample_updated.csv')
    od_matrix = od_matrix.merge(ind_data[['work_ID', 'gender', 'age','income','income_after_tax']], on = ['work_ID'])


    ### cal time
    od_matrix['car_cost'] = od_matrix['distance'] / 1000 * 1.7 / 1.3  # 1.7 is cost per km for car, 1.3 is avg person per car trip
    od_matrix['car_time'] = od_matrix['distance'] / 1000 / 30  # 30 km/h


    od_matrix['pt_time'] = (2.1931 * od_matrix['distance'] / 1000 + 23.607) / 60  # obtained by regression R² = 0.8875


    od_matrix['pt_cost'] = 3.0
    od_matrix.loc[(od_matrix['distance']/1000 > 6) & (od_matrix['distance']/1000 <= 12), 'pt_cost'] = 4.0
    od_matrix.loc[(od_matrix['distance'] / 1000 > 6) & (od_matrix['distance'] / 1000 <= 12), 'pt_cost'] = 4.0
    od_matrix.loc[(od_matrix['distance'] / 1000 > 12) & (od_matrix['distance'] / 1000 <= 22), 'pt_cost'] = 5.0
    od_matrix.loc[(od_matrix['distance'] / 1000 > 22) & (od_matrix['distance'] / 1000 <= 32), 'pt_cost'] = 6.0
    od_matrix.loc[(od_matrix['distance'] / 1000 > 32), 'pt_cost'] = 6 + np.ceil((od_matrix.loc[(od_matrix['distance'] / 1000 > 32), 'distance'] / 1000 - 32) * 1.0)


    od_matrix['bike_time'] = od_matrix['distance'] / 1000 / 10  # 10 km/h
    od_matrix['bike_cost'] = 0.0
    od_matrix['walk_time'] = od_matrix['distance'] / 1000 / 4.43  #
    od_matrix['walk_cost'] = 0.0


    para = pd.read_csv('regression_res/dcm_mode_choice_bj.csv',index_col=0)


    category_var = ['gender','age'] #['gender','age','hhsize','edu']#['gender','age','income_cat','hhsize']


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




    od_matrix, category_var_dummy = add_dummy(od_matrix, category_var)
    all_modes = ['car','pt','bike','walk']
    base_mode = 'walk'
    for mode in all_modes:
        if mode == base_mode:
            od_matrix[f'V_{mode}'] = 0.0
        else:
            od_matrix[f'V_{mode}'] = para.loc[f'ASC_{mode.upper()}']['Value']
        for key in category_var_dummy:
            if mode != base_mode:
                od_matrix[f'V_{mode}'] += para.loc[f'B_{mode.upper()}_{key.upper()}']['Value'] * od_matrix[key]
        od_matrix[f'V_{mode}'] += para.loc[f'B_TIME']['Value'] * od_matrix[f'{mode}_time']
        od_matrix[f'V_{mode}'] += para.loc[f'B_COST']['Value'] * (od_matrix[f'{mode}_cost'] / od_matrix['income'])


    od_matrix['exp_sum'] = 0.0
    for mode in all_modes:
        if mode == 'car':
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}'])
        else:
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}'])


    for mode in all_modes:
        if mode == 'car':
            od_matrix[f'prob_{mode}'] = np.exp(od_matrix[f'V_{mode}']) / od_matrix['exp_sum']
        else:
            od_matrix[f'prob_{mode}'] = np.exp(od_matrix[f'V_{mode}']) / od_matrix['exp_sum']


    for mode in all_modes:
        print(f'{mode}: avg prob: {np.mean(od_matrix[f"prob_{mode}"])}')


    # weighted avg travel time and cost
    od_matrix['t_ij'] = 0
    for mode in all_modes:
        od_matrix['t_ij'] += od_matrix[f'prob_{mode}'] * od_matrix[f'{mode}_time']


    od_matrix['vot_i'] = para.loc[f'B_TIME']['Value'] / para.loc[f'B_COST']['Value'] * od_matrix['income']
    od_matrix['c_ij'] = 0




    for mode in all_modes:
        od_matrix['c_ij'] += od_matrix[f'prob_{mode}'] * od_matrix[f'{mode}_cost']




    # save data
    prob_col = []
    for mode in all_modes:
        prob_col.append(f'prob_{mode}')
        od_matrix[prob_col] = np.round(od_matrix[prob_col],4)
    saved_col = ['t_ij','c_ij','vot_i','income_after_tax']
    for col in saved_col:
        od_matrix[col] = np.round(od_matrix[col],4)
    od_matrix[['work_ID', 'home_ID', 'distance']+prob_col+saved_col].to_parquet('../Beijing/OD_matrix_with_info.parquet',index=False)


def process_data_sg_ind():

    ind_data = pd.read_csv('../Singapore_ind/processed_individual_data.csv')
    survey_data = pd.read_csv("../Singapore_ind/HITS 2012 Database.csv", encoding='latin1')
    data = ind_data.merge(survey_data, on=['HP_ID'])
    data['car_avail'] = 0
    data['bike_avail'] = 0
    data.loc[data['H5_VehAvailable'] == 'Yes', 'car_avail'] = 1
    data.loc[data['H7_Bike'] == 'Yes', 'bike_avail'] = 1
    data = data.loc[data['T6_Purpose'].isin(
        ['Working for paid employment', 'Work Related Trip', 'Work-related (meetings, sales etc)'])]
    data['travel_mode'] = data['T10_Mode']
    data.loc[data['travel_mode'].isna(), 'travel_mode'] = data.loc[data['travel_mode'].isna(), 'T6cTripMode']

    all_modes = list(set(data['travel_mode']))
    mode_map = {
        1: ['Van / Lorry driver', 'Taxi', 'Motorcycle rider', 'Car driver', 'Car passenger', 'Motorcycle passenger',
            'Van / Lorry passenger'],
        2: ['Public bus', 'MRT', 'Company bus', 'LRT', 'School bus', 'Shuttle bus'],
        3: ['Cycle'],
        4: ['Walk Only']}
    data['choice'] = -1
    for mode_id, mode_list in mode_map.items():
        data.loc[data['travel_mode'].isin(mode_list), 'choice'] = mode_id
    # test = data.loc[data['choice'] == -1]
    most_frequent = data['choice'].mode()[0]
    data.loc[~data['choice'].isin([1, 2, 3, 4]), 'choice'] = most_frequent  # fill by most frequent
    data['choice'] = data['choice'].astype(int)
    # data['walk_only_time'] = data['']
    data['trip_time'] = (
            data['T10a_WalkTime'].fillna(0) + data['T13_WaitTime'].fillna(0) + data['T14_InVehTime'].fillna(0) +
            data['T22_LastWalkTime'].fillna(0))
    data = data[['HP_ID', 'car_avail', 'bike_avail', 'choice', 'trip_time']].drop_duplicates()
    data['num_trips'] = data.groupby(['HP_ID'])['choice'].transform('count')
    data['total_trip_time'] = data.groupby(['HP_ID'])['trip_time'].transform('sum')
    data['total_trip_time'] /= 60  # to hr
    mode_priority = [2, 1, 3, 4]  # multi trips, select by the above sequence
    data['priority'] = 999
    for idx, mode in enumerate(mode_priority):
        data.loc[data['choice'] == mode, 'priority'] = idx
    data = data.sort_values(['priority'])
    data = data.groupby(['HP_ID']).first().reset_index()
    data = data.drop(columns=['num_trips', 'priority', 'trip_time'])
    print(len(data))
    data = ind_data.merge(data, on=['HP_ID'], how='left')
    print(len(data))
    miss_info = data.loc[data['choice'].isna()]
    print(f'% missing: {len(miss_info) / len(data)}')

    most_frequent = data['choice'].mode()[0]
    data.loc[~data['choice'].isin([1, 2, 3, 4]), 'choice'] = most_frequent  # fill by most frequent
    data['choice'] = data['choice'].astype(int)

    for mode in range(1, 5):
        print(f'{mode} prob: {sum(data["choice"] == mode) / len(data)}')

    # fix inconsistency
    data.loc[data['choice'] == 1, 'car_avail'] = 1
    data['car_avail'].fillna(0, inplace=True)
    data.loc[data['choice'] == 3, 'bike_avail'] = 1
    data['bike_avail'].fillna(0, inplace=True)


    ind_category_var = ['Ethnic', 'Age', 'Citizen', 'Gender', 'Occup', 'income']
    all_choices = set(data['choice'])

    def fill_by_most_freq(data, col):
        most_frequent = data[col].mode()[0]
        data.loc[data[col].isna(), col] = most_frequent  # fill by most frequent
        return data

    #
    for key in ind_category_var:
        data = fill_by_most_freq(data, key)

    time_correction_factor = {}

    od_matrix = pd.read_parquet('../Singapore_ind/processed_od_matrix.parquet')
    data = data.merge(od_matrix[['work_ID', 'home_ID', 'distance']], on=['work_ID', 'home_ID'])
    data['car_cost'] = data[
                           'distance'] / 1000 * 0.93 / 1.75  # 1.13 is cost per km for car, 1.75 is avg person per car trip
    data['car_time'] = data['distance'] / 1000 / 45  # 45 km/h
    correction_factor = sum(data.loc[
                                (data['choice'] == 1) & (~data['total_trip_time'].isna()) & (
                                            data['total_trip_time'] > 0), 'total_trip_time']) / sum(data.loc[
                                                                                                        (data[
                                                                                                             'choice'] == 1) & (
                                                                                                            ~data[
                                                                                                                'total_trip_time'].isna()) & (
                                                                                                                    data[
                                                                                                                        'total_trip_time'] > 0), 'car_time'])

    time_correction_factor['car'] = correction_factor

    data.loc[
        (data['choice'] == 1) & (~data['total_trip_time'].isna()) & (data['total_trip_time'] > 0), 'car_time'] = \
    data.loc[
        (data['choice'] == 1) & (~data['total_trip_time'].isna()) & (
                    data['total_trip_time'] > 0), 'total_trip_time']
    #

    data['pt_time'] = (0.1898 * data['distance'] + 775.91) / 3600  # obtained by regression R² = 0.8875
    correction_factor = sum(data.loc[
                                (data['choice'] == 2) & (~data['total_trip_time'].isna()) & (
                                            data['total_trip_time'] > 0), 'total_trip_time']) / sum(data.loc[
                                                                                                        (data[
                                                                                                             'choice'] == 2) & (
                                                                                                            ~data[
                                                                                                                'total_trip_time'].isna()) & (
                                                                                                                    data[
                                                                                                                        'total_trip_time'] > 0), 'pt_time'])
    time_correction_factor['pt'] = correction_factor

    data.loc[
        (data['choice'] == 2) & (~data['total_trip_time'].isna()) & (data['total_trip_time'] > 0), 'pt_time'] = \
    data.loc[
        (data['choice'] == 2) & (~data['total_trip_time'].isna()) & (
                    data['total_trip_time'] > 0), 'total_trip_time']



    data['pt_cost'] = -1e-09 * data['distance'] ** 2 + 7e-05 * data['distance'] + 0.8546  # (R² = 0.9842)

    data['bike_time'] = data['distance'] / 1000 / 20  # 20 km/h

    correction_factor = sum(data.loc[
                                (data['choice'] == 3) & (~data['total_trip_time'].isna()) & (
                                            data['total_trip_time'] > 0), 'total_trip_time']) / sum(data.loc[
                                                                                                        (data[
                                                                                                             'choice'] == 3) & (
                                                                                                            ~data[
                                                                                                                'total_trip_time'].isna()) & (
                                                                                                                    data[
                                                                                                                        'total_trip_time'] > 0), 'bike_time'])
    data['bike_time'] *= correction_factor
    time_correction_factor['bike'] = correction_factor

    data.loc[
        (data['choice'] == 3) & (~data['total_trip_time'].isna()) & (data['total_trip_time'] > 0), 'bike_time'] = \
    data.loc[
        (data['choice'] == 3) & (~data['total_trip_time'].isna()) & (
                    data['total_trip_time'] > 0), 'total_trip_time']
    data['bike_cost'] = 0.0

    data['walk_time'] = data['distance'] / 1000 / 4.43  #

    # no samples for walk time
    correction_factor = 1.0
    # data['walk_time'] *= correction_factor

    time_correction_factor['walk'] = correction_factor

    data['walk_cost'] = 0.0

    data_len = len(data)
    data = data.merge(tax_df_sg, how = 'cross')
    data = data.loc[(data['income']>data['min_income']) & (data['income']<=data['max_income'])]
    assert len(data) == data_len
    data['income_after_tax'] = data['income'] - (data['income'] - data['min_income']) * data['tax_rate'] - data['cum_tax']


    data['income'] /= 160  # assume 160  hr per month

    print(f"num individuals {len(data)}")

    # age_mapping = {
    #     '15-19 yrs old': '15-34 yrs',
    #     '20-24 yrs old': '15-34 yrs',
    #     '25-29 yrs old': '15-34 yrs',
    #     '30-34 yrs old': '15-34 yrs',
    #     '35-39 yrs old': '35-54 yrs',
    #     '40-44 yrs old': '35-54 yrs',
    #     '45-49 yrs old': '35-54 yrs',
    #     '50-54 yrs old': '35-54 yrs',
    #     '55-59 yrs old': '55-74 yrs',
    #     '60-64 yrs old': '55-74 yrs',
    #     '65-69 yrs old': '55-74 yrs',
    #     '70-74 yrs old': '55-74 yrs',
    #     '75-79 yrs old': '75 yrs & above',
    #     '80-84 yrs old': '75 yrs & above',
    #     '85 yrs & above': '75 yrs & above'
    # }
    # data['Age'] = data['Age'].apply(lambda x: age_mapping[x])

    data.to_csv('../Singapore_ind/mode_choice_sample_updated.csv', index=False)
    correction_factor_df = pd.DataFrame(time_correction_factor.items(), columns=['mode', 'time_correction_factor'])
    correction_factor_df.to_csv('../Singapore_ind/time_correction_factor.csv', index=False)


def estimate_dcm_for_sg_ind():
    data = pd.read_csv('../Singapore_ind/mode_choice_sample_updated.csv')

    # print(data['Age'].unique())
    ### change units to hr
    # category_var = ['Ethnic', 'Age', 'Citizen', 'Gender', 'Occup']
    category_var = ['Age',
                    'Gender']  # ['Ethnic','Age'] #['gender','age','hhsize','edu']#['gender','age','income_cat','hhsize']
    continuous_var = ['car_cost', 'car_time', 'pt_cost', 'pt_time', 'bike_cost', 'bike_time', 'walk_cost', 'walk_time']

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
    data, category_var_dummy = add_dummy(data, category_var)

    data_used = data.copy()
    # data_used = data.loc[data['distance'] > 0].copy()
    # data_used['pt_time'] #*= 1.5
    cost_var = ['car_cost', 'pt_cost', 'walk_cost', 'bike_cost']

    avg_hr_income = np.mean(data_used["income"])
    print(f'avg_hr_income: {avg_hr_income}')
    for key in cost_var:
        data_used[key] /= data_used['income']

    av_var = ['car_avail', 'bike_avail']
    # data_used['car_cost2'] = (data_used['distance'] / 1000) * 1.4 #0.52 is cost per km for car
    # Create a Biogeme database
    database = Database('CommuteMode', data_used[category_var_dummy + continuous_var + av_var + ['choice']])

    # Define the choice column
    CHOICE = database.variables['choice']

    # Define social demographic variables (transform categorical variables into dummies)
    var_dict = {}
    for var in category_var_dummy:
        var_dict[var] = database.variables[var]

    # Define cost and time variables
    car_cost = database.variables['car_cost']
    car_time = database.variables['car_time']
    pt_cost = database.variables['pt_cost']
    pt_time = database.variables['pt_time']
    bike_cost = database.variables['bike_cost']
    bike_time = database.variables['bike_time']
    walk_cost = database.variables['walk_cost']
    walk_time = database.variables['walk_time']

    AV_Car = database.variables['car_avail']
    AV_Bike = database.variables['bike_avail']

    # Define parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_PT = Beta('ASC_PT', 0, None, None, 0)
    ASC_BIKE = Beta('ASC_BIKE', 0, None, None, 0)
    ASC_WALK = Beta('ASC_WALK', 0, None, None, 1)  # Base mode (reference)

    B_COST = Beta('B_COST', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    # B_CAR_COST = Beta('B_CAR_COST', 0, None, None, 0)
    # B_CAR_TIME = Beta('B_CAR_TIME', 0, None, None, 0)
    #
    # B_PT_COST = Beta('B_PT_COST', 0, None, None, 0)
    # B_PT_TIME = Beta('B_PT_TIME', 0, None, None, 0)
    #
    # B_WALK_COST = Beta('B_WALK_COST', 0, None, None, 0)
    # B_WALK_TIME = Beta('B_WALK_TIME', 0, None, None, 0)

    mode_list = ['car','pt','bike']

    beta_dict = {}
    for var in category_var_dummy:
        beta_dict[var] = {}
        for mode in mode_list:
            beta_dict[var][mode] = Beta(f'B_{mode.upper()}_{var.upper()}', 0, None, None, 0)

    # Define utility functions
    V_CAR = (
            ASC_CAR
            + B_COST * car_cost
            + B_TIME * car_time
    )
    for var in category_var_dummy:
        V_CAR += beta_dict[var]['car'] * var_dict[var]

    V_PT = (
            ASC_PT
            + B_COST * pt_cost
            + B_TIME * pt_time
    )
    for var in category_var_dummy:
        V_PT += beta_dict[var]['pt'] * var_dict[var]

    V_BIKE = (
            ASC_BIKE
            + B_COST * bike_cost
            + B_TIME * bike_time
    )
    for var in category_var_dummy:
        V_BIKE += beta_dict[var]['bike'] * var_dict[var]

    # base mode, not include social demo
    V_WALK = (
            ASC_WALK
            + B_COST * walk_cost
            + B_TIME * walk_time
    )

    # Associate utility functions with choice options
    V = {
        1: V_CAR,
        2: V_PT,
        3: V_BIKE,
        4: V_WALK,
    }

    # Availability of each mode (assume all are available)
    avail = {1: AV_Car, 2: 1, 3: AV_Bike, 4: 1}

    # Logit model
    logprob = models.loglogit(V, avail, CHOICE)

    # Define the Biogeme object

    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'CommuteModeModel'
    biogeme.generateHtml = False
    biogeme.generatePickle = False
    biogeme.saveIterations = False
    # Estimate the parameters
    results = biogeme.estimate()

    # Display results


    print(results.shortSummary())
    est_para = results.getEstimatedParameters()
    print(est_para)
    b_time = est_para.loc['B_TIME']['Value']
    b_cost_div_income = est_para.loc['B_COST']['Value']
    val_to_time = b_time / b_cost_div_income * data_used['income']  # Euro per hour
    print(f'avg value per hr {np.mean(val_to_time)}')
    print(f'max value per hr {np.max(val_to_time)}')
    print(f'min value per hr {np.min(val_to_time)}')
    print(f'avg % vot divided by income {round(np.mean(val_to_time / data_used["income"].values) * 100, 2)}%')
    print(f'max % vot divided by income {round(np.max(val_to_time / data_used["income"].values) * 100, 2)}%')
    print(f'min % vot divided by income {round(np.min(val_to_time / data_used["income"].values) * 100, 2)}%')
    # a=1
    est_para.to_csv('regression_res/dcm_mode_choice_sg_ind.csv', index=True)


    summary_stats = {
        'Log-likelihood': results.data.logLike,
        'Initial Log-likelihood': results.data.initLogLike,
        'Null log-likelihood': results.data.nullLogLike,
        'Likelihood ratio test statistic': results.data.likelihoodRatioTest,
        'Rho-square (R^2)': results.data.rhoSquare,
        'Rho-bar-square': results.data.rhoBarSquare,
        'Number of observations': results.data.numberOfObservations,
        'Number of parameters': results.data.nparam,
        'Akaike Information Criterion (AIC)': results.data.akaike,
        'Bayesian Information Criterion (BIC)': results.data.bayesian
    }

    # Convert summary statistics to a DataFrame
    summary_stats_df = pd.DataFrame(summary_stats.items(), columns=['Metric', 'Value'])
    # Save summary statistics to a CSV file
    summary_stats_df.to_csv('regression_res/dcm_mode_choice_sg_ind_summary.csv', index=True)

def calculate_new_travel_time_and_mode_sg_ind():
    od_matrix = pd.read_parquet('../Singapore_ind/processed_od_matrix.parquet')[['home_ID', 'work_ID', 'distance']]
    correction_factor_df = pd.read_csv('../Singapore_ind/time_correction_factor.csv')

    od_matrix['car_cost'] = od_matrix[
                                'distance'] / 1000 * 0.93 / 1.75  # 1.13 is cost per km for car, 1.3 is avg person per car trip
    od_matrix['car_time'] = od_matrix['distance'] / 1000 / 45  #
    correction_factor = correction_factor_df.loc[correction_factor_df['mode'] == 'car']['time_correction_factor'].iloc[
        0]
    # od_matrix['car_time'] *= correction_factor

    od_matrix['pt_time'] = (0.1898 * od_matrix['distance'] + 775.91) / 3600  #
    correction_factor = correction_factor_df.loc[correction_factor_df['mode'] == 'pt']['time_correction_factor'].iloc[0]
    # od_matrix['pt_time'] *= correction_factor
    od_matrix['pt_cost'] = -1e-09 * od_matrix['distance'] ** 2 + 7e-05 * od_matrix['distance'] + 0.8546

    od_matrix['bike_time'] = od_matrix['distance'] / 1000 / 20  # 20 km/h
    correction_factor = correction_factor_df.loc[correction_factor_df['mode'] == 'bike']['time_correction_factor'].iloc[
        0]
    od_matrix['bike_time'] *= correction_factor
    od_matrix['bike_cost'] = 0.0

    od_matrix['walk_time'] = od_matrix['distance'] / 1000 / 4.43  #
    correction_factor = correction_factor_df.loc[correction_factor_df['mode'] == 'walk']['time_correction_factor'].iloc[
        0]
    od_matrix['walk_time'] *= correction_factor
    od_matrix['walk_cost'] = 0.0

    print('finish calculate time cost...')
    ind_data = pd.read_csv('../Singapore_ind/mode_choice_sample_updated.csv')
    od_matrix = od_matrix.merge(
        ind_data[['work_ID', 'Gender', 'Age', 'car_avail', 'bike_avail', 'income','income_after_tax']], on=['work_ID'])

    print('finish join home with individual...')

    para = pd.read_csv('regression_res/dcm_mode_choice_sg_ind.csv', index_col=0)
    category_var = ['Gender', 'Age']  # ['gender','age','hhsize','edu']#['gender','age','income_cat','hhsize']

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

    od_matrix, category_var_dummy = add_dummy(od_matrix, category_var)
    all_modes = ['car', 'pt', 'bike', 'walk']
    base_mode = 'walk'
    for mode in all_modes:
        if mode == base_mode:
            od_matrix[f'V_{mode}'] = 0.0
        else:
            od_matrix[f'V_{mode}'] = para.loc[f'ASC_{mode.upper()}']['Value']
        for key in category_var_dummy:
            if mode != base_mode:
                od_matrix[f'V_{mode}'] += para.loc[f'B_{mode.upper()}_{key.upper()}']['Value'] * od_matrix[key]
        od_matrix[f'V_{mode}'] += para.loc[f'B_TIME']['Value'] * od_matrix[f'{mode}_time']
        od_matrix[f'V_{mode}'] += para.loc[f'B_COST']['Value'] * (od_matrix[f'{mode}_cost'] / od_matrix['income'])

    od_matrix['exp_sum'] = 0.0
    for mode in all_modes:
        if mode == 'car':
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}']) * od_matrix[f'car_avail']
        elif mode == 'bike':
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}']) * od_matrix[f'bike_avail']
        else:
            od_matrix['exp_sum'] += np.exp(od_matrix[f'V_{mode}'])

    for mode in all_modes:
        if mode == 'car':
            od_matrix[f'prob_{mode}'] = np.exp(od_matrix[f'V_{mode}']) * od_matrix[f'car_avail'] / od_matrix['exp_sum']
        else:
            od_matrix[f'prob_{mode}'] = np.exp(od_matrix[f'V_{mode}']) / od_matrix['exp_sum']

    for mode in all_modes:
        print(f'{mode}: avg prob: {np.mean(od_matrix[f"prob_{mode}"])}')

    # weighted avg travel time and cost
    od_matrix['t_ij'] = 0
    for mode in all_modes:
        od_matrix['t_ij'] += od_matrix[f'prob_{mode}'] * od_matrix[f'{mode}_time']

    od_matrix['vot_i'] = para.loc[f'B_TIME']['Value'] / para.loc[f'B_COST']['Value'] * od_matrix['income']
    od_matrix['c_ij'] = 0

    for mode in all_modes:
        od_matrix['c_ij'] += od_matrix[f'prob_{mode}'] * od_matrix[f'{mode}_cost']

    # save data
    prob_col = []
    for mode in all_modes:
        prob_col.append(f'prob_{mode}')
        od_matrix[prob_col] = np.round(od_matrix[prob_col], 4)
    saved_col = ['t_ij', 'c_ij', 'vot_i','income_after_tax']
    for col in saved_col:
        od_matrix[col] = np.round(od_matrix[col], 4)
    od_matrix[['work_ID', 'home_ID', 'distance'] + prob_col + saved_col].to_parquet(
        '../Singapore_ind/OD_matrix_with_info.parquet', index=False)


if __name__ == '__main__':
    # ######### Munich
    # process_data_munich()
    # ###
    # estimate_dcm_for_munich()
    # ##
    # calculate_new_travel_time_and_mode_munich()
    # #
    # # ######### Singapore
    # process_data_sg()
    # estimate_dcm_for_sg()
    # calculate_new_travel_time_and_mode_sg()
    #
    # ########## Beijing
    # process_data_bj()
    # ###
    # estimate_dcm_for_bj()
    # # ##
    # calculate_new_travel_time_and_mode_bj()

    ########## Singapore_Ind
    process_data_sg_ind()
    ###
    estimate_dcm_for_sg_ind()
    ##
    # calculate_new_travel_time_and_mode_sg_ind()