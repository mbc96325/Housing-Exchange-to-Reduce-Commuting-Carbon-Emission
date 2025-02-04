from ortools.linear_solver import pywraplp
import pandas as pd
import os
import time

def solve_optimal_targeting(data, participation_threshold, scenario, city_folder):

    data['obj_weight'] = data['distance'] # minimize distance
    # Map work_ID and home_ID to contiguous integers
    data['work_ID'] = data['work_ID'].astype(int)
    data['home_ID'] = data['home_ID'].astype(int)

    # Extract unique IDs
    work_ids = data['work_ID'].unique()
    home_ids = data['home_ID'].unique()
    n = len(work_ids)  # Assuming work_ID_map and home_ID_map have the same length
    assert len(work_ids) == len(home_ids)
    # # Set participation threshold (e.g., 10%)
    # participation_threshold = 0.1

    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP') # CBC, SCIP, # SAT
    if not solver:
        raise Exception("Solver not created.")

    s_time = time.time()
    # Decision variables: x_ij (binary)
    x = {}
    for i, j in zip(data['work_ID'], data['home_ID']):
        x[i, j] = solver.BoolVar(f'x_{i}_{j}')
    print(f'finish add var, time {time.time() - s_time}')
    # Objective: minimize sum x_ij * obj_weight_ij

    s_time = time.time()
    objective = solver.Objective()
    for i, j, weight in zip(data['work_ID'], data['home_ID'], data['obj_weight']):
        objective.SetCoefficient(x[i, j], weight)
    objective.SetMinimization()
    print(f'finish add obj, time {time.time() - s_time}')

    s_time = time.time()
    # Constraint: Each work_ID_map is matched to exactly one home_ID_map
    for i in work_ids:
        solver.Add(solver.Sum([x[i, j] for j in home_ids if (i, j) in x]) == 1)
    # Constraint: Each home_ID_map is matched to exactly one work_ID_map
    for j in home_ids:
        solver.Add(solver.Sum([x[i, j] for i in work_ids if (i, j) in x]) == 1)
    print(f'finish add constraints 1 ID, time {time.time() - s_time}')

    s_time = time.time()
    # Constraint: At least x% of people do not participate in matching
    not_matching_count = solver.Sum([x[i, i] for i in work_ids if (i, i) in x])
    solver.Add(not_matching_count >= n * (1 - participation_threshold))
    print(f'finish add constraints participants, time {time.time() - s_time}')

    # Solve the problem
    s_time = time.time()
    print('start solving')
    status = solver.Solve()
    print(f'finish SOLVING, time {time.time() - s_time}')

    matched_res_dict = {'work_ID': [], 'home_ID': []}
    if status == pywraplp.Solver.OPTIMAL:
        s_time = time.time()
        for i, j in x:
            if x[i, j].solution_value() > 0.5:  # Binary variable threshold
                matched_res_dict['work_ID'].append(i)
                matched_res_dict['home_ID'].append(j)
        print(f'finish get solutions, time {time.time() - s_time}')
    else:
        print("The problem does not have an optimal solution.")
    matched_res = pd.DataFrame(matched_res_dict)
    matched_res.to_csv(f'../{city_folder}/matched_res_{scenario}.csv',index=False)


if __name__ == '__main__':
    city_folder_list = ['Munich', 'Beijing', 'Singapore_ind']
    refer_scenario_name = '3_0_realistic' # refer
    step = 2
    optimal_percent = list(range(1, 17+step, step))
    participation_rate_list = [rate / 100 for rate in optimal_percent]
    skip_existing = True
    for city_folder in city_folder_list:
        for participation_rate in participation_rate_list:
            scenario = f'4_0_optimal_{int(participation_rate*100)}_percent'
            print(f'########{city_folder} {scenario}#############')
            # Load your data
            if skip_existing and os.path.exists(f'../{city_folder}/matched_res_{scenario}.csv'):
                print(f'{scenario} exist, skip...')
            else:
                ### read data
                data = pd.read_parquet(f'../{city_folder}/final_pair_for_matching_{refer_scenario_name}.parquet')
                solve_optimal_targeting(data, participation_rate, scenario, city_folder)