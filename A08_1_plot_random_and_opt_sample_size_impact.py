import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns



def plot_legend(res_stats, save_fig, save_name, city_folder, refer_scenario_name, percent_scenario_prefix):

    refer = res_stats.loc[res_stats['scenario'] == refer_scenario_name].copy()
    refer['scenario'] = f'{percent_scenario_prefix}random_100_percent'
    res_stats = pd.concat([res_stats, refer])
    refer['scenario'] = f'{percent_scenario_prefix}optimal_100_percent'
    res_stats = pd.concat([res_stats, refer])

    random_data = res_stats.loc[(res_stats['scenario'].str.contains('random_') & (res_stats['attributes'] == 'distance'))].copy()
    total_num_inds = res_stats.loc[(res_stats['scenario'].str.contains('random_') & (res_stats['attributes'] == 'num_inds'))].copy()
    total_num_inds['num_samples'] = total_num_inds['old']
    random_data = random_data.merge(total_num_inds[['scenario','num_samples']], on=['scenario'])
    random_data['sample_percent'] = random_data['scenario'].apply(lambda x: int(x.split('_')[1]) / 100)
    random_data = random_data.sort_values(['sample_percent'])
    random_data['sample_size'] = random_data['num_samples'] * random_data['sample_percent']
    random_data['avg_dis_reduction'] = (random_data['old'] - random_data['new']) / random_data['sample_size']


    total_Pareto_cost_reduc = -random_data['diff_percent'].values * 100
    Pareto_cost_reduc = random_data['avg_dis_reduction'].values

    optimal_data = res_stats.loc[(res_stats['scenario'].str.contains('optimal_') & (res_stats['attributes'] == 'distance'))].copy()
    total_num_inds = res_stats.loc[(res_stats['scenario'].str.contains('optimal_') & (res_stats['attributes'] == 'num_inds'))].copy()
    total_num_inds['num_samples'] = total_num_inds['old']
    optimal_data = optimal_data.merge(total_num_inds[['scenario','num_samples']], on=['scenario'])
    optimal_data['sample_percent'] = optimal_data['scenario'].apply(lambda x: int(x.split('_')[1]) / 100)
    optimal_data = optimal_data.sort_values(['sample_percent'])
    optimal_data['sample_size'] = optimal_data['num_samples'] * optimal_data['sample_percent']
    optimal_data['avg_dis_reduction'] = (optimal_data['old'] - optimal_data['new']) / optimal_data['sample_size']

    total_Pareto_cost_reduc_opt = -optimal_data['diff_percent'].values * 100
    Pareto_cost_reduc_opt = optimal_data['avg_dis_reduction'].values


    font_size = 25
    matplotlib.rcParams['font.size'] #= font_size - 2
    N = len(optimal_data)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4  # the width of the bars
    scale_para = 1.2
    fig, ax = plt.subplots(figsize=(22,5))#figsize=(6.8*scale_para, 5.7*scale_para))

    x_lim_l = -0.5*width - 0.1
    x_lim_r = ind[-1] + width + 2*width + width
    # x_base = [x_lim_l,x_lim_r]
    # y_base = [sum_cost_old/1000, sum_cost_old/1000]
    # line = ax.plot(x_base,y_base, '--',color = 'gray', linewidth = 2.0,label = 'Status quo')
    # rects1 = ax.bar(ind, total_system_cost_reduc, width, color=colors[1], label = 'SO (Random)',alpha = 0.5)
    # rects11 = ax.bar(ind+ width, total_system_cost_reduc_opt, width, color=colors[1], label = 'SO (Total)',alpha = 0.8)
    rects2 = ax.bar(ind + width, total_Pareto_cost_reduc, width, color=colors[0],label = 'Random, Total',alpha = 0.75)
    rects22 = ax.bar(ind + 2*width, total_Pareto_cost_reduc_opt, width, color=colors[1],label = 'Purpose, Total',alpha = 0.75)

    #

    y_max = max(total_Pareto_cost_reduc)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # p1 = ax2.plot(ind+2*width, system_cost_reduc, marker = 's', markersize = 8, color=colors[1], label = '',alpha = 0.5)
    # p11 = ax2.plot(ind+1*width, system_cost_reduc_opt, color=colors[1], label = 'SO (Avg)')
    # ax2.scatter(ind+1*width, system_cost_reduc_opt, s=45, c= colors[1], edgecolor='black', linewidth=1, marker = 's')
    p2 = ax2.plot(ind + 1*width, Pareto_cost_reduc, color=colors[0], label = 'Random, Avg')
    ax2.scatter(ind + 1*width, Pareto_cost_reduc, s=45, c=colors[0], edgecolor='black', linewidth=1, marker='s')

    p22 = ax2.plot(ind +2*width, Pareto_cost_reduc_opt, color=colors[1], label = 'Purpose, Avg')
    ax2.scatter(ind + 2*width, Pareto_cost_reduc_opt, s=45, c=colors[1], edgecolor='black', linewidth=1, marker='s')
    if city_folder == 'Munich':
        y_ticks = [0, 10, 20, 30, 40]
        y_tick_label = [f'{rate}%' for rate in y_ticks]
    else:
        y_ticks = np.round(ax.get_yticks(), 2)
    ax.set_yticks(y_ticks, y_tick_label, fontsize=font_size)
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=font_size)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.yaxis.major.formatter._useMathText = True
    # ax2.yaxis.major.formatter._useMathText = True
    plt.xlim(x_lim_l,x_lim_r)
    # ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])
    ax.set_ylabel('Total reduction (%)', fontsize=font_size)
    ax2.set_ylabel('Avg reduction (km)', fontsize=font_size)
    ax.set_xlabel('Participation rate', fontsize=font_size)
    x_ticks_rate = [10, 40, 70, 100]
    x_ticks = np.array([ind[int((ele-10) / 10)] for ele in x_ticks_rate])
    x_ticks_list = [f'{rate}%' for rate in x_ticks_rate]
    ax.set_xticks(x_ticks + 1.5* width, x_ticks_list, fontsize=font_size)
    # ax.set_xticklabels(x_ticks + 1.5* width, x_ticks_list, fontsize=font_size)


    ax.set_ylim([0,100])
    ax2.set_ylim([0, 100])
    lns = [rects2] + [rects22] + p2 + p22
    labs = [l.get_label() for l in lns]

    ax.legend(lns, labs, fontsize=font_size - 2, loc = 'upper right', ncol = 4)

    # base_cost = y_base[0]
    # plt.ylim(0, 6)

    # add words
    # for idx in range(len(ind)):
    #     reduction_sys = max((base_cost - system_cost[idx]) / base_cost, 0)
    #     x_word = ind[idx] - 0.7*width
    #     y_word = system_cost[idx] + base_cost*0.03
    #     plt.text(x_word,y_word, str(round(reduction_sys*100, 1)) + '%', fontsize = font_size - 3)
    #     reduction_Pare = max((base_cost - Pareto_cost[idx]) / base_cost,0) # avoid Precision errors
    #     x_word = ind[idx] + width - 0.3*width
    #     y_word = Pareto_cost[idx] + base_cost*0.03
    #     plt.text(x_word,y_word, str(round(reduction_Pare*100, 1)) + '%',fontsize = font_size - 3)
    #     a=1
    #
    # pos_x = int(np.round(len(ind)/2))
    # ax.text(ind[pos_x], y_max * 1.1, text_city[city_folder], size=font_size*1.2,
    #          ha="center", va="center",
    #          bbox=dict(boxstyle="round",
    #                    ec=(1., 0.5, 0.5),
    #                    fc=(1., 0.8, 0.8),
    #                    )
    #          )

    # plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/' + save_name + '.jpg', dpi=300)

def plot_random_sample_size_impact(res_stats, save_fig, save_name, city_folder, refer_scenario_name, percent_scenario_prefix):

    refer = res_stats.loc[res_stats['scenario'] == refer_scenario_name].copy()
    refer['scenario'] = f'{percent_scenario_prefix}random_100_percent'
    res_stats = pd.concat([res_stats, refer])
    refer['scenario'] = f'{percent_scenario_prefix}optimal_100_percent'
    res_stats = pd.concat([res_stats, refer])

    measure_value = 'CO2_ij'
    unit_scale = 1000 * 1000
    time_span_scaler = 52*5*2
    # if measure_value == 'CO2_ij':
    #     unit_scale = 1

    random_data = res_stats.loc[(res_stats['scenario'].str.contains('random_') & (res_stats['attributes'] == measure_value))].copy()
    total_num_inds = res_stats.loc[(res_stats['scenario'].str.contains('random_') & (res_stats['attributes'] == 'num_inds'))].copy()
    total_num_inds['num_samples'] = total_num_inds['old']
    random_data = random_data.merge(total_num_inds[['scenario','num_samples']], on=['scenario'])
    random_data['sample_percent'] = random_data['scenario'].apply(lambda x: int(x.split('random_')[1].split('_percent')[0]) / 100)
    random_data = random_data.sort_values(['sample_percent'])
    random_data['sample_size'] = random_data['num_samples'] * random_data['sample_percent']
    random_data['avg_dis_reduction'] = (random_data['old'] - random_data['new']) / random_data['sample_size']


    dis_reduction_percent_random = -random_data['diff_percent'].values * 100
    avg_dis_reduction_random = random_data['avg_dis_reduction'].values / unit_scale * time_span_scaler

    optimal_data = res_stats.loc[(res_stats['scenario'].str.contains('optimal_') & (res_stats['attributes'] == measure_value))].copy()
    total_num_inds = res_stats.loc[(res_stats['scenario'].str.contains('optimal_') & (res_stats['attributes'] == 'num_inds'))].copy()
    total_num_inds['num_samples'] = total_num_inds['old']
    optimal_data = optimal_data.merge(total_num_inds[['scenario','num_samples']], on=['scenario'])
    optimal_data['sample_percent'] = optimal_data['scenario'].apply(lambda x: int(x.split('optimal_')[1].split('_percent')[0]) / 100)
    optimal_data = optimal_data.sort_values(['sample_percent'])
    optimal_data['sample_size'] = optimal_data['num_samples'] * optimal_data['sample_percent']
    optimal_data['avg_dis_reduction'] = (optimal_data['old'] - optimal_data['new']) / optimal_data['sample_size']

    dis_reduction_percent_opt = -optimal_data['diff_percent'].values * 100
    avg_dis_reduction_opt = optimal_data['avg_dis_reduction'].values / unit_scale * time_span_scaler


    font_size = 25
    matplotlib.rcParams['font.size'] #= font_size - 2
    N = len(optimal_data)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4  # the width of the bars
    scale_para = 1.2
    fig, ax = plt.subplots(figsize=(6.8*scale_para, 5.7*scale_para))

    wid_scaler = 1.5

    x_lim_l = -0.5*width - 0.1
    x_lim_r = ind[-1] + width + 2*width + width
    # x_base = [x_lim_l,x_lim_r]
    # y_base = [sum_cost_old/1000, sum_cost_old/1000]
    # line = ax.plot(x_base,y_base, '--',color = 'gray', linewidth = 2.0,label = 'Status quo')
    # rects1 = ax.bar(ind, total_system_cost_reduc, width, color=colors[1], label = 'SO (Random)',alpha = 0.5)
    # rects11 = ax.bar(ind+ width, total_system_cost_reduc_opt, width, color=colors[1], label = 'SO (Total)',alpha = 0.8)
    rects2 = ax.bar(ind + wid_scaler*width, dis_reduction_percent_random, width, color=colors[0],label = 'Random, Total',alpha = 0.75)
    # rects22 = ax.bar(ind + 2*width, dis_reduction_percent_opt, width, color=colors[1],label = 'Selected, Total',alpha = 0.75)

    #

    y_max = max(dis_reduction_percent_random)
    y_max2 = max(avg_dis_reduction_random)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # p1 = ax2.plot(ind+2*width, system_cost_reduc, marker = 's', markersize = 8, color=colors[1], label = '',alpha = 0.5)
    # p11 = ax2.plot(ind+1*width, system_cost_reduc_opt, color=colors[1], label = 'SO (Avg)')
    # ax2.scatter(ind+1*width, system_cost_reduc_opt, s=45, c= colors[1], edgecolor='black', linewidth=1, marker = 's')
    p2 = ax2.plot(ind + wid_scaler*width, avg_dis_reduction_random, color=colors[0], label = 'Random, Avg')
    ax2.scatter(ind + wid_scaler*width, avg_dis_reduction_random, s=45, c=colors[0], edgecolor='black', linewidth=1, marker='s')

    # p22 = ax2.plot(ind +2*width, avg_dis_reduction_opt, color=colors[1], label = 'Selected, Avg')
    # ax2.scatter(ind + 2*width, avg_dis_reduction_opt, s=45, c=colors[1], edgecolor='black', linewidth=1, marker='s')

    if city_folder == 'Munich':
        coef_adjust = 1/10 * time_span_scaler
        y_ticks = [0, 3, 6, 9, 12]
        y_tick_label = [f'{rate}%' for rate in y_ticks]
        y_ticks2 = [0,0.01,0.02,0.03,0.04]#list(np.array([0, 6, 12, 18]) * coef_adjust)
        y_tick_label2 = [f'{rate}' for rate in y_ticks2]
    elif city_folder == 'Beijing':
        y_ticks = [0, 3, 6, 9, 12]
        coef_adjust = 1/10 * time_span_scaler
        y_tick_label = [f'{rate}%' for rate in y_ticks]
        y_ticks2 = [0,0.02,0.04,0.06,0.08]#list(np.array([0, 6, 12, 18]) * coef_adjust)
        y_tick_label2 = [f'{rate}' for rate in y_ticks2]
    elif city_folder == 'Singapore_ind':
        coef_adjust = 1/10 * time_span_scaler
        y_ticks = [0, 3, 6, 9, 12]
        y_tick_label = [f'{rate}%' for rate in y_ticks]
        y_ticks2 = [0,0.03,0.06,0.09,0.12]#list(np.array([0, 6, 12, 18]) * coef_adjust)
        y_tick_label2 = [f'{rate}' for rate in y_ticks2]
    else:
        y_ticks = np.round(ax.get_yticks(), 2)
        y_tick_label = y_ticks
        y_ticks2 = np.round(ax2.get_yticks(), 2)
        y_tick_label2 = y_ticks2
    ax.set_yticks(y_ticks, y_tick_label, fontsize=font_size)
    ax2.set_yticks(y_ticks2, y_tick_label2, fontsize=font_size)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.yaxis.major.formatter._useMathText = True
    # ax2.yaxis.major.formatter._useMathText = True
    plt.xlim(x_lim_l,x_lim_r)
    # ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])
    ax.set_ylabel('Total CO2 reduction (%)', fontsize=font_size)
    ax2.set_ylabel('Avg CO2 reduction (t)', fontsize=font_size)
    ax.set_xlabel('Participation rate', fontsize=font_size)
    x_ticks_rate = [10, 40, 70, 100]
    x_ticks = np.array([ind[int((ele-10) / 10)] for ele in x_ticks_rate])
    x_ticks_list = [f'{rate}%' for rate in x_ticks_rate]
    ax.set_xticks(x_ticks + 1.5* width, x_ticks_list, fontsize=font_size)
    # ax.set_xticklabels(x_ticks + 1.5* width, x_ticks_list, fontsize=font_size)


    ax.set_ylim([0,y_max * 1.05])
    ax2.set_ylim([0, y_max2 * 1.05])
    # lns = [rects2] + [rects22] + p2 + p22
    lns = [rects2] + p2
    labs = [l.get_label() for l in lns]

    # ax.legend(lns, labs, fontsize=font_size - 2, loc = 'upper right', ncol = 2)

    # base_cost = y_base[0]
    # plt.ylim(0, 6)

    # add words
    # for idx in range(len(ind)):
    #     reduction_sys = max((base_cost - system_cost[idx]) / base_cost, 0)
    #     x_word = ind[idx] - 0.7*width
    #     y_word = system_cost[idx] + base_cost*0.03
    #     plt.text(x_word,y_word, str(round(reduction_sys*100, 1)) + '%', fontsize = font_size - 3)
    #     reduction_Pare = max((base_cost - Pareto_cost[idx]) / base_cost,0) # avoid Precision errors
    #     x_word = ind[idx] + width - 0.3*width
    #     y_word = Pareto_cost[idx] + base_cost*0.03
    #     plt.text(x_word,y_word, str(round(reduction_Pare*100, 1)) + '%',fontsize = font_size - 3)
    #     a=1

    # pos_x = int(np.round(len(ind)/2))
    # ax.text(ind[pos_x], y_max * 1.1, text_city[city_folder], size=font_size*1.2,
    #          ha="center", va="center",
    #          bbox=dict(boxstyle="round",
    #                    ec=(1., 0.5, 0.5),
    #                    fc=(1., 0.8, 0.8),
    #                    )
    #          )

    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/' + save_name + '.jpg', dpi=300)




def plot_optimal_sample_size_impact(res_stats, save_fig, save_name, city_folder, refer_scenario_name, percent_scenario_prefix, scenario_percent):

    # refer = res_stats.loc[res_stats['scenario'] == refer_scenario_name].copy()
    # refer['scenario'] = f'{percent_scenario_prefix}random_100_percent'
    # res_stats = pd.concat([res_stats, refer])
    # refer['scenario'] = f'{percent_scenario_prefix}optimal_100_percent'
    # res_stats = pd.concat([res_stats, refer])

    measure_value = 'CO2_ij'
    unit_scale = 1000 * 1000
    time_span_scaler = 52*5*2
    # if measure_value == 'CO2_ij':
    #     unit_scale = 1

    random_data = res_stats.loc[(res_stats['scenario'].str.contains('random_') & (res_stats['attributes'] == measure_value))].copy()
    total_num_inds = res_stats.loc[(res_stats['scenario'].str.contains('random_') & (res_stats['attributes'] == 'num_inds'))].copy()
    total_num_inds['num_samples'] = total_num_inds['old']
    random_data = random_data.merge(total_num_inds[['scenario','num_samples']], on=['scenario'])
    random_data['sample_percent'] = random_data['scenario'].apply(lambda x: int(x.split('random_')[1].split('_percent')[0]) / 100)
    random_data = random_data.sort_values(['sample_percent'])
    random_data['sample_size'] = random_data['num_samples'] * random_data['sample_percent']
    random_data['avg_dis_reduction'] = (random_data['old'] - random_data['new']) / random_data['sample_size']


    dis_reduction_percent_random = -random_data['diff_percent'].values * 100
    avg_dis_reduction_random = random_data['avg_dis_reduction'].values / unit_scale * time_span_scaler

    optimal_data = res_stats.loc[(res_stats['scenario'].str.contains('optimal_') & (res_stats['attributes'] == measure_value))].copy()
    total_num_inds = res_stats.loc[(res_stats['scenario'].str.contains('optimal_') & (res_stats['attributes'] == 'num_inds'))].copy()
    total_num_inds['num_samples'] = total_num_inds['old']
    optimal_data = optimal_data.merge(total_num_inds[['scenario','num_samples']], on=['scenario'])
    optimal_data['sample_percent'] = optimal_data['scenario'].apply(lambda x: int(x.split('optimal_')[1].split('_percent')[0]) / 100)
    optimal_data = optimal_data.sort_values(['sample_percent'])
    optimal_data['sample_size'] = optimal_data['num_samples'] * optimal_data['sample_percent']
    optimal_data['avg_dis_reduction'] = (optimal_data['old'] - optimal_data['new']) / optimal_data['sample_size']

    dis_reduction_percent_opt = -optimal_data['diff_percent'].values * 100
    avg_dis_reduction_opt = optimal_data['avg_dis_reduction'].values / unit_scale * time_span_scaler


    font_size = 25
    matplotlib.rcParams['font.size'] #= font_size - 2
    N = len(optimal_data)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4  # the width of the bars
    scale_para = 1.2
    fig, ax = plt.subplots(figsize=(6.8*scale_para, 5.7*scale_para))

    wid_scaler = 1.5

    x_lim_l = -0.5*width - 0.1
    x_lim_r = ind[-1] + width + 2*width + width
    # x_base = [x_lim_l,x_lim_r]
    # y_base = [sum_cost_old/1000, sum_cost_old/1000]
    # line = ax.plot(x_base,y_base, '--',color = 'gray', linewidth = 2.0,label = 'Status quo')
    # rects1 = ax.bar(ind, total_system_cost_reduc, width, color=colors[1], label = 'SO (Random)',alpha = 0.5)
    # rects11 = ax.bar(ind+ width, total_system_cost_reduc_opt, width, color=colors[1], label = 'SO (Total)',alpha = 0.8)
    # rects2 = ax.bar(ind + 1.5*width, dis_reduction_percent_random, width, color=colors[0],label = 'Random, Total',alpha = 0.75)
    rects22 = ax.bar(ind + wid_scaler*width, dis_reduction_percent_opt, width, color=colors[1],label = 'Selected, Total',alpha = 0.75)

    #

    y_max = max(dis_reduction_percent_opt)
    y_max2 = max(avg_dis_reduction_opt)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    # p2 = ax2.plot(ind + 1*width, avg_dis_reduction_random, color=colors[0], label = 'Random, Avg')
    # ax2.scatter(ind + 1*width, avg_dis_reduction_random, s=45, c=colors[0], edgecolor='black', linewidth=1, marker='s')

    p22 = ax2.plot(ind +wid_scaler*width, avg_dis_reduction_opt, color=colors[1], label = 'Selected, Avg')
    ax2.scatter(ind + wid_scaler*width, avg_dis_reduction_opt, s=45, c=colors[1], edgecolor='black', linewidth=1, marker='s')

    if city_folder == 'Munich':
        coef_adjust = 1/10 * time_span_scaler
        y_ticks = [0, 3, 6, 9, 12]
        y_tick_label = [f'{rate}%' for rate in y_ticks]
        y_ticks2 = [0,0.3,0.6,0.9,1.2]#list(np.array([0, 6, 12, 18]) * coef_adjust)
        y_tick_label2 = [f'{rate}' for rate in y_ticks2]
    elif city_folder == 'Beijing':
        y_ticks = [0, 3, 6, 9, 12]
        coef_adjust = 1/10 * time_span_scaler
        y_tick_label = [f'{rate}%' for rate in y_ticks]
        y_ticks2 = [0,0.3,0.6,0.9,1.2]#list(np.array([0, 6, 12, 18]) * coef_adjust)
        y_tick_label2 = [f'{rate}' for rate in y_ticks2]
    elif city_folder == 'Singapore_ind':
        coef_adjust = 1/10 * time_span_scaler
        y_ticks = [0, 3, 6, 9, 12]
        y_tick_label = [f'{rate}%' for rate in y_ticks]
        y_ticks2 = [0,0.6,1.2,1.8,2.4]#list(np.array([0, 6, 12, 18]) * coef_adjust)
        y_tick_label2 = [f'{rate}' for rate in y_ticks2]
    else:
        y_ticks = np.round(ax.get_yticks(), 2)
        y_tick_label = y_ticks
        y_ticks2 = np.round(ax2.get_yticks(), 2)
        y_tick_label2 = y_ticks2
    ax.set_yticks(y_ticks, y_tick_label, fontsize=font_size)
    ax2.set_yticks(y_ticks2, y_tick_label2, fontsize=font_size)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.yaxis.major.formatter._useMathText = True
    # ax2.yaxis.major.formatter._useMathText = True
    plt.xlim(x_lim_l,x_lim_r)
    # ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])
    ax.set_ylabel('Total CO2 reduction (%)', fontsize=font_size)
    ax2.set_ylabel('Avg CO2 reduction (t)', fontsize=font_size)
    ax.set_xlabel('Participation rate', fontsize=font_size)

    x_ticks_rate = [1, 5, 9, 13, 17]
    x_ticks = np.array([ind[scenario_percent.index(ele)] for ele in x_ticks_rate])
    x_ticks_list = [f'{rate}%' for rate in x_ticks_rate]
    ax.set_xticks(x_ticks + 1.5* width, x_ticks_list, fontsize=font_size)
    # ax.set_xticklabels(x_ticks + 1.5* width, x_ticks_list, fontsize=font_size)


    ax.set_ylim([0,y_max * 1.07])
    ax2.set_ylim([0, y_max2 * 1.07])
    # lns = [rects2] + [rects22] + p2 + p22
    lns = [rects22] + p22
    labs = [l.get_label() for l in lns]

    # ax.legend(lns, labs, fontsize=font_size - 2, loc = 'upper right', ncol = 2)

    # base_cost = y_base[0]
    # plt.ylim(0, 6)

    # add words
    # for idx in range(len(ind)):
    #     reduction_sys = max((base_cost - system_cost[idx]) / base_cost, 0)
    #     x_word = ind[idx] - 0.7*width
    #     y_word = system_cost[idx] + base_cost*0.03
    #     plt.text(x_word,y_word, str(round(reduction_sys*100, 1)) + '%', fontsize = font_size - 3)
    #     reduction_Pare = max((base_cost - Pareto_cost[idx]) / base_cost,0) # avoid Precision errors
    #     x_word = ind[idx] + width - 0.3*width
    #     y_word = Pareto_cost[idx] + base_cost*0.03
    #     plt.text(x_word,y_word, str(round(reduction_Pare*100, 1)) + '%',fontsize = font_size - 3)
    #     a=1

    # pos_x = int(np.round(len(ind)/2))
    # ax.text(ind[pos_x], y_max * 1.1, text_city[city_folder], size=font_size*1.2,
    #          ha="center", va="center",
    #          bbox=dict(boxstyle="round",
    #                    ec=(1., 0.5, 0.5),
    #                    fc=(1., 0.8, 0.8),
    #                    )
    #          )

    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/' + save_name + '.jpg', dpi=300)

if __name__ == '__main__':
    colors = ['#D8A19E','#7394BD']
    # colors = sns.color_palette("muted")
    city_folder_list = ['Munich', 'Beijing', 'Singapore_ind']#['Munich', 'Beijing', 'Singapore_ind']'Munich',
    save_fig = 1
    text_city={'Beijing':'Beijing',
               'Munich':'Munich',
               'Singapore':'Singapore',
               'Singapore_ind':'Singapore'}

    refer_scenario_name = '3_0_realistic' # 'refer'
    percent_scenario_prefix = '4_0_' # ''

    ############ plot legend

    city_folder = 'Munich'
    save_name = 'impact_sample_size_legend'
    res_stats = pd.read_csv(f'../{city_folder}/final_matched_res_stats.csv')
    plot_legend(res_stats, 1, save_name, city_folder, refer_scenario_name, percent_scenario_prefix)

    # #############
    for city_folder in city_folder_list:
        step = 10
        scenario_percent = list(range(10, 100 + step, step))
        save_name = f'impact_random_sample_size_{text_city[city_folder]}'
        res_stats = pd.read_csv(f'../{city_folder}/final_matched_res_stats.csv')
        plot_random_sample_size_impact(res_stats, save_fig, save_name, city_folder, refer_scenario_name, percent_scenario_prefix)


    # # #############
    # for city_folder in city_folder_list:
    #     step = 2
    #     scenario_percent = list(range(1, 17 + step, step))
    #     save_name = f'impact_optimal_sample_size_{text_city[city_folder]}'
    #     res_stats = pd.read_csv(f'../{city_folder}/final_matched_res_stats.csv')
    #     plot_optimal_sample_size_impact(res_stats, save_fig, save_name, city_folder, refer_scenario_name, percent_scenario_prefix, scenario_percent)