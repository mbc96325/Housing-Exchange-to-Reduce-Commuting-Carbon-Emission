import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from sklearn.linear_model import LinearRegression
import math


rgb1 = (178,80,95)
rgb1 = tuple([x / 255.0 for x in rgb1])
rgb2 = (64,112,171)
rgb2 = tuple([x / 255.0 for x in rgb2])
rgb3 = (120,160,210)
rgb3 = tuple([x / 255.0 for x in rgb3])
# colors = sns.color_palette("muted")
colors_dict = {'Optimal': rgb2, 'Current': rgb1, 'Ideal': rgb3}
text_city = {'Beijing': 'Beijing', 'Munich': 'Munich', 'Singapore_ind': 'Singapore','sg_one_ind_per_HH': 'Singapore'}

def plot_commute_dist_distribution(data_plot_dict, city, save_fig):
    plot_inner = False
    color_idx = 0
    font_size = 20
    matplotlib.rcParams['font.size'] = font_size - 2
    fig, ax = plt.subplots(figsize=(6*1.1, 5*1.1))

    for scenario, data_plot in data_plot_dict.items():
        dist_int = 3000
        data_plot['HP_ID'] = data_plot['work_ID']

        data_plot['Dist'] = data_plot['distance']#*dist_int

        data_plot['Dist_int'] = data_plot['Dist'] // dist_int *  dist_int
        avg = np.mean(data_plot['Dist']) / 1000

        print(city, 'avg dist', round(avg, 2))

        data_plot = data_plot.groupby(['Dist_int'])['HP_ID'].count().reset_index(drop=False)
        data_plot['p_dist'] = data_plot['HP_ID'] / data_plot['HP_ID'].sum()
        data_plot['Dist_int'] /= 1000
        data_plot['Dist_int'] += (dist_int / 2 / 1000)
        print('greater than 20 fraction', sum(data_plot.loc[data_plot['Dist_int']>=20,'p_dist']))
        # data_plot = data_plot.loc[data_plot['Dist_int']<=40]
        # create some data to use for the plot

        # the main axes is subplot(111) by default
        # plt.scatter(data_plot['Dist_int'], data_plot['p_dist'], marker = 'o', s = 5,
        #          color = colors[2])
        plt.plot(data_plot['Dist_int'], data_plot['p_dist'], marker = 'o', markersize = 10,
                 color = colors_dict[scenario],linewidth = 3, label=scenario + '\nAvg = ' + str(round(avg,2))+ ' km')
        color_idx += 1

        plt.xlabel('CD: Commuting distance (km)', fontsize=font_size)
        plt.ylabel(r'$P(CD)$', fontsize=font_size)
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax.yaxis.major.formatter._useMathText = True
        # x_ticks = list(range(0,120+20,20))

        if 'Singapore' in city:
            plt.xticks([0,10,20,30,40,50], fontsize=font_size)
            log_dist_max = 40
            plt.xlim([0, 50])
            avg_cd_word_y = 0.001
            # plt.ylim([0,0.06])
            y_ticks = [0.0,0.1,0.2,0.3]
            y_tick_label = [f'{num}' for num in y_ticks]
            ax.set_yticks(y_ticks, y_tick_label, fontsize=font_size)
        elif city == 'sg_one_ind_per_HH':
            plt.xticks([0,10,20,30,40,50], fontsize=font_size)
            log_dist_max = 40
            plt.xlim([0, 55])
            avg_cd_word_y = 0.001
            plt.yticks(fontsize=font_size)
        elif 'Beijing' in city:
            plt.xticks([0,10,20,30,40], fontsize=font_size)
            log_dist_max = 37
            plt.xlim([0, 40])
            avg_cd_word_y = 0.001
            plt.yticks(fontsize=font_size)
        elif 'Munich' in city:
            plt.xticks([0,10,20,30,40], fontsize=font_size)
            plt.xlim([0, 40])
            log_dist_max = 37
            avg_cd_word_y = 0.001
            plt.yticks(fontsize=font_size)
        # plt.yscale('log')
        # x_ticks

        #
        # plt.text(1, avg_cd_word_y, 'Avg CD = ' + str(round(avg,2))+ ' km', fontsize=font_size)
        #========================================

    plt.legend(fontsize=font_size)


    if plot_inner:
        if 'Singapore' in city:
            inset_axes(ax, width=2.8, height=2.8, loc=3, bbox_to_anchor=(0.55, 0.45, .3, .3), bbox_transform=ax.transAxes)
        else:
            inset_axes(ax, width=2.8, height=2.8, loc=3, bbox_to_anchor=(0.55,0.45,.3,.3), bbox_transform=ax.transAxes)
        data_plot_used = data_plot.loc[data_plot['Dist_int'] < log_dist_max]
        plt.plot(data_plot_used['Dist_int'],data_plot_used['p_dist'], marker = 'o', markersize = 5,
                 color = colors[3],linewidth = 2)
        y_max = np.max(data_plot_used['p_dist'])*1.05
        plt.yscale('log')
        # ax.set_yscale('log', basey=2)
        # plt.title('Probability')
        plt.xlabel('CD: Commuting distance (km)', fontsize=font_size)
        plt.ylabel(r'$P(CD)$ (log-scale)', fontsize=font_size)
        plt.yticks(fontsize=font_size)
        if city == 'sg':
            plt.xticks([0,10,20,30,40],fontsize=font_size) #x_ticks
            min_dist = 25
            trip_freq_lim = 40
            pos_x = np.round(trip_freq_lim/3)
        elif 'Singapore' in city:
            plt.xticks([0,10,20,30,40],fontsize=font_size) #x_ticks
            min_dist = 25
            trip_freq_lim = 40
            pos_x = np.round(trip_freq_lim/2.5)
        elif 'Beijing' in city:
            plt.xticks([0,10,20,30],fontsize=font_size) #x_ticks
            min_dist = 5
            trip_freq_lim = 30
            pos_x = np.round(trip_freq_lim/3)
        elif 'Munich' in city:
            plt.xticks([0,10,20,30],fontsize=font_size) #x_ticks
            min_dist = 5
            trip_freq_lim = 30
            pos_x = np.round(trip_freq_lim/3)


        # ax.text(pos_x, y_max * 0.925, text_city[city], size=font_size*1.2,
        #          ha="center", va="center",
        #          bbox=dict(boxstyle="round",
        #                    ec=(1., 0.5, 0.5),
        #                    fc=(1., 0.8, 0.8),
        #                    )
        #          )




        #============Fit and add parameters
        X = data_plot.loc[(data_plot['Dist_int']>=min_dist) & (data_plot['Dist_int']<=trip_freq_lim),'Dist_int'].values.reshape(-1,1)
        y = np.log(data_plot.loc[(data_plot['Dist_int']>=min_dist) & (data_plot['Dist_int']<=trip_freq_lim),'p_dist'].values).reshape(-1,1)
        reg = LinearRegression().fit(X, y)
        _lambda = reg.coef_[0][0]
        if 'Singapore' in city:
            word_x = trip_freq_lim - 21
            word_y = 0.078
        elif city == 'sg_one_ind_per_HH':
            word_x = trip_freq_lim - 21
            word_y = 0.078
        elif 'Beijing' in city:
            word_x = trip_freq_lim - 18
            word_y = 0.078
        elif 'Munich' in city:
            word_x = trip_freq_lim - 18
            word_y = 0.09


        # plt.text(word_x, word_y, r'$\lambda = $' + str(round(-1/_lambda,2))+ ' km')
        if 'Singapore' in city:
            x_pred = np.arange(1,trip_freq_lim + 20,1).reshape(-1,1)
        elif city == 'sg_one_ind_per_HH':
            x_pred = np.arange(1, trip_freq_lim + 20, 1).reshape(-1, 1)
        elif 'Beijing' in city:
            x_pred = np.arange(1,trip_freq_lim + 20,1).reshape(-1,1)
        elif 'Munich' in city:
            x_pred = np.arange(1,trip_freq_lim + 20,1).reshape(-1,1)
        y_pred = reg.predict(x_pred) + 0.8 # -> move it a little bit upward
        plt.plot(x_pred.ravel(),np.exp(y_pred.ravel()), 'k--', linewidth = 1.5) #
        plt.xlim([0, trip_freq_lim + 1])
        plt.ylim([math.pow(10,-3.3), math.pow(10,-0.6)])
        #======================


    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('img/commuting_dist_NS_' + city + '.jpg', dpi = 200)
    else:
        plt.show()


if __name__ == '__main__':
    city_scenario = ['Munich','Beijing','Singapore_ind']#'bj', 'mu',# sample distribution. no sg_one_ind
    # city_scenario = ['sg_one_ind_per_HH']
    scenario_to_plot = {'Optimal': '3_0_realistic', 'Ideal': '1_0_ideal',
                        'Current': ''}
    refer_scenario = '3_0_realistic'
    for city in city_scenario:
        plot_data = {}
        for scenario, name in scenario_to_plot.items():
            if scenario == 'Current':
                pair = pd.read_parquet(f'../{city}/final_pair_for_matching_{refer_scenario}.parquet')
                data = pair.loc[pair['home_ID']==pair['work_ID'],['home_ID', 'work_ID', 'distance']].copy()
            else:
                pair = pd.read_parquet(f'../{city}/final_pair_for_matching_{name}.parquet')
                match_res = pd.read_csv(f'../{city}/matched_res_{name}.csv')
                data = match_res.merge(pair[['home_ID', 'work_ID', 'distance']], on=['home_ID', 'work_ID'])
            plot_data[scenario] = data

        plot_commute_dist_distribution(plot_data, city, save_fig = 1)