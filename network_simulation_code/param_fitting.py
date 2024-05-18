import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
from scipy.stats import linregress
import pickle
import statsmodels.api as sm
import pandas as pd

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica'], 'size':10})

denim = [39/255, 93/255, 173/255] #275dad 
gray = [91/255, 97/255, 106/255] #5b616a
orange = [238/255, 123/255, 48/255] #ee7b30
verdigris = [58/255, 175/255, 185/255] #3aafb8
colors = {1: verdigris, 2: gray, 3: denim, 4: orange}


def TC_LARB():

    x_select_val = 'hull_area'
    y_select_val = 'eLen_2D'
    z_select_val = 'mean_void_size'
    w_select_val = 'n_edges'

    with open('hybrid_mCherry_TCs_projected_analyzed.p', 'rb') as f:
        data = pickle.load(f)

    bad_rows = data[(data['instar'] == 4) & (data['number'] == 9)].index
    data = data.drop(index=bad_rows)

    all_xs = []
    all_ys = []
    all_zs = []
    all_ws = []

    for ind in [1, 2, 4]:

        select = data[(data['instar'] == ind)]

        all_xs += list(np.log10(np.array(select[x_select_val])))
        all_ys += list(np.log10(np.array(select[y_select_val])))
        all_zs += list(np.log10(np.array(select[z_select_val])))
        all_ws += list(np.log10(np.array(select[w_select_val])))

    return all_xs, all_ys, all_zs, all_ws


def linplot(X, Y):

    model = sm.OLS(Y, sm.add_constant(X))
    res = model.fit()
    sim_m = res.params[1]
    sim_b = res.params[0]

    # plt.scatter(TCx, TCy, color='royalblue', zorder = 0, alpha = 0.8)
    # xs = np.array((np.min(TCx), np.max(TCx)))
    # ys = fit[0] * xs + fit[1]
    # plt.plot(xs, ys, color='k', zorder=2)

    # plt.errorbar(np.mean(TCx), TCy_av, xerr=TCx_std, yerr=TCy_std, marker='o', linestyle='none', color='orange')
    plt.scatter(X, Y, color='k', marker='x', zorder = 1, s = 30)
    xs = np.array((np.min(X), np.max(X)))
    ys = sim_m * xs + sim_b
    plt.plot(xs, ys, color='orange', zorder=2)
    #plt.title(r'$r^2 = $' + str(round(r**2, 2)))

def linfit_and_plot(L, A, R, B, name):

    opt = '2_separate_plots'

    if opt == '3_plot':
        fig, ax = plt.subplots(1, 3, figsize=(13, 2))

        plt.subplot(1, 3, 1)
        linplot(A, L)
        plt.scatter(TCA, TCL, zorder = 0, alpha = 0.5)
        # plt.xlim(2, 6)
        # plt.ylim(1.5, 4)
        plt.xlabel('log hull area')
        plt.ylabel('log total edge length')

        plt.subplot(1, 3, 2)
        linplot(A, R)
        plt.scatter(TCA, TCR, zorder = 0, alpha = 0.5)
        plt.xlabel('log hull area')
        plt.ylabel('mean void radius')

        plt.subplot(1, 3, 3)
        linplot(B, L)
        plt.scatter(TCB, TCL, zorder = 0, alpha = 0.5)
        plt.xlabel('number of edges')
        plt.ylabel('total edge length')


    elif opt == '2_plot':
        fig, ax = plt.subplots(1, 2, figsize=(8, 2))

        plt.subplot(1, 2, 1)
        linplot(A, L)
        plt.scatter(TCA, TCL, zorder=0, alpha=0.5)
        # plt.xlim(2, 6)
        # plt.ylim(1.5, 4)
        plt.xlabel('log hull area')
        plt.ylabel('log total edge length')

        plt.subplot(1, 2, 2)
        linplot(A, R)
        plt.scatter(TCA, TCR, zorder=0, alpha=0.5)
        plt.xlabel('log hull area')
        plt.ylabel('mean void radius')

        plt.savefig('fitplots/' + name + '.png', bbox_inches = 'tight')
        plt.close()

    else:
        fig, ax = plt.subplots(figsize=(2, 2))
        linplot(A, L)
        plt.scatter(TCA, TCL, zorder=0, s=60, alpha=0.5, color = 'gray', edgecolors='none')
        plt.axis([2.4, 5.5, 1.6, 4.0])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('log A')
        plt.ylabel('log L')

        alpha_fit, a_err, a_intercept, a_res = linfit_with_err(A, L, TCA, TCL)
        plt.text(3.6, 1.8, 'RSS = ' + str(round(a_res, 2)))
        plt.tight_layout()

        plt.savefig('fitplots/AL_' + name + '.pdf', bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(2, 2))
        linplot(A, R)
        plt.scatter(TCA, TCR, zorder=0, s=60, alpha=0.5, color = 'gray', edgecolors='none')
        plt.axis([2.4, 5.5, -0.4, 2.0])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('log A')
        plt.ylabel('log R')

        beta_fit, b_err, b_intercept, b_res = linfit_with_err(A, R, TCA, TCR)
        plt.text(3.6, -0.2, 'RSS = ' + str(round(b_res, 2)))
        plt.tight_layout()

        plt.savefig('fitplots/AR_' + name + '.pdf', bbox_inches='tight')
        plt.close()

def linfit_with_err(xs, ys, xData, yData):
    Y = ys
    X = sm.add_constant(xs)
    model = sm.OLS(Y, X)
    res = model.fit()
    slope = res.params[1]
    slope_err = res.bse[1]
    intercept = res.params[0]


    rs = np.array(yData) - slope * np.array(xData) - intercept * np.ones(len(xData))
    RSS = np.sum(rs ** 2)

    #print(slope)
    # print(RSS)


    # print("Parameters: ", res.params)
    # print("Standard errors: ", res.bse)

    #pred = res.get_prediction(X).summary_frame()

    #err_below = np.mean(pred['mean_ci_lower'])
    #err_above = np.mean(pred['mean_ci_upper'])

    # ax.plot(x_pred,,linestyle='--',color='blue')
    # ax.plot(x_pred,pred['mean_ci_upper'],linestyle='--',color='blue')

    return slope, slope_err, intercept, RSS


data = pickle.load(open('param_sweep_info_full.p', 'rb'))
ss = sorted(set(list(data['s'])))
bs = sorted(set(list(data['b'])))


prop_list = ['s', 'b', 'alpha', 'beta', 'gamma', 'alpha_err', 'beta_err',
             'gamma_err', 'alpha_res', 'beta_res', 'gamma_res']
scaling_fits = {}
for key in prop_list:
    scaling_fits[key] = []

TCA, TCL, TCR, TCB = TC_LARB()

for s in ss:
    for b in bs:

        select = data[(data['s'] == s) & (data['b'] == b)]
        A = select['hull_area']
        L = select['tot_len']
        R = select['void_mean']
        B = select['n_edges']

        mean_branch_length = np.log10(np.array(L) / np.array(B))
        sqrtA = np.log10(np.sqrt(np.array(A)))
        A = np.log10(np.array(A))
        L = np.log10(np.array(L))
        R = np.log10(np.array(R))
        B = np.log10(np.array(B))

        alpha_fit, a_err, a_intercept, a_res = linfit_with_err(A, L, TCA, TCL)
        beta_fit, b_err, b_intercept, b_res = linfit_with_err(A, R, TCA, TCR)
        gamma_fit, g_err, g_intercept, g_res = linfit_with_err(A, B, TCA, TCB)

        #print(alpha_fit, beta_fit, gamma_fit)

        # a_res = compute_residual(TCA, TCL, alpha_fit, a_intercept)
        # b_res = compute_residual(TCA, TCR, beta_fit, b_intercept)
        # g_res = compute_residual(TCA, TCB, gamma_fit, g_intercept)

        residual = a_res + b_res #+ g_res

        #linfit_and_plot(A, L)
        #linfit_and_plot(L, A, R, B, 's_' + str(round(s, 5)) + '_b_' + str(round(b, 5)))


        if (s == 0.001 and b == 0.02) or (s == 0.0007 and b == 0.055) or (s == 0.0002 and b == 0.08):
            linfit_and_plot(L, A, R, B, 's_' + str(round(s, 5)) + '_b_' + str(round(b, 5)))
            print('s_' + str(round(s, 5)) + '_b_' + str(round(b, 5)), round(a_res, 2), round(b_res, 2))

        scaling_fits['s'].append(s)
        scaling_fits['b'].append(b)
        scaling_fits['alpha'].append(alpha_fit)
        scaling_fits['beta'].append(beta_fit)
        scaling_fits['gamma'].append(gamma_fit)
        scaling_fits['alpha_res'].append(a_res)
        scaling_fits['beta_res'].append(b_res)
        scaling_fits['gamma_res'].append(g_res)
        scaling_fits['alpha_err'].append(a_err)
        scaling_fits['beta_err'].append(b_err)
        scaling_fits['gamma_err'].append(g_err)


dataframe = pd.DataFrame(scaling_fits, columns = prop_list)
pickle.dump(dataframe, open('param_sweep_results.p', "wb"))