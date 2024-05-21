import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
import pandas
import pickle
import scipy.ndimage
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.cm as cm
import seaborn

#from pylr2 import regress2
import statsmodels.api as sm

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica'], 'size':12})

def MA_regress(xs, ys):

    result_Ma_pyrl2 = regress2(np.array(xs), np.array(ys), _method_type_2="major axis")
    m = result_Ma_pyrl2['slope']
    b = result_Ma_pyrl2['intercept']
    m_err = result_Ma_pyrl2['std_slope']
    r = result_Ma_pyrl2['r']

    return m

def OLS_regress(xs, ys):

    model = sm.OLS(ys, sm.add_constant(xs))
    result = model.fit()
    b_coef, m_coef = result.params
    b_err, m_err = result.bse

    return m_coef, m_err, b_coef

def TC_exponents():

    x_select_val = 'hull_area'
    y_select_val = 'eLen_2D'
    z_select_val = 'mean_void_size'
    w_select_val = 'n_edges'

    with open('TCs_Tr9_analyzed.p', 'rb') as f:
        data = pickle.load(f)

    # bad_rows = data[(data['instar'] == 4) & (data['number'] == 9)].index
    # data = data.drop(index=bad_rows)

    As = []
    Ls = []
    Rs = []
    Bs = []

    for ind in [1, 2, 3]:
        select = data[(data['instar'] == ind)]
        As += list(np.log10(np.array(select[x_select_val])))
        Ls += list(np.log10(np.array(select[y_select_val])))
        Rs += list(np.log10(np.array(select[z_select_val])))
        Bs += list(np.log10(np.array(select[w_select_val])))

    alpha, alpha_err, alpha_b = OLS_regress(np.array(As), np.array(Ls))
    beta, beta_err, beta_b = OLS_regress(np.array(As), np.array(Rs))
    gamma, gamma_err, gamma_b = OLS_regress(np.array(As), np.array(Bs))

    return alpha, alpha_err, beta, beta_err, gamma, gamma_err

def make_contour_plot(A, xvals, yvals, append = ''):

    blurred_data = gaussian_filter(A, sigma=1)
    # zoomed_data = scipy.ndimage.zoom(A, 50)

    levs = [2, 3, 5, 10, 20, 30]

    fig, ax = plt.subplots(figsize=(5.6, 3))

    # plt.contourf(D, levels = levs, cmap = 'plasma_r')
    plt.imshow(A, cmap=colorMap, origin='lower')
    cbar = plt.colorbar()
    plt.clim((lower_clim, upper_clim))

    C = plt.contour(blurred_data, levels=levs, colors=('k'), linewidths=0.75)
    plt.clabel(C, fmt='%2.f', colors='k', fontsize=10)

    xticks = np.arange(0, len(xvals))
    xticklabels = xvals
    xticks = xticks[3::8]
    xticklabels = xticklabels[3::8]

    yticks = np.arange(0, len(yvals))
    yticklabels = yvals
    yticks = yticks[::10]
    yticklabels = yticklabels[::10]

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.xticks(xticks, xticklabels)
    plt.yticks(yticks, yticklabels)

    cbar.set_label(r'residual')
    plt.xlabel('branching factor b')
    plt.ylabel('scaling factor s')

    #plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    plt.savefig('plots/smooth_contour_filled' + append + '.pdf', bbox_inches='tight')


def make_scatter_plot(alphas, betas, alpha_errs, beta_errs, color_arr, inds = [], append = '', title = ''):

    fig, ax = plt.subplots(figsize=(5.2, 2.8))

    # create a scatter plot
    sc = plt.scatter(alphas, betas, s=0, c=color_arr, cmap=colorMap)

    # create colorbar according to the scatter plot
    cbar = plt.colorbar(sc)
    plt.clim((lower_clim, upper_clim))
    cbar.set_ticks([0, 10, 20, 30])
    cbar.set_ticklabels([0, 10, 20, 30])

    ### change location of 2 label in contours

    # convert time to a color tuple using the colormap used for scatter
    norm = matplotlib.colors.Normalize(vmin=lower_clim, vmax=upper_clim, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colorMap)
    time_color = np.array([(mapper.to_rgba(v)) for v in color_arr])


    ##loop over each data point to plot
    for x, y, ex, ey, color in zip(alphas, betas, alpha_errs, beta_errs, time_color):

        plt.errorbar(x, y, xerr=ex, yerr=ey, capsize=0, zorder=0, color=color, alpha=0.7, mfc='white')
        plt.scatter(x, y, marker='o', color=color, zorder = 2, s=12, edgecolors='none', alpha = 0.7)

        # plt.errorbar(x, y, xerr=ex, yerr=ey, markersize=3,
        #              ls='none', marker='o', capsize=0, zorder = 0, color = color, alpha = 0.6)

    fit_m, fit_m_err, fit_b = OLS_regress(alphas, betas)

    fit_x = [np.min(alphas), np.max(alphas)]
    fit_y = [fit_m*x + fit_b for x in fit_x]

    plt.plot(fit_x, fit_y, color = 'k')

    plt.text(0.5, 0.02, r'slope = ' + str(round(fit_m, 2)) + r' $\pm$ ' + str(round(fit_m_err, 2)))

    params = title.split('_')

    plt.text(1, 0.45, r'$\theta_1 = \pi/$' + params[1])
    if params[3] == '1':
        plt.text(1, 0.4, r'$\theta_2 = \pi$')
    else:
        plt.text(1, 0.4, r'$\theta_2 = \pi/$' + params[3])


    for ind in inds:
        plt.scatter(alphas[ind], betas[ind], color = 'k', marker = 'o', s=20, zorder=3)

    cbar.set_label(r'RSS')
    plt.xlabel(r'scaling exponent $\alpha$')
    plt.ylabel(r'scaling exponent $\beta$')

    plt.scatter(0.5, 0.5, facecolors='none', edgecolors='k', s=50, marker = 'o', lw=1)
    plt.scatter(1.0, 0.0, facecolors='none', edgecolors='k', s=50, marker = 'o', lw=1)
    plt.scatter(TC_alpha, TC_beta, color='k', marker='x', s=50, zorder=3, lw=1)
    #plt.errorbar(TC_alpha, TC_beta, xerr=TC_alpha_err, yerr=TC_beta_err, capsize=0, zorder=5, color='k')


    plt.axis([0.45, 1.25, -0.05, 0.55])
    plt.gca().set_aspect('equal', adjustable='box')

    #if title != '': plt.title(title)

    plt.savefig('plots/alpha_beta_scatter_' + append + '_no_example_pts' + '.pdf', bbox_inches='tight')
    plt.clf()

TC_alpha, TC_alpha_err, TC_beta, TC_beta_err, TC_gamma, TC_gamma_err = TC_exponents()
print('TC exponents:', TC_alpha, TC_alpha_err, TC_beta, TC_beta_err, TC_gamma, TC_gamma_err)

colorMap = 'plasma_r'
cmapName = 'plasma'

colorMap = seaborn.blend_palette(['gold', 'dodgerblue'], n_colors=2, as_cmap=True)
cmapName = 'yellow_blue'

lower_clim = 0
upper_clim = 30

def main_plot():

    lower_clim = 0
    upper_clim = 30
    identifier = '_' + cmapName + '_abridged_lim_30_no_b0'

    with open('param_sweep_results.p', 'rb') as f:
        data = pickle.load(f)

    ss = sorted(set(list(data['s'])))
    bs = sorted(set(list(data['b'])))
    ss = ss[:21]
    bs = bs[1:23]

    print(ss)
    print(bs)

    D = np.zeros((len(ss), len(bs)))
    for i in range(len(ss)):
        for j in range(len(bs)):
            select = data[(data['s'] == ss[i]) & (data['b'] == bs[j])]
            D[i, j] = select['alpha_res'] + select['beta_res']

    print('minimal res:', np.min(D))
    inds = np.unravel_index(D.argmin(), D.shape)
    print('best fit params:', ss[inds[0]], bs[inds[1]])

    make_contour_plot(D, bs, ss, append = identifier + '10_20')
    plt.clf()

    select = data[(data['s'] <= 0.002) & (data['b'] <= 0.1) & (data['b'] != 0)]
    #select = data

    alphas = select['alpha']
    betas = select['beta']
    alpha_errs = select['alpha_err']
    beta_errs = select['beta_err']
    residuals = select['alpha_res'] + select['beta_res']

    ss = data['s']
    bs = data['b']

    ##find the three sample points to draw
    [ind1] = set(np.where(ss == 0.001)[0]) & set(np.where(bs == 0.02)[0])
    [ind2] = set(np.where(ss == 0.0007)[0]) & set(np.where(bs == 0.055)[0])
    [ind3] = set(np.where(ss == 0.0002)[0]) & set(np.where(bs == 0.08)[0])

    make_scatter_plot(alphas, betas, alpha_errs, beta_errs, residuals, inds = [ind1, ind2, ind3], append = identifier)

def supplemental_plots():

    for f1 in [4, 9, 14]:
        for f2 in [1, 2, 4]:

            type = 'f1_' + str(f1) + '_f2_' + str(f2)

            #data = pickle.load(open('mini_param_sweeps/info_' + type + '.p', 'rb'))
            savefile = 'mini_param_sweeps/results_' + type

            with open(savefile + '.p', 'rb') as f:
                data = pickle.load(f)

            # ss = sorted(set(list(data['s'])))
            # bs = sorted(set(list(data['b'])))
            # ss = ss[:21]
            # bs = bs[1:23]
            #
            # print(ss)
            # print(bs)
            #
            # D = np.zeros((len(ss), len(bs)))
            # for i in range(len(ss)):
            #     for j in range(len(bs)):
            #         select = data[(data['s'] == ss[i]) & (data['b'] == bs[j])]
            #         D[i, j] = select['alpha_res'] + select['beta_res']
            #
            # print('minimal res:', np.min(D))
            # inds = np.unravel_index(D.argmin(), D.shape)
            # print('best fit params:', ss[inds[0]], bs[inds[1]])
            #
            # make_contour_plot(D, bs, ss, append=identifier + '10_20')
            # plt.clf()

            select = data[(data['s'] <= 0.002) & (data['b'] <= 0.1) & (data['b'] != 0)]

            alphas = select['alpha']
            betas = select['beta']
            alpha_errs = select['alpha_err']
            beta_errs = select['beta_err']
            residuals = select['alpha_res'] + select['beta_res']

            make_scatter_plot(alphas, betas, alpha_errs, beta_errs, residuals, append=type, title = type)


supplemental_plots()