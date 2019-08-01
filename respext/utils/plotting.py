# imports -- standard
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

# plotting style defaults
rcParams['font.family'] = 'serif'
rcParams['xtick.minor.visible'] = True
rcParams['xtick.minor.size'] = 4
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.pad'] = 10
rcParams['ytick.minor.visible'] = True
rcParams['ytick.minor.size'] = 4
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.pad'] = 10
rcParams['axes.grid'] = False
rcParams['xtick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.direction'] = 'in'
rcParams['ytick.right'] = True

__all__ = ['setup_plot', 'plot_spec', 'plot_filled_spec', 'plot_continuum', 'plot_lines', 'define_continuum']

def setup_plot(title = None, xlabel = 'Rest Wavelength (\u212B)', ylabel = 'Normalized Flux', figsize = 'auto'):
    '''setup and return plot'''

    if figsize == 'auto':
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = plt.subplots(1, 1, figsize = figsize)
    if title is not None:
        ax.set_title(title, size = 20)
    if xlabel is not None:
        ax.set_xlabel(xlabel, size=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, size=14)
    return fig, ax

def plot_spec(ax, wave, flux, spec_color = 'black', spec_alpha = 1):
    '''add plot of spectrum to ax'''

    ax.plot(wave, flux, color = spec_color, alpha = spec_alpha)

def plot_filled_spec(ax, x, mean, conf, fill_color = 'red', fill_alpha = 0.3):
    '''add filled spectrum to ax'''

    ax.fill_between(x, mean - conf, mean + conf, color = fill_color, alpha = fill_alpha)

def plot_continuum(ax, cont_points, cp_color = 'black', cl_color = 'blue', cl_alpha = 0.5, show_conf = True, conf_alpha = 0.3):
    '''plot continuum points and line with optional uncertainties'''

    for feature in cont_points.index:
        if cont_points.loc[feature].notnull().all():
            if show_conf:
                _, bot, top, _ = cont_points.loc[feature, 'cont'](cont_points.loc[feature, ['wav1', 'wav2']].values.astype(float))
                ax.fill_between(cont_points.loc[feature, ['wav1', 'wav2']].values.astype(float), bot, top,
                                color = cl_color, alpha = conf_alpha)
            ax.plot(cont_points.loc[feature, ['wav1', 'wav2']], cont_points.loc[feature, ['flux1', 'flux2']],
                    color = cl_color, alpha = cl_alpha)
            ax.scatter(cont_points.loc[feature, ['wav1', 'wav2']], cont_points.loc[feature, ['flux1', 'flux2']],
                       color = cp_color, s = 80)

def plot_lines(ax, absorptions, line_color = 'black', show_line_labels = True):
    '''plot absorption lines'''

    for feature in absorptions.index:
        # if absorption measured, label it
        if absorptions.loc[feature, ['wava', 'fluxa', 'cont']].notnull().all():
            # absorption line below continuum
            ax.plot([absorptions.loc[feature, 'wava']] * 2,
                    [absorptions.loc[feature, 'fluxa'], absorptions.loc[feature, 'cont'](absorptions.loc[feature, 'wava'])[0]],
                    color = line_color)

            # points from which absorption was measured
            ax.scatter([absorptions.loc[feature, 'wava']] * 2,
                       [absorptions.loc[feature, 'fluxa'], absorptions.loc[feature, 'cont'](absorptions.loc[feature, 'wava'])[0]],
                       color = line_color, s = 40)

            # absorption line above continuum
            ax.plot([absorptions.loc[feature, 'wava']] * 2,
                    [absorptions.loc[feature, 'cont'](absorptions.loc[feature, 'wava'])[0], 1.1],
                    color = line_color, ls = '--')
            if show_line_labels:
                ax.text(absorptions.loc[feature, 'wava'], 1.1, feature, rotation = 'vertical',
                        horizontalalignment = 'right', verticalalignment = 'top')

        # if absorption not measured, still label using center or derived continuum points
        elif show_line_labels and absorptions.loc[feature, ['wav1', 'wav2']].notnull().all():
            wav_tmp = absorptions.loc[feature, ['wav1', 'wav2']].mean()

            ax.plot([wav_tmp] * 2, [absorptions.loc[feature, 'cont'](wav_tmp)[0], 1.1],
                    color = line_color, ls = '--')

            ax.text(wav_tmp, 1.1, feature, rotation = 'vertical',
                    horizontalalignment = 'right', verticalalignment = 'top')

def _dc_onpick(event, cont_points, wave, flux, ax, fig):
    '''handle click events triggered by define_continuum'''

    # identify nearest point (in wavelength space)
    nearest = np.abs(wave - event.xdata).argmin()

    #global cp
    global c_line
    cp = cont_points.get_offsets()
    if cp.shape[0] == 2: # then was already 2 and should be reset
        cont_points.set_offsets(np.array([[], []]).T)
        c_line.set_data([], [])
        fig.canvas.draw()
        #cp = []
        return
    if cp.shape[0] < 2: # don't have both sides yet
        cp = np.concatenate([cp, np.array([[wave[nearest], flux[nearest]]])])
        cont_points.set_offsets(cp)
    if cp.shape[0] == 2: # have both sides, plot continuum
        c_line, = ax.plot(cp[:, 0], cp[:, 1], color = 'blue', alpha = 0.4)

    fig.canvas.draw()

def define_continuum(wave, flux, absorption):
    '''interactively determine continuum by clicking on boundaries'''

    # setup plot, add model, show low and high continuum boundaries
    fig, ax = setup_plot(title = absorption.name, figsize = (6, 6))
    plot_spec(ax, wave, flux, spec_color = 'red')
    for edge in ['low_1', 'high_1', 'low_2', 'high_2']:
        if edge in ['low_1', 'high_1']:
            col = 'orangered'
            sty = '--'
        else:
            col = 'royalblue'
            sty = ':'
        ax.axvline(absorption[edge], color = col, ls = sty)

    # manually select continuum points
    cont_points = ax.scatter([], [], color = 'black', s = 80)
    plt.ion()
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: _dc_onpick(event, cont_points, wave, flux, ax, fig))
    fig.show()
    print('click points to two points to define continuum')
    input('accept continuum [enter] or reset [click again] > ')
    fig.canvas.mpl_disconnect(cid)
    plt.ioff()
    plt.clf()
    plt.close()

    cp = cont_points.get_offsets()
    # return continuum points
    try:
        if len(cp) == 0:
            return [np.nan]*4
        else:
            return cp.flatten()
    except NameError:
        return [np.nan]*4

