# imports -- standard
import matplotlib.pyplot as plt

__all__ = ['setup_plot', 'plot_spec', 'plot_filled_spec', 'plot_continuum', 'plot_lines']

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

def plot_continuum(ax, cont_points, cp_color = 'black', cl_color = 'blue', cl_alpha = 0.4):
    '''plot continuum points and line'''

    for feature in cont_points.index:
        ax.plot(cont_points.loc[feature, ['wav1', 'wav2']], cont_points.loc[feature, ['flux1', 'flux2']],
                color = cl_color, alpha = cl_alpha)
        ax.scatter(cont_points.loc[feature, ['wav1', 'wav2']], cont_points.loc[feature, ['flux1', 'flux2']],
                   color = cp_color, s = 80)

def plot_lines(ax, absorptions, line_color = 'black', show_line_labels = True):
    '''plot absorption lines'''

    for feature in absorptions.index:
        ax.plot([absorptions.loc[feature, 'wava']] * 2,
                [absorptions.loc[feature, 'fluxa'], absorptions.loc[feature, 'cont'](absorptions.loc[feature, 'wava'])],
                color = line_color)
        ax.scatter([absorptions.loc[feature, 'wava']] * 2,
                   [absorptions.loc[feature, 'fluxa'], absorptions.loc[feature, 'cont'](absorptions.loc[feature, 'wava'])],
                   color = line_color, s = 40)
        ax.plot([absorptions.loc[feature, 'wava']] * 2,
                [absorptions.loc[feature, 'cont'](absorptions.loc[feature, 'wava']), 1.1],
                color = line_color, ls = '--')
        if show_line_labels:
            ax.text(absorptions.loc[feature, 'wava'], 1.1, feature, rotation = 'vertical',
                    horizontalalignment = 'right', verticalalignment = 'top')