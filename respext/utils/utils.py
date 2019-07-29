# imports -- standard
import warnings
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic

__all__ = ['load_spectrum', 'de_redshift', 'rebin', 'auto_prune', 'normalize_flux']

def load_spectrum(filename, scale = 1):
    '''load spectrum from file'''

    # load data, handling case if eflux is available or not
    data = pd.read_csv(filename, delim_whitespace = True, header = None, comment = '#')
    wave = data.loc[:, 0].values
    flux = data.loc[:, 1].values
    if len(data.columns) > 2:
        eflux = data.loc[:, 2].values
    else:
        eflux = np.nan * np.ones(len(flux))

    # automatically determine flux scale if needed
    if scale == 'auto':
        if np.median(flux) > 1e-10:
            scale = 1e-15
        else:
            scale = 1
    return wave, flux * scale, eflux * scale

def de_redshift(wave, z):
    '''correct wavelength for redshift'''

    return wave / (1 + z)

def rebin(wave, flux, eflux, factor):
    '''rebin spectrum by factor'''

    if (type(factor) != type(1)) or (factor < 1):
        warnings.warn('rebin factor must be an integer greater than 1 (not {}) -- doing nothing'.format(factor))
        return wave, flux, eflux

    if factor == 1:
        return wave, flux, eflux

    # bin wave, flux, and eflux (with error propagation)
    binned = binned_statistic(wave, (wave, flux), statistic = 'mean', bins = int(len(wave) / factor))[0]
    eflux = np.sqrt(binned_statistic(wave, eflux**2, statistic = 'sum', bins = int(len(wave) / factor))[0]) / factor
    wave, flux = binned[0], binned[1]
    return wave[~np.isnan(wave)], flux[~np.isnan(wave)], eflux[~np.isnan(wave)]

def auto_prune(wave, flux, eflux, lines, prune_leeway = 100):
    '''restrict to range of lines to be measured, with some leeway'''

    # skip if argument is bad
    if ((type(prune_leeway) != type(0.1)) and (type(prune_leeway) != type(1))) or (prune_leeway < 0):
        warnings.warn('prune amount invalid ({}) -- doing nothing'.format(prune_leeway))
        return wave, flux, eflux

    wav_min = lines['low_1'].min() - prune_leeway
    wav_max = lines['high_2'].max() + prune_leeway
    i0, i1 = np.searchsorted(wave, [wav_min, wav_max])
    return wave[i0:i1], flux[i0:i1], eflux[i0:i1]

def normalize_flux(flux, eflux, norm_method = 'max'):
    '''normalize flux according to given method'''

    if norm_method == 'max':
        scale = flux.max()
    elif norm_method == 'median':
        scale = np.median(flux)
    else:
        warnings.warn('{} is not a supported flux normalization method, using max instead')
        scale = flux.max()
    return flux / scale, eflux / scale, scale
