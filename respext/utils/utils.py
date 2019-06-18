# imports -- standard
import warnings
import pandas as pd
import numpy as np
import GPy

__all__ = ['load_spectrum', 'de_redshift', 'normalize_flux', 'remove_gaps', 'auto_prune', 'downsample', 'filter_outliers']

def load_spectrum(filename, Si_check = True, normalize = True, **kwargs):
    '''load spectrum from file'''

    # load data
    data = pd.read_csv(filename, delim_whitespace = True, header = None, comment = '#')
    wave = data.loc[:, 0]
    flux = data.loc[:, 1]
    # check for data in proximity of Si II 6355
    if Si_check and ((wave.min() > 5800) or (wave.max() < 7000)):
        warnings.warn('No data detected near Si II 6355 feature.')
    # return raw wavelength and flux
    if normalize:
        return wave.values, normalize_flux(flux.values, **kwargs)
    else:
        return wave.values, flux.values

def de_redshift(wave, z):
    '''correct wavelength for redshift'''

    return wave / (1 + z)

def normalize_flux(flux, norm_method = 'max'):
    '''normalize flux according to given method'''

    if norm_method == 'max':
        return flux / flux.max()
    elif norm_method == 'median':
        return flux / np.median(flux)
    else:
        warnings.warn('{} is not a supported flux normalization method, using max instead')
        return flux / flux.max()

def remove_gaps(wave, flux):
    '''remove gaps by masking regions where flux is zero'''

    keep = flux != 0
    return wave[keep], flux[keep]

def auto_prune(wave, flux, lines, prune_leeway = 500, normalize = True, **kwargs):
    '''restrict to range of lines to be measured, with some leeway'''

    wav_min = lines.min().min() - prune_leeway
    wav_max = lines.max().max() + prune_leeway
    i0, i1 = np.searchsorted(wave, [wav_min, wav_max])
    flux = flux[i0:i1]
    if normalize:
        flux = normalize_flux(flux, **kwargs)
    return wave[i0:i1], flux

def downsample(wave, flux, downsampling):
    '''down sample spectrum by a factor of <downsampling>'''

    return wave[::downsampling], flux[::downsampling]

def filter_outliers(wave, flux, sigma_outliers, sig_downsampling = 20, normalize = True, **kwargs):
    '''
    attempt to remove sharp lines (telluric, cosmic rays, etc.)
    applies a heavy downsampling, then discards points that are
    further than <sigma_outliers> standard deviations
    '''

    x = wave[::sig_downsampling, np.newaxis]
    y = flux[::sig_downsampling, np.newaxis]

    kernel = GPy.kern.Matern32(input_dim = 1, lengthscale = 300, variance = 0.001)
    m = GPy.models.GPRegression(x, y, kernel)
    m.optimize()

    pred, var = m.predict(wave[:, np.newaxis])
    sigma = np.sqrt(var.squeeze())
    valid = np.abs(flux - pred.squeeze()) < sigma_outliers * sigma

    flux = flux[valid]
    if normalize:
        flux = normalize_flux(flux, **kwargs)

    return wave[valid], flux
