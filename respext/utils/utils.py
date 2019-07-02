# imports -- standard
import warnings
import pandas as pd
import numpy as np
import GPy

__all__ = ['load_spectrum', 'de_redshift', 'normalize_flux', 'remove_gaps', 'auto_prune', 'downsample', 'filter_outliers']

def load_spectrum(filename, Si_check = True, normalize = True, return_scale = False, **kwargs):
    '''load spectrum from file'''

    # load data
    data = pd.read_csv(filename, delim_whitespace = True, header = None, comment = '#')
    wave = data.loc[:, 0].values
    flux = data.loc[:, 1].values
    if len(data.columns) > 2:
        flux_err = data.loc[:, 2].values
    else:
        flux_err = np.nan * np.ones(len(flux))
    scale = flux.max()
    # check for data in proximity of Si II 6355
    if Si_check and ((wave.min() > 5800) or (wave.max() < 7000)):
        warnings.warn('No data detected near Si II 6355 feature.')
    # return raw wavelength and flux
    if normalize:
        flux, flux_err = normalize_flux(flux, flux_err, **kwargs)
    if return_scale:
        return wave, flux, flux_err, scale
    else:
        return wave, flux, flux_err

def de_redshift(wave, z):
    '''correct wavelength for redshift'''

    return wave / (1 + z)

def normalize_flux(flux, flux_err, norm_method = 'max'):
    '''normalize flux according to given method'''

    if norm_method == 'max':
        scale = flux.max()
    elif norm_method == 'median':
        scale = np.median(flux)
    else:
        warnings.warn('{} is not a supported flux normalization method, using max instead')
        scale = flux.max()
    return flux / scale, flux_err / scale

def remove_gaps(wave, flux, flux_err):
    '''remove gaps by masking regions where flux is zero'''

    keep = flux != 0
    return wave[keep], flux[keep], flux_err[keep]

def auto_prune(wave, flux, flux_err, lines, prune_leeway = 500, normalize = True, **kwargs):
    '''restrict to range of lines to be measured, with some leeway'''

    wav_min = lines.min().min() - prune_leeway
    wav_max = lines.max().max() + prune_leeway
    i0, i1 = np.searchsorted(wave, [wav_min, wav_max])
    flux = flux[i0:i1]
    flux_err = flux_err[i0:i1]
    if normalize:
        flux, flux_err = normalize_flux(flux, flux_err, **kwargs)
    return wave[i0:i1], flux, flux_err

def downsample(wave, flux, flux_err, downsampling):
    '''down sample spectrum by a factor of <downsampling>'''

    return wave[::downsampling], flux[::downsampling], flux_err[::downsampling]

def filter_outliers(wave, flux, flux_err, sigma_outliers, sig_downsampling = 20, normalize = True, **kwargs):
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
    flux_err = flux_err[valid]
    if normalize:
        flux, flux_err = normalize_flux(flux, flux_err, **kwargs)

    return wave[valid], flux, flux_err
