# imports -- standard
import math
import warnings
import pandas as pd
import numpy as np
from scipy import signal

__all__ = ['LINES', 'get_speed', 'pseudo_continuum', 'pEW', 'absorption_depth']

# Ia lines from Silverman et al. (2012)
LINES_Ia = pd.DataFrame(index = ['Ca II H&K', 'Si II 4000', 'Mg II', 'Fe II', 'S II W',
                                 'Si II 5972', 'Si II 6355', 'O I triplet', 'Ca II near-IR triplet'],
                        columns = ['rest_wavelength', 'low_1', 'high_1', 'low_2', 'high_2', 'blue_deriv', 'red_deriv'],
                        data = [(3945.28, 3400, 3800, 3800, 4100, 0, 0),
                                (4129.73, 3850, 4000, 4000, 4150, 0, 0),
                                (np.nan, 4000, 4150, 4350, 4700, 0, 0),
                                (np.nan, 4350, 4700, 5050, 5550, 0, 0),
                                (5624.32, 5100, 5300, 5450, 5700, 0, 0),
                                (5971.85, 5400, 5700, 5750, 6000, 0, 0),
                                (6355.21, 5750, 6060, 6200, 6600, 0, 0),
                                (7773.37, 6800, 7450, 7600, 8000, -2e-18, 0),
                                (8578.75, 7500, 8100, 8200, 8900, 0, 0)])

LINES_Ib = pd.DataFrame(index = ['Fe II', 'He I'],
                        columns = ['rest_wavelength', 'low_1', 'high_1', 'low_2', 'high_2', 'blue_deriv', 'red_deriv'],
                        data = [(5169, 4950, 5050, 5150, 5250, 0, 0),
                                (5875, 5350, 5450, 5850, 6000, 0, 0)])

LINES_Ic = pd.DataFrame(index = ['Fe II', 'O I'],
                        columns = ['rest_wavelength', 'low_1', 'high_1', 'low_2', 'high_2', 'blue_deriv', 'red_deriv'],
                        data = [(5169, 4950, 5050, 5150, 5250, 0, 0),
                                (7773, 7250, 7350, 7750, 7950, 0, 0)])

LINES = dict(Ia=LINES_Ia, Ib=LINES_Ib, Ic=LINES_Ic)

def get_speed(lambda_m, lambda_m_err, lambda_rest, c = 299.792458):
    '''
    calculate speed of feature from relativistic Doppler formula

    Parameters
    ----------
    lambda_m(_err) : measured wavelength(uncertainty) of de-redshifted feature
    lambda_rest : rest wavelength of feature
    c : speed of light in km/s

    Returns
    -------
    velocity and uncertainty of feature
    '''

    l_quot = lambda_m / lambda_rest
    velocity = -c * (l_quot ** 2 - 1) / (l_quot ** 2 + 1)
    velocity_err = c * 4 * l_quot * lambda_m_err / (lambda_rest * (l_quot ** 2 + 1)**2)
    return velocity, velocity_err

def pseudo_continuum(x, y, y_err):
    '''
    given points, define pseudo continuum and uncertainty as a function of wavelength

    Parameters
    ----------
    x, y, y_err : endpoints (and flux errors) which define the continuum

    Returns
    -------
    function the returns the continuum flux and uncertainty (upper, lower, mean) at the specified wavelength(s)
    '''

    # slopes
    m = (np.diff(y) / np.diff(x))[0]
    m_bot = (np.diff(y - y_err) / np.diff(x))[0]
    m_top = (np.diff(y + y_err) / np.diff(x))[0]

    # functions
    pc = lambda wave: m * (wave - x[0]) + y[0]
    e_pc_bot = lambda wave: m_bot * (wave - x[0]) + y[0] - y_err[0]
    e_pc_top = lambda wave: m_top * (wave - x[0]) + y[0] + y_err[0]

    return lambda wave: (pc(wave), e_pc_bot(wave), e_pc_top(wave),
                         np.mean(np.abs([e_pc_bot(wave) - pc(wave), e_pc_top(wave) - pc(wave)]), axis = 0))

def _pEW(wavelength, flux, eflux, cont, cont_coords):
    '''internal pEW calculation -- only pEW should be exposed for external use'''

    val = 0
    sq_e_val = 0
    fc, _, _, e_fc = cont(wavelength)
    for i in range(len(wavelength)):
        if (wavelength[i] > cont_coords[0, 0]) and (wavelength[i] < cont_coords[0, 1]):
            dwave = 0.5 * (wavelength[i + 1] - wavelength[i - 1])
            val += dwave * (1 - flux[i] / fc[i])
            sq_e_val += (eflux[i]**2 + e_fc[i]*(flux[i] / fc[i])**2) * (dwave / fc[i])**2
    return val, np.sqrt(sq_e_val)

def pEW(wavelength, flux, cont, cont_coords, err_method = 'default', eflux = np.array([np.nan])):
    '''
    calculates the pEW between two chosen points
    cont should be the return of a call to <pseudo_continuum>
    '''

    # calculate pEW
    pEW_val, pEW_err = _pEW(wavelength, flux, eflux, cont, cont_coords)

    # calculate pEW uncertainty
    if err_method == 'data':
        if ~np.isnan(pEW_err):
            return pEW_val, pEW_err
        else:
            warnings.warn('NaN in flux err, computing pEW error using default method instead of from data')

    # otherwise compute error using default method
    eflux = np.sqrt(np.mean(signal.cwt(flux, signal.ricker, [1])**2))
    pEW_stat_err = eflux
    pEW_cont_err = np.abs(cont_coords[0, 0] - cont_coords[0, 1]) * eflux
    pEW_err = math.hypot(pEW_stat_err, pEW_cont_err)
    
    return pEW_val, pEW_err

def absorption_depth(lambda_m, flux_m, flux_m_err, cont):
    '''compute absorption depth relative to the pseudo continuum'''

    fc, _, _, e_fc = cont(lambda_m) # continuum flux and uncertainty at minimum

    a = ( fc - flux_m ) / fc
    a_err = np.sqrt(flux_m_err**2 + e_fc*(flux_m / fc)**2) / fc
    return a, a_err
