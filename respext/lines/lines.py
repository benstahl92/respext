# imports -- standard
import math
import pandas as pd
import numpy as np
from scipy import interpolate, signal

# Element, rest wavelength, low_1, high_1, low_2, high_2
LINES_Ia = [('Ca II H&K', 3945.12, 3450, 3800, 3800, 3950),
            ('Si 4000A', 4129.73, 3840, 3950, 4000, 4200),
            ('Mg II 4300A', 4481.2, 4000, 4250, 4300, 4700),
            ('Fe II 4800A', 5083.42, 4300, 4700, 4950, 5600),
            ('S W', 5536.24, 5050, 5300, 5500, 5750),
            ('Si II 5800A', 6007.7, 5400, 5700, 5800, 6000),
            ('Si II 6150A', 6355.1, 5800, 6100, 6200, 6600)
            ]

LINES_Ia = pd.DataFrame(index = ['Ca II H&K', 'Si 4000A', 'Mg II 4300A', 'Fe II 4800A',
                                 'S W', 'Si II 5800A', 'Si II 6150A'],
                        columns = ['rest_wavelength', 'low_1', 'high_1', 'low_2', 'high_2'],
                        data = [(3945.12, 3450, 3800, 3800, 3950),
                                (4129.73, 3840, 3950, 4000, 4200),
                                (4481.2, 4000, 4250, 4300, 4700),
                                (5083.42, 4300, 4700, 4950, 5600),
                                (5536.24, 5050, 5300, 5500, 5750),
                                (6007.7, 5400, 5700, 5800, 6000),
                                (6355.1, 5800, 6100, 6200, 6600)])

LINES_Ib = [('Fe II', 5169, 4950, 5050, 5150, 5250),
            ('He I', 5875, 5350, 5450, 5850, 6000)]

LINES_Ic = [('Fe II', 5169, 4950, 5050, 5150, 5250),
            ('O I', 7773, 7250, 7350, 7750, 7950)]

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

def pseudo_continuum(cont_coords):
	'''
	get pseudo continuum as a function of wavelength from feature edges
	
	Parameters
	----------
	cont_coords : endpoints of continuum, given as np.array([x1,x2], [y1,y2]

	Returns
	-------
	interpolate.inter1d object of the continuum
	'''

	return interpolate.interp1d(cont_coords[0], cont_coords[1], bounds_error = False, fill_value = 1)

def pEW(wavelength, flux, cont, cont_coords):
    '''
    calculates the pEW between two chosen points
    cont should be the return of a call to <pseudo_continuum>
    '''

    # normalize flux within pseudo continuum
    nflux = flux / cont(wavelength)

    # calculate pEW
    pEW = 0
    for i in range(len(wavelength)):
        if wavelength[i] > cont_coords[0, 0] and wavelength[i] < cont_coords[0, 1]:
            dwave = 0.5 * (wavelength[i + 1] - wavelength[i - 1])
            pEW += dwave * (1 - nflux[i])

    # calculate pEW uncertainty
    flux_err = np.abs(signal.cwt(flux, signal.ricker, [1])).mean()
    pEW_stat_err = flux_err
    pEW_cont_err = np.abs(cont_coords[0, 0] - cont_coords[0, 1]) * flux_err
    pEW_err = math.hypot(pEW_stat_err, pEW_cont_err)
    
    return pEW, pEW_err

def absorption_depth(lambda_m, flux_m, flux_m_err, cont):
	'''compute absorption depth relative to the pseudo continuum'''

	a = cont(lambda_m) - flux_m
	a_err = flux_m_err
	return a, a_err