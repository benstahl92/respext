# ------------------------------------------------------------------------------
# respext --- redux of spextractor (https://github.com/astrobarn/spextractor)
#
#   an automated pEW, velocity, and absorption "depth" extractor
#   optimized for SN Ia spectra, though minimal support is 
#   available for SNe Ib and Ic (and could be further developed)
#
# this code base is a re-factorization and extension of the spextractor
# package written by Seméli Papadogiannakis (linked above)
# ------------------------------------------------------------------------------

# imports -- standard
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy

# imports -- internal
from . import utils
from .lines import LINES, get_speed, pseudo_continuum, pEW, absorption_depth

class SpExtractor:
    '''container for a SN spectrum, with methods for all processing'''

    def __init__(self, spec_file, z, sn_type = 'Ia', remove_gaps = True, auto_prune = True,
                 sigma_outliers = None, downsampling = None, plot = False, **kwargs):

        # store arguments from instantiation
        self.spec_file = spec_file
        self.z = z
        self.sn_type = sn_type
        self.plot = plot

        # select appropriate set of spectral lines
        if self.sn_type not in ['Ia', 'Ib', 'Ic']:
            warnings.warn('{} is not a supported type, defaulting to Ia'.format(self.sn_type))
            self.sn_type = 'Ia'
        self.lines = LINES[self.sn_type]

        # load and prepare spectrum
        self.prepare_spectrum(remove_gaps, auto_prune, sigma_outliers, downsampling, **kwargs)

        # setup model
        self.setup_model()

        # setup plot
        if self.plot:
            self.setup_plot(**kwargs)

    def prepare_spectrum(self, remove_gaps, auto_prune, sigma_outliers, downsampling, **kwargs):
        '''
        perform preparation steps of loading, de-redshifting, and normalizing flux of spectrum
        optional intermediate steps: remove gaps, prune, remove outliers, downsample
        '''

        wave, flux = utils.load_spectrum(self.spec_file, **kwargs)
        wave = utils.de_redshift(wave, self.z)
        if remove_gaps:
            wave, flux = utils.remove_gaps(wave, flux)
        if auto_prune:
            wave, flux = utils.auto_prune(wave, flux, self.lines, **kwargs)
        if sigma_outliers is not None:
            wave, flux = utils.filter_outliers(wave, flux, sigma_outliers, **kwargs)
        if downsampling is not None:
            wave, flux = utils.downsample(wave, flux, downsampling)
        self.wave, self.flux = wave, flux

    def setup_plot(self, plot_title = None):
        '''setup plot'''

        fig, ax = plt.subplots(1, 1)
        if plot_title is None:
            plot_title = self.spec_file
        ax.set_title(plot_title)
        ax.set_xlabel('Rest Wavelength (\u212B)', size=14)
        ax.set_ylabel('Normalized Flux', size=14)
        ax.plot(self.wave, self.flux, color='k', alpha=0.5, label = 'Processed Input Spectrum')
        self.plotter = (fig, ax)

    def setup_model(self):
        '''set up model'''

        self.x, self.y = self.wave[:, np.newaxis], self.flux[:, np.newaxis]
        kernel = GPy.kern.Matern32(input_dim = 1, lengthscale = 300, variance = 0.001)
        m = GPy.models.GPRegression(self.x, self.y, kernel)
        m['Gaussian.noise.variance'][0] = 0.0027
        self.model = m

    def fit_model(self):
        '''fit model to data --- this step may take some time'''

        self.model.optimize()
        self.mod_mean, self.mod_var = self.model.predict(self.x)
        conf = np.sqrt(self.mod_var)

        if self.plot:
            fig, ax = self.plotter
            ax.plot(self.x, self.mod_mean, color = 'red')
            ax.fill_between(self.x[:, 0], self.mod_mean[:, 0] - conf[:, 0],
                             self.mod_mean[:, 0] + conf[:, 0],
                             alpha = 0.3, color = 'red')

    def get_feature_min(self, lambda_0, x_values, y_values, feature):
        '''compute location and flux of feature minimum'''

        # find deepest absorption
        min_pos = y_values.argmin()
        if (min_pos == 0) or (min_pos == y_values.shape[0]):
            return np.nan, np.nan, np.nan, np.nan

        # measured wavelength and flux of feature
        lambda_m, flux_m = x_values[min_pos], y_values[min_pos]

        # sample possible spectra from posterior and find the minima
        samples = self.model.posterior_samples_f(x_values[:, np.newaxis], 100).squeeze().argmin(axis = 0)

        # exclude points at either end
        samples = samples[np.logical_and(samples != 0, samples != x_values.shape[0])]
        if samples.size == 0:
            return np.nan, np.nan, np.nan, np.nan

        # do error estimation as standard deviation of suitable realizations
        lambda_m_samples, flux_m_samples = x_values[samples], y_values[samples]
        lambda_m_err, flux_m_err = lambda_m_samples.std(), flux_m_samples.std()

        # add to plot
        if self.plot:
            fig, ax = self.plotter
            ax.axvline(lambda_m, color = 'k', linestyle = '--')
            # add text label for each line
            # perhaps also add a marker for identified feature minimum

        return lambda_m, lambda_m_err, flux_m, flux_m_err

    def measure_feature(self, feature):
        '''measure feature'''

        # run optimization if it has not already been done
        if not hasattr(self, 'mod_mean'):
            self.fit_model()

        # unpack feature information
        rest_wavelength, low_1, high_1, low_2, high_2 = self.lines.loc[feature]

        # identify indices of feature edge bounds
        cp_1 = np.searchsorted(self.x[:, 0], (low_1, high_1))
        index_low, index_hi = cp_1
        cp_2 = np.searchsorted(self.x[:, 0], (low_2, high_2))
        index_low_2, index_hi_2 = cp_2

        # check if feature outside range of spectrum
        if (index_low == index_hi) or (index_low_2 == index_hi_2):
            return pd.Series([np.nan] * 6, index = ['pEW', 'e_pEW', 'vel', 'e_vel', 'abs', 'e_abs'])

        # identify indices of feature edges from where model peaks
        max_point = index_low + np.argmax(self.mod_mean[index_low: index_hi])
        max_point_2 = index_low_2 + np.argmax(self.mod_mean[index_low_2: index_hi_2])

        # get wavelength, model flux at the feature edges
        cp1_x, cp1_y = self.x[max_point, 0], self.mod_mean[max_point, 0]
        cp2_x, cp2_y = self.x[max_point_2, 0], self.mod_mean[max_point_2, 0]

        # get feature minimum
        lambda_m, lambda_m_err, flux_m, flux_m_err = self.get_feature_min(rest_wavelength, self.x[max_point:max_point_2, 0],
                                                                          self.y[max_point:max_point_2, 0], feature)

        # in future, may want to separate the above into a parametrize_feature function and then do calcs afterward
        # also store many of the results in class attributes

        # compute and store velocity
        velocity, velocity_err = get_speed(lambda_m, lambda_m_err, rest_wavelength)

        # if velocity is not detected, don't do pEW
        if np.isnan(velocity):
           return pd.Series([np.nan] * 6, index = ['pEW', 'e_pEW', 'vel', 'e_vel', 'abs', 'e_abs'])

        # get pseudo continuum
        if self.plot:
            ax_arg = self.plotter[1]
        else:
            ax_arg = None
        cont = pseudo_continuum(np.array([[cp1_x, cp2_x], [cp1_y, cp2_y]]), ax = ax_arg)

        # compute pEWs
        pew_results, pew_err_results = pEW(self.wave, self.flux, cont, np.array([[cp1_x, cp2_x], [cp1_y, cp2_y]]))

        # compute absorption depth
        a, a_err = absorption_depth(lambda_m, flux_m, flux_m_err, cont)

        return pd.Series([pew_results, pew_err_results, velocity, velocity_err, a, a_err],
                         index = ['pEW', 'e_pEW', 'vel', 'e_vel', 'abs', 'e_abs'])

    def process_spectrum(self):
        '''do full processing of spectrum by measuring each feature'''

        return self.lines.apply(lambda feature: self.measure_feature(feature.name), axis = 1, result_type = 'expand')


### make plotting its own method
