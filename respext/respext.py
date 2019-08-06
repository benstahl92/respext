# ------------------------------------------------------------------------------
# respext --- redux of spextractor (https://github.com/astrobarn/spextractor)
#
#   an automated pEW, velocity, and absorption "depth" extractor
#   optimized for SN Ia spectra, though minimal support is 
#   available for SNe Ib and Ic (and could be further developed)
#
# this code base is a re-factorization and extension of the spextractor
# package written by Seméli Papadogiannakis (linked above)
#
# NB: this implementation (using Savitzky-Golay smoothing) is a significant
# departure from the original codebase -- refer to the GP branch for the
# now non-supported Gaussian Processes implementation
# ------------------------------------------------------------------------------

# imports -- standard
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage.filters import generic_filter
from scipy.interpolate import CubicSpline
import dill as pkl

# imports -- internal
from . import utils
from .lines import LINES, get_speed, pseudo_continuum, pEW, absorption_depth

class SpExtractor:
    '''container for a SN spectrum, with methods for all processing'''

    def __init__(self,
                 spec_file = None, z = None, save_file = None, sn_type = 'Ia', spec_flux_scale = 'auto', # SN/spectrum information
                 rebin = 1, prune = 200, ebv = None, # spectrum preprocessing information
                 no_overlap = True, lambda_m_err = 'measure', pEW_measure_from = 'data', pEW_err_method = 'default',
                 **kwargs):

        # store arguments from instantiation
        self.spec_file = spec_file
        self.z = z
        self.save_file = save_file
        self.sn_type = sn_type
        self.ebv = ebv
        self.no_overlap = no_overlap
        self.lambda_m_err = lambda_m_err
        self.pEW_measure_from = pEW_measure_from
        self.pEW_err_method = pEW_err_method

        # features to skip
        self.skip_features = []

        # determine how to instantiate
        if save_file is not None:
            self.load()
            self.save_file = save_file
            return
        elif (isinstance(self.spec_file, str) and (isinstance(self.z, float) or (isinstance(self.z, int)))):
            self.save_file = self.spec_file + '.respext.sav'
            pass
        else:
            warnings.warn('Instantiation failed. Must give a valid save file OR spectrum AND redshift')
            return

        # smoothing params
        self.signal_window_angstroms = 100

        # select appropriate set of spectral lines
        if self.sn_type not in ['Ia', 'Ib', 'Ic']:
            warnings.warn('{} is not a supported type, defaulting to Ia'.format(self.sn_type))
            self.sn_type = 'Ia'
        self.lines = LINES[self.sn_type].copy()

        # load and prepare spectrum
        self._wave, self._flux, self._eflux = utils.load_spectrum(self.spec_file, scale = spec_flux_scale)
        self.prepare(rebin = rebin, prune = prune, **kwargs)

    def save(self):
        '''save current state'''
        with open(self.save_file, 'wb') as f:
            pkl.dump(self.__dict__, f)

    def load(self):
        '''load from save file'''
        with open(self.save_file, 'rb') as f:
            tmp = pkl.load(f)
        self.__dict__.update(tmp)

    def prepare(self, rebin = 1, prune = 200, **kwargs):
        '''
        perform preparation steps of de-redshifting and normalizing flux of spectrum
        optional intermediate steps: rebin, prune
        '''

        # process spectrum
        wave, flux, eflux = self._wave, self._flux, self._eflux
        flux = utils.extinction_correction(wave, flux, self.ebv)
        wave = utils.de_redshift(wave, self.z)
        wave, flux, eflux = utils.rebin(wave, flux, eflux, rebin)
        self.wave, flux, eflux = utils.auto_prune(wave, flux, eflux, self.lines, prune_leeway = prune)
        self.flux, self.eflux, self.flux_norm_factor = utils.normalize_flux(flux, eflux, **kwargs)
        self.angstroms_per_pixel = np.abs(np.mean(self.wave[1:] - self.wave[:-1]))

        # (re)perform smoothing and instantiation of continuum DataFrame
        self._smooth()
        # columns - 1/2: left/right continuum points, a: absorption minimum point, cont: continuum interpolator
        self.continuum = pd.DataFrame(columns = ['wav1', 'flux1', 'e_flux1', 'wava', 'fluxa', 'wav2', 'flux2', 'e_flux2', 'cont'],
                                      index = self.lines.index)

    def _smooth(self):
        '''fit model to data --- this step may take some time'''

        # compute window sizes in pixels
        signal_window_pixels = int(np.ceil( (self.signal_window_angstroms / self.angstroms_per_pixel) / 2) * 2 - 1)

        # smooth and compute noise and derivative
        self.sflux = savgol_filter(self.flux, signal_window_pixels, 3)
        self.nflux = np.sqrt(generic_filter((self.flux - self.sflux)**2, np.mean, signal_window_pixels))
        self.sfluxprime = savgol_filter(self.flux, signal_window_pixels, 3, deriv = 1)

    def _get_continuum(self, feature):
        '''given a feature, automatically determine the continuum'''

        if feature in self.skip_features:
            return False

        # optionally enforce non-overlapping of continuum points
        rest_wavelength, low_1, high_1, low_2, high_2, blue_deriv, red_deriv = self.lines.loc[feature]
        if self.no_overlap:
            prev_iloc = self.continuum.index.get_loc(feature) - 1
            prev_blue_edge = self.continuum.loc[:, 'wav2'].iloc[prev_iloc]
            if (prev_iloc >= 0) and (prev_blue_edge > low_1) and (prev_blue_edge < high_1):
                 low_1 = self.continuum.loc[:, 'wav2'].iloc[prev_iloc]
            elif (prev_iloc >= 0) and (~np.isnan(prev_blue_edge)) and (prev_blue_edge > low_1):
                return False

        # identify indices of feature edge bounds
        cp_1 = np.searchsorted(self.wave, (low_1, high_1))
        index_low, index_hi = cp_1
        cp_2 = np.searchsorted(self.wave, (low_2, high_2))
        index_low_2, index_hi_2 = cp_2

        # check if feature outside range of spectrum
        if (index_low == index_hi) or (index_low_2 == index_hi_2):
            return False

        # find boundary by finding where derivative passes through specified slope (usually zero) from + to - 
        # blue side
        pos = self.sfluxprime * self.flux_norm_factor - blue_deriv > 0 # where derivative is positive
        p_ind = (pos[:-1] & ~pos[1:]).nonzero()[0] # indices where last positive occurs before going negative
        max_point_cands = p_ind[(p_ind >= index_low) & (p_ind <= index_hi)]
        # red side
        pos = self.sfluxprime * self.flux_norm_factor - red_deriv > 0 # where derivative is positive
        p_ind = (pos[:-1] & ~pos[1:]).nonzero()[0] # indices where last positive occurs before going negative
        max_point_2_cands = p_ind[(p_ind >= index_low_2) & (p_ind <= index_hi_2)]
        # if at least one candidate for each, use those that have the highest maxima
        if (len(max_point_cands) >= 1) and (len(max_point_2_cands) >= 1):
            max_point = max_point_cands[np.argmax(self.sflux[max_point_cands])]
            max_point_2 = max_point_2_cands[np.argmax(self.sflux[max_point_2_cands])]
        else:
            return False

        # get wavelength, model flux at the feature edges and define continuum
        self.continuum.loc[feature, ['wav1', 'flux1', 'e_flux1']] = self.wave[max_point], self.sflux[max_point], self.nflux[max_point]
        self.continuum.loc[feature, ['wav2', 'flux2', 'e_flux2']] = self.wave[max_point_2], self.sflux[max_point_2], self.nflux[max_point_2]
        self.continuum.loc[feature, 'cont'] = pseudo_continuum(self.continuum.loc[feature, ['wav1', 'wav2']].values,
                                                               self.continuum.loc[feature, ['flux1', 'flux2']].values,
                                                               self.continuum.loc[feature, ['e_flux1', 'e_flux2']].values)

        return True

    def pick_continuum(self, features = None):
        '''interactively select continuum points'''

        # reset skip features
        self.skip_features = []

        if features is None:
            print('Select number(s) feature (or features separated by commas):')
            for idx, feature in enumerate(self.lines.index):
                print('  {}:  {}'.format(idx, feature))
            response = input('Selection > ')
            features = self.lines.index[[int(i) for i in response.split(',')]]
        elif type(features) == type('this is a string'):
            features = [features]
        # not checking that everything else is iterable, or that it has real features, so use correctly!

        for feature in features:
            selection = (self.wave > self.lines.loc[feature, 'low_1'] - 250) & (self.wave < self.lines.loc[feature, 'high_2'] + 250)
            self.continuum.loc[feature, ['wav1', 'flux1', 'wav2', 'flux2']] = utils.define_continuum(self.wave[selection],
                                                                                                     self.sflux[selection],
                                                                                                     self.lines.loc[feature])
            selection1 = np.argmin(np.abs(self.wave - self.continuum.loc[feature, 'wav1']))
            selection2 = np.argmin(np.abs(self.wave - self.continuum.loc[feature, 'wav2']))
            self.continuum.loc[feature, ['e_flux1', 'e_flux2']] = self.nflux[selection1], self.nflux[selection2]
            if np.isnan(self.continuum.loc[feature, 'wav1']):
                self.skip_features.append(feature)
                self.continuum.loc[feature] = np.nan
            else:
                self.continuum.loc[feature, 'cont'] = pseudo_continuum(self.continuum.loc[feature, ['wav1', 'wav2']].values,
                                                                       self.continuum.loc[feature, ['flux1', 'flux2']].values,
                                                                       self.continuum.loc[feature, ['e_flux1', 'e_flux2']].values)

    def _get_feature_min(self, lambda_0, x_values, y_values, ey_values, feature):
        '''compute location and flux of feature minimum'''

        # find deepest absorption
        min_pos = y_values.argmin()
        if (min_pos < 5) or (min_pos > y_values.shape[0] - 5):
            return np.nan, np.nan, np.nan, np.nan

        # interpolate the feature with a Cubic Spline and use it to derive the absorption minimum
        cs = CubicSpline(x_values, y_values, extrapolate = False)
        extrema = cs.derivative().roots()
        lambda_m = extrema[np.argmin(cs(extrema))]
        flux_m = cs(lambda_m)

        # rough calculation of flux minimum uncertainty
        flux_m_err = np.median(ey_values)

        # compute wavelength error has std of all wavelengths corresponding to fluxes within noise from minimum if not overridden
        lambda_m_err = np.std(x_values[y_values < (flux_m + flux_m_err)])

        # optionally override lambda uncertainty with a specified value
        if (self.lambda_m_err != 'measure') and ((type(self.lambda_m_err) == type(1)) or (type(self.lambda_m_err) == type(1.1))):
            lambda_m_err = self.lambda_m_err

        return lambda_m, lambda_m_err, flux_m, flux_m_err

    def pick_feature_min(self, feature):
        '''interactively set absorption minimum'''

        # reset skip features
        self.skip_features = []

        selection = (self.wave > self.lines.loc[feature, 'low_1'] - 250) & (self.wave < self.lines.loc[feature, 'high_2'] + 250)
        wav1, flux1, wav2, flux2 = utils.define_continuum(self.wave[selection], self.sflux[selection], self.lines.loc[feature])
        selection = (self.wave > wav1) & (self.wave < wav2)
        self._min_data = self._get_feature_min(self.lines.loc[feature, 'rest_wavelength'], self.wave[selection],
                                               self.sflux[selection], self.nflux[selection], feature)
        self.skip_features.append(feature)

    def _measure_feature(self, feature):
        '''measure feature'''

        ### this check needs updating b/c shouldn't be done in in interactive mode

        # run optimization if it has not already been done, and check if successful
        if np.isnan(self.continuum.loc[feature, 'wav1']):
            if not self._get_continuum(feature):
                return pd.Series([np.nan] * 10,
                                 index = ['Fb', 'e_Fb', 'Fr', 'e_Fr', 'pEW', 'e_pEW', 'vel', 'e_vel', 'abs', 'e_abs'])

        # compute pEW
        if self.pEW_measure_from == 'model':
            tmp_flux, tmp_err = self.sflux, self.nflux
        else:
            tmp_flux, tmp_err = self.flux, self.eflux
        pew_results, pew_err_results = pEW(self.wave, tmp_flux, self.continuum.loc[feature, 'cont'],
                                           np.array([self.continuum.loc[feature, ['wav1', 'wav2']],
                                                     self.continuum.loc[feature, ['flux1', 'flux2']]]),
                                           err_method = self.pEW_err_method, eflux = tmp_err)

        # get feature minimum
        if feature not in self.skip_features:
            selection = (self.wave > self.continuum.loc[feature, 'wav1']) & (self.wave < self.continuum.loc[feature, 'wav2'])
            lambda_m, lambda_m_err, flux_m, flux_m_err = self._get_feature_min(self.lines.loc[feature, 'rest_wavelength'],
                                                                               self.wave[selection], self.sflux[selection], 
                                                                               self.nflux[selection], feature)
        elif hasattr(self, '_min_data'):
            lambda_m, lambda_m_err, flux_m, flux_m_err = self._min_data
        else:
            warnings.warn('manual minimum is not set for {}'.format(feature))
            return pd.Series([np.nan] * 10,
                             index = ['Fb', 'e_Fb', 'Fr', 'e_Fr', 'pEW', 'e_pEW', 'vel', 'e_vel', 'abs', 'e_abs'])

        self.continuum.loc[feature, ['wava', 'fluxa']] = lambda_m, flux_m

        # compute velocity
        velocity, velocity_err = get_speed(lambda_m, lambda_m_err, self.lines.loc[feature, 'rest_wavelength'])

        # compute absorption depth if velocity successful
        if np.isnan(velocity):
            a, a_err = np.nan, np.nan
            self.continuum.loc[feature, ['wava', 'fluxa']] = np.nan, np.nan
        else:
            a, a_err = absorption_depth(lambda_m, flux_m, flux_m_err, self.continuum.loc[feature, 'cont'])

        return pd.Series([self.continuum.loc[feature, 'flux1'] * self.flux_norm_factor / 1e-15, 
                          self.continuum.loc[feature, 'e_flux1'] * self.flux_norm_factor / 1e-15,
                          self.continuum.loc[feature, 'flux2'] * self.flux_norm_factor / 1e-15,
                          self.continuum.loc[feature, 'e_flux2'] * self.flux_norm_factor / 1e-15,
                          pew_results, pew_err_results, velocity, velocity_err, a, a_err],
                         index = ['Fb', 'e_Fb', 'Fr', 'e_Fr', 'pEW', 'e_pEW', 'vel', 'e_vel', 'abs', 'e_abs'])

    def process(self, features = 'all'):
        '''do full processing of spectrum by measuring each feature'''

        if type(features) == type('this is a string'):
            self.results = self.lines.apply(lambda feature: self._measure_feature(feature.name), axis = 1, result_type = 'expand')
        # otherwise ind should be a list of features to do, but not checking so use wisely!
        else:
            self.results.loc[features] = self.lines.loc[features].apply(lambda feature: self._measure_feature(feature.name), axis = 1, result_type = 'expand')

    def plot(self, initial_spec = True, model = True, continuum = True, lines = True, show_conf = True, show_line_labels = True,
             save = False, display = True, **kwargs):
        '''make plot'''

        self.plotter = utils.setup_plot(**kwargs)
        if initial_spec:
            utils.plot_spec(self.plotter[1], self.wave, self.flux, spec_color = 'black', spec_alpha = 0.4)
        if model:
            utils.plot_filled_spec(self.plotter[1], self.wave, self.sflux,
                                   self.nflux, fill_color = 'red', fill_alpha = 0.3)
        if continuum:
            utils.plot_continuum(self.plotter[1], self.continuum.loc[:, ['wav1', 'wav2', 'flux1', 'flux2', 'cont']],
                                 cp_color = 'black', cl_color = 'blue', cl_alpha = 0.6, show_conf = show_conf, conf_alpha = 0.15)
        if model:
            utils.plot_spec(self.plotter[1], self.wave, self.sflux, spec_color = 'red')
        if lines:
            utils.plot_lines(self.plotter[1], self.continuum,
                             show_line_labels = show_line_labels)
        plt.tight_layout()
        if save is not False:
            self.plotter[0].savefig(save)
        elif display:
            self.plotter[0].show()

    def report(self):
        '''print report'''

        print(self.results.round(2).to_string())
