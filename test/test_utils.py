# imports -- standard
import os
import warnings
import numpy as np

# imports -- internal
from respext import utils

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
Ia_SPEC_FILE = os.path.join(os.path.dirname(TEST_DIR), 'example', 'sn2006mo', 'sn2006mo-20061113.21-fast.flm')

def test_load_spectrum():
	wave, flux, flux_err = utils.load_spectrum(Ia_SPEC_FILE, normalize = False)
	assert type(wave) == type(np.array([0.]))
	assert np.abs(np.max(flux) - 1) > 0.0001 # max should not be one if not normalized by max
	assert len(wave) == len(flux)
	assert len(flux) == len(flux_err)

def test_load_spectrum_median():
	wave, flux, flux_err = utils.load_spectrum(Ia_SPEC_FILE, norm_method = 'median')
	assert np.max(flux) > 1
	assert np.min(flux) < 1

def test_load_spectrum_badnorm():
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('always')
		wave, flux, flux_err = utils.load_spectrum(Ia_SPEC_FILE, norm_method = 'UNSUPPORTED MODE')
		assert np.abs(np.max(flux) - 1) < 0.0001