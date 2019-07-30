# imports -- standard
import os
import warnings
import numpy as np

# imports -- internal
from respext import utils

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
Ia_SPEC_FILE = os.path.join(os.path.dirname(TEST_DIR), 'example', 'sn2006mo', 'sn2006mo-20061113.21-fast.flm')
Ia_REDSHIFT = 0.0459
Ia_SPEC_FILE_NOERR = os.path.join(os.path.dirname(TEST_DIR), 'example', 'sn2006mo', 'sn2006mo-20061113.21-fast_noerr.flm')
SCALED_SPEC_FILE = os.path.join(os.path.dirname(TEST_DIR), 'example', 'sn2013gd', 'sn2013gd-20131128.396-ui.flm')

def test_load_spectrum():
	wave, flux, eflux = utils.load_spectrum(Ia_SPEC_FILE)
	assert type(wave) == type(np.array([0.]))
	assert len(wave) == len(flux)
	assert len(flux) == len(eflux)
	wave, flux, eflux = utils.load_spectrum(Ia_SPEC_FILE_NOERR)
	assert np.isnan(eflux).all()
	wave, flux, eflux = utils.load_spectrum(SCALED_SPEC_FILE, scale = 'auto')
	assert np.median(flux) < 1e-10

def test_extinction_correction():
	wave, flux, eflux = utils.load_spectrum(Ia_SPEC_FILE)
	f = utils.extinction_correction(wave, flux, None)
	assert (f == flux).all()
	f = utils.extinction_correction(wave, flux, 1)
	assert np.median(f[wave > 5000])/np.median(f) < np.median(flux[wave > 5000])/np.median(flux)

def test_de_redshift():
	wave, flux, eflux = utils.load_spectrum(Ia_SPEC_FILE)
	w = utils.de_redshift(wave, Ia_REDSHIFT)
	assert (w < wave).all()
	w = utils.de_redshift(wave, 0)
	assert (w == wave).all()

def test_rebin():
	wave, flux, eflux = utils.load_spectrum(Ia_SPEC_FILE)
	w, f, ef = utils.rebin(wave, flux, eflux, 2)
	assert len(w) <= len(wave) / 2
	w, f, ef = utils.rebin(wave, flux, eflux, 1)
	assert len(w) == len(wave)
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('always')
		w, f, ef = utils.rebin(wave, flux, eflux, 0)
		assert len(w) == len(wave)

def test_prune():
	from respext.lines import LINES
	lines = LINES['Ia'].loc[['Si II 5972', 'Si II 6355']]
	wave, flux, eflux = utils.load_spectrum(Ia_SPEC_FILE)
	w, f, ef = utils.auto_prune(wave, flux, eflux, lines, prune_leeway = 100)
	assert wave.min() < w.min()
	assert wave.max() > w.max()
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('always')
		w, f, ef = utils.auto_prune(wave, flux, eflux, lines, prune_leeway = -1)
		assert (wave == w).all()

def test_normalization():
	wave, flux, eflux = utils.load_spectrum(Ia_SPEC_FILE)
	f, ef, scale = utils.normalize_flux(flux, eflux, norm_method = 'max')
	assert len(f) == len(flux)
	assert (flux / scale == f).all()
	assert scale == flux.max()
	f, ef, scale = utils.normalize_flux(flux, eflux, norm_method = 'median')
	assert scale == np.median(flux)
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('always')
		f, ef, scale = utils.normalize_flux(flux, eflux, norm_method = 'UNSUPPORTED')
		assert scale == flux.max()