# imports -- standard
import os
import warnings
import numpy as np
import pandas as pd

# imports -- internal
import respext

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
Ia_SPEC_FILE = os.path.join(os.path.dirname(TEST_DIR), 'example', 'sn2006mo', 'sn2006mo-20061113.21-fast.flm')
Ia_SPEC_FILE_NOERR = os.path.join(os.path.dirname(TEST_DIR), 'example', 'sn2006mo', 'sn2006mo-20061113.21-fast_noerr.flm')
Ia_REDSHIFT = 0.0459

def test_default_Ia():
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('always')
		s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, sn_type = 'UNSUPPORTED TYPE')
		assert s.sn_type == 'Ia'

def test_plotter():
	'''make sure plotter gets set up correctly'''
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT)
	s.process()
	s.plot(save = False, display = False, title = 'Ia test', xlabel = 'x', ylabel = 'y')
	assert len(s.plotter) == 2
	assert s.plotter[1].title.get_text() == 'Ia test'
	s.plot(save = False, display = False, xlabel = 'x', ylabel = 'y', figsize = (6, 3))
	assert s.plotter[1].xaxis.get_label_text() == 'x'
	assert s.plotter[1].yaxis.get_label_text() == 'y'
	assert s.plotter[0].get_figwidth() == 6
	assert s.plotter[0].get_figheight() == 3

def test_report(capfd):
	'''test reporting'''
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT)
	s.process()
	s.report()
	out, err = capfd.readouterr()
	assert 'Ca II H&K' in out

def test_e_pEW():
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, pEW_err_method = 'LEGACY')
	s.process()
	assert s.results.loc[:, 'e_pEW'].notnull().any()
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, pEW_err_method = 'data')
	s.process()
	assert s.results.loc[:, 'e_pEW'].notnull().any()
	# now force NaN and check that defaults correctly
	s.eflux = [np.nan]*len(s.eflux)
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('always')
		s = respext.SpExtractor(spec_file = Ia_SPEC_FILE_NOERR, z = Ia_REDSHIFT, pEW_err_method = 'data')
		s.process()
		assert 'using default method' in str(w[0].message)

