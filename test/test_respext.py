# imports -- standard
import os
import warnings
import pandas as pd

# imports -- internal
import respext

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
Ia_SPEC_FILE = os.path.join(os.path.dirname(TEST_DIR), 'example', 'sn2006mo', 'sn2006mo-20061113.21-fast.flm')
Ia_REDSHIFT = 0.0459

def test_default_Ia():
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('always')
		s = respext.SpExtractor(Ia_SPEC_FILE, Ia_REDSHIFT, sn_type = 'UNSUPPORTED TYPE')
		assert s.sn_type == 'Ia'

def test_Ia_ds4():
	'''compare against results derived from running original v04 code'''
	s = respext.SpExtractor(Ia_SPEC_FILE, Ia_REDSHIFT, downsampling = 4)
	s.process_spectrum()
	v04_result = pd.DataFrame([
				  {'Ca II H&K': 70.35277629914613,
				  'Fe II 4800A': 148.75978861487167,
				  'Mg II 4300A': 89.27372695873309,
				  'S W': 54.87262613646505,
				  'Si 4000A': 18.052962141956847,
				  'Si II 5800A': 32.14803826168107,
				  'Si II 6150A': 92.66242302398456},
				 {'Ca II H&K': 8.364596002854674,
				  'Fe II 4800A': 20.04006429775402,
				  'Mg II 4300A': 14.289453677386549,
				  'S W': 12.721106840115947,
				  'Si 4000A': 5.402186753522786,
				  'Si II 5800A': 9.41015694292241,
				  'Si II 6150A': 13.418149759285184},
				 {'Ca II H&K': 13.794027999084344,
				  'Fe II 4800A': 14.961065371575067,
				  'Mg II 4300A': 14.606080401458689,
				  'S W': 6.336146873234958,
				  'Si 4000A': 10.5519680227087,
				  'Si II 5800A': 15.371923633969024,
				  'Si II 6150A': 13.06951586975049},
				 {'Ca II H&K': 0.5734110353439821,
				  'Fe II 4800A': 1.1086681036155808,
				  'Mg II 4300A': 1.6117489811462913,
				  'S W': 0.6508691028539108,
				  'Si 4000A': 0.559168482283103,
				  'Si II 5800A': 0.7312428191931065,
				  'Si II 6150A': 0.6566419372045513}],
				  index = ['pEW', 'e_pEW', 'vel', 'e_vel']).T
	# cannot compare ev because generate from random samples
	assert (s.results.loc[:, ['pEW', 'e_pEW', 'vel']].sort_index().round(4) == 
		    v04_result.loc[:, ['pEW', 'e_pEW', 'vel']].sort_index().round(4)).all().all()

def test_Ia_ds8_sigma_outliers():
	'''compare against results derived from running original v04 code'''
	s = respext.SpExtractor(Ia_SPEC_FILE, Ia_REDSHIFT, downsampling = 8, sigma_outliers = 3)
	s.process_spectrum()
	v04_result = pd.DataFrame([
				  {'Ca II H&K': 64.8942130293112,
				  'Fe II 4800A': 144.24826091678415,
				  'Mg II 4300A': 81.22127191544115,
				  'S W': 54.981174160129655,
				  'Si 4000A': 27.968181103738985,
				  'Si II 5800A': 30.9913871731691,
				  'Si II 6150A': 97.80639491405469},
				 {'Ca II H&K': 7.671018812357933,
				  'Fe II 4800A': 19.115898710960252,
				  'Mg II 4300A': 11.198824010114148,
				  'S W': 10.829634560190732,
				  'Si 4000A': 7.3838726132206745,
				  'Si II 5800A': 8.86062598539543,
				  'Si II 6150A': 13.126814544634636},
				 {'Ca II H&K': 13.570903663553109,
				  'Fe II 4800A': 15.482971308274221,
				  'Mg II 4300A': 13.426294893371361,
				  'S W': 4.322479142163436,
				  'Si 4000A': 10.446429962194053,
				  'Si II 5800A': 13.755893848599149,
				  'Si II 6150A': 14.038714795668415},
				 {'Ca II H&K': 0.5386187930622782,
				  'Fe II 4800A': 0.5719131831755743,
				  'Mg II 4300A': 1.1260620309725693,
				  'S W': 1.0095912705446528,
				  'Si 4000A': 0.6085026634057323,
				  'Si II 5800A': 0.6696959503312099,
				  'Si II 6150A': 0.7592544681971937}],
				  index = ['pEW', 'e_pEW', 'vel', 'e_vel']).T
	# cannot compare ev because generate from random samples
	assert (s.results.loc[:, ['pEW', 'e_pEW', 'vel']].sort_index().round(4) == 
		    v04_result.loc[:, ['pEW', 'e_pEW', 'vel']].sort_index().round(4)).all().all()

