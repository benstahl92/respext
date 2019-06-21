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
		s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, sn_type = 'UNSUPPORTED TYPE')
		assert s.sn_type == 'Ia'

def test_no_args():
	with warnings.catch_warnings(record = True) as w:
		warnings.simplefilter('always')
		s = respext.SpExtractor()
		assert not hasattr(s, 'lines')

def test_Ia_ds4():
	'''compare against results derived from running original v04 code'''
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, sn_type = 'Ia_LEGACY', downsampling = 4)
	s.process_spectrum()
	v04_result = pd.DataFrame([
				 {'Ca II H&K': 70.35277629914837,
				  'Fe II 4800A': 148.7597886148739,
				  'Mg II 4300A': 89.27372695873125,
				  'S W': 54.872626136466515,
				  'Si 4000A': 18.052962141957,
				  'Si II 5800A': 32.1480382616784,
				  'Si II 6150A': 92.66242302398052},
				 {'Ca II H&K': 8.364596002854674,
				  'Fe II 4800A': 20.04006429775402,
				  'Mg II 4300A': 14.289453677386549,
				  'S W': 12.721106840115947,
				  'Si 4000A': 5.402186753522786,
				  'Si II 5800A': 9.41015694292241,
				  'Si II 6150A': 13.418149759285182},
				 {'Ca II H&K': 14.24073024798393,
				  'Fe II 4800A': 15.657121797409314,
				  'Mg II 4300A': 13.426294893371361,
				  'S W': 5.715167048246831,
				  'Si 4000A': 11.397517592080256,
				  'Si II 5800A': 13.90247981192803,
				  'Si II 6150A': 13.622994140594392},
				 {'Ca II H&K': 0.5463522358894352,
				  'Fe II 4800A': 1.5772728021002953,
				  'Mg II 4300A': 1.548214921525901,
				  'S W': 0.6149101363774669,
				  'Si 4000A': 0.527022136125741,
				  'Si II 5800A': 0.7647588257732787,
				  'Si II 6150A': 0.6077254280342788}],
				  index = ['pEW', 'e_pEW', 'vel', 'e_vel']).T
	# cannot compare ev because generate from random samples
	assert (s.results.loc[:, ['pEW', 'e_pEW', 'vel']].sort_index().round(4) == 
		    v04_result.loc[:, ['pEW', 'e_pEW', 'vel']].sort_index().round(4)).all().all()

def test_Ia_ds8_sigma_outliers():
	'''compare against results derived from running original v04 code'''
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, sn_type = 'Ia_LEGACY', downsampling = 8, sigma_outliers = 3)
	s.process_spectrum()
	v04_result = pd.DataFrame([
				 {'Ca II H&K': 64.89421302931024,
				  'Fe II 4800A': 144.24826091678375,
				  'Mg II 4300A': 81.22127191544212,
				  'S W': 54.98117416012838,
				  'Si 4000A': 27.968181103738765,
				  'Si II 5800A': 30.991387173167816,
				  'Si II 6150A': 97.80639491405525},
				 {'Ca II H&K': 7.671018812357933,
				  'Fe II 4800A': 19.115898710960252,
				  'Mg II 4300A': 11.198824010114148,
				  'S W': 10.829634560190732,
				  'Si 4000A': 7.383872613220675,
				  'Si II 5800A': 8.86062598539543,
				  'Si II 6150A': 13.126814544634636},
				 {'Ca II H&K': 14.576153683867679,
				  'Fe II 4800A': 15.482971308274221,
				  'Mg II 4300A': 11.85981327648231,
				  'S W': 4.940679329342198,
				  'Si 4000A': 11.291702871835378,
				  'Si II 5800A': 14.342628839676998,
				  'Si II 6150A': 12.931292055519176},
				 {'Ca II H&K': 0.6177218733276749,
				  'Fe II 4800A': 0.6080828718249356,
				  'Mg II 4300A': 1.2176238278016764,
				  'S W': 0.9548865250811992,
				  'Si 4000A': 1.0859146794295496,
				  'Si II 5800A': 0.6607352581784063,
				  'Si II 6150A': 0.6934893762788}],
				  index = ['pEW', 'e_pEW', 'vel', 'e_vel']).T
	# cannot compare ev because generate from random samples
	assert (s.results.loc[:, ['pEW', 'e_pEW', 'vel']].sort_index().round(4) == 
		    v04_result.loc[:, ['pEW', 'e_pEW', 'vel']].sort_index().round(4)).all().all()

def test_Ia_ds8_bad_feature_fail():
	'''give a bad continuum region to measure and ensure fails as expected'''
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, downsampling = 8)
	s.lines = pd.DataFrame(index = ['Ca II H&K'],
                        columns = ['rest_wavelength', 'low_1', 'high_1', 'low_2', 'high_2', 'blue_deriv', 'red_deriv'],
                        data = [(3945.12, 3770, 3800, 3800, 3950, 0, 0)])
	s.process_spectrum()
	assert (s.results.loc['Ca II H&K'].isnull().all())

def test_Ia_ds8_bad_meas_feat_fail():
	'''give a bad feature to measure and ensure fails as expected'''
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, downsampling = 8)
	s.lines = pd.DataFrame(index = ['Ca II H&K'],
                        columns = ['rest_wavelength', 'low_1', 'high_1', 'low_2', 'high_2', 'blue_deriv', 'red_deriv'],
                        data = [(3945.12, 3800, 3800, 3800, 3950, 0, 0)])
	s.process_spectrum()
	assert (s.results.loc['Ca II H&K'].isnull().all())

def test_plotter_init():
	'''make sure plotter gets set up correctly'''
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, downsampling = 8)
	s.process_spectrum()
	s.plot(save = False, display = False, title = 'Ia test', xlabel = 'x', ylabel = 'y', figsize = (6, 3))
	assert len(s.plotter) == 2
	assert s.plotter[1].title.get_text() == 'Ia test'
	assert s.plotter[1].xaxis.get_label_text() == 'x'
	assert s.plotter[1].yaxis.get_label_text() == 'y'
	assert s.plotter[0].get_figwidth() == 6
	assert s.plotter[0].get_figheight() == 3

def test_report(capfd):
	'''test reporting'''
	s = respext.SpExtractor(spec_file = Ia_SPEC_FILE, z = Ia_REDSHIFT, downsampling = 8)
	s.process_spectrum()
	s.report()
	out, err = capfd.readouterr()
	assert 'Ca II H&K' in out
