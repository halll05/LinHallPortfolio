import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from spot import bidSPOT
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

global_N_INIT = 300

filenames=['data/nonrigid_rollall_processed.csv', 'data/nonrigid_pitch2_processed.csv',
'data/nonrigid_x4_processed.csv', 'data/nonrigid_y2_processed.csv',  'data/nonrigid_yaw2_processed.csv',
'data/nonrigid_z_processed.csv']

def extreme_event_split(filename):
	df = pd.read_csv(filename, sep=',')
	print(df.size)
	# df = pd.read_csv('data/nonrigid_rollall_processed.csv', sep=',')
	df = df.dropna()
	# print(df)

	X = df.iloc[:, 0]
	# X = df.dropna()
	X = pd.to_numeric(X, errors='coerce')
	X = X.values

	n_init = global_N_INIT
	init_data = X[:n_init] 	# initial batch
	data = X[n_init:]  		# stream

	q = 1e-3				# risk parameter
	d = 100  				# depth parameter
	s = bidSPOT(q,d)     	# biDSPOT object
	s.fit(init_data,data) 	# data import
	s.initialize() 	  		# initialization step
	results = s.run()    	# run
	s.plot(results)

	q_name = str(q)+'_depth'+str(d)

	# output_dir = "C:\\Users\\12525\\Documents\\Thesis\\code\\test\\figures\\extreme_event"
	name = Path(filename).stem
	# name = name + '_' + str(d)
	name = name + '_' + q_name+'.png'

	plt.title(str(q))

	plt.savefig('C:/Users/12525/Documents/Thesis/code/test/figures/extreme_event/{}'.format(name))

	plt.clf()

	run_ee_split(filename)

def run_ee_split(filename):

	# df = pd.read_csv(filename, sep=',')
	df = pd.read_csv(filename, sep=',')


	# print(df.isnull().any().any())
	# print(df.isnull().any())
	# df = df.astype(float)
	df = df.dropna()


	# print(df.isnull().any().any())
	# print(df.isnull().any())

	d = np.isfinite(df).any()
	# print(d)

	values = df.values
	# values = values.astype('float32')

	scaler = MinMaxScaler(feature_range=(0,1))
	scaled = scaler.fit_transform(values)

	reframed = DataFrame(scaled)
	X = reframed.iloc[:, 0].dropna()
	X = pd.to_numeric(X, errors='coerce')
	X = X.values

	# X = reframed.iloc[:, 0].dropna(inplace=True)
	# X = reframed.iloc[:, 0].to_numpy(dtype='float32')

# # reframed = DataFrame(scaled)

	# X = pd.to_numeric(X, errors='coerce')
	# X = X.values

	CV = 5
	# defines which chunk of the CV to visualize
	# e.g. CV=5 and VISUALIZE_CHUNK, this means visualize the 2nd of 5 chunks
	VISUALIZE_CHUNK = 5

	N_INIT = global_N_INIT

	# for threshold
	# for q_cms
	# results are totally different
	# q = 1e-5
	q = 1e-3				# risk parameter, probability to have extreme events
							# like percentile to be the threshold, the higher the number
							# the less alarms (outliers)
	d = 100  				# depth parameter

	# for Turb
	# q = 0.03

	# # for SRP_mgPL
	# q = 0.1
	# d = 300


	# KERNEL = DecisionTreeRegressor(max_features='sqrt')
	# KERNEL = KNeighborsRegressor(n_neighbors=3)
	# KERNEL = MLPRegressor(learning_rate='adaptive', max_iter=500)
	KERNEL_EXTREME = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
	       subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
	       min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None,
	       init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, 
	       warm_start=False, validation_fraction=0.1, tol=0.0001)

	KERNEL_NORMAL = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
	       subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
	       min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None,
	       init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, 
	       warm_start=False, validation_fraction=0.1, tol=0.0001)


	TESTING_PERCENTAGE = 1/float(CV)

	X_backup = X.copy()

	# find threshold
	init_data = X[:N_INIT] 	# initial batch
	data = X[N_INIT:]  		# stream


	s = bidSPOT(q,d)     	# biDSPOT object
	# s = SPOT(q)		     	# biDSPOT object
	s.fit(init_data,data) 	# data import
	s.initialize() 	  		# initialization step
	results = s.run()    	# run

	s.plot(results) 	 	# plot

	original = df.iloc[N_INIT:,:].copy()
	# restart index
	original = original.reset_index(drop='True')

	total_len = original.shape[0]
	# index of peaks results['alarms']
	# 20% for testing
	# 80% for training
	test_data_len = int(total_len*TESTING_PERCENTAGE)
	train_data_len = total_len - test_data_len


	r_2_list = []
	rmse_list = []
	vis_prediction = []
	vis_gap = []
	normal_event_id = np.setdiff1d(range(total_len),results['alarms'])
	for i in range(CV):
		df_backup = df.iloc[N_INIT:,:].copy()
		df_backup =df_backup.reset_index(drop='True')
		# decide where to make holes
		count = 0
		print("currently dealing with loop:", i)

		testing_start = test_data_len * i
		testing_end = testing_start + test_data_len
		testing_data_df = df_backup.iloc[testing_start:testing_end]

		if i == 0:
			training_data = df_backup.iloc[testing_end:]
		elif i<=4:
			training_data_first = df_backup.iloc[:testing_start]
			training_data_second = df_backup.iloc[testing_end:]
			training_data = pd.concat([training_data_first,training_data_second])
		else:
			training_data = df_backup.iloc[:testing_start]


		flag_array_extreme = np.array([m for m in results['alarms'] if m<testing_start or m>testing_end])
		# flag_array_extreme = flag_array_extreme - global_N_INIT
		# extreme_training = training_data.reindex(columns=flag_array_extreme)  
		extreme_training = training_data.loc[flag_array_extreme]
		# extreme_training = training_data.reindex(flag_array_extreme)
		flag_array_normal = np.array([m for m in normal_event_id if m<testing_start or m>testing_end])  
		# normal_training = training_data.reindex(columns=flag_array_normal)
		# normal_training = training_data[flag_array_normal].reindex(columns = flag_array_extreme)
		normal_training = training_data.reindex(flag_array_normal)

		# print("normal: ", normal_training.shape)
		# normal_training = training_data.loc[flag_array_normal]

	# 	extreme_training.replace([np.inf, -np.inf], np.nan, inplace=True)
	# 	normal_training.replace([np.inf, -np.inf], np.nan, inplace=True)

	# 	extreme_training.fillna(flag_array_extreme.mean())
	# 	normal_training.fillna(flag_array_normal.mean())
		print("flag: ", flag_array_extreme.shape)
		print("extreme: ", extreme_training)
		print("max: ", np.max(flag_array_extreme))
		KERNEL_EXTREME.fit(extreme_training.iloc[:,1:].values,extreme_training.iloc[:,0].values) 
		KERNEL_NORMAL.fit(normal_training.iloc[:,1:].values,normal_training.iloc[:,0].values)

		# testing
		x_testing_data = testing_data_df.iloc[:,1:].values
		y_testing_data = testing_data_df.iloc[:,0].values
		predictions = []
		for count in range(len(y_testing_data)):
			record_index = testing_start+count
			if record_index in results['alarms']:
				# extreme event
				tmp_pred = KERNEL_EXTREME.predict([x_testing_data[count]])
			else:
				# normal event
				tmp_pred = KERNEL_NORMAL.predict([x_testing_data[count]])
			predictions.append(tmp_pred)

		tmp_r_2 = r2_score(y_testing_data, predictions)
		r_2_list.append(tmp_r_2)

		tmp_rmse = mean_squared_error(y_testing_data, predictions, squared=False)
		rmse_list.append(tmp_rmse)
		print("length of rmse: ", len(rmse_list))

	# print("rmse list: ", rmse_list)
	print(r_2_list)
	print("avg_r2 is: ", sum(r_2_list) / len(r_2_list))

	print("RMSE of Testing Extreme Event Split is displayed below:")
	# print(np.average(mean_squared_error(test_y, reg_gbr.predict(test_x), squared=False)))
	print("avg_rmse is: ", sum(rmse_list) / len(rmse_list))


# for file in filenames:
# 	extreme_event_split(file)

filenames=['data/nonrigid_rollall_processed.csv', 'data/nonrigid_pitch2_processed.csv',
'data/nonrigid_x4_processed.csv', 'data/nonrigid_y2_processed.csv',  'data/nonrigid_yaw2_processed.csv',
'data/nonrigid_z_processed.csv']

tri_filenames=['data_triangleconfig/pitch7_triangle_processed.csv', 'data_triangleconfig/x_triangle_processed.csv', 'data_triangleconfig/y_triangle_processed.csv',
'data_triangleconfig/z_triangle_processed.csv', 'data_triangleconfig/rollfinal_triangle_processed.csv',
'data_triangleconfig/yaw_triangle_processed.csv']

# extreme_event_split('data/nonrigid_yaw2_processed.csv')
extreme_event_split('data_triangleconfig/pitch7_triangle_processed.csv')


