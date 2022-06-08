
# import numpy as np
# import matplotlib.pyplot as plt
# from spot import bidSPOT

# import pandas as pd

# df = pd.read_csv('../data/data_tracker1_roll.csv')
# X = df.iloc[:,0].values

# n_init = 1000
# init_data = X[:n_init] 	# initial batch
# data = X[n_init:]  		# stream

# q = 1e-3 				# risk parameter
# d = 450  				# depth parameter
# s = bidSPOT(q,d)     	# biDSPOT object
# s.fit(init_data,data) 	# data import
# s.initialize() 	  		# initialization step
# results = s.run()    	# run
# s.plot(results) 	 	# plot


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import r2_score
import random

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline


from datetime import datetime
from dateutil.parser import parse
from spot import bidSPOT,dSPOT,SPOT
# import util

# q_cms use bidSPOT(q,d) 
# Turb SPOT
# this defines cross validation times to get average accuracy value
# N means every time we loop through 1/N data points
CV = 5
# defines which chunk of the CV to visualize
# e.g. CV=5 and VISUALIZE_CHUNK, this means visualize the 2nd of 5 chunks
VISUALIZE_CHUNK = 5

COL_NAME = 't1_r'

N_INIT = 1000

# for threshold
# for q_cms
# results are totally different
# q = 1e-5
q = 1e-4				# risk parameter, probability to have extreme events
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

df = pd.read_csv('../data/data_tracker1_roll_header.csv',"r",delimiter=',')
X = df[COL_NAME].to_numpy(dtype='float')
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

plt.show()

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
	elif i<4:
		training_data_first = df_backup.iloc[:testing_start]
		training_data_second = df_backup.iloc[testing_end:]
		training_data = pd.concat([training_data_first,training_data_second])
	else:
		training_data = df_backup.iloc[:testing_start]


	flag_array_extreme = np.array([m for m in results['alarms'] if m<testing_start or m>testing_end])  
	extreme_training = training_data.loc[flag_array_extreme]
	flag_array_normal = np.array([m for m in normal_event_id if m<testing_start or m>testing_end])  
	normal_training = training_data.loc[flag_array_normal]

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
			tmp_pred = KERNEL_EXTREME.predict([x_testing_data[count]])
		predictions.append(tmp_pred)

	tmp_r_2 = r2_score(y_testing_data, predictions)
	r_2_list.append(tmp_r_2)
	print("current is cv fold", i, "r^2 value is", tmp_r_2)

print("avg_r2 is:", sum(r_2_list) / len(r_2_list))



# 	# for index in results['alarms'][-test_data_len:]:
# 	# based on cross validation idea applied to the peak data
# 	data_size = int(len(results['alarms'])/CV)
# 	# from front to back
# 	start_index = i*data_size
# 	end_index = start_index + data_size
# 	# # from back to front
# 	# start_index = (CV-i-1)*data_size
# 	# end_index = start_index + data_size
# 	for index in results['alarms'][start_index:end_index]:
# 		df_no_nan_no_time_backup[COL_NAME].iloc[index] = np.nan



# 	imp = IterativeImputer(max_iter=10, estimator=KERNEL)
# 	# imp = IterativeImputer(KNeighborsRegressor(n_neighbors=15))

# 	array_filled = imp.fit_transform(df_no_nan_no_time_backup)

# 	# calculate accuracy between backup and df
# 	ground_truth = []
# 	prediction = []

# 	for index in results['alarms'][start_index:end_index]:
# 		# rule 1: prediction should be above 0
# 		if array_filled[:,0][index]<0:
# 			prediction.append(0)
# 			array_filled[:,0][index] = 0
# 		else:
# 			prediction.append(array_filled[:,0][index])

# 		ground_truth.append(original[COL_NAME].iloc[index])

# 	tmp_r_2 = r2_score(ground_truth, prediction)
# 	r_2_list.append(tmp_r_2)

# 	if i+1 == VISUALIZE_CHUNK:
# 		vis_prediction = array_filled[:,0].copy()
# 		vis_gap = df_no_nan_no_time_backup[COL_NAME].copy()

# avg_r2 = sum(r_2_list) / len(r_2_list)
# print("average R^2 is: ", avg_r2)
# print(r_2_list)



# x_time = util.get_time_stamp_array(original)
# # original
# y1 = original[COL_NAME]
# # with gap
# y1_gap = vis_gap

# # filled with predictions
# y1_filled = vis_prediction 

# # # vis
# plt.scatter( x_time[results['alarms']], s.data[results['alarms']], marker="x", label=COL_NAME+"_original")
# plt.scatter( x_time, y1_gap, marker="v", label=COL_NAME+"_with_gap")
# plt.scatter( x_time[results['alarms']], y1_filled[results['alarms']], marker="+", label=COL_NAME+"_filled")
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.legend()
# plt.show()

# TODO 
# draw the graph x-axis gap length, y-axis R^2 values
# test other parameters