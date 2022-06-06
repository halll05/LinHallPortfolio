import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

filenames=['data/nonrigid_rollall_processed.csv', 'data/nonrigid_pitch2_processed.csv',
'data/nonrigid_x4_processed.csv', 'data/nonrigid_y2_processed.csv',  'data/nonrigid_yaw2_processed.csv',
'data/nonrigid_z_processed.csv']

tri_filenames=['data_triangleconfig/pitch7_triangle_processed.csv', 'data_triangleconfig/x_triangle_processed.csv', 'data_triangleconfig/y_triangle_processed.csv',
'data_triangleconfig/z_triangle_processed.csv', 'data_triangleconfig/rollfinal_triangle_processed.csv',
'data_triangleconfig/yaw_triangle_processed.csv']

df = pd.read_csv('data_triangleconfig/x_triangle_processed.csv')
# df = pd.read_csv('data/nonrigid_y2_processed.csv')
df = df.dropna()
print('check if y has strong connections with your X:')
print(df.corr())

# you can choose other models
# try GBR https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
# try SVR
reg = linear_model.Ridge(alpha=.5)

y = df.values[:,0]
X = df.values[:,1:]

print("LR cross validation score is displayed below:")
print(np.average(cross_val_score(reg, X, y, cv=5)))
print(cross_val_score(reg, X, y, cv=5))

# -11

reg_svr = make_pipeline(StandardScaler(), SVR(C=0.5, epsilon=0.1))
print("RMSE of SVR is displayed below:")
print(np.average(cross_val_score(reg_svr, X, y, cv=5)))

reg_gbr = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=600, 
    max_depth=5, 
    learning_rate=0.01, 
    min_samples_split=3))
print("R-Squared of GBR is displayed below:")
print(np.average(cross_val_score(reg_gbr, X, y, cv=5)))


# print("RMSE of Training GBR is displayed below:")
# print(np.average(mean_squared_error(reg_gbr, X, y, squared=False)))


train_x, test_x, train_y, test_y = train_test_split(X, y,test_size=0.25,random_state=0)

reg.fit(train_x, train_y)

print("RMSE of Testing LR is displayed below:")
print(np.average(mean_squared_error(test_y, reg.predict(test_x), squared=False)))

reg_gbr.fit(train_x, train_y)

# print(np.average(mean_squared_error(train_y, reg_gbr.predict(train_x), squared=False)))

print("RMSE of Testing GBR is displayed below:")
print(np.average(mean_squared_error(test_y, reg_gbr.predict(test_x), squared=False)))