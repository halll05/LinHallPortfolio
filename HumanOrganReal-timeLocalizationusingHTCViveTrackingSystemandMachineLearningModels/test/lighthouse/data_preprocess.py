# import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('4-lighthouse.csv', header=None)
# first_three_col = []
# second_three_col = []
# third_three_col = []
for i in range(df.shape[0]):
	# sensor 1
	if i%3 == 0:
		if i == 0:
			first_three_col = df.iloc[i,:]
		else:
			first_three_col = pd.concat([first_three_col, df.iloc[i,:]], axis=1, ignore_index=True)
	# sensor 2
	elif i%3 == 1:
		if i == 1:
			second_three_col = df.iloc[i,:]
		else:
			second_three_col = pd.concat([second_three_col, df.iloc[i,:]], axis=1, ignore_index=True)
	# sensor 3
	elif i%3 == 2:
		if i == 2:
			third_three_col = df.iloc[i,:]
		else:
			third_three_col = pd.concat([third_three_col, df.iloc[i,:]], axis=1, ignore_index=True)

combined_df = pd.concat([first_three_col.T, second_three_col.T], axis=1, ignore_index=True)
combined_df = pd.concat([combined_df, third_three_col.T], axis=1, ignore_index=True)
combined_df.to_csv('4-combined_csv.csv')
