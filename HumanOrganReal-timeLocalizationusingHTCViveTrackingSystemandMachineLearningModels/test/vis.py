import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/yaw_combined_csvPLOT.csv')
y1 = df['rotation_ground_truth']
y2 = df['prediction']
x = df['time']
plt.plot(x, y1, label='ground_truth')
plt.plot(x, y2, label='prediction')
plt.title('Rotation')
plt.legend()
plt.show()
