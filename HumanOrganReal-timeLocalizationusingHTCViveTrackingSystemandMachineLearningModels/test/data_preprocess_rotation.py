# import matplotlib.pyplot as plt
import pandas as pd

import math
 
def euler_from_quaternion(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
 
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
 
    return roll_x, pitch_y, yaw_z # in radians

df = pd.read_csv('data/rollall_nonrigid_pre.csv', header=None)
first_chunk = []
second_chunk = []
third_chunk = []
for i in range(df.shape[0]):
    # sensor 1
    if i%3 == 0:
        # if i == 0:
        first_three_col = df.iloc[i,:]
        roll_x, pitch_y, yaw_z = euler_from_quaternion(float(first_three_col[0]),float(first_three_col[1]),float(first_three_col[2]),float(first_three_col[3]))
        first_chunk.append([roll_x, pitch_y, yaw_z])
        # else:
        #   first_three_col = pd.concat([first_three_col, df.iloc[i,:]], axis=1, ignore_index=True)
    # sensor 2
    elif i%3 == 1:
        # if i == 1:
        second_three_col = df.iloc[i,:]
        roll_x, pitch_y, yaw_z = euler_from_quaternion(float(second_three_col[0]),float(second_three_col[1]),float(second_three_col[2]),float(first_three_col[3]))
        second_chunk.append([roll_x, pitch_y, yaw_z])
        # else:
        #   second_three_col = pd.concat([second_three_col, df.iloc[i,:]], axis=1, ignore_index=True)
    # sensor 3
    elif i%3 == 2:
        # if i == 2:
        third_three_col = df.iloc[i,:]
        roll_x, pitch_y, yaw_z = euler_from_quaternion(float(third_three_col[0]),float(third_three_col[1]),float(third_three_col[2]),float(first_three_col[3]))
        third_chunk.append([roll_x, pitch_y, yaw_z])
        # else:
        #   third_three_col = pd.concat([third_three_col, df.iloc[i,:]], axis=1, ignore_index=True)

df_1=pd.DataFrame(first_chunk, columns = ['yaw_1', 'pitch_1', 'roll_1'])
df_2=pd.DataFrame(second_chunk, columns = ['yaw_2', 'pitch_2', 'roll_2'])
df_3=pd.DataFrame(third_chunk, columns = ['yaw_3', 'pitch_3', 'roll_3'])

combined_df = pd.concat([df_1, df_2], axis=1, ignore_index=False)
combined_df = pd.concat([combined_df, df_3], axis=1, ignore_index=False)
combined_df.to_csv('data/nonrigid_rollall_processed.csv')