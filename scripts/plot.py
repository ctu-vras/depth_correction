import pandas as pd
import matplotlib.pyplot as plt

icp_dist = 'point-to-point'
# icp_dist = 'point-to-plane'
df = pd.read_csv('/home/ruslan/Desktop/bias_estimation-%s_depth_correction_0.csv' % icp_dist)
df_corr = pd.read_csv('/home/ruslan/Desktop/bias_estimation-%s_depth_correction_1.csv' % icp_dist)

plt.figure()
plt.grid()
plt.plot(df['Incidence angle [deg]'][:len(df)//2], df[' ICP distance [m]'][:len(df)//2], color='r')
plt.plot(df_corr['Incidence angle [deg]'][:len(df_corr)//2], df_corr[' ICP distance [m]'][:len(df_corr)//2], color='k')

plt.show()
