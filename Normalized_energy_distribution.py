import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# df = pd.read_csv("./Non_ergodic/cpp_setup/fermi_pasta_cpp.dat")

energy_k = []

df = pd.read_csv("./cpp_setup/fermi_pasta_cpp.dat")

num_rows = len(df)

# num_rows = 1

N = 31

inv_N = 1/N

time = []


for i in range(num_rows):
    energy_k_row = []
    
    x_frame = df.iloc[i,2:34]
    xdot_frame = df.iloc[i,34:]
    x_vals = x_frame.values
    xdot_vals = xdot_frame.values
    x_vals[0] = np.float64(x_frame.iloc[0][1:])
    x_vals[-1] = np.float64(x_frame.iloc[-1][:-1])
    xdot_vals[0] = np.float64(xdot_frame.iloc[0][1:])
    xdot_vals[-1] = np.float64(xdot_frame.iloc[-1][:-1])
    x_vals = np.array(x_vals,dtype=np.float64)
    xdot_vals = np.array(xdot_vals,dtype=np.float64)

    for k in range(1,N):
        Qk = 0
        Pk = 0
        wk = 2*np.sin(np.pi*k*0.5*inv_N)
        for j in range(1,N):
            phi = np.sqrt(2*inv_N) * np.sin(np.pi*k*j*inv_N)
            Qk += phi*x_vals[j]
            Pk += phi*xdot_vals[j]
        energy_k_row.append(0.5*(Pk**2+(wk**2)*Qk**2))
    energy_k.append(energy_k_row)
    time.append(df.iloc[i,1])

energy_k = np.array(energy_k).T

total_energy = np.sum(energy_k,axis=0)

sns.set_palette("husl",5)

norm_energy_k = []

for i in range(len(energy_k)):
    norm_energy_k.append(energy_k[i]/total_energy[i])

time = np.array(time)

# plt.gca().set_aspect(50)
plt.figure(figsize=(14,4),dpi=100)
ax = plt.axes()
# plt.contourf(norm_energy_k,cmap="inferno",levels=25)
cnt = ax.pcolor(norm_energy_k,cmap="inferno")
# sns.heatmap(norm_energy_k)
# plt.colorbar(cnt)
ax.set_xticks(time[::1000]/2)
ax.set_xticklabels(time[::1000])
plt.xlabel(r"$t$",size=13)
plt.ylabel(r"$k$",size=13)
plt.title("Normalized Energy distribution",size=14)
cnt.set_edgecolor("face")
plt.show()
# plt.savefig("output2.png")