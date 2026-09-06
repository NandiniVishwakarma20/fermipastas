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

R_t = []
time = []

inv_QP_coeff = 0

Qk_a = []
Pk_a = []

x_frame = df.iloc[0,2:34]
xdot_frame = df.iloc[0,34:]
x_vals = x_frame.values
xdot_vals = xdot_frame.values
x_vals[0] = np.float64(x_frame.iloc[0][1:])
x_vals[-1] = np.float64(x_frame.iloc[-1][:-1])
xdot_vals[0] = np.float64(xdot_frame.iloc[0][1:])
xdot_vals[-1] = np.float64(xdot_frame.iloc[-1][:-1])
x_vals = np.array(x_vals,dtype=np.float64)
xdot_vals = np.array(xdot_vals,dtype=np.float64)

for k in range(1,N):
    Qk_o = 0
    Pk_o = 0
    wk = 2*np.sin(np.pi*k*0.5*inv_N)
    for j in range(1,N):
        phi = np.sqrt(2*inv_N) * np.sin(np.pi*k*j*inv_N)
        Qk_o += phi*x_vals[j]
        Pk_o += phi*xdot_vals[j]
    Qk_a.append(Qk_o)
    Pk_a.append(Pk_o)
    inv_QP_coeff += (Qk_o**2 +Pk_o**2)

inv_QP_coeff = 1/inv_QP_coeff

for i in range(0,num_rows):
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

    Rm_t = 0

    for k in range(1,N):
        Qk = 0
        Pk = 0
        wk = 2*np.sin(np.pi*k*0.5*inv_N)
        for j in range(1,N):
            phi = np.sqrt(2*inv_N) * np.sin(np.pi*k*j*inv_N)
            Qk += phi*x_vals[j]
            Pk += phi*xdot_vals[j]

        Rm_t += (Qk*Qk_a[k-1] + Pk*Pk_a[k-1])
    
    R_t.append(Rm_t*inv_QP_coeff)
    time.append(df.iloc[i,1])

sns.set_palette("husl",5)

# plt.ylim((-0.2,0.2))
plt.plot(time,R_t)

# plt.legend()
plt.xlabel(r"$t$",size=13)
plt.ylabel(r"$R(t)$",size=13)
plt.title("Recurrence Measure",size=14)

plt.show()