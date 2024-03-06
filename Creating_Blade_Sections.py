import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Inputs
P = float(input('Power Output (KW): '))
H = float(input('Net Head (m): '))
eta = float(input('Efficiency: '))
Z = float(input('Number of Blades: '))
attack = float(input('Optimum Angle of Attack (deg): '))
bladesections = int(input('Number of Blade Sections: '))  # New input for number of blade sections
section_distance = float(input('Distance between sections (m): '))  # New input for section distance
density = float(input('Density of fluid (kg/m^3): '))  # New input for fluid density

# Discharge
Q = P * 1000 / (eta * density * 9.81 * H)
# Specific Speed
Ns = 885.5 / (H ** 0.25)
# Turbine Speed
N = Ns * (H ** 1.25) / (P ** 0.5)
phi = 0.0242 * (Ns ** (2 / 3))

d_runner = 84.5 * phi * (H ** 0.5) / N
m = 0.4  # m=d_hub/d_runner
d_hub = m * d_runner

flowarea = np.pi * ((d_runner ** 2) - (d_hub ** 2)) / 4

# Flow Velocity
V_f = Q / flowarea

# Whirl velocity
d_avg = (d_runner + d_hub) / 2
V_avg = np.pi * d_avg * N / 60
V_w = P * 1000 / (density * Q * V_avg)

s = np.linspace(1.3, 0.75, bladesections)
i = 1

results = []

for d in np.linspace(d_hub, d_runner, bladesections):
    U = (np.pi * d * N) / 60
    beta_1 = np.rad2deg(np.arctan(V_f / (U - V_w)))
    beta_2 = np.rad2deg(np.arctan(V_f / U))

    # Blade Spacing
    t = (np.pi * d) / Z
    chord = s[i - 1] * t
    theta = 180 - beta_1 + attack
    naca = pd.read_excel('naca.xlsx', header=None)
    x = naca.iloc[:, 0].values
    y = naca.iloc[:, 1].values
    x = x * chord
    y = y * chord
    x = x - (chord / 2)
    R = np.array([[np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))],
                  [-np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
    rot_matrix = np.dot(R, np.vstack([x, y]))
    X_cord = rot_matrix[0, :]
    Y_cord = rot_matrix[1, :]
    Z_cord = np.zeros_like(X_cord)
    X_cord = np.round(X_cord, 6)
    Y_cord = np.round(Y_cord, 6)
    plt.figure()
    plt.plot(X_cord, Y_cord)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=False)
    section = np.vstack([X_cord, Y_cord, Z_cord]).T
    np.savetxt(f'section{i}.txt', section, delimiter='\t')
    
    # Append results
    results.append({
        'Section': i,
        'Diameter (m)': d,
        'Chord (m)': chord,
        'Theta (deg)': theta,
        'Blade Sections': bladesections,
        'Power Output (KW)': P,
        'Head Available (m)': H,
        'Efficiency': eta,
        'Number of Blades': Z,
        'Optimum Angle of Attack (deg)': attack,
        'Discharge (m^3/s)': Q,
        'Specific Speed': Ns,
        'Turbine Speed (rpm)': N,
        'Flow Velocity (m/s)': V_f,
        'Whirl velocity (m/s)': V_w,
    })

    i += 1

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to Excel file
results_df.to_excel('turbine_results.xlsx', index=False)

# Print inputs and results
print("Inputs:")
print(f"Power Output (KW): {P}")
print(f"Head Available (m): {H}")
print(f"Efficiency: {eta}")
print(f"Number of Blades: {Z}")
print(f"Optimum Angle of Attack (deg): {attack}")
print("\nResults:")
print(results_df)

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(1, bladesections + 1):
    section = np.loadtxt(f'section{i}.txt', delimiter='\t')
    ax.plot(section[:, 0], section[:, 1], section[:, 2] - (i - 1) * section_distance)  # Adjust Z coordinate based on section distance

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Blade Sections')
plt.show()
