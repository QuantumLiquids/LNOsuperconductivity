import json
import numpy as np
import matplotlib.pyplot as plt

# Define lattice parameters
parameters = {
    "Ly": 6,
    "Lx": 8,
    "t": 3,
    "J": 1,
    "delta": 0.5,
    "Hole": 24,
    "D": 12000,
    "NoPin": True
}
Ly = parameters["Ly"]
Lx = parameters["Lx"]

# Load correlation data: <n(0)n(r)>
with open('nf0nf6x8t3J1delta0.5Hole24D12000NoPin.json', 'r') as f:
    corr_data = json.load(f)

# Load density data: <n(r)>
with open('nf6x8t3J1delta0.5Hole24D12000NoPin.json', 'r') as f:
    density_data = json.load(f)

# Extract density into a dictionary: {site: <n(site)>}
density = {item[0][0]: complex(item[1][0], item[1][1]) for item in density_data}

# Extract correlation into a dictionary: {r: <n(0)n(r)>}
corr = {item[0][1]: complex(item[1][0], item[1][1]) for item in corr_data}

# Get <n(0)>
n0 = density[0]

# Compute reduced correlation
reduced_corr_data = []

# For r=0: <n(0)> - <n(0)>^2
reduced_corr_0 = n0 - n0 * n0
reduced_corr_data.append([[0], [reduced_corr_0.real, reduced_corr_0.imag]])

# For r=1 to 47: <n(0)n(r)> - <n(0)><n(r)>
for r in range(1, Ly * Lx):
    if r in corr:
        val = corr[r] - n0 * density[r]
        reduced_corr_data.append([[r], [val.real, val.imag]])
    else:
        raise ValueError(f"Correlation data missing for r={r}")

# Compute averaged reduced correlation over y for each delta x
delta_x = list(range(Lx))  # delta x from 0 to 7
averaged_reduced_corr = []
for x in range(Lx):
    # Sites with x-coordinate equal to 'x'
    r_list = [x * Ly + y for y in range(1)]  # e.g., for x=0: r=0,1,2,3,4,5
    values = [reduced_corr_data[r][1][0] for r in r_list]  # Real parts
    avg = np.mean(values)
    averaged_reduced_corr.append(avg)

abs_reduced_corr = [abs(x) for x in averaged_reduced_corr]
# Plot the averaged reduced correlation as a curve
#plt.figure(figsize=(8, 6))
plt.loglog(delta_x, abs_reduced_corr, marker='o', linestyle='-', color='b')
plt.xlabel('Delta x')
plt.ylabel('Averaged Reduced Correlation')
plt.title('Averaged Reduced Correlation over y for each Delta x\nLy=6, Lx=8, t=3, J=1, delta=0.5')
plt.grid(True)
plt.show()
