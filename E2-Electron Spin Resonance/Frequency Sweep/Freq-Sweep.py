import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# =========================
# CONSTANTS
# =========================
B_values = np.array([0.5, 0.35, 0.645, 0.25, 0.757])

# store results
f_res_list = []
f_res_err_list = []
delta_f_list = []
delta_f_err_list = []


# =========================
# ANALYSIS FUNCTION
# =========================
def analyze_sweep(freq, signal_raw):

    # smooth signal
    signal = gaussian_filter1d(signal_raw, sigma=0.8)

    # restrict region
    mask = (freq >= 20) & (freq <= 80)
    f = freq[mask]
    y = signal[mask]

    if len(f) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # =========================
    # RESONANCE (tangent from origin)
    # =========================
    ratio = y / np.maximum(f, 1e-12)
    i_res = np.argmax(ratio)

    f_res = f[i_res]
    V_res = y[i_res]

    if f_res <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    V_half = V_res / np.sqrt(2)

    # =========================
    # LINES
    # =========================
    m_half = V_half / f_res

    def half_line(x):
        return m_half * x

    # =========================
    # FIND CROSSINGS + LOCAL SLOPES
    # =========================
    def crossing(x, y, func):
        d = y - func(x)
        idx = np.where(np.diff(np.sign(d)))[0]

        if len(idx) == 0:
            return np.nan, np.nan

        i = idx[0]

        x0, x1 = x[i], x[i + 1]
        y0, y1 = d[i], d[i + 1]

        x_cross = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)

        # LOCAL slope (key fix)
        dy = y[i+1] - y[i]
        dx = x[i+1] - x[i]
        slope_local = dy / dx if dx != 0 else 1e-12

        return x_cross, slope_local

    left = freq < f_res
    right = freq > f_res

    f1, slope1 = crossing(freq[left], signal[left], half_line)
    f2, slope2 = crossing(freq[right], signal[right], half_line)

    delta_f = abs(f2 - f1) if np.isfinite(f1) and np.isfinite(f2) else np.nan

    # =========================
    # NOISE (DATA-BASED)
    # =========================
    noise = np.std(signal_raw - signal)

    # =========================
    # ERROR: RESONANCE (LOCAL CURVATURE)
    # =========================
    if 1 < i_res < len(f) - 1:
        slope_res = (y[i_res+1] - y[i_res-1]) / (f[i_res+1] - f[i_res-1])
    else:
        slope_res = 1e-12

    f_res_err = abs(noise / (slope_res + 1e-12))

    # =========================
    # ERROR: DELTA F
    # =========================
    f1_err = abs(noise / (slope1 + 1e-12))
    f2_err = abs(noise / (slope2 + 1e-12))

    delta_f_err = np.sqrt(f1_err**2 + f2_err**2)

    return f_res, f_res_err, delta_f, delta_f_err, V_res


# =========================
# FILES
# =========================
files = [
    "Sweep_0.5.txt",
    "Sweep_0.35.txt",
    "Sweep_0.645.txt",
    "Sweep_0.25.txt",
    "Sweep_0.757.txt"
]

labels = ["0.5 T", "0.35 T", "0.645 T", "0.25 T", "0.757 T"]


# =========================
# LOOP
# =========================
for file, label, B in zip(files, labels, B_values):

    data = np.loadtxt(file)

    freq = data[:, 0]
    signal_raw = data[:, 1]

    mask = (freq >= 0) & (freq <= 100)
    freq = freq[mask]
    signal_raw = signal_raw[mask]

    f_res, f_err, d_f, d_f_err, V_res = analyze_sweep(freq, signal_raw)

    f_res_list.append(f_res)
    f_res_err_list.append(f_err)
    delta_f_list.append(d_f)
    delta_f_err_list.append(d_f_err)

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(freq, gaussian_filter1d(signal_raw, 0.8), label="Signal")
    plt.scatter(f_res, V_res, label="f_res")

    plt.title(label)
    plt.legend()
    plt.grid()
    plt.show()

    print(f"\n--- {label} ---")
    print(f"f_res = {f_res:.4f} ± {f_err:.4f} Hz")
    print(f"Δf = {d_f:.4f} ± {d_f_err:.4f} Hz")


# =========================
# FINAL SLOPE
# =========================
f_res_list = np.array(f_res_list)
f_res_err_list = np.array(f_res_err_list)

weights = 1 / np.maximum(f_res_err_list, 1e-6)

fit = np.polyfit(B_values, f_res_list, 1, w=weights)
slope = fit[0]

model = np.polyval(fit, B_values)
chi2 = np.sum(((f_res_list - model) / f_res_err_list) ** 2)
dof = len(B_values) - 2
chi2_red = chi2 / dof

# =========================
# HALL COEFFICIENT
# =========================
N = 1
d = 0.8e-3
mu0 = 4*np.pi*1e-7

R_H = (2 * mu0 * d**2 / (np.pi * N**2)) * slope

# =========================
# OUTPUT
# =========================
print("\n================ FINAL RESULTS ================")
print(f"df_res/dB = {slope:.4f} Hz/T")
print(f"Chi^2 = {chi2:.3f}")
print(f"Reduced Chi^2 = {chi2_red:.3f}")
print(f"Hall coefficient R_H = {R_H:.3e} m^3/C")

""""
import numpy as np
import matplotlib.pyplot as plt

# List your files and labels (edit these)
files = [
    "Sweep_0.5.txt",
    "Sweep_0.35.txt",
    "Sweep_0.645.txt",
    "Sweep_0.25.txt",
    "Sweep_0.757.txt"
]

labels = [
    "0.5 T",
    "0.35 T",
    "0.645 T",
    "0.25 T",
    "0.757 T"
]

# Loop through files and plot
for file, label in zip(files, labels):
    data = np.loadtxt(file)

    freq = data[:, 0]     # first column
    signal = data[:, 1]   # second column

    plt.plot(freq, signal, label=label)

# Labels and legend
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (mV)")
plt.title("Magnetic Fields for Frequency Sweeps")
#plt.xlim(0, 200)  
plt.legend()
plt.grid()

plt.show()
"""