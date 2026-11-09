"""

Happines arrive, hunger like a storm

"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# DATA (SI units).  a values are taken from Table 3 of the report.
# ------------------------------------------------------------------
g = 9.8                                              # m/s^2
L2 = 0.100                                           # m, flag length

# Series 1: m_a changes, m + m_a = 0.280 kg
ma1 = np.array([0.010, 0.030, 0.050, 0.070])         # kg
t1_s1 = np.array([0.423, 0.229, 0.230, 0.180])       # s
t2_s1 = np.array([0.216, 0.109, 0.082, 0.073])       # s
t3_s1 = np.array([1.651, 0.799, 0.594, 0.645])       # s
a1 = np.array([0.137, 0.602, 1.322, 1.262])          # m/s^2
F1 = ma1 * g                                         # N

# Series 2: m changes, m_a = 0.010 kg
ma2 = 0.010                                          # kg
m2 = np.array([0.270, 0.250, 0.230, 0.210])          # kg
t1_s2 = np.array([0.323, 0.302, 0.282, 0.271])       # s
t2_s2 = np.array([0.158, 0.151, 0.141, 0.134])       # s
t3_s2 = np.array([1.156, 1.111, 1.018, 0.975])       # s
a2 = np.array([0.279, 0.298, 0.348, 0.387])          # m/s^2

# ------------------------------------------------------------------
# READING UNCERTAINTIES  (!!! check these against your Table 1 !!!)
# ------------------------------------------------------------------
dt = 0.001        # s,  photogate timer resolution
dL2 = 0.0005      # m,  reading uncertainty of the length of the flag
dm = 0.0001       # kg, reading uncertainty of the masses (0.1 g)


def delta_a(t1, t2, t3, a):
    """Uncertainty propagation as in the lab manual (fractional errors add)."""
    v1, v2 = L2 / t1, L2 / t2
    dv1 = v1 * (dL2 / L2 + dt / t1)
    dv2 = v2 * (dL2 / L2 + dt / t2)
    d_dv = dv1 + dv2                                  # error of (v2 - v1): sum
    return a * (d_dv / (v2 - v1) + dt / t3)


da1 = delta_a(t1_s1, t2_s1, t3_s1, a1)
da2 = delta_a(t1_s2, t2_s2, t3_s2, a2)
dF1 = g * dm * np.ones_like(F1)                       # dF_a = g * dm_a
d_inv_a2 = da2 / a2**2                                # error of 1/a


# ------------------------------------------------------------------
# LEAST-SQUARES FITS  (slope/intercept and their standard errors)
# ------------------------------------------------------------------
def linfit(x, y):
    (k, b), cov = np.polyfit(x, y, deg=1, cov=True)
    dk, db = np.sqrt(np.diag(cov))
    yhat = k * x + b
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return k, b, dk, db, r2


k1, b1, dk1, db1, r2_1 = linfit(F1, a1)               # a = k F_a + b
k2, b2, dk2, db2, r2_2 = linfit(m2, 1 / a2)           # 1/a = k m + b

# theory
M1 = 0.280                                            # kg, m + m_a (Series 1)
k1_th = 1 / M1                                        # kg^-1
k2_th = 1 / (ma2 * g)                                 # N^-1
b2_th = 1 / g                                         # s^2/m

print("--- Series 1: a = k*F_a + b ---")
print(f"k = {k1:.3f} +- {dk1:.3f} kg^-1   (theory {k1_th:.3f})")
print(f"b = {b1:.3f} +- {db1:.3f} m/s^2   (theory 0)")
print(f"R^2 = {r2_1:.3f}")
print(f"delta a = {np.round(da1, 3)}   delta F_a = {dF1[0]:.4f} N")
print("--- Series 2: 1/a = k*m + b ---")
print(f"k = {k2:.2f} +- {dk2:.2f} N^-1   (theory {k2_th:.2f})")
print(f"b = {b2:.2f} +- {db2:.2f} s^2/m   (theory {b2_th:.3f})")
print(f"R^2 = {r2_2:.3f}")
print(f"delta a = {np.round(da2, 3)}   delta(1/a) = {np.round(d_inv_a2, 3)}")
print("a_exp / a_theory, Series 1:", np.round(a1 / (F1 / M1), 2))
print("a_exp / a_theory, Series 2:", np.round(a2 / (ma2 * g / (m2 + ma2)), 2))

# ------------------------------------------------------------------
# FIGURE 1: a vs F_a
# ------------------------------------------------------------------
fig1, ax = plt.subplots(figsize=(6, 4.5))
xx = np.linspace(0, 0.75, 100)
ax.errorbar(F1, a1, xerr=dF1, yerr=da1, fmt='o', color='blue', capsize=3,
            label='Measured data')
ax.plot(xx, k1 * xx + b1, color='red',
        label=f'Linear fit: $a$ = ({k1:.2f})$F_a$ + ({b1:.2f})')
ax.plot(xx, k1_th * xx, color='green', linestyle='--',
        label=r'Theory: $a = F_a/(m+m_a)$')
ax.set_xlabel(r'Applied force $F_a$ (N)')
ax.set_ylabel(r'Acceleration $a$ (m/s$^2$)')
ax.set_xlim(0, 0.75)
ax.set_ylim(0, 2.8)
ax.grid(alpha=0.3)
ax.legend()
fig1.tight_layout()
fig1.savefig('figure1_a_vs_F.png', dpi=300)

# ------------------------------------------------------------------
# FIGURE 2: (a) a vs m   (b) 1/a vs m
# ------------------------------------------------------------------
fig2, (axa, axb) = plt.subplots(1, 2, figsize=(10, 4.2))
mm = np.linspace(0.19, 0.29, 100)

axa.errorbar(m2, a2, xerr=dm, yerr=da2, fmt='o', color='blue', capsize=3,
             label='Measured data')
axa.plot(mm, ma2 * g / (mm + ma2), color='green', linestyle='--',
         label=r'Theory: $a = m_a g/(m+m_a)$')
axa.set_xlabel(r'Glider mass $m$ (kg)')
axa.set_ylabel(r'Acceleration $a$ (m/s$^2$)')
axa.set_title('(a)')
axa.grid(alpha=0.3)
axa.legend()

axb.errorbar(m2, 1 / a2, xerr=dm, yerr=d_inv_a2, fmt='o', color='blue',
             capsize=3, label='Measured data')
axb.plot(mm, k2 * mm + b2, color='red',
         label=f'Linear fit: 1/$a$ = ({k2:.1f})$m$ + ({b2:.2f})')
axb.plot(mm, k2_th * mm + b2_th, color='green', linestyle='--',
         label=r'Theory: 1/$a$ = $m/(m_a g)$ + 1/$g$')
axb.set_xlabel(r'Glider mass $m$ (kg)')
axb.set_ylabel(r'1/$a$ (s$^2$/m)')
axb.set_title('(b)')
axb.grid(alpha=0.3)
axb.legend()

fig2.tight_layout()
fig2.savefig('figure2_a_vs_m.png', dpi=300)

plt.show()