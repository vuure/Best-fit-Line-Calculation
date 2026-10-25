"""
Kiss, suddenly alive.
"""

import numpy as np
import matplotlib.pyplot as plt

m_distances = np.array([70, 60, 50, 40, 30, 20, 10])
m_velocities = np.array([34.465, 35.608, 36.469, 36.900, 37.453, 36.900, 40.161])


m_bestfitlineM, m_bestfitlineB = np.polyfit(m_distances, m_velocities, deg=1)

m_yfit = m_bestfitlineM * m_distances + m_bestfitlineB

plt.scatter(m_distances, m_velocities, color='blue', label='Measured Velocities')
plt.plot(m_distances, m_yfit, color='red', label=f'Best Fit Line: y = {m_bestfitlineM:.2f}x + {m_bestfitlineB:.2f}')
plt.legend()
plt.show()
