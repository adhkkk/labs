import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_model(x, a, b):
    return a * x + b

data1 = np.loadtxt('2sem/131/data/1.txt')  
# data2 = np.loadtxt('data/1.txt')  
# data3 = np.loadtxt('data/1.txt')  




x1 = data1[:, 0]
y1 = data1[:, 1] 

mask = x1 < 5.2

x1_laminar = x1[mask]
y1_laminar = y1[mask]

x1_turbulent = x1[~mask]
y1_turbulent = y1[~mask]


params_laminar_1, covariance_laminar_1 = curve_fit(linear_model, x1_laminar, y1_laminar)

a1_laminar, b1_laminar = params_laminar_1
a_err1_laminar, b_err1_laminar = np.sqrt(np.diag(covariance_laminar_1))
y_fit_laminar_1 = linear_model(x1_laminar,a1_laminar,b1_laminar)


params_turbulent_1, covariance_turbulent_1 = curve_fit(linear_model, x1_turbulent, y1_turbulent)

a1_turbulent, b1_turbulent = params_turbulent_1
a_err1_turbulent, b_err1_turbulent = np.sqrt(np.diag(covariance_turbulent_1))
y_fit_turbulent_1 = linear_model(x1_turbulent,a1_turbulent,b1_turbulent)


# x2 = data2[:, 0]
# y2 = data2[:, 1] 

# x3 = data3[:, 0]
# y3 = data3[:, 1] 

plt.figure()

# ПЕРВАЯ ТРУБКА

# plt.scatter(x1, y1) 

text_box_laminar_1 = (
    f"Параметры прямой:\n"
    f"a = {a1_laminar:.2f} ± {a_err1_laminar:.2f}\n"
    f"b = {b1_laminar:.2f} ± {b_err1_laminar:.2f}"
)

text_box_turbulent_1 = (
    f"Параметры прямой:\n"
    f"a = {a1_turbulent:.2f} ± {a_err1_turbulent:.2f}\n"
    f"b = {b1_turbulent:.2f} ± {b_err1_turbulent:.2f}"
)

# plt.text(0.5, 0.95, text_box_laminar_1, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
# plt.scatter(x1_laminar, y1_laminar, color='orange', label='Эксперимент')
# plt.plot(x1_laminar, y_fit_laminar_1, label='Аппроксимация')

plt.text(0.5, 0.95, text_box_turbulent_1, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.scatter(x1_turbulent, y1_turbulent, color='orange', label='Эксперимент')
plt.plot(x1_turbulent, y_fit_turbulent_1, label='Аппроксимация')

# ВТОРАЯ ТРУБКА

# plt.plot(x2, y2)

# ТРЕТЬЯ ТРУБКА

# plt.plot(x3, y3)

plt.xlabel(r'Объемный расход $Q$, л/мин', fontsize=12)
plt.ylabel(r'Разница давлений $\Delta P$, Па', fontsize=12)

plt.minorticks_on()

plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)  
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5) 

plt.legend()
plt.tight_layout()

plt.show()