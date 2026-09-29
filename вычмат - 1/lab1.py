# Барабанов Иван Сергеевич
# Вариант №16


import numpy as np
import matplotlib.pyplot as plt
import math

print("=" * 60)
print("Лабораторная работа №1")
print("=" * 60)

# ========================================================
# ЗАДАНИЕ 1: Сравнение точности float 64 и float 32
# ========================================================


def f(x):
    if (abs(x) < 1e-3):
        return 1/6
    return (x-np.sin(x))/(x**3)


x_vals = np.logspace(-1, -15, 15)

print("\n --- Задание 1: Сравнение float64 and float32 ---")
print(f"{'x':<10} {'float64:':<20} {'float32:':<20} {'Отн. ошибка':<15}")
print("-" * 65)

for x in x_vals:
    res64 = f(np.float64(x))
    res32 = f(np.float32(x))
    if res64 != 0:
        rel_err = abs(res64 - res32) / abs(res64)
    else:
        res_err = 0
    print(f"{x:<10.1e} {res64:<20.10f} {res32:<20.10f} {rel_err:<15.2e}")


# ========================================================
# ЗАДАНИЕ 2: Машинный эпсилон
# ========================================================


def machine_epsilon(dtype):
    eps = 1.0
    while dtype(1.0 + eps) != dtype(1.0):
        eps /= 2.0
    return eps * 2.0


print("\n --- Задание 2: Машинный эпсилон ---")
print(f"float64: {machine_epsilon(np.float64): .5e}")
print(f"float32: {machine_epsilon(np.float32): .5e}")


# ========================================================
# ЗАДАНИЕ 3: Ряд Тейлора
# ========================================================

def my_logf(x, eps=1e-10):
    term = -x*x
    s = term
    n = 2
    while abs(term) > eps:
        term *= x*x*(n-1)/n
        s += term
        n += 1
    return s, n-1


print("\n --- Задание 3: Ряд Тейлора для log(1-x^2) ---")
for x in [0.2, -0.6, 0.8]:
    for eps in [1e-6, 1e-10]:
        approx, terms = my_logf(x, eps)
        exact = math.log(1-x*x)
        print(f"x={x:3}, eps={eps:.0e}: approx={approx:.10f}, "
              f"exact={exact:.10f}, terms={terms}, error={abs(approx-exact):.2e}")

# ========================================================
# ЗАДАНИЕ 4: График погрешности для x=0.7
# ========================================================

print("\n --- Задание 4: График поогрешности ---")
x_fixed = 0.7
exact_val = math.log(1-x_fixed*x_fixed)
terms_list = []
errors_list = []

term = -x_fixed*x_fixed
s = term
n = 2
while n < 50:
    term *= x_fixed*x_fixed*(n-1)/n
    s += term
    n += 1
    terms_list.append(n-1)
    errors_list.append(abs(exact_val-s))

plt.figure(figsize=(8, 5))
plt.semilogy(terms_list, errors_list, 'o-', color='red')
plt.title('Погрешность частичных сумм для ln(1-x^2)')
plt.xlabel('Число членов')
plt.ylabel('Абсолютная погрешность (лог. шкала)')
plt.grid(True, which='both', linestyle='--')
plt.show()
