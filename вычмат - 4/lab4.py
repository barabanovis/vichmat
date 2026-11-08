from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №4 (Вариант 8)")
print("=" * 60)

# ============================================================
# ЗАДАНИЕ 1: Решение системы методом Ньютона
# ============================================================

# Система уравнений по Варианту 8:
# f1 = x^2 + y^2 - 1 = 0
# f2 = ln(x) + y = 0


def F(x):
    return np.array([
        x[0]**2 + x[1]**2 - 1,
        np.log(x[0]) + x[1]
    ])


def J(x):
    """
    Матрица Якоби для системы.
    df1/dx = 2x,  df1/dy = 2y
    df2/dx = 1/x,  df2/dy = 1
    """
    return np.array([
        [2 * x[0], 2 * x[1]],
        [1 / x[0], 1]
    ])


def newton_system(F, J, x0, eps=1e-8, max_iter=100):
    """
    Решает систему F(x)=0 методом Ньютона.
    x0 - начальное приближение (вектор).
    Возвращает (корень, число итераций, список приближений).
    """
    x = x0.astype(float).copy()
    history = [x.copy()]
    for i in range(max_iter):
        Fx = F(x)
        if np.linalg.norm(Fx) < eps:
            return x, i, history
        Jx = J(x)
        if abs(np.linalg.det(Jx)) < 1e-15:
            raise ValueError("Матрица Якоби вырождена")
        delta = np.linalg.solve(Jx, -Fx)
        x_new = x + delta
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < eps:
            return x_new, i + 1, history
        x = x_new
    raise RuntimeError("Метод Ньютона не сошёлся за заданное число итераций")


# Начальное приближение по Варианту 8
x0 = np.array([1.5, 0.5])
root, iterations, history = newton_system(F, J, x0)
print("\n--- Задание 1: Решение системы методом Ньютона ---")
print(f"Начальное приближение: x0 = {x0}")
print(f"Найденный корень: x = {root}")
print(f"Число итераций: {iterations}")
print(f"Невязка F(x): {np.linalg.norm(F(root)):.2e}")

# ============================================================
# ЗАДАНИЕ 2: Проверка точности
# ============================================================

try:
    from scipy.optimize import fsolve
    root_scipy = fsolve(F, x0)
    print("\n--- Задание 2: Проверка с помощью scipy.optimize.fsolve ---")
    print(f"Решение scipy: {root_scipy}")
    print(f"Невязка: {np.linalg.norm(F(root_scipy)):.2e}")
    print(f"Разница: {np.linalg.norm(root - root_scipy):.2e}")
except ImportError:
    print("\n--- Задание 2: scipy не установлен. Пропускаем. ---")

# ============================================================
# ЗАДАНИЕ 3: Исследование сходимости
# ============================================================

print("\n--- Задание 3: Исследование сходимости ---")
print(
    f"{'Нач. x':<10} {'Нач. y':<10} {'Итераций':<10} {'Корень x':<12} {'Корень y':<12} {'Невязка':<10}"
)
print("-" * 75)

x_start = 0.8
y_start = -0.3
try:
    x0_vec = np.array([x_start, y_start])
    root_tmp, it_tmp, _ = newton_system(F, J, x0_vec)
    res_tmp = np.linalg.norm(F(root_tmp))
    print(
        f"{x_start:<10.2f} {y_start:<10.2f} {it_tmp:<10} {root_tmp[0]:<12.6f} {root_tmp[1]:<12.6f} {res_tmp:<10.2e}"
    )
except Exception as e:
    print(
        f"{x_start:<10.2f} {y_start:<10.2f} {'не сошлась':<10} {'-':<12} {'-':<12} {'-':<10}")

# ============================================================
# ЗАДАНИЕ 4: Визуализация
# ============================================================
print("\n--- Задание 4: График сходимости ---")

# 1) x > 0, т.к. ln(x) определён только при x > 0
x_vals = np.linspace(0.05, 2.5, 400)      # было (-2.5, 2.5)  ← источник NaN
y_vals = np.linspace(-2.5, 2.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# 2) Уравнения совпадают с F(x)
F1 = X**2 + Y**2 - 1
F2 = np.log(X) + Y                        # теперь X > 0, NaN нет

# 3) ОДИН plt.figure — иначе первая фигура остаётся пустой
plt.figure(figsize=(8, 6))

# 4) levels=, без label= (contour не принимает label)
plt.contour(X, Y, F1, levels=[0], colors='blue', linewidths=2)
plt.contour(X, Y, F2, levels=[0], colors='red',  linewidths=2)

# 5) Итерации и корень
history = np.array(history)
plt.plot(history[:, 0], history[:, 1], 'go-',
         linewidth=1.5, markersize=4, label='Итерации')
plt.plot(root[0], root[1], 'ks', markersize=10, label='Корень')

# 6) Ручная легенда (у contour нет .get_label())
handles = [
    Line2D([0], [0], color='blue', lw=2, label='x² + y² − 1 = 0'),
    Line2D([0], [0], color='red',  lw=2, label='ln(x) + y = 0'),
    Line2D([0], [0], color='green', marker='o', lw=1.5, label='Итерации'),
    Line2D([0], [0], color='black', marker='s',
           lw=0, markersize=10, label='Корень'),
]
plt.legend(handles=handles, loc='upper right')

plt.title('Метод Ньютона для системы нелинейных уравнений (Вариант 8)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.xlim(0, 2.5)
plt.ylim(-2.5, 2.5)
plt.show()
