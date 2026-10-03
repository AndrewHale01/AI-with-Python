import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


x = np.linspace(-10, 10, 1000)
y = x ** 2 * np.sin(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, color="royalblue", label=r"$f(x)=x^2 \cdot \sin(x)$")
plt.title("Графік функції f(x) = x² · sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("task1_function.png")
plt.show()


data = np.random.normal(loc=5, scale=2, size=1000)

plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color="steelblue", edgecolor="black")
plt.title("Гістограма: нормальний розподіл (μ = 5, σ = 2)")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.tight_layout()
plt.savefig("task2_histogram.png")
plt.show()



hobbies = ["Програмування", "Музика", "Подорожі", "Спорт", "Читання"]
shares = [35, 20, 20, 15, 10]

plt.figure(figsize=(6, 6))
plt.pie(shares, labels=hobbies, autopct="%1.1f%%", startangle=90)
plt.title("Мої хобі")
plt.tight_layout()
plt.savefig("task3_pie.png")
plt.show()


fruits = ["Яблука", "Банани", "Апельсини", "Груші"]
weights = [
    np.random.normal(180, 15, 100),
    np.random.normal(120, 10, 100),
    np.random.normal(210, 20, 100),
    np.random.normal(170, 12, 100),
]

plt.figure(figsize=(8, 5))
plt.boxplot(weights, tick_labels=fruits)
plt.title("Розподіл маси фруктів")
plt.xlabel("Вид фрукта")
plt.ylabel("Маса, г")
plt.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("task4_boxplot.png")
plt.show()


px = np.random.uniform(0, 1, 100)
py = np.random.uniform(0, 1, 100)

plt.figure(figsize=(6, 6))
plt.scatter(px, py, color="green", alpha=0.6)
plt.title("Точкова діаграма (рівномірний розподіл на [0, 1])")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("task5_scatter.png")
plt.show()


t = np.linspace(-2 * np.pi, 2 * np.pi, 500)

plt.figure(figsize=(9, 5))
plt.plot(t, np.sin(t), color="red", label="f(x) = sin(x)")
plt.plot(t, np.cos(t), color="blue", label="g(x) = cos(x)")
plt.plot(t, np.sin(t) + np.cos(t), color="green", label="h(x) = sin(x) + cos(x)")
plt.title("Графіки функцій sin(x), cos(x) та sin(x) + cos(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("task6_functions.png")
plt.show()