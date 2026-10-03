import pandas as pd
import matplotlib.pyplot as plt

data = {
    "OrderID": [1001, 1002, 1003],
    "Customer": ["Alice", "Bob", "Alice"],
    "Product": ["Laptop", "Chair", "Mouse"],
    "Category": ["Electronics", "Furniture", "Electronics"],
    "Quantity": [1, 2, 3],
    "Price": [1500, 180, 25],
    "OrderDate": ["2023-06-01", "2023-06-03", "2023-06-05"],
}
pd.DataFrame(data).to_csv("orders.csv", index=False)
df = pd.read_csv("orders.csv")
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
print("Таблиця замовлень:\n", df, "\n")

df["TotalAmount"] = df["Quantity"] * df["Price"]
print("Із TotalAmount:\n", df, "\n")

print("Сумарний дохід магазину:", df["TotalAmount"].sum())
print("Середнє значення TotalAmount:", df["TotalAmount"].mean())
print("Кількість замовлень по клієнтах:\n", df["Customer"].value_counts(), "\n")

print("Замовлення із сумою > 500:\n", df[df["TotalAmount"] > 500], "\n")

print("Сортування за OrderDate (спадання):\n",
      df.sort_values("OrderDate", ascending=False), "\n")

mask = df["OrderDate"].between("2023-06-05", "2023-06-10")
print("Замовлення з 5 по 10 червня:\n", df[mask], "\n")

by_cat = df.groupby("Category").agg(
    Orders=("OrderID", "count"),
    ItemsQuantity=("Quantity", "sum"),
    TotalSales=("TotalAmount", "sum"),
)
print("Групування за Category:\n", by_cat, "\n")

top3 = (df.groupby("Customer")["TotalAmount"].sum()
          .sort_values(ascending=False).head(3))
print("ТОП-3 клієнтів:\n", top3, "\n")
orders_by_date = df.groupby("OrderDate")["OrderID"].count()
plt.figure(figsize=(7, 4))
orders_by_date.plot(kind="line", marker="o")
plt.title("Кількість замовлень по датах")
plt.xlabel("Дата")
plt.ylabel("Кількість замовлень")
plt.yticks(range(0, int(orders_by_date.max()) + 2))
plt.grid(True)
plt.tight_layout()
plt.savefig("orders_by_date.png")
plt.show()

revenue_by_cat = df.groupby("Category")["TotalAmount"].sum()
plt.figure(figsize=(5, 5))
revenue_by_cat.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Розподіл доходів по категоріях")
plt.ylabel("")
plt.tight_layout()
plt.savefig("revenue_by_category.png")
plt.show()
