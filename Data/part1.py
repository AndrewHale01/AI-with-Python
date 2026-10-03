import pandas as pd


data = {
    "OrderID": [1001, 1002, 1003],
    "Customer": ["Alice", "Bob", "Alice"],
    "Product": ["Laptop", "Chair", "Mouse"],
    "Category": ["Electronics", "Furniture", "Electronics"],
    "Quantity": [1, 2, 3],
    "Price": [1500, 180, 25],
    "OrderDate": ["2023-06-01", "2023-06-03", "2023-06-05"],
}
df = pd.DataFrame(data)
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
