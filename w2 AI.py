import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 讀取資料集
df = pd.read_csv("dataset.csv")

# 轉換日期格式，並加上星期幾欄位
df["Date"] = pd.to_datetime(df["Date"])
df["Weekday"] = df["Date"].dt.dayofweek  # 星期一為0，星期日為6

# 數據準備：使用連續天數當作 X
df = df.sort_values("Date").reset_index(drop=True)
df["DayIndex"] = np.arange(len(df))

# 訓練資料
X = df[["DayIndex"]]
y = df["Sales"]

# ========== 線性回歸 ==========
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# ========== 非線性回歸（3次多項式） ==========
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# ========== 繪圖 ==========
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], y, 'o-', label="Actual Sales")
plt.plot(df["Date"], y_pred_linear, 'r--', label="Linear Regression")
plt.plot(df["Date"], y_pred_poly, 'g-.', label="Polynomial Regression")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Prediction: Linear vs Nonlinear")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== 加分題 1: 預測未來7天（星期一到星期日）的銷售 ==========
# 假設未來接續 7 天（接在 DayIndex 後面）
future_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
future_days_poly = poly.transform(future_days)
future_sales_pred = poly_model.predict(future_days_poly)

print("\n📈 預測未來一週銷售金額（非線性模型）:")
weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
for i in range(7):
    print(f"{weekdays[i]}：預測銷售金額 = {future_sales_pred[i]:.2f}")

# ========== 加分題 2: 每星期幾的實際平均銷售 ==========
weekday_avg = df.groupby("Weekday")["Sales"].mean()
print("\n📊 每星期幾的實際平均銷售金額:")
for i in range(7):
    print(f"{weekdays[i]}：平均銷售 = {weekday_avg.get(i, 0):.2f}")
