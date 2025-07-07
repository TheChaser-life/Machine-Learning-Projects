import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Source: https://www.kaggle.com/datasets/rahulverma07/linear-regression-house-price-prediction
df = pd.read_csv("Housing.csv",usecols=['price','area'])

# chuyển đổi file về mảng và chuẩn hóa do số lớn
sto = df.values
scaler = StandardScaler()
sto = scaler.fit_transform(sto)

# khởi tạo các giá trị ban đầu
w = 0
b = 0
a = 0.01
k = 1000

# Gradient decent
def training(w, b, a, k, sto):
    while k:
        k = k - 1
        db = 0
        dw = 0
        for i in range(len(sto)):
            x = sto[i][1]
            y = sto[i][0]
            tmp_val = w*x + b
            dw = dw + (tmp_val - y)*x
            db = db + (tmp_val - y)
        db = db*(1/len(sto))
        dw = dw*(1/len(sto))
        b = b - a*db
        w = w - a*dw
    return w,b

w, b = training(w,b,a,k,sto)
val = float(input())

# chuẩn hóa số được nhập vào
# công thức: val chuẩn hóa = (val - mean_val)/độ lệch chuẩn
val_mean = scaler.mean_[1]
val_std = np.sqrt(scaler.var_[1])
val = (val - val_mean)/val_std

# dự đoán kết quả 
predicted_val = w*val + b

# khôi phục về giá trị ban đầu
# công thức: val cuối cùng = (val chuẩn hóa)*(độ lệch chuẩn) + mean_val
rs_mean = scaler.mean_[0]
rs_std = np.sqrt(scaler.var_[0])
rs = predicted_val*rs_std + rs_mean
print(rs)

area_orig = df['area'].values
price_orig = df['price'].values

# Tạo trục x: area từ min đến max
# np.linspace là một hàm trong thư viện NumPy, dùng để tạo một mảng gồm các giá trị cách đều nhau trong một khoảng nhất định.
x_plot = np.linspace(area_orig.min(), area_orig.max(), 100)

# Chuẩn hóa x_plot theo area
area_mean = scaler.mean_[1]
area_std = np.sqrt(scaler.var_[1])
x_plot_scaled = (x_plot - area_mean) / area_std

# Dự đoán price (ở dạng chuẩn hóa)
y_plot_scaled = w * x_plot_scaled + b

# Khôi phục về đơn vị price thật
price_mean = scaler.mean_[0]
price_std = np.sqrt(scaler.var_[0])
y_plot = y_plot_scaled * price_std + price_mean

# 5. Vẽ biểu đồ
# kích thước: chiều ngang: 10 inch / chiều cao: 6 inch
plt.figure(figsize=(10, 6))
# vẽ các điểm dữ liệu thực tế từ dataset
plt.scatter(area_orig, price_orig, color='blue', label='Dữ liệu thực tế')
# vẽ đường hồi quy
plt.plot(x_plot, y_plot, color='red', label='Đường hồi quy tuyến tính')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Biểu đồ hồi quy tuyến tính: Price theo Area')
plt.legend()
plt.grid(True)
plt.show()
