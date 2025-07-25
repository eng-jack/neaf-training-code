







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(0)


statistics = {
    'Image Index': [],
    'Max': [],
    'Min': [],
    'Mean': [],
    'Std': []
}

for i in range(10):
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        
    axes[0].imshow(image)
    axes[0].set_title(f'color graph No.{i+1}')
    #plt.show()
    #img = np.stack((image, image, image), axis=-1)
    gray = np.mean(image, axis=2).astype(np.uint8)
    axes[1].imshow(gray,cmap='gray')
    axes[1].set_title(f'gray graph No.{i+1}')
    plt.show()
    max_val = np.max(gray)
    min_val = np.min(gray)
    mean_val = np.mean(gray)
    std_val = np.std(gray)

    
    statistics['Image Index'].append(f'Image_{i+1}')
    statistics['Max'].append(max_val)
    statistics['Min'].append(min_val)
    statistics['Mean'].append(mean_val)
    statistics['Std'].append(std_val)


df = pd.DataFrame(statistics)


df.to_excel('statistics.xlsx', index=False)

print("已儲存影像統計資料至 statistics.xlsx")



'''
plt.imshow(image)
plt.imshow(gray)

plt.title('graph')
plt.show()




# 折線線圖
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.title('Simple Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# 散佈圖
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
# 設置隨機種子
np.random.seed(0)

# 隨機生成三通道影像
images = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

# 轉換為灰階並計算統計數據
gray_images = np.mean(images, axis=2).astype(np.uint8)

# 計算統計數據
max_values = np.max(gray_images, axis=(0, 1))

# show出統計結果
print(max_values)

plt.imshow(images)
plt.show()
one = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
zero = np.zeros((10, 10), dtype=np.uint8)
img = np.stack((one, one, one), axis=-1)
plt.imshow(img)
plt.show()
data = {
    'Name': ['John', 'Emma', 'Alex'],
    'Age': [28, 32, 24],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)
print(df)

# 數據選擇
print("\nSelect 'Name' column:")
print(df['Name'])

# 數據過濾
print("\nFilter age > 25:")
print(df[df['Age'] > 25])

# 基本數據操作
print("\nSort by Age:")
print(df.sort_values('Age'))
# 生成線性數列
sequence_1 = np.linspace(1, 3, 10)
sequence_2 = np.linspace(1, 10, 10)

# 寫成字典型態，後面可以是 list 或 numpy array
d = {'s1': sequence_1, 
     's2': sequence_2,
    }

# 以字典型態寫入xlsl檔案
df = pd.DataFrame(data=d)
df.to_excel(f'test.xlsx')
np.random.seed(0)

# 生成10張隨機的三通道影像
images = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

# 轉換為灰階
gray_images = np.mean(images, axis=2).astype(np.uint8)

# 顯示選中的灰階影像
plt.figure(figsize=(6, 6))
plt.imshow(gray_images, cmap='gray')
plt.title(f"Gray Image")
plt.axis('off')
plt.show()

# 顯示選中的彩色影像
plt.figure(figsize=(6, 6))
plt.imshow(images)
plt.title(f"Color Image")
plt.axis('off')
plt.show()
'''
