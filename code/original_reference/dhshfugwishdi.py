import matplotlib.pyplot as plt
import numpy as np

# 初始化
n = 1009
a = [1.0]  # a1 = 1

# 计算数列
for i in range(1, n):
    next_a = a[i-1] + 1 / a[i-1]
    a.append(next_a)

# 绘制图像
x = np.arange(1, n+1)
y = np.array(a)
z = np.sqrt(2 * x)
w = z + np.log(x) / (4 * z)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='$a_n$')
plt.plot(x, z, label='$\sqrt{2n}$')
plt.plot(x, w, label='$\sqrt{2n} + \\frac{\ln n}{4\sqrt{2n}}$')
plt.title('a(n+1) = a(n) + 1/a(n), a1 = 1')
plt.xlabel('n')
plt.ylabel('a(n)')
plt.grid(True)
plt.legend()
plt.show()