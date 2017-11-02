import numpy as np
import matplotlib.pyplot as plt
plt.plot([1,3],[2,6],'r-') #以(1,2)、(3,6)两点做直线，参数'r-'指定颜色red
plt.axis([1, 2, 1, 6])#指定坐标范围(Xmin,Xmax,Ymin,Ymax)
plt.xlabel('time')
plt.ylabel('price')
plt.show()

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2) #0~5每0.2取一个数，即0.0,0.2,0.4...4.8
# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

