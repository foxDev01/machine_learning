from sklearn.datasets import fetch_openml 
from matplotlib import pyplot as plt

mnist = fetch_openml('mnist_784')

x,y = mnist["data"] , mnist["target"]
print("MINST数据集大小为：{}".format(x.shape))
for i in range(25):
    digit =x.loc[i*2500].to_numpy()
    digit_image = digit.reshape(28,28)
    plt.subplot(5,5, i +1)
    plt.axis('off')
    plt.imshow(digit_image, cmap='gray')
plt.show()
