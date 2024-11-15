from scipy import stats  # stats modülünü içe aktardık
import matplotlib.pyplot as plt

# Örnek kullanım
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Doğrusal regresyon örneği
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("Eğim:", slope)
print("Kesişim:", intercept)
print("Korelasyon Katsayısı (r):", r_value)
print("P-değeri:", p_value)


def myfun(x):
    return slope *x +intercept

print("-----------------")
speed=myfun(10)
print(speed)
mymodel=list(map(myfun,x))
plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()


