import seaborn, pandas
import matplotlib.pyplot as plt

data = pandas.read_csv("optimization.csv")
print(data.columns)
res = seaborn.lineplot(x=data.Epoch,y=data.Loss)
res.set(yscale='log')

plt.savefig("optimization.svg")
plt.show()
