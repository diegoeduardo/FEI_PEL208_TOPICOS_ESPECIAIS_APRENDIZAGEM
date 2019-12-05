# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:34:07 2019

@author: SilvaDE
	Implementação da rede neural Perceptron
	w = w + N * (d(k) - y) * x(k)
"""
#Importação das bibliotecas que serão usadas durante o código
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#definição da classe que fará a chamada dos metodos para calculo do Perceptron
class Perceptron(object):
    #metodo para definicao e geração do radom state
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state #atributo com o radom state para geracao aleatoria
    
    #metodo para treinamento dos dados e posterior uso e plotagem
    def treinar(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
        self.errors_ = []
        #loop para reprocesamento caso o resultado desejado ainda nao  tenha sido atingido
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    #metodo para eentrada dos dados
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    #metodo para predizer os resultados 
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
#leitura do arquivo e posterior criação de um dataframe na variavel df
df = pd.read_csv('iris.data',header=None)
df.tail()
print(df)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plotagem dos dados de acordo com as cores desejadas
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

#repasse do objeto perceptron para posterior plotagem 
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.treinar(X, y)
plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

#plotagem gerado dos dadosm, importação do objeto ListedColormap
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('orange', 'green', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

