#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Primeiro importamos o modulo.
from sklearn.datasets import load_iris
 
#Aqui eu instancio o objeto.
iris = load_iris()

#Aqui eu instancio o objeto.
iris = load_iris()
data = iris.data #Armazeno os dados para treino.
labels = iris.target #Aqui são os Labels
labels_names = iris.target_names # E aqui o nome de cada Label

#print(iris)

#Primeiro definimos a função para calculo de distância.
from math import sqrt
 
def euclidian(v1,v2):
    """Essa função recebe duas
       listas e retorna a
       distancia entre elas"""
 
    #Armazena o quadrado da distância
    dist = 0.0
    for x in range(len(v1)):
        dist += pow((v1[x] - v2[x]),2)
 
    #Tira a raiz quadrada da soma
    eucli = sqrt(dist)
    return eucli

#Precisamos do modulo random
import random
 
def Kcluster(data,distance=euclidian,k=3):
    #Determina o valor máximo e mínimo para cada atributo
    #Cria uma lista de tuplas que contem valores máximos e mínimos de cada atributo
    ranges = [(min([row[i] for row in data]),
               max([row[i] for row in data]))
               for i in range(len(data[0]))]
    
   #Cria K centroides aleatórias
   #Cria uma lista contendo os K centroides em posições aleatorias.
   #No nosso caso serão 3
    clusters=[[random.random()*(ranges[i][1] - ranges[i][0])+ranges[i][0]
               for i in range(len(data[0]))] for j in range(k)]
 
    lastmatches = None
    
    #O número de iterações será no máximo 100
    for t in range(100):
        bestmatches = [[] for i in range(k)] #Cria uma lista contendo 3 lista vazias
 
        #Verifica qual centroide esta mais perto de cada instância
        for j in range(len(data)):
            row=data[j]
            bestmatche = 0 #Aqui armazeno o índice da menor distância para comparação
            for i in range(k):
                d = distance(clusters[i],row) #Calcula a distancia em relação ao centroide
                #if d 0:
                if d>distance(clusters[bestmatch],row): bestmatch=i
                for rowid in bestmatches[i]:
                    for m in range(len(data[rowid])):
                        avgs[m] += data[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i]=avgs
 
    return bestmatches

    #Aqui retorna um lista de duas dimensões com os índices de cada cluster
    cluster = Kcluster(data,k=3)
    #seleciono as instâncias no dataset original de acordo com os seus índices
    c1 = data[[cluster[0]]]
    c2 = data[[cluster[1]]]
    c3 = data[[cluster[2]]]
    
    #a lista de valores que serão plotados
    plots = [c1,c2,c3]
    
    fig = plt.figure(236)
    x = 0   
    y = 1
    
    #Os títulos de cada gráfico
    titles = ['sepal length x sepal width','sepal length x petal length',
              'sepal length x petal width','sepal width x petal length',
              'sepal width x petal width','petal length x petal width']
    
    
#    Aqui trato de gerar todos os gráficos.
    for h in range(1,7):
        fig.add_subplot(2,3,h)
        for plot,color in zip(plots,['r','b','g']):
            plt.scatter(plot[:,x],plot[:,y],c=color, alpha=0.7)
        if (y==3):
            x += 1
            y = x + 1
        plt.title(titles[h - 1])
        plt.xticks(())
        plt.yticks(())
    plt.show()
    