# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:15:50 2019

@author: SilvaDE
"""

######################################
import pandas as pd             #biblioteca para criacao de dataframes
import xlrd                     #biblioteca abrir arquivo  excel
import matplotlib.pyplot as plt #biblioteca para os graficos
import numpy as np
import copy
from mpl_toolkits.mplot3d import Axes3D

## Método met_matriz_mult faz multiplicação de 2 matrizes quaisquer
## Recebe como parametro MatrizA e MatrizB que serão multiplicadas
## 1 - Checa o numero de colunas da Matriz A com o numero de linhas da Matriz B
## Recebe 2 Matrizes (nXm) e (mX?) e retorna uma matriz (nx?)
def met_matriz_mult(MatrizA,MatrizB):
    if len(MatrizA[0]) != len(MatrizB): return -1 ## faz checagem se o numero de colunas de MatrizA é igual ao de linhas em MatrizB
    else:
        NumLinhaMatrizA=len(MatrizA)            ## NumLinhaMatrizA guarda o numero de linhas da matriz MatrizA
        NumColMatrizA=len(MatrizA[0])           ## NumColMatrizA guarda o numero de colunas da matriz MatrizA
        NumColMatriz=len(MatrizB[0])            ## NumColMatriz guarda o numero de colunas da matriz MatrizB
        MatrizC_Result_A_x_B = np.empty((NumLinhaMatrizA,NumColMatriz))  # variavel MatrizC_Result_A_x_B sera uma matriz com o numero de linhas da matriz MatrizA e numero de colunas da matriz MatrizB
        # O loop ira fazer a multiplicação e soma de cada Coluna de uma linha da matriz MatrizA pelas Linhas da matriz MatrizB
        for i in range(0,NumLinhaMatrizA,1):
            for j in range(0,NumColMatriz,1):
                MatrizC_Result_A_x_B[i][j]=0
                for k in range(0,NumColMatrizA,1):
                    MatrizC_Result_A_x_B[i][j]=MatrizC_Result_A_x_B[i][j]+MatrizA[i][k]*MatrizB[k][j]
        return MatrizC_Result_A_x_B

## Metodo para Matriz Transposta: faz a transformação das linhas das de uma matriz em colunas na matriz transposta
## Recebe uma matriz A (nXm) e retorna uma matriz (mxn)
def met_matriz_transposta(MatrizA):
    NumLinhaMatrizA=len(MatrizA)               ## variavel NumLinhaMatrizA guarda o numero de linhas da matriz MatrizA
    NumColMatriz=len(MatrizA[0])               ## variavel NumLinhaMatrizA guarda o numero de colunas da matriz MatrizA
    MatrizC_Result_A_x_B = np.empty((NumColMatriz,NumLinhaMatrizA))      ## variavel MatrizC_Result_A_x_B tera o numero de linhas igual ao numero de colunas da matriz MatrizA e numero de linhas igual ao numero de colunas da matriz MatrizA
    # loop para a transposicao, linha = coluna
    for i in range(0,NumLinhaMatrizA,1):
        for j in range(0,NumColMatriz,1):
            MatrizC_Result_A_x_B[j][i]=MatrizA[i][j]
    return MatrizC_Result_A_x_B

## Metodo para Matriz Determinante: Faz o calculo do determinante de uma matriz
## Recebe uma matriz MatrizA (nXm) e retorna o valor numérico do determinante
def met_matriz_determinante(MatrizA):
    NumLinhaMatrizA = len(MatrizA)      ## variavel NumLinhaMatrizA para guardar numero de linhas da matriz MatrizA
    NumColMatriz = len(MatrizA[0])      ## variavel NumColMatriz para guardar numero de colunas da matriz MatrizA
    ### Se a matriz tiver apenas 1 elemento o determinante é o prório elemento
    if NumLinhaMatrizA ==  1:
        ValorDeterminante = MatrizA[0][0]
    ### Se a matriz tiver apenas 2 elemento o determinante é calculado pela regra da diferença do produto da diagonal principal com o produto da diagonal secundária
    elif NumLinhaMatrizA ==  2:
        ValorDeterminante = MatrizA[0][0]*MatrizA[1][1]-MatrizA[0][1]*MatrizA[1][0]
    else:
    ### Se a matriz tiver mais de dois elementos o algoritmo realiza o Teorema de Laplace com chamadas das matrizes reduzidas pelos cofatores
        ValorDeterminante=0
        for j in range(0,NumColMatriz,1):
            ValorDeterminante=ValorDeterminante+MatrizA[0][j]*met_matriz_cofator(MatrizA,0,j)  ##aciona o metodo met_matriz_cofator
    return ValorDeterminante

## Metodo met_matriz_cofator usado para calcular o complemento algebrico (cofator) de uma matriz
## Recebe uma matriz MatrizA (nXm) e coordenada que deseja o cofator e retorna o valor numérico do cofator
def met_matriz_cofator(MatrizA,i,j):
    NumLinhaMatrizA = len(MatrizA)      ## variavel NumLinhaMatrizA guarda o numero de linhas da matriz MatrizA
    NumColMatriz = len(MatrizA[0])      ## variavel NumColMatriz guarda o numero de colunas da matriz MatrizA
    indexAncoraLinha=0    ##varivel indexAncoraLinha guarda o index para ancora a ser utilizados na formacao da matriz reduzida
    indexAncoraColuna=0   ##varivel indexAncoraColuna guarda o index para ancora a ser utilizados na formacao da matriz reduzida
    ## Variavel ValorCofator para matriz reduzida para a formacao do cofator
    ValorCofator = np.empty(((NumLinhaMatrizA-1), (NumColMatriz-1)))
    ## loop para redução da matriz, removendo a linha e coluna em que o cofator se encontra (reduçao das matrizes pelo teorema de Laplace)
    for l in range(0,NumLinhaMatrizA,1):
        ## Checa se a linha é a linha do cofator para ser removida
        if l == i:
            indexAncoraLinha=1
        else:
            for k in range(0,NumColMatriz,1):
                ## Checa se se a coluna é a coluna do cofator para ser removida
                if k == j:
                    indexAncoraColuna=1
                else:
                    ## variavel ValorCofator para guardar a matriz reduzida
                    ValorCofator[(l-indexAncoraLinha)][(k-indexAncoraColuna)]=MatrizA[l][k]
            indexAncoraColuna = 0
    ## realiza a chamada da rotina de calculo do ValorDeterminante com a matriz reduzida, inicia um ciclo de chamadas recursivas entre cofator e ValorDeterminante
    ValorDeterminanteC = met_matriz_determinante(ValorCofator)
    ## calculo do cafator segundo o teorema de Laplace
    resultCofator=((-1)**((i+1)+(j+1))) * ValorDeterminanteC
    return resultCofator

## Metodo para formacao da matriz de cofatores (matriz Adjacente), utilizada no calculo da Matriz Inversa
## Recebe uma matriz MatrizA e retorna uma matriz de cofatores
def met_MatrizAdjacente(MatrizA):
    NumLinhaMatrizA = len(MatrizA)     ## variavel NumLinhaMatrizA guarda numero de linhas da matriz MatrizA
    NumColMatriz = len(MatrizA[0])     ## variavel NumColMatriz guarda numero de colunas da matriz MatrizA
    ## Variavel MatrizC_Result_A_x_B para matriz de cofatores
    MatrizC_Result_A_x_B = np.empty(((NumLinhaMatrizA), (NumColMatriz)))
    for i in range(0,NumLinhaMatrizA,1):
        for j in range(0,NumColMatriz,1):
            ## acionamento da rotina para obter o cofator de cada ponto da matriz
            MatrizC_Result_A_x_B[i][j]=met_matriz_cofator(MatrizA,i,j)
    ## a matriz de cofatores é uma transposta, antes de retornar a matriz de cofatores, realiza a transposta
    MatrizTranspCoef=met_matriz_transposta(MatrizC_Result_A_x_B)
    return MatrizTranspCoef

## Metodo met_MatrizInversa para formacao da matriz inversa
## Recebe uma matriz MatrizA e retorna a matriz inversa MatrizA^-1
def met_MatrizInversa(MatrizA):
    NumLinhaMatrizA = len(MatrizA)  ## variavel NumLinhaMatrizA guarda numero de linhas da matriz MatrizA
    NumColMatriz = len(MatrizA[0])  ## variavel NumColMatriz guarda numero de colunas da matriz MatrizA
    ## Variavel MatrizCoafatores para matriz de cofatores - chama o metodo matriz met_MatrizAdjacente
    MatrizCoafatores=met_MatrizAdjacente(MatrizA)
    ## variavel DeterminanteMatrizA guarda o calculo do Determinante da Matriz MatrizA para utilizar no calculo da inversa sendo MatrizA^-1=MatrizCoafatores(MatrizA)/Det(MatrizA)
    DeterminanteMatrizA=met_matriz_determinante(MatrizA)
    MtInversa = np.empty(((NumLinhaMatrizA), (NumColMatriz)))
    ## loop para calcular cada elemento da matriz inversa pela formula  MatrizA^-1=MatrizCoafatores(MatrizA)/Det(MatrizA)
    for i in range(0,NumLinhaMatrizA,1):
        for j in range(0, NumColMatriz, 1):
            MtInversa[i][j] = MatrizCoafatores[i][j]*(1/DeterminanteMatrizA)
    return MtInversa ## retorna variavel MtInversa

def met_Media_Adj(matrizInicial):
  print("Matriz INICIAL:", matrizInicial)
  
  linha = np.size(matrizInicial,0)           # numero de linhas da matriz A
  coluna = np.size(matrizInicial, 1)         # numero de colunas da matriz A
  
  mtx_Media=np.zeros((coluna,1))             #matriz media media com numero de colunas 
  mtx_DataAjuste = np.empty(((linha), (coluna)))  
  
#  print("MATRIZ MEDIA VAZIA *****")
#  print(mtx_Media)
#  print("MATRIZ AJUSTE VAZIA *****")
#  print(mtx_Media)
  
  for i in range(0, coluna, 1):
      for j in range(0, linha, 1):
          print(j)
          mtx_Media[i]=mtx_Media[i]+matrizInicial[j][i]
          #print (mtx_DataAjuste)
          
  for i in range(0, coluna, 1):
      mtx_Media[i]=mtx_Media[i]/linha
          
  for i in range(0, linha, 1):
      for j in range(0,coluna,1):
          mtx_DataAjuste[i][j]=matrizInicial[i][j]-mtx_Media[j]
            
            
  print("Matriz Media ")
  print(mtx_Media)
    
  print("Matriz Ajuste ")
  print(mtx_DataAjuste)
    
      
  return mtx_Media, mtx_DataAjuste
    
## Rotina para formacao da matriz de covariancia
## Recebe uma matriz DataAdjust e retorna a matriz covariancia Σ
def met_Mtx_Covariancia(mtx_DataAjuste):
    linha = np.size(mtx_DataAjuste,0)  # numero de linhas da matriz mtx_DataAjuste
    coluna = np.size(mtx_DataAjuste, 1)  # numero de linhas da matriz mtx_DataAjuste
    
    mtx_Somatoria = np.empty(((coluna), (coluna)))
    mtx_BaseSomat = np.zeros(((coluna**coluna), 1))

    for i in range(0,linha,1):
        for j in range(0,coluna,1):
            for k in range(0,coluna,1):
                mtx_BaseSomat[(coluna*j+k)]=mtx_BaseSomat[(coluna*j+k)]+(mtx_DataAjuste[i][j] * mtx_DataAjuste[i][k])

    for j in range(0,coluna,1):
        for k in range(0, coluna, 1):
            mtx_Somatoria[j][k]=mtx_BaseSomat[(coluna * j + k)]/(linha - 1)

    print("Matriz Somatoria ")
    print(mtx_Somatoria)
    
    print("Matriz BaseSomat ")
    print(mtx_BaseSomat)    
    
    return mtx_Somatoria

## Rotina para calculo dos autovalores - eigenvalues
## Recebe uma matriz covariancia Σ e retorna a matriz de autovalores Λ
def met_Mtx_AutoValores(mtx_Somatoria):
    linha = np.size(mtx_Somatoria, 0)
    if linha==1:                   # calculo dos auto valores de uma funcao linear
        mtx_AltoValores = np.empty((1))
        mtx_AltoValores[0] = mtx_Somatoria[0]
    else:
        if linha==2:              # calculo dos auto valores de uma funcao quadratica
            mtx_AltoValores=np.empty((2))
            delta=(-(mtx_Somatoria[0][0] + mtx_Somatoria[1][1]) )**2 - 4 * (mtx_Somatoria[0][0] * mtx_Somatoria[1][1] - mtx_Somatoria[0][1] * mtx_Somatoria[1][0])
            mtx_AltoValores[0]=((mtx_Somatoria[0][0] + mtx_Somatoria[1][1])-delta**(1/2))/2
            mtx_AltoValores[1]=((mtx_Somatoria[0][0] + mtx_Somatoria[1][1])+delta**(1/2))/2
        else:
            if linha>2:           # calculo dos auto valores de uma funcao polinomial de ordem maior, pelo scikit learn
                mtx_AltoValores=np.linalg.eigvals(mtx_Somatoria)
            else:
                return -1
    return mtx_AltoValores

## Rotina para calculo dos Autovetores - eigenvectores
## Recebe uma matriz covariancia Σ e a matriz de autovalores Λ e retorna a matriz de autovetores Φ
def met_Mtx_AltoVetores(mtx_Somatoria, mtx_AltoValores):
    linha = np.size(mtx_Somatoria, 0)
    if linha==1:                   # calculo dos auto valores de uma funcao linear que será sempre 0
        mtx_AltoVetores = np.empty((1))
        mtx_AltoVetores[0] = mtx_Somatoria[0]*1-mtx_AltoValores[0]
    else:
        if linha==2:              # calculo dos auto vetores de uma funcao quadratica
            mtx_AltoVetores = np.empty((2, 2))

            mtx_AltoVetores[0][0] = 1

            if (mtx_Somatoria[0][1] != 0):
                mtx_AltoVetores[1][0] = -(mtx_Somatoria[0][0] - mtx_AltoValores[0]) * mtx_AltoVetores[0][0] / mtx_Somatoria[0][1]
            else:
                mtx_AltoVetores[1][0] = -(mtx_Somatoria[1][0]) * mtx_AltoVetores[0][0] / (mtx_Somatoria[1][1] - mtx_AltoValores[0])
            mtx_AltoVetores[1][1] = mtx_AltoVetores[0][0]

            if (mtx_Somatoria[1][0] != 0):
                mtx_AltoVetores[0][1] = -(mtx_Somatoria[1][1] - mtx_AltoValores[1]) * mtx_AltoVetores[1][1] / mtx_Somatoria[1][0]
            else:
                mtx_AltoVetores[0][1] = -(mtx_Somatoria[0][1]) * mtx_AltoVetores[1][1] / (mtx_Somatoria[0][0] - mtx_AltoValores[1])
        else:
            if linha>2:           # calculo dos auto vetores de uma funcao polinomial de ordem maior, pelo scikit learn
                mtx_AltoValores, mtx_AltoVetores  = np.linalg.eig(mtx_Somatoria)
            else:
                return -1
    return mtx_AltoVetores

## Rotina para calculo dos Vetores principais - Feature Vector
## Recebe os auto valores e auto vetores (Φ, Λ) e ordena os vetores de acordo com os eixos de maior importancia
def met_Mtx_VetoresPrincipais(mtx_AltoVetores, mtx_AltoValores):
    VectorOrdem=mtx_AltoValores.argsort()[::-1]
    linha = np.size(mtx_AltoVetores, 0)
    coluna = np.size(mtx_AltoVetores, 1)
    mtx_VetorPrinc= np.empty((linha, coluna))
    for j in range(0, coluna, 1):
        for i in range(0, linha, 1):
            mtx_VetorPrinc[i][j] = mtx_AltoVetores[i][(VectorOrdem[j])]
    return mtx_VetorPrinc, VectorOrdem

## Rotina para Calculo da dispersao intra grupo
## Recebe a matriz com os dados segmentados, quantidade de grupos e variaveis/dimensoes e retorna a matriz de dispersao de cada grupo
def met_Mtx_SW(tpDados, medDados, quantidadeGrupos, quantidadeVarGrupos):
    valorSW = 0
    valor_ClassSC_Mat = 0
    for i in range(0, quantidadeGrupos, 1):
        N = np.size(tpDados[i], 0)
        for j in range(0, N, 1):
            valorDadosX = np.asarray(tpDados[i][j]).reshape(quantidadeVarGrupos, 1)
            mv = np.asarray(medDados[i]).reshape(quantidadeVarGrupos, 1)
            valor_ClassSC_Mat += (valorDadosX - mv).dot((valorDadosX - mv).T)
    valorSW += valor_ClassSC_Mat
    return valorSW

## Rotina para Calculo da dispersao intre grupos
## Recebe a matriz com os dados segmentados e quantidade de variaveis/dimensoes e retorna a matriz de dispersao entre grupos
def met_Mtx_SB(tpDados, medDados, quantidadeVarGrupos):
    ##Calculo das Médias Total  
    medTotal = []
    for i in range(0, quantidadeVarGrupos, 1):
        medTotal.append(np.mean(medDados[:, i]))

    valorSB = np.zeros((quantidadeVarGrupos, quantidadeVarGrupos))
    for i in range(0, 3, 1):
        n = np.size(tpDados[i], 0)
        vetorPrinc = medDados[i].reshape(quantidadeVarGrupos, 1)
        medTotal = np.array(medTotal).reshape(quantidadeVarGrupos, 1)
        valorSB += n * (vetorPrinc - medTotal).dot((vetorPrinc - medTotal).T)
    return valorSB	
	
def main():
     
    ########### leitura do arquivo que contem a base de dados em excel
    ########### Arquivo no formato: Y, X, X^2, onde Y é variável dependente, X explicativa e X^2  as outras explicativas
     import xlrd
     book = xlrd.open_workbook("LDAdb.xlsx") ##nome do arquivo excel
     print ("Número de abas na planilha do excel: ", book.nsheets)
     print ("Nomes das Planilhas de excel:", book.sheet_names())
      
     for vSheet in book.sheet_names():
        print("xxxxxxxxx******NOME DA SHEET", vSheet)
        sh = book.sheet_by_name(vSheet)
        
        labelChart=sh.name
        print("SHEET: ",labelChart )
        
        quantElementos=sh.nrows
        
        sheet = book.sheet_by_index(0)
        nrows = sheet.nrows
        print(sheet.nrows)
        
        #ncols = 1
        ncols = sheet.ncols                
        print(sheet.ncols)

        tbCol = sh.ncols  #numero de colunas
        variaveis=tbCol-1		
        Label = []        #vetor
        quantidadeGrupos=3
        np.set_printoptions(precision=4)
        dic_Type = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
		 
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
       
        for i in range (0, tbCol,1):
            Label.append(sh.cell_value(rowx=0, colx=i))
        labelY = sh.cell_value(rowx=0, colx=0)
        labelX = sh.cell_value(rowx=0, colx=1)
        QuantElementos=sh.nrows
                
        ### Matrizes para carregar as informaçoes das bases de dados do arquivo excel
        mtx_DBasePCA=np.empty(((QuantElementos - 1), tbCol-1))
        mtx_DBaseLDA=np.empty(((QuantElementos - 1), tbCol))
        
        for i in range (1,QuantElementos,1):
            ## Carga Variável dependente Y 
            #Y[(i-1)][0]=sh.cell_value(rowx=i, colx=0)
            ###Busca dados do PCA para carga
            for j in range(0, tbCol,1):
                if j ==0:
                    mtx_DBasePCA[(i-1)][(j-1)] = sh.cell_value(rowx=i, colx=j)
                    mtx_DBaseLDA[(i-1)][j] = sh.cell_value(rowx=i, colx=j)
                else:
                    mtx_DBaseLDA[(i - 1)][j] = int(sh.cell_value(rowx=i, colx=j))

        ## Segmenta os Grupos para o LDA  
        tpDados = []
        medDados = []
        for i in range(0, quantidadeGrupos, 1):
            tpDados.append([])
            medDados.append([])
        for i in range(0, (QuantElementos - 1), 1):
            valorObj = int(mtx_DBaseLDA[i][0])
            tpDados[valorObj].append(mtx_DBaseLDA[i][1:, ])

        fig, ax = plt.subplots(figsize=(12, 12))

        ##Plotagem dos dados Orignais em 2d - Eixos originais
        dataFrameOriginalBase = pd.DataFrame({'idx': mtx_DBaseLDA[:,0], 'X': mtx_DBaseLDA[:, 1], 'Y': mtx_DBaseLDA[:, 2]})

        line = dict(linewidth=1, linestyle='--', color='k')

        ax = plt.gca()
        ax = plt.subplot((111))

        plt.title(" Dados Orinais 2D - 3 Grupos")

        for grupo, marker, color in zip(range(0, 3), ('^', 's', 'o'), ('blue', 'orange', 'green')):
            varFiltro = dataFrameOriginalBase["idx"] == grupo
            varAgrup="Amostra "+dic_Type[grupo]
            PlotBase=dataFrameOriginalBase.where(varFiltro)
            PlotBase.plot(kind='scatter', x='X', y='Y', marker=marker, color=color, s=40, ax=ax, label=varAgrup)
        plt.grid(True)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.show()
		
        
        ##LDA - trecho para calular as medias entre os grupos
        quantidadeVarGrupos = np.size(tpDados[0], 1)
        for i in range(0, quantidadeGrupos, 1):
            tpDados[i] = np.asarray(tpDados[i])
            for j in range(0, quantidadeVarGrupos, 1):
                medDados[i].append(np.mean(tpDados[i][:, j]))
        medDados = np.array(medDados).reshape(quantidadeGrupos, variaveis)

        ##variavel para guardar o valor da dispersao dentro dos Grupos - Sw
        valorSW= met_Mtx_SW(tpDados, medDados, quantidadeGrupos,quantidadeVarGrupos)
        ## variavel para guardar o valor da  Dispersao entre os Grupos - Sb
        valorSB=met_Mtx_SB(tpDados, medDados, quantidadeVarGrupos)
        ## variavel para guardar o valor da matriz de projecao   
        valorMatrizProjecao = np.linalg.inv(valorSW).dot(valorSB)
        ##variaveis para os  autovetores e autovalores   
        varAltoValorLDA, varAltoVetorLDA = np.linalg.eig(valorMatrizProjecao)
        vetorPrinc_LDA, vetorOrder_LDA = met_Mtx_VetoresPrincipais(varAltoVetorLDA, varAltoValorLDA)

        ##linha e coluna do vetor
        linhaVetor = np.size(vetorPrinc_LDA, 0)
        colunaVetor = np.size(vetorPrinc_LDA, 1)

		##trecho para geração das Matrizes dimensionais - VtDimensoes segmenta cada dimensao em um vetor próprio 

        vetDimensoes = []
        vetDadosFinais = []
        vetDadosFinaisT = []

        for i in range(0, colunaVetor, 1):
            vetDimensao = np.zeros((linhaVetor, colunaVetor))
            for j in range(0, linhaVetor, 1):
                vetDimensao[j][i] = vetorPrinc_LDA[j][i]
            vetDimensoes.append(vetDimensao)
        vetDimensaoT = []

        DataAdjust = mtx_DBaseLDA[:, 1:np.size(mtx_DBaseLDA, 1)]
        DataAdjustT = met_matriz_transposta(DataAdjust)
        for i in range(0, colunaVetor, 1):
            vetDimensaoT.append(met_matriz_transposta(vetDimensoes[i]))
        for i in range(0, colunaVetor, 1):
            vetDadosFinais.append(met_matriz_mult(vetDimensaoT[i], DataAdjustT))
        for i in range(0, colunaVetor, 1):
            vetDadosFinaisT.append(met_matriz_transposta(vetDadosFinais[i]))

        vetDadosFinaisPrincT = vetDadosFinaisT[0]  # met_matriz_transposta(FinalDataPriVetor)
        vetDadosFinaisSecT = vetDadosFinaisT[1]  # met_matriz_transposta(FinalDataSecVetor)
        dataFrameEixoLDA_vec = pd.DataFrame({'XPriVet': vetDadosFinaisPrincT[:, 0], 'YPriVet': vetDadosFinaisPrincT[:, 1],
                                     'XSecVet': vetDadosFinaisSecT[:, 0], 'YSecVet': vetDadosFinaisSecT[:, 1]})
        # Transformando as amostras no novo subespaço
        eixoX_LDA = []
        for i in range(0, quantidadeGrupos, 1):
            eixoX_LDA.append(tpDados[i].dot(vetorPrinc_LDA))

        linhaVetor = np.size(vetorPrinc_LDA, 0)
        colunaVetor = np.size(vetorPrinc_LDA, 1)

        print("Numero de Dimensoes:  ", colunaVetor)

        ##Matrizes dimensionais - VtDimensoes segmenta cada dimensao em um vetor próprio

        fig, ax = plt.subplots(figsize=(24, 12))
        ax = plt.gca()
        ax = plt.subplot((121))

        dataFrameEixoLDA_vec.plot(kind='line', marker='o', ms=6, x='XPriVet', y='YPriVet', color='red', ax=ax,
                          label='LDAVetor Principal')
        dataFrameEixoLDA_vec.plot(kind='line', marker='o', ms=6, x='XSecVet', y='YSecVet', color='green', ax=ax,
                          label='LDAVetor Secundario')

        ##Plota LDA no Eixo 

        for grupo, marker, color in zip(range(0, 3), ('^', 's', 'o'), ('blue', 'orange', 'green')):
            plt.scatter(x=eixoX_LDA[grupo][:, 0], y=eixoX_LDA[grupo][:, 1], marker=marker, color=color, label=dic_Type[grupo])
        plt.xlabel(Label[vetorOrder_LDA[0]])
        plt.ylabel(Label[vetorOrder_LDA[1]])
        leg = plt.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.title('ANÁLISE DISCRIMINANTE LINEAR (LDA) - Base Iris')

        plt.axvline(x=0)
        x = np.linspace(-2, 3, 100)
        y = 0 * x
        plt.plot(x, y, '-r')
        plt.grid()

        ##DADOS DO PCA 
        ax = plt.subplot((122))
        # Calcula matriz Ajustada pelas médias
        PCA_Mean, PCA_DataAdjust = matriz_DataAdjust(mtx_DBasePCA)
        # Calcula a covariancia
        varPCA_somatoria = met_Mtx_Covariancia(PCA_DataAdjust)
        print("covariância PCA Σ:", varPCA_somatoria)
        # Calcula os autovalores
        print("varPCA_somatoria PCA_Σ", varPCA_somatoria)
        varPCA_AutoVal = matriz_autovalores(varPCA_somatoria)
        print('autovalores PCA Λ:', varPCA_AutoVal)
        # Calcula os autovetores
        varPCA_AutoVet = met_Mtx_AltoVetores(varPCA_somatoria, varPCA_AutoVal)
        print("autovetores PCA Φ: ", varPCA_AutoVet)
        # Ordena pelo vetor mais significativo
        vetorPrincipalBase_PCA, vetorOrder_PCA = met_Mtx_VetoresPrincipais(varPCA_AutoVet, varPCA_AutoVal)

        vetorPrincipal_PCA = copy.deepcopy(vetorPrincipalBase_PCA)
      
        # Exclui vetor menos significativo
        PCA_reduction = 1
        vetorPrincipal_PCA = vetorPrincipal_PCA[:, 0:(np.size(vetorPrincipal_PCA, 1) - PCA_reduction)]

        # trecho para calcular os vetores normalizados ao vetor mais significativo

        PCA_DataAdjustT = met_matriz_transposta(PCA_DataAdjust)
        PCA_FeatureVectorT = met_matriz_transposta(vetorPrincipal_PCA)
        PCA_FinalData = met_matriz_mult(PCA_FeatureVectorT, PCA_DataAdjustT)
        PCA_FinalDataT = met_matriz_transposta(PCA_FinalData)

        dfPCA_Vec = pd.DataFrame({'idx': mtx_DBaseLDA[:, 0], 'X': PCA_FinalDataT[:, 0], 'Y': PCA_FinalDataT[:, 1]})

        line = dict(linewidth=1, linestyle='--', color='k')
        ax.axhline(**line)
        ax.axvline(**line)
        ax = plt.gca()

        for grupo, marker, color in zip(range(0, 3), ('^', 's', 'o'), ('blue', 'orange', 'green')):
            PCA_filtro = dfPCA_Vec["idx"] == grupo
            varAgrup = "Amostra " + dic_Type[grupo]
            PCA_basePlot = dfPCA_Vec.where(PCA_filtro)
            PCA_basePlot.plot(kind='scatter', x='X', y='Y', marker=marker, color=color, s=40, ax=ax, label=varAgrup)
        plt.grid(True)

        PCA_colVet = np.size(vetorPrincipal_PCA, 1)

        ######     EIXOS   ###########

        linhaVetor = np.size(vetorPrincipal_PCA, 0)
        colunaVetor = np.size(vetorPrincipal_PCA, 1)

        PCA_vtDimensoes = []
        PCA_FinalDataVetor = []
        PCA_FinalDataVetorT = []

        for i in range(0, colunaVetor, 1):
            PCA_vtDimensao = np.zeros((linhaVetor, colunaVetor))
            for j in range(0, linhaVetor, 1):
                PCA_vtDimensao[j][i] = vetorPrincipal_PCA[j][i]
            PCA_vtDimensoes.append(PCA_vtDimensao)

        PCA_vtDimensoesT = []
        for i in range(0, colunaVetor, 1):
            PCA_vtDimensoesT.append(met_matriz_transposta(PCA_vtDimensoes[i]))
        for i in range(0, colunaVetor, 1):
            PCA_FinalDataVetor.append(met_matriz_mult(PCA_vtDimensoesT[i], PCA_DataAdjustT))
        for i in range(0, colunaVetor, 1):
            PCA_FinalDataVetorT.append(met_matriz_transposta(PCA_FinalDataVetor[i]))

        dfPCA_Eixo = pd.DataFrame({'X': PCA_FinalDataT[:, 0], 'Y': PCA_FinalDataT[:, 1]})

        for i in range(0, (colunaVetor - 1), 1):
            labelA = "X" + str(i) + "Vet"
            labelB = "EqVetor_" + str(i) + "_Vet_" + Label[vetorOrder_PCA[i]]
            dfPCA_Eixo[labelA] = PCA_FinalDataVetorT[i][:, 0]
            dfPCA_Eixo[labelB] = PCA_FinalDataVetorT[i][:, 1]

        for i in range(0, (colunaVetor - 1), 1):
            labelA = "X" + str(i) + "Vet"
            labelB = "EqVetor_" + str(i) + "_Vet_" + Label[vetorOrder_PCA[i]]
            dfPCA_Eixo.plot(kind='line', linewidth=3, x=labelA, y=labelB, ax=ax, label=labelB)

        plt.title("PCA Plotagem Eixo Principal em 2D para " + str(PCA_colVet) + "  dimensoes")
        plt.xlabel(Label[vetorOrder_PCA[0]])
        plt.ylabel(Label[vetorOrder_PCA[1]])

        plt.show()
                   

        ########################################               MDF            ##########################################
        dimensoes=np.size(vetorPrincipalBase_PCA,1)

        fig, ax = plt.subplots(figsize=(18, 18))
        for dim in range (0, (dimensoes), 1):

            vetorPrincipal_PCA=copy.deepcopy(vetorPrincipalBase_PCA)
            ###################################################################################################################
            # Exclui vetor menos significativo
            PCA_reduction = dim
            vetorPrincipal_PCA = vetorPrincipal_PCA[:, 0:(np.size(vetorPrincipal_PCA, 1) - PCA_reduction)]
            ###################################################################################################################


            # Calcula os vetores normalizados ao vetor mais significativo

            PCA_DataAdjustT = met_matriz_transposta(PCA_DataAdjust)
            PCA_FeatureVectorT = met_matriz_transposta(vetorPrincipal_PCA)
            PCA_FinalData = met_matriz_mult(PCA_FeatureVectorT, PCA_DataAdjustT)
            PCA_FinalDataT = met_matriz_transposta(PCA_FinalData)

            ## LDA       
            # trecho para separação de dados em grupos de tipos
            tpDados = []
            medDados = []
            for i in range(0, quantidadeGrupos, 1):
                tpDados.append([])
                medDados.append([])
            for i in range(0, (QuantElementos - 1), 1):
                valorObj = int(mtx_DBaseLDA[i][0])
                tpDados[valorObj].append(PCA_FinalDataT[i])

            #calcula as médias de cada grupo do LDA
            quantidadeVarGrupos = np.size(tpDados[0], 1)
            for i in range(0, quantidadeGrupos, 1):
                tpDados[i] = np.asarray(tpDados[i])
                for j in range(0, quantidadeVarGrupos, 1):
                    medDados[i].append(np.mean(tpDados[i][:, j]))
            medDados = np.array(medDados).reshape(quantidadeGrupos, quantidadeVarGrupos)

            ##trecho para calcular a Dispersao dentro dos Grupos - Sw
            valorSW = met_Mtx_SW(tpDados, medDados, quantidadeGrupos, quantidadeVarGrupos)
            print("Dispersao dentro dos Grupos - Sw", valorSW)
            ##Calculo Dispersao entre os Grupos - Sb  
            valorSB = met_Mtx_SB(tpDados, medDados, quantidadeVarGrupos)
            print("Dispersao entre dos Grupos - Sb", valorSB)
            ##Calculo da matriz de projecao   
            valorMatrizProjecao = np.linalg.inv(valorSW).dot(valorSB)
            # Calcula os autovetores e autovalores
            varAltoValorLDA, varAltoVetorLDA = np.linalg.eig(valorMatrizProjecao)
            print("Autovalores - LDA_Λ", varAltoValorLDA)
            print("Autovetores - LDA_Φ", varAltoVetorLDA)
            vetorPrinc_LDA, vetorOrder_LDA = met_Mtx_VetoresPrincipais(varAltoVetorLDA, varAltoValorLDA)

            ###########
            linhaVetor = np.size(vetorPrinc_LDA, 0)
            colunaVetor = np.size(vetorPrinc_LDA, 1)

            ######     Matrizes dimensionais - VtDimensoes segmenta cada dimensao em um vetor próprio ###########

            vetDimensoes = []
            vetDadosFinais = []
            vetDadosFinaisT = []

            for i in range(0, colunaVetor, 1):
                vetDimensao = np.zeros((linhaVetor, colunaVetor))
                for j in range(0, linhaVetor, 1):
                    vetDimensao[j][i] = vetorPrinc_LDA[j][i]
                vetDimensoes.append(vetDimensao)
            vetDimensaoT = []
            DataAdjust = PCA_FinalData

            for i in range(0, colunaVetor, 1):
                vetDimensaoT.append(met_matriz_transposta(vetDimensoes[i]))
            for i in range(0, colunaVetor, 1):
                vetDadosFinais.append(vetDimensaoT[i].dot(DataAdjust))
            for i in range(0, colunaVetor, 1):
                vetDadosFinaisT.append(met_matriz_transposta(vetDadosFinais[i]))


            ax = plt.subplot((221+dim))

            vetDadosFinaisPrincT = vetDadosFinaisT[0]

            colunaVetor = np.size(vetorPrinc_LDA, 1)

            if colunaVetor >1 :
                vetDadosFinaisSecT = vetDadosFinaisT[1]  # met_matriz_transposta(FinalDataSecVetor)
                dataFrameEixoLDA_vec = pd.DataFrame({'XPriVet': vetDadosFinaisPrincT[:, 0], 'YPriVet': vetDadosFinaisPrincT[:, 1],
                                         'XSecVet': vetDadosFinaisSecT[:, 0], 'YSecVet': vetDadosFinaisSecT[:, 1]})
                dataFrameEixoLDA_vec.plot(kind='line', marker='o', ms=6, x='XPriVet', y='YPriVet', color='red', ax=ax,
                                  label='LDAVetor Principal')
                dataFrameEixoLDA_vec.plot(kind='line', marker='o', ms=6, x='XSecVet', y='YSecVet', color='green', ax=ax,
                                  label='LDAVetor Secundario')

            # Transformando as amostras no novo subespaço
            eixoX_LDA = []
            for i in range(0, quantidadeGrupos, 1):
                eixoX_LDA.append(tpDados[i].dot(vetorPrinc_LDA))

            if colunaVetor <2 :
                for i in range(0, quantidadeGrupos, 1):
                    addVec = np.zeros((np.size(eixoX_LDA[i], 0), 1))
                    ReducVector = (np.append(eixoX_LDA[i], addVec, axis=1))
                    eixoX_LDA[i]=ReducVector


            print("Numero de Dimensoes:  ", colunaVetor)

            ##Plota MDF no Eixo

            for grupo, marker, color in zip(range(0, 3), ('^', 's', 'o'), ('blue', 'orange', 'green')):
                plt.scatter(x=eixoX_LDA[grupo][:, 0], y=eixoX_LDA[grupo][:, 1], marker=marker, s=50, color=color, label=dic_Type[grupo])
            plt.xlabel(Label[vetorOrder_LDA[0]])
            if colunaVetor > 1:
                plt.ylabel(Label[vetorOrder_LDA[1]])

            leg = plt.legend(loc='upper right', fancybox=True)
            leg.get_frame().set_alpha(0.5)

            plt.title('ANÁLISE Most Discriminant Features (PCA&LDA) - Base Iris - ' + str(colunaVetor) + ' dimensoes')

            plt.axvline(x=0)
            x = np.linspace(-2, 3, 100)
            y = 0 * x
            plt.plot(x, y, '-r')
            plt.grid()
        plt.show()

        
main()
