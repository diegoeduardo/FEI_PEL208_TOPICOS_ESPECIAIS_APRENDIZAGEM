# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:11:40 2019

@author: SilvaDE
"""

##########################################################################
import pandas as pd             #biblioteca para criacao de dataframes
import xlrd                     #biblioteca abrir arquivo  excel
import matplotlib.pyplot as plt #biblioteca para os graficos
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def main():
    
    ########### leitura do arquivo que contem a base de dados em excel
    ########### Arquivo no formato: Y, X, X^2, onde Y é variável dependente, X explicativa e X^2  as outras explicativas
     book = xlrd.open_workbook("DB_Rel2.xlsx") ##nome do arquivo excel
     print ("Número de abas na planilha do excel: ", book.nsheets)
     print ("Nomes das Planilhas de excel:", book.sheet_names())
      
     for vSheet in book.sheet_names():
        print("NOME DA SHEET", vSheet)
        sh = book.sheet_by_name(vSheet)
        
        labelChart=sh.name
        print("SHEET: ",labelChart )
        
        #labelY = sh.cell_value(rowx=0, colx=0)
        #print("labelY: ",labelY )
        
        #labelX=sh.cell_value(rowx=0, colx=1)
        #print("labelX: ",labelX )
        
        quantElementos=sh.nrows
        #print("QUANTIDADE DE ELEMENTOS: ", quantElementos)
        
        sheet = book.sheet_by_index(0)
        nrows = sheet.nrows
        print(sheet.nrows)
        
        #ncols = 1
        ncols = sheet.ncols                
        print(sheet.ncols)

        tbCol = sh.ncols  #numero de colunas
        Label = []        #vetor
        
        #print(sheet.cell_value(0,0))
        
       ## print("Totas as headers - colunas")
       ## for col in range(sheet.ncols):             
       ##     print(sheet.cell_value(0, col))
        
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
       ## for col in range(sheet.ncols):             
       ##     print(sheet.cell_value(2, col))
   
        for i in range (0, tbCol,1):
            Label.append(sh.cell_value(rowx=0, colx=i))
        labelY = sh.cell_value(rowx=0, colx=0)
        labelX=sh.cell_value(rowx=0, colx=1)
        QuantElementos=sh.nrows
        grp_YReal = []
        grp_XReal = []            
        
        ### Matrizes para carregar as informaçoes das bases de dados do arquivo excel
        mtx_DBasePCA=np.empty(((QuantElementos - 1), tbCol))   #Matriz das variáveis explicativa X
        XLinear = np.empty(((QuantElementos - 1), 2))  ##variavel XLinear para matriz das variáveis explicativa X
        Y = np.empty(((QuantElementos-1), 1))          ##variavel Y para matriz das variáveis dependente Y
       
        Xmin =sh.cell_value(rowx=1, colx=1)            ##variavel para ser usado como ponteiro para minimo a ser utilizado na formacao do gráfico
        Xmax =sh.cell_value(rowx=1, colx=1)            ##variavel para ser usado como ponteiro para máximo a ser utilizado na formacao do gráfico
                
        for i in range (1,QuantElementos,1):
            ## Carga Variável dependente Y 
            Y[(i-1)][0]=sh.cell_value(rowx=i, colx=0)
            ###Busca dados do PCA para carga
            for j in range(0, tbCol,1):
                if j ==0:
                    mtx_DBasePCA[(i-1)][(tbCol - 1)] = sh.cell_value(rowx=i, colx=j)
                else:
                    if j == (tbCol - 1):
                        mtx_DBasePCA[(i - 1)][0] = sh.cell_value(rowx=i, colx=j)
                    else:
                        mtx_DBasePCA[(i - 1)][j] = sh.cell_value(rowx=i, colx=j)        
           
            ## Carga Variável explicativa X 
            XLinear[(i - 1)][0] = 1     # Add 1 - "BIAS" conforme metodo dos minimos quadrados
            XLinear[(i - 1)][1] = sh.cell_value(rowx=i, colx=1)
        
            # variavel de ponteiro para minimo a ser utilizado na formacao do gráfico
            if Xmin>sh.cell_value(rowx=i, colx=1):Xmin=sh.cell_value(rowx=i, colx=1)
            if Xmax<sh.cell_value(rowx=i, colx=1):Xmax=sh.cell_value(rowx=i, colx=1)
            ## Valores das amostras para apresentar no gráfico
            grp_YReal.append(sh.cell_value(rowx=i, colx=0))
            grp_XReal.append(sh.cell_value(rowx=i, colx=1))
        
            ## Define intervalo para plotar gráficos 
            step=(Xmax-Xmin)/QuantElementos
            
#            if step <1 :
#                step=1
#            else:
#                if step>5 and step< 10:
#                    step=10
#                else:
#                    step=int(step)            
            
    ##Entrada dos dados da base de dados para o calculo do PCA
    #  Metodo irá calcular matriz Ajustada pelas médias - recebe as matrizes mtx_Media, mtx_DataAjuste
        media, dataAjuste=met_Media_Adj(mtx_DBasePCA) # passa como parametro de entada os dados oriundos do excel
    #  Metodo irá calcular os covariancia - recebe mtx_Somatoria
        somatoria = met_Mtx_Covariancia(dataAjuste)
        print("\n***********Covariância :\n", somatoria)
    #  Metodo irá calcular os autovalores - recebe mtx_AltoValores
        autoValores=met_Mtx_AutoValores(somatoria)
        print('\n***********Autovalores :', autoValores)
    # Metodo irá calcular os autovetores - recebe mtx_AltoVetores 
        autoVetores=met_Mtx_AltoVetores(somatoria, autoValores)
        print("\n***********Autovetores:\n ", autoVetores)
    # Metodo irá ordenar pelo vetor mais significativo- recebe mtx_VetorPrinc
        vetorPrincipal=met_Mtx_VetoresPrincipais(autoVetores, autoValores)
    
    # Metodo ira calcular os auto vetores e considerar o mais significativo
        dataAjusteTrasposto=met_matriz_transposta(dataAjuste)
        vetorPrincipalTransposto=met_matriz_transposta(vetorPrincipal)
#
        resultFinal=met_matriz_mult(vetorPrincipalTransposto, dataAjusteTrasposto)
        resultFinalTranspoto=met_matriz_transposta(resultFinal)
        
        linhaVetor = np.size(vetorPrincipal, 0)
        colunaVetor = np.size(vetorPrincipal, 1)    
        
        print("\n*************Numero de Dimensoes do vetor:  \n", colunaVetor)
        ######     Matrizes dimensionais - VtDimensoes segmenta cada dimensao em um vetor próprio ###########

        vetDim = []
        dataVetFin = []
        dataVetFinTransp = []

        for i in range (0, colunaVetor,1):
            dimVt=np.zeros((linhaVetor, colunaVetor))
            for j in range(0,linhaVetor,1):
                dimVt[j][i]=vetorPrincipal[j][i]
            vetDim.append(dimVt)

        vetDimT = []
        for i in range(0, colunaVetor,1):
            vetDimT.append(met_matriz_transposta(vetDim[i]))
        for i in range(0, colunaVetor, 1):
            dataVetFin.append(met_matriz_mult(vetDimT[i],dataAjusteTrasposto))
        for i in range(0, colunaVetor, 1):
            dataVetFinTransp.append(met_matriz_transposta(dataVetFin[i]))


        vetDataFinalPrim=dataVetFin[0] #met_matriz_mult(VtPrimarioT,dataAjusteTrasposto)
        vetDataFinalPrimTransp=dataVetFinTransp[0] #met_matriz_transposta(vetDataFinalPrim)

        vetDataFinalSec=dataVetFin[1] #met_matriz_mult(VtSecundarioT,dataAjusteTrasposto)
        vetDataFinalSecTransp=dataVetFinTransp[1] #met_matriz_transposta(vetDataFinalSec)


        dataFrameVet = pd.DataFrame({'X': resultFinalTranspoto[:, 0], 'Y': resultFinalTranspoto[:, 1], 'XPriVet': vetDataFinalPrimTransp[:, 0],'YPriVet': vetDataFinalPrimTransp[:, 1], 'XSecVet': vetDataFinalSecTransp[:, 0] ,'YSecVet': vetDataFinalSecTransp[:, 1]})

################################## Plota PCA no Eixo Principal####################################

        line = dict(linewidth=1, linestyle='--', color='k')
        fPlot, ax = plt.subplots(figsize=(12, 8))

        ax.axhline(**line)
        ax.axvline(**line)

        ax = plt.gca()

        dataFrameVet.plot(kind='scatter', x='X', y='Y', color='blue',ax=ax, label='Todos Vetores')
        dataFrameVet.plot(kind='line', marker='o', ms=6, x='XPriVet', y='YPriVet', color='red', ax=ax, label='EqVetor Principal')
        dataFrameVet.plot(kind='line', marker='o', ms=6, x='XSecVet', y='YSecVet', color='green', ax=ax, label='EqVetor Secundario')

        plt.title("PCA Eixo Principal "+labelChart)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.grid(True)

        plt.show()

####################################   PCA todos Eixos  ############################################################
        indexVec=[1,0]
        vetPrincTI=met_MatrizInversa(vetorPrincipalTransposto)

        DataOrigemMediaPrim = np.full((np.size(resultFinal,0), np.size(resultFinal,1)), media[indexVec[0]] )

        DataOrigem = met_matriz_transposta(met_matriz_mult(vetPrincTI, resultFinal)+DataOrigemMediaPrim)
        DataOrigemVetPrim = met_matriz_transposta(met_matriz_mult(vetPrincTI, vetDataFinalPrim)+DataOrigemMediaPrim)
        DataOrigemVetPrim.sort(axis=0)
        DataOrigemVetSec = met_matriz_transposta(met_matriz_mult(vetPrincTI, vetDataFinalSec)+DataOrigemMediaPrim)


        dataFrame1 = pd.DataFrame({'X': DataOrigem[:, 0], 'Y': DataOrigem[:, 1], 'XPriVet': DataOrigemVetPrim[:, 0], 'YPriVet': DataOrigemVetPrim[:, 1], 'XSecVet': DataOrigemVetSec[:, 1], 'YSecVet': DataOrigemVetSec[:, 0] })

        print("DataFrame CPA\n")
        print("####################################################################################################\n")
        print (dataFrame1)
        print("####################################################################################################\n")


################################## Plota PCA todos Eixos####################################
        line = dict(linewidth=1, linestyle='--', color='k')
        fPlot, ax = plt.subplots(figsize=(12, 12))

        ax = plt.gca()

        dataFrame1.plot(kind='scatter', x='X', y='Y', color='blue', ax=ax, label='TodosEqVetoresS')
        dataFrame1.plot(kind='line', marker='o', ms=6,  x='XPriVet', y='YPriVet', color='red',ax=ax, label='EqVetor Principal')
        dataFrame1.plot(kind='line', marker='o', ms=6, x='XSecVet', y='YSecVet', color='green', ax=ax, label='EqVetore Secundario')

        plt.title("PCA Todos Eixos "+labelChart)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.grid(True)

        plt.show()


        ##Calculo para o LMS  Linear            
        XLinearT=met_matriz_transposta(XLinear)           ## Calculo da Matriz X transposta
        XLinearTXLinear=met_matriz_mult(XLinearT,XLinear) ## Calculo da Matriz X transposta multiplicada pela Matriz X
        XLinearTY=met_matriz_mult(XLinearT,Y)             ## Calculo da Matriz X transposta multiplicada pela Matriz Y
        XLinearTI = met_MatrizInversa(XLinearTXLinear)    ## Calculo da Matrix inversa (XT*X)
        βXLinear=met_matriz_mult(XLinearTI,XLinearTY)     ## Calculo da Fatores da regressao LMS
        print("βXLinear:",βXLinear)
        
        #########################################################################################
        ##Variáveis para formação dos gráficos 
        grp_YXLinear= []
#        grp_YXLinearRob = []
#        grp_YXQuad = []
        grp_X = []

        DataOrigemMediaPrim = np.full((np.size(resultFinal,0), np.size(resultFinal,1)), media[indexVec[0]] )
        y = np.full((np.size(resultFinal,1)), media[indexVec[0]] )
        x = np.full((np.size(resultFinal,1)), media[indexVec[1]] )
        DataOrigemMediaPrim[0, :]=x
        DataOrigemMediaPrim[1, :]=y

        DataOrigemVetPrim = met_matriz_transposta(met_matriz_mult(vetPrincTI, vetDataFinalPrim)+DataOrigemMediaPrim)
        DataOrigemVetPrim.sort(axis=0)

        grp_XPCA = DataOrigemVetPrim[:, 0]
        grp_PriPCA = DataOrigemVetPrim[:, 1]


        plt.show()

        Xmin=np.min(grp_XPCA)



        for index in range(0, (QuantElementos-1), 1):
            ################## Prepara Gráfico Regressao Linear Simples ##############################
            varXLinear = np.empty((1, 2))
            varXLinear[0][0] = 1
            varXLinear[0][1] = (index*step+Xmin)
            yRegresaaoXLinear = met_matriz_mult(varXLinear, βXLinear)
            grp_YXLinear.insert(index, yRegresaaoXLinear[0][0])
            ##########################################################################################
            grp_X.insert(index, varXLinear[0][1])	
            
            if index > (QuantElementos-2):
                grp_YReal.append(np.nan)
                grp_XReal.append(np.nan)            

########### Data Frame conjugando todos os modelos para a formação de gráfico único ##########
        ########### Data Frame conjugando todos os modelos para a formação de gráfico único ##########

        df = pd.DataFrame({'X': grp_X, 'Linear': grp_YXLinear, 'XReal': grp_XReal, 'YReal': grp_YReal, 'XPCA': grp_XPCA, 'PriPCA': grp_PriPCA})
        ########################## Imprime tabela com todos os resultados ############################

        print("Comparação dos graficos de DataFrame Regressão x PCA \n")
        print("####################################################################################################\n")
        print (df)
        ################### Fixa eixo de referencia unico para todos os graficos #####################
        line = dict(linewidth=1, linestyle='--', color='k')
        fig, ax = plt.subplots(figsize=(12, 12))

        ax = plt.gca()
        ax.set_xlim([(-2*Xmin), (2*Xmax)])

        ################################## Plota todos os Gráficos####################################

        df.plot(kind='line', marker='o', ms=6, x='XPCA', y='PriPCA', color='red', ax=ax, label='EixoPrimarioPCA')
        df.plot(kind='scatter', x='XReal', y='YReal', color='blue',ax=ax)
        df.plot(kind='line', x='X', y='Linear', color='green', ax=ax ,  label='RegressaoLinear')
        plt.title(labelChart)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.grid(True)
        plt.show()

                
                
#        
#        ### Matrizes para carregar as informaçoes das bases de dados do arquivo excel
#        # matrizInicial = np.empty(((quantElementos - 1), 2))  ##variavel XLinear para matriz das variáveis explicativa X
#        matrizInicial = np.empty
#        matrizInicial = np.empty(((quantElementos -1), ncols))
#        
#        matrizInicial2 = np.empty
#        matrizInicial2 = np.empty(((quantElementos -1), ncols))
#        
#        
#        y = np.empty(((quantElementos-1), 1))          ##variavel Y para matriz das variáveis dependente Y
#        # print(matrizInicial) #mostrar a matriz vazia
#                    
#        # loop para criar a matriz com os dados do excel - for linha in range(1, quantElementos, 1): ... a cada linha da planilha excell ...
#        for linha in range(1, quantElementos, 1):
#            ## Carga Y - como a primeira linha da planilha do excel tem o nome dos campos (linha - 1 no campo linha da matriz) 
#            y[(linha - 1)][0] = sh.cell_value(rowx=linha, colx=0)
#            ## Carga X
#            #matrizInicial[(linha - 1)][0] = 1 # BIAS 
#            #matrizInicial[(linha - 1)][1] = sh.cell_value(rowx=linha, colx=1)
#            
#            matrizInicial[(linha - 1)][0] = sh.cell_value(rowx=linha, colx=0) ## carregando a primeira coluna 
#            matrizInicial[(linha - 1)][1] = sh.cell_value(rowx=linha, colx=1) ## carregando a segunda coluna 

    
        #print("Matriz:", matrizInicial)             
     
        
        #teste = np.cov(np.transpose(matrizInicial))
        #print(teste)
       # met_Media_Adj(matrizInicial)
        #met_Mtx_Covariancia(met_Media_Adj(matrizInicial))
               
main()
      

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
    return mtx_VetorPrinc
