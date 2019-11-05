##########################################################################
## Importação das bibliotecas que serão usadas no programa
##########################################################################
import numpy as np
import xlrd                     #biblioteca abrir arquivo  excel
import pandas as pd             #biblioteca para criacao de dataframes
import matplotlib.pyplot as plt #biblioteca para os graficos
##########################################################################
##########################################################################
##      Algoritmo em Python para realizar o calculo da Regressão Linear
##      Metodo dos Minimos Quadrados (LMS) Simples, Quadrático e Robusto (com pesos)
##########################################################################
##########################################################################


def main():
    import xlrd
    ########### leitura do arquivo que contem a base de dados em excel
    ########### Arquivo no formato: Y, X, X^2, onde Y é variável dependente, X explicativa e X^2  as outras explicativas
    book = xlrd.open_workbook("DB_Rel1.xlsx") ##nome do arquivo excel
    print ("Número de abas na planilha do excel: ", book.nsheets)
    print ("Nomes das Planilhas de excel:", book.sheet_names())
    for vSheet in book.sheet_names():
        print(vSheet)
        sh = book.sheet_by_name(vSheet)
        labelChart=sh.name
        labelY = sh.cell_value(rowx=0, colx=0)
        labelX=sh.cell_value(rowx=0, colx=1)
        QuantElementos=sh.nrows
        qtX=3
        grp_YReal = []
        grp_XReal = []
        ### Matrizes para carregar as informaçoes das bases de dados do arquivo excel
        XLinear = np.empty(((QuantElementos - 1), 2))  ##variavel XLinear para matriz das variáveis explicativa X
        XQuad = np.empty(((QuantElementos - 1), qtX))  ##variavel XQuad para matriz das variáveis explicativa X para modelo de regressao quadratica
        Y = np.empty(((QuantElementos-1), 1))          ##variavel Y para matriz das variáveis dependente Y
        Xmin =sh.cell_value(rowx=1, colx=1)            ##variavel para ser usado como ponteiro para minimo a ser utilizado na formacao do gráfico
        Xmax =sh.cell_value(rowx=1, colx=1)            ##variavel para ser usado como ponteiro para máximo a ser utilizado na formacao do gráfico
        for i in range (1,QuantElementos,1):
            ## Carga Variável dependente Y 
            Y[(i-1)][0]=sh.cell_value(rowx=i, colx=0)
            ## Carga Variável explicativa X 
            XLinear[(i - 1)][0] = 1     # Add 1 - "BIAS" conforme metodo dos minimos quadrados
            XLinear[(i - 1)][1] = sh.cell_value(rowx=i, colx=1)
            ## Carga Variável explicativa X para modelo quadratico 
            XQuad[(i - 1)][0] = 1   # Add 1 - "BIAS" conforme metodo dos minimos quadrados
            XQuad[(i - 1)][1] = sh.cell_value(rowx=i, colx=1)
            XQuad[(i - 1)][2] = sh.cell_value(rowx=i, colx=2)
            # variavel de ponteiro para minimo a ser utilizado na formacao do gráfico
            if Xmin>sh.cell_value(rowx=i, colx=1):Xmin=sh.cell_value(rowx=i, colx=1)
            if Xmax<sh.cell_value(rowx=i, colx=1):Xmax=sh.cell_value(rowx=i, colx=1)
            ## Valores das amostras para apresentar no gráfico
            grp_YReal.append(sh.cell_value(rowx=i, colx=0))
            grp_XReal.append(sh.cell_value(rowx=i, colx=1))

        ## Define intervalo para plotar gráficos 
        step=(Xmax-Xmin)/QuantElementos
        if step <1 :
            step=1
        else:
            if step>5 and step< 10:
                step=10
            else:
                step=int(step)
        ########################################################################################            
        ########################################################################################            
        ########################################################################################            
        ##Calculo para o LMS  Linear            
        XLinearT=met_matriz_transposta(XLinear)           ## Calculo da Matriz X transposta
        XLinearTXLinear=met_matriz_mult(XLinearT,XLinear) ## Calculo da Matriz X transposta multiplicada pela Matriz X
        XLinearTY=met_matriz_mult(XLinearT,Y)             ## Calculo da Matriz X transposta multiplicada pela Matriz Y
        XLinearTI = met_MatrizInversa(XLinearTXLinear)    ## Calculo da Matrix inversa (XT*X)
        βXLinear=met_matriz_mult(XLinearTI,XLinearTY)     ## Calculo da Fatores da regressao LMS
        print("βXLinear:",βXLinear)
        ########################################################################################
        ########################################################################################            
        ########################################################################################            
        ## Calculo para o  LMS Robusta            
        W = np.empty((QuantElementos - 1, 1))                  ## cria matriz W de pesos para robusta
        WXLinear = np.empty((QuantElementos - 1, 2))           ## cria matriz WXLinear X ponderados
        WY = np.empty((QuantElementos - 1, 1))                 ## cria matriz WY para Y ponderados
        yRegresaaoXLinear = met_matriz_mult(XLinear, βXLinear) ## regressao linear de X utilizada para obter pesos de W
        ## Loop para obter os pesos de W pela interação com a regressao linear de X
        for i in range(0,len(Y),1):
            W[i]= abs(1/(Y[i]-yRegresaaoXLinear[i]))        ## Peso W = |1/(Y-y)|
            WXLinear[i][0]=W[i]
            WXLinear[i][1]=W[i]*XLinear[i][1]
            WY[i][0]=W[i]*Y[i][0]
        XLinearTWZ=met_matriz_mult(XLinearT,WXLinear)     ## Matriz XLinearT transposta multiplicada pela Matriz WXLinear
        XLinearTWY=met_matriz_mult(XLinearT,WY)           ## Matriz XLinearT transposta multiplicada pela Matriz WY
        XLinearTWI = met_MatrizInversa(XLinearTWZ)        ## Matriz inversa (XLinearTWZ*WX)
        βXLinearRob=met_matriz_mult(XLinearTWI,XLinearTWY)## Fatores da regressao por minimo
        print("βXLinearRobusta:",βXLinearRob)
        #########################################################################################
        #########################################################################################
        #########################################################################################
        ##Calculo para o LMS  Quadratica            
        XQuadT=met_matriz_transposta(XQuad)               ## Matriz XQuad transposta
        XQuadTXQuad=met_matriz_mult(XQuadT,XQuad)         ## Matriz XQuad transposta multiplicada pela Matriz XQuad
        XQuadTY=met_matriz_mult(XQuadT,Y)                 ## Matriz XQuad transposta multiplicada pela Matriz Y
        XQuadTI = met_MatrizInversa(XQuadTXQuad)          ## Matriz inversa (XQuadTI*XQuadTY)
        βXQuad=met_matriz_mult(XQuadTI,XQuadTY)           ## Fatores da regressao por minimo
        print("βXQuad:",βXQuad)
        #########################################################################################
        #########################################################################################
        #########################################################################################
        ##Variáveis para formação dos gráficos 
        grp_YXLinear= []
        grp_YXLinearRob = []
        grp_YXQuad = []
        grp_X = []
        for index in range(0, (int(1.2*QuantElementos)), 1):
            ## Prepara para o Gráfico Regressao Linear Simples 
            varXLinear = np.empty((1, 2),int)
            varXLinear[0][0] = 1
            varXLinear[0][1] = index*step+Xmin
            yRegresaaoXLinear = met_matriz_mult(varXLinear, βXLinear)
            grp_YXLinear.insert(index, yRegresaaoXLinear[0][0])
            ## Prepara para o Gráfico Regressao Linear Robusta 
            yRegresaaoXLinearRob = met_matriz_mult(varXLinear, βXLinearRob)
            grp_YXLinearRob.insert(index, yRegresaaoXLinearRob[0][0])
            ## Prepara para o Gráfico Regressao Quadratica 
            varXQuad = np.empty((1, qtX),int)
            varXQuad[0][0] = 1
            varXQuad[0][1] = index*step+Xmin
            varXQuad[0][2] = (index * step + Xmin)**2
            yRegresaaoXQuad = met_matriz_mult(varXQuad, βXQuad)
            grp_YXQuad.insert(index, yRegresaaoXQuad[0][0])
            ##########################################################################################
            ##########################################################################################
            ##########################################################################################
            grp_X.insert(index, varXQuad[0][1])
            if index > (QuantElementos-2):
                grp_YReal.append(np.nan)
                grp_XReal.append(np.nan)
        ## Data Frame conjugando todos os modelos para a formação de gráfico único 
        df = pd.DataFrame({'X': grp_X, 'Quadratica': grp_YXQuad, 'Linear': grp_YXLinear, 'LinearRobusta': grp_YXLinearRob,'XReal': grp_XReal, 'YReal': grp_YReal})
        ## Imprime tabela com todos os resultados 
        print (df)
        ## Fixa eixo de referencia unico para todos os graficos 
        ax = plt.gca()
        ## Faz a plotagem de todos os Gráficos
        df.plot(kind='scatter', x='XReal', y='YReal', color='red',ax=ax)
        df.plot(kind='line', x='X', y='Quadratica', color='green', ax=ax)
        df.plot(kind='line', x='X', y='Linear', color='blue', ax=ax)
        df.plot(kind='line', x='X', y='LinearRobusta', color='orange', ax=ax)
        plt.title(labelChart)
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        plt.show()

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
