"""
Created on Tue Nov 26 19:15:50 2019

@author: SilvaDE

perceptron MultiLayer
"""
# Importação das bibliotecas que serão usadas para o programa
import numpy as np
import math
import copy


def main ():
    import xlrd
    metTreinamento = xlrd.open_workbook("dataset_traning.xlsx")
    valida = xlrd.open_workbook("datasetValid.xlsx")

    print("Quantidade de abas: ", metTreinamento.nsheets)
    print("Nome da planilha de excel:", metTreinamento.sheet_names())
    for vSheet in metTreinamento.sheet_names():
        print(vSheet)
        saidas=3
        maxNeuronios=10
        maxCamada2=10
        for Camada2 in range(1, maxCamada2,1):
            for neuronios in range(1,maxNeuronios,1):
                camadas=[neuronios, Camada2, saidas]

                sh = metTreinamento.sheet_by_name(vSheet)
                vl = valida.sheet_by_name(vSheet)
                tbCol = sh.ncols
                Label = []
                np.set_printoptions(precision=4)
                for i in range(1, (tbCol), 1):
                    Label.append(sh.cell_value(rowx=0, colx=i))
                qtElementos = sh.nrows
                qtVal = vl.nrows

                ## Matrizes para carregar as informaçoes das bases de dados 
                X = np.empty(((qtElementos - 1), (tbCol - saidas)))
                XVal = np.empty(((qtVal - 1), (tbCol - saidas)))
                Y = np.empty(((qtElementos - 1), saidas))
                YVal = np.empty(((qtVal - 1), saidas))

                for i in range(1, qtElementos, 1):
                ## Carrega Variável Y e X  - Base de Treinamento 
                    for j in range(0, tbCol, 1):
                        if j > (saidas-1):
                            X[(i - 1)][(j - saidas)] = sh.cell_value(rowx=i, colx=j)
                        else:
                            Y[(i - 1)][j] = int(sh.cell_value(rowx=i, colx=j))

                for i in range(1, qtVal, 1):
                ## Carrega Variável Y e X  Base de Validacao 
                    for j in range(0, tbCol, 1):
                        if j > (saidas-1):
                            XVal[(i - 1)][(j - saidas)] = vl.cell_value(rowx=i, colx=j)
                        else:
                            YVal[(i - 1)][j] = int(vl.cell_value(rowx=i, colx=j))

                print("qtdeDimensoes=", np.size(X,1))
                ## inicializa da RNA MLP 
                rna= metMLP(layer=camadas, indSaturacao=0.5, coefAprendizado=0.01, qtdeDimensoes=np.size(X,1), numEpisodios=20000, valorErro=0.01)
                ## metTreinamento com a base de treinamento X e Y 
                rna.metTreinamento(X, Y)
                ## realiza a validacao com a pridicao dos valores da base de aprendizado 
                tbConf = np.zeros((2,saidas))
                tbErros = np.zeros((2, saidas))
                ## Valida com a base de validacao XVal e YVal 
                print("====================Predicao====================")
                for i in range(0,np.size(XVal,0),1):
                    YPred=rna.metPrediz(XVal[i])

                    for j in range(0,saidas,1):
                        if YVal[i][j]==1:
                            if YPred[j]==1:
                                tbConf[0][j]+=1
                            else:
                                tbConf[1][j] += 1
                        if YVal[i][j]==0:
                            if YPred[j]==1:
                                tbErros[0][j]+=1
                            else:
                                tbErros[1][j] += 1
                print("================================================")
                print("Tabela Confusao - acertos", neuronios, Camada2)
                print("================================================")
                print(tbConf)
                print("================================================")
                print("================================================")
                print("Tabela Confusao - Erros", neuronios, Camada2)
                print("================================================")
                print(tbErros)
                print("================================================")


main()

## Objeto referente ao Perceptron rede neural artificial (RNA)
## inicializa com o indice de saturacao α_sat =, 
## coeficiente de aprendizado η_aprend, 
## numero de dimensoes e episodios/epocas
class MetPerceptron(object):
    ## metodo de construçao recebe indice de saturacao α_sat=, 
	##coeficiente de aprendizado(η_aprend), 
	##qtde de dimensoes e episodios/epocas
    def __init__(self,  indSaturacao=0.5, coefAprendizado=0.01, qtdeDimensoes=2, numEpisodios=2000, ε=0.01):
        self.indSaturacao=indSaturacao
        self.coefAprendizado = coefAprendizado
        self.numEpisodios = numEpisodios
        self.entrada = 0
        self.alfaErro = 0
        self.valorNet = 0
        self.varDFnet = 0
        self.varFNet = 0
        self.valorErro=valorErro

    ## Trecho para realização do calculo do primeiro vetor w com numeros radomicos 
        self.w = np.random.uniform(-0.1, 0.1, qtdeDimensoes + 1)
    ## Metodo referente a função de saturação 
    def metSaturacao(self, x):
        if x >= self.indSaturacao:  ##OBS: recebe o resultante da funcao f(x), se valor for maior que o indice de saturacao retorna 1
            return 1
        else:
            return 0
    ## Metodo referente a funcao de ativacao do neuronio
    def metFunAtivacao(self, x):
        return 1/(1+math.exp(-x))

    ##  Metodo referente a funcao de derivada da ativacao do neuronio 
    def metFunDerivadaAtNeuronio(self, x):
        return self.metFunAtivacao(x)*(1-self.metFunAtivacao(x))

    ##  Metodo referente ao  retornar e impressao dos valores do vetor com os pesos sinápticos 
    def metRetornaValVetores(self):
        return self.w	
    ## Metodo referente ao retorno e impressao dos valores do erro com os pesos sinápticos 
    def metRetornaValErro(self):
        return self.alfaErro
    ## Metodo referente ao retorno e impressao dos valores do fnet do neuronio 
    def metRetornaValFNet(self):
        return self.varFNet
    ## Metodo referente ao retorno e impressao dos valores do net do neuronio 
    def metRetornaValNet(self):
        return self.valorNet
    ## método para treinamento do perceptron, recete a amostra X e as classificacoes reais da amostra Y 
    def metTreinamento(self, X, Y):
		##  amostras + BIAS 
        X = np.insert(X[:, ], len(X[0]), 1, axis=1)
		## for para loop de treinamento para o número de epsodios/epocas ou até confirmacao de aprendizado conforme desejado
        for i in range(self.numEpisodios):
			## checagem se existe o erro ou não (aprendizado concluído para saida antecipada ao termino dos episodios 
            erro_referencia = 0
            for j in range(len(X)):
                ## calculo da saida estimada com o vetor de pesos sinapticos atual 
                self.valorNet=self.w.dot(X[j])
                self.varFNet = self.metFunAtivacao(self.valorNet)
                self.varDFnet= self.metFunAtivacao(self.valorNet)*(1-self.fAtiva(self.valorNet))
                ## checagem de resultado: se é o esperado ou nao para calibrar os pesos sinapticos 
                if ((Y[j] - self.varFNet) **2)/2 > self.valorErro :
                    erro_referencia += 1
                    alfaErro = Y[j] - self.varFNet
                    self.w += self.coefAprendizado * alfaErro * X[j] * (1- self.varFNet)* X[j]
            if erro_referencia == 0:
                print("Iterações", i)
                break
    ## metodo para predizer os valores com base no aprendizado do treinamento 
    def metPrediz(self, ponto):
        if np.ndim(ponto) == 1:
            ponto = np.insert(ponto, len(ponto), 1)
    ## realiza as multiplicacoes do vetor de pesos e o ponto para predicao 
            self.varFNet = 1 / (1 + math.exp(-self.w.dot(ponto)))
            prediction=self.metSaturacao(self.varFNet)
            return prediction
        else:
            ponto = np.insert(ponto[:, ], len(ponto[0]), 1, axis=1)
    ## realiza as multiplicacoes do vetor de pesos e o ponto para predicao 
            prediction = [self.metSaturacao(self.varFNet(x)) for x in ponto]
            return predicao



## inicializa com o numero de camadas e neuronios, o indice de saturacao α_sat=, 
## coeficiente de aprendizado η_aprend, numero de dimensoes e episodios/epocas 
class metMLP(object):
    ## metodo de construçao recebe indice de saturacao α_sat=,
	## coeficiente de aprendizado η_aprend, 
	## numero de dimensoes e episodios/epocas
    def __init__(self, layer=[10,2], indSaturacao=0.5, coefAprendizado=0.01, qtdeDimensoes=2, numEpisodios=2000, valorErro=0.01):
        self.indSaturacao=indSaturacao
        self.coefAprendizado = coefAprendizado
        self.numEpisodios = numEpisodios
        self.qtdeDimensoes=qtdeDimensoes
        self.valorErro=valorErro
        self.alfaErro = []
        self.qtCamadas = len(layer)
        self.valorNet = []
        self.varFNet = []
        self.varDFnet = []
        self.layer=layer
        self.layers = []

        for i in range (0, self.qtCamadas,1):
            self.layers.append([])
            self.valorNet.append([])
            self.varFNet.append([])
            self.alfaErro.append([])
            if i==0:
                dim=self.qtdeDimensoes
            else:
                dim=self.layer[(i-1)]
            for j in range (0,layer[i],1):
                self.alfaErro[i].append([])
                self.layers[i].append(MetPerceptron(indSaturacao=self.indSaturacao, coefAprendizado=self.coefAprendizado, qtdeDimensoes=dim, numEpisodios=self.numEpisodios, valorErro=self.valorErro))
        print("RNA - Quantidade de Camadas:", self.qtCamadas)
    def metSaturacao(self, x):
        if x >= self.indSaturacao:
            return 1
        else:
            return 0
    ## método par retornar e imprimir os valores do erro com os pesos sinápticos 
    def metRetornaValErro(self):
        for i in range(0, len(self.layer), 1):
            for j in range(0, layer[i], 1):
                self.erro+=(self.layers[i][j].metRetornaValErro())**2
        self.erro=self.erro/2
        return self.erro
    ## método para treinamento do perceptron, recete a amostra X e as classificacoes reais da amostra Y 
    def metTreinamento(self, X, Y):
		## amostras + BIAS 
        X = np.insert(X[:, ], len(X[0]), 1, axis=1)
		##loop para treinamento para o número de epsodios/epocas ou até confirmacao de aprendizado 
        for i in range(self.numEpisodios):
            varErroCiclo = 0
			## referencia de existencia de erro (aprendizado concluído para saida antecipada ao termino dos episodios 
            for j in range(len(X)):
				## Feed Forward das entradas 
                for k in range(0, self.qtCamadas, 1):  # camada k
                    for n in range(0, self.layer[k], 1):  # neuronio n
                        if k == 0:
                            entrada = copy.deepcopy(X[j])
                        else:
                            entrada=[]
                            for imp in range(0, self.layer[(k - 1)],1):
                                entrada.append(self.layers[(k - 1)][imp].varFNet)
                            entrada.append(1)
                        ## calculo da saida estimada com o vetor de pesos sinapticos atual 
                        self.layers[k][n].entrada=copy.deepcopy(entrada)
                        self.layers[k][n].valorNet=self.layers[k][n].w.dot(self.layers[k][n].entrada)
                        ## calculo da saida estimada com o vetor de pesos sinapticos atual 
                        self.layers[k][n].varFNet = self.layers[k][n].fAtiva(self.layers[k][n].valorNet)
                        self.layers[k][n].varDFnet = self.layers[k][n].fAtiva(self.layers[k][n].valorNet) * (1-self.layers[k][n].fAtiva(self.layers[k][n].valorNet))

				## Feed Back dos erros das entradas 
                for k in range(1, (self.qtCamadas +1), 1):  # camada k
                    camadaNeural=self.qtCamadas-k
                    if k == 1:
                        saida=Y[j]
                        for n in range(0, self.layer[camadaNeural], 1):
                            qtSaidas=len(saida)
                            if qtSaidas > 1:
                                if ((saida[n] - self.metSaturacao(self.layers[camadaNeural][n].varFNet))**2)/2 > self.valorErro:
                                #if ((saida[qtS] - self.layers[camadaNeural][n].fnet) ** 2) / 2 > self.ε:
                                    varErroCiclo += 1
                                    self.alfaErro[camadaNeural][n] = (saida[n] - self.layers[camadaNeural][n].varFNet) * (self.layers[camadaNeural][n].varDFnet)
                                else:
                                    self.alfaErro[camadaNeural][n] =0
                            else:
                                if ((saida - self.metSaturacao(self.layers[camadaNeural][n].varFNet))**2)/2 > self.valorErro:
                                #if ((saida - self.layers[camadaNeural][n].fnet) ** 2) / 2 > self.ε:
                                    varErroCiclo += 1
                                    self.alfaErro[camadaNeural][n] = (saida - self.layers[camadaNeural][n].varFNet) * (self.layers[camadaNeural][n].varDFnet)
                                else:
                                    self.alfaErro[camadaNeural][n]=0
                    else:
                        for n in range(0, self.layer[camadaNeural], 1):  # neuronio n
                            self.alfaErro[camadaNeural][n]=0
                            for ne in range (0, self.layer[(camadaNeural+1)],1):
                                for w in range (0, len(self.layers[(camadaNeural+1)][ne].w),1):
                                    self.alfaErro[camadaNeural][n] += self.alfaErro[(camadaNeural+1)][ne]*self.layers[(camadaNeural+1)][ne].w[w]
                            self.alfaErro[camadaNeural][n] =self.alfaErro[camadaNeural][n] * self.layers[camadaNeural][n].varDFnet

                ## retoanr e calcula valor de W 
                for k in range(1, (self.qtCamadas+1), 1):  # camada k
                    camadaNeural=self.qtCamadas - k
                    for n in range(0, self.layer[camadaNeural], 1):
                        if k == self.qtCamadas:
                            entrada = copy.deepcopy(X[j])
                            qtImput = len(entrada)
                        else:
                            entrada = copy.deepcopy(self.layers[camadaNeural][n].entrada)
                            qtImput = len(entrada)
                        if qtImput > 1:
                            for qtS in range(0,qtImput,1):
                                self.layers[camadaNeural][n].w[qtS] += self.layers[camadaNeural][n].coefAprendizado*self.alfaErro[camadaNeural][n]*entrada[qtS]
                        else:
                            self.layers[camadaNeural][n].w += self.layers[camadaNeural][n].coefAprendizado*self.alfaErro[camadaNeural][n] * entrada
            if varErroCiclo==0:
                print("\nIterações RNA", i)
                break

    ## metodo para predizer os valores com base no aprendizado do treinamento 
    def metPrediz(self, ponto):
        if np.ndim(ponto) == 1:
            prediction=[]
			## realiza as multiplicacoes do vetor de pesos e o ponto para predicao 
            for k in range (0, self.qtCamadas, 1): 	# camada k
                for n in range (0, self.layer[k],1): #neuronio n
                    if k == 0:
                        entrada = np.insert(ponto, len(ponto), 1)
                    else:
                        entrada=[]
                        for imp in range (0, self.layer[(k-1)],1):
                            entrada.append(self.layers[(k - 1)][imp].varFNet)
                        entrada.append(1)
                    ## calculo da saida estimada com o vetor de pesos sinapticos atual 
                    self.layers[k][n].entrada =  copy.deepcopy(entrada)
                    self.layers[k][n].valorNet = self.layers[k][n].w.dot(self.layers[k][n].entrada)
                    ## calculo da saida estimada com o vetor de pesos sinapticos atual 
                    self.layers[k][n].varFNet = self.layers[k][n].fAtiva(self.layers[k][n].valorNet)
                    self.layers[k][n].varDFnet = self.layers[k][n].fAtiva(self.layers[k][n].valorNet) * (1 - self.layers[k][n].fAtiva(self.layers[k][n].valorNet))
                    if k==(self.qtCamadas-1):
                        prediction.append(self.metSaturacao(self.layers[k][n].varFNet))
            return prediction
        else:
            ponto = np.insert(ponto[:, ], len(ponto[0]), 1, axis=1)
			## realiza as multiplicacoes do vetor de pesos e o ponto para predicao 
            prediction=[self.metSaturacao(1 / (1 + math.exp(-self.w.dot(x)))) for x in ponto]
            return  prediction