import numpy as np

# x são as entradas para serem passadas para teste
# índice 0 = entrada1
# índice 1 = entrada2
# índice 2 = saída esperada
# logo, para trabalhar, serão utilizados apenas i 0 e 1, e aí
# se compara com o i = 2 para saber se foi o esperado

def main(): #{
    x = [[0, 0, 0],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 0]]
    rn = RedeNeural()

    oculta, saida = rn.feedForward(x[1])
    print(oculta)
    print(saida)
#}

# A classe da minha rede
# Aqui vai estar todas os métodos necessários para
# se gerar uma rede e fazer seu treinamento e teste

# A ideia básica inicial é fazer uma rede neural para
# aprender a resolver um XOR.
# Caso isso funcione, pensarei em outro problema, caso
# ainda tenha tempo para continuar mexendo

class RedeNeural(object):#{
    # Construtor, inicio ele recebendo as camadas
    # da rede específica. Então cria-se uma lista para os pesos
    # que serão preenchidos aleatóriamente, e a mesma coisa para obias

    # Então, é feito um for que irá rodar as camadas.
    # Nesse problema, como temos a camadade entrada, oculta e saída, ele vai rodar de 0 a 2
    # Aqui a ideia é: primeiro ele vai criar os pesos entre a entrada e a oculta
    # depois ele vai criar os pesos entre a oculta e a saída
    # É levado em consideração o valor presente em cada camada, ou seja, no índice 0 se tem
    # o valor de 2, ou seja, 2 entradas, e 2 no índice 1, ou seja, 2 neurönios da oculta, e
    # 1 no índice 2, ou seja, apenas uma saída.
    # Utiliza-se aqui a função randn do numpy passando 2 parämetros para cada iteração do for
    # sendo que o que é passado é a camadas[i+1], que no i=0 é a camada oculta, e o segundo
    # parämetro é camadas[i], que é a entrada no i=0. O que a função irá fazer é, ela vai gerar uma
    # matriz 2x2, porém, se tivéssemos 4 neuônios na camada, então o parâmetro passado será (4,2)
    # ou seja, uma matriz 4x2, pois estes serão os pesos, e como visto na teoria, os pesos
    # tem os índices ao contrário, ou seja, a entrada1 pra oculta2 será w21, e não w12, logo
    # a matriz ficará ao contrário do comumente esperado, por isso ela está sendo gerada assim
    # para o bias a ideia é a mesma, porém, em vez de camadas[i] como segundo parämetro, é
    # usado o calor 1. Nesse problema do XOR isso não muda o resultado, mas se fosse um problema
    # com n saídas, aí teria uma diferença, pois usando 1, ele será então uma matriz de uma
    # única coluna, com 1 linha para cada saída.

    def __init__(self): #{
        self.camadas = [2, 2, 1]
        self.pesos = []
        self.biases = []
        self.oculta = []
        self.pesos.append(np.random.randn(2, 2))
        self.pesos.append(np.random.randn(2, 1))
        self.biases.append(np.random.randn(2, 1))
        self.biases.append(np.random.randn(1, 1))

    #}

    def feedForward(self, entrada): #{
        a = np.copy(entrada[0:2])
        ocultaSemSig = a.dot(self.pesos[0]) + self.biases[0]
        ocultaComSig = self.sigmoide(ocultaSemSig[-1])
        saidaSemSig = ocultaComSig.dot(self.pesos[1] + self.biases[1])
        saidaComSig = self.sigmoide((saidaSemSig))
        return ocultaComSig, saidaComSig
    #}

    def sigmoide(self, valor): #{
        y = 1 / (1 + 1/np.exp(valor))
        return y
    #}
#}
if (__name__ == "__main__"):
    main()