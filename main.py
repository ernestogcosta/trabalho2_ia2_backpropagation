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