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
    y = np.copy(x)

    taxaAprencizado = float(input('Valor taxa de aprendizado: '))
    qtdEpocas = int(input('Quantidade de épocas: '))

    # rn = RedeNeural(0.2, 10001)
    rn = RedeNeural(taxaAprencizado, qtdEpocas)
    rn.treinar(rn, x)

    # oculta, saida = rn.feedForward(x[1])
    # novoPesoEO, novoPesoOS = rn.backpropagation([0, 0, 0], oculta, saida)

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
    def __init__(self, taxaAprend, epocas): #{
        self.camadas = [2, 2, 1]
        self.taxaAprend = taxaAprend
        self.epoca = epocas
        self.pesos = []
        self.biases = []
        self.oculta = []
        self.pesos.append(np.random.randn(2, 2))
        self.pesos.append(np.random.randn(2, 1))
        self.biases.append(np.random.randn(2, 1))
        self.biases.append(np.random.randn(1, 1))

        # A ideia no construtor é já criar uma lista com os pesos, gerados especificamente praesse problema
        # E também gerando aleatóriamente uma lista de bias para essa rede
        # Também vai se receber o valor para a taxa de aprendizado e também um
        # valor para a quantidade de épocas que se deseja rodar
    #}

    def feedForward(self, entrada): #{
        # Fazendo uma cópia pra não mexer nas entradas originais e para
        # ficar no formato de array
        a = np.copy(entrada[0:2])

        # Agora, calcula primeiro o valor da oculta, ou seja,
        # a entrada1 vezes o peso + entrada2 vezes o peso
        # e somando isso ao bias gerado aleatoriamente
        # e então calculando função de ativação dele
        ocultaSemSig = a.dot(self.pesos[0]) + self.biases[0]
        ocultaComSig = self.sigmoide(ocultaSemSig[-1])

        # Aqui é feita exatamente a mesma coisa entre oculta e saída
        saidaSemSig = ocultaComSig.dot(self.pesos[1] + self.biases[1])
        saidaComSig = self.sigmoide((saidaSemSig))

        # Retorno então os valores encontrados para a oculta e para a saída
        return ocultaComSig, saidaComSig
    #}

    def backpropagation(self, entrada, oculta, saida): #{
        # Primeiro, estou copiando a entrada para não alterá-la e ter ela como array
        entradaArray = np.copy(entrada)

        # Agora, calculo aqui o erro referente ä saída, ou seja, o valor
        # da saída em relação ao esperado, e depois aplicando a derivada da sigmoide
        # nesse resultado
        erroSaida = entrada[2] - saida
        deltaErroSaida = erroSaida * self.derivadaSig(saida)

        # Agora, nós temos o tanto que cada neurônio influenciou
        # Então se pode aplicar a mesma ideia para a oculta, ou seja
        # efetuar a propagação do erro para trás até a entrada
        erroOculta = deltaErroSaida.dot(self.pesos[1].T)
        deltaErroOculta = erroOculta * self.derivadaSig(oculta)

        # E com isso se pode calcular os novos erros
        # utilizando-se a entrada original com a derivada do erro da oculta
        # e a saída obtida com o valor da derivada do erro da saída
        novoPesoEO = entradaArray[0:2].dot(deltaErroOculta)
        novoPesoOS = np.array(saida.dot(deltaErroSaida))

        # E então retorno esses valores para serem somados nos pesos no fim do processo
        # na função onde o backpropagation foi chamado
        return novoPesoEO, novoPesoOS
    #}

    def treinar(self, redeNeural, entrada): #{
        for epoca in range(self.epoca):
            countIter = 0
            i = 0
            perda = 1
            while(i<len(entrada) and perda > 0.22):
                oculta, saida = redeNeural.feedForward(entrada[i])
                novoPesoEO, novoPesoOS = redeNeural.backpropagation([0, 0, 0], oculta, saida)

                # O erro/perda é calculada pelo MSE (Mean Sum Squared Loss, Média da soma quadrática da perda)
                # sendo que este é dado pela função de média do numpy
                # é a média da raiz das entradas subtraindo-se a saída
                perda = float(np.mean(np.square(entrada[0:2] - saida)))

                print(f'Época: {epoca}')
                print(f'Entrada: {entrada[i][0:2]}')
                print(f'Saída esperada: {entrada[i][2]}')
                print(f'Saída obtida: {saida}')
                # print(f'Erro quadrático: {erro}')
                print(f'Erro: {perda}')
                print(f'PesosEO originais: {self.pesos[0]}')
                print(f'PesosOS originais: {self.pesos[1]}')
                self.pesos[0] += novoPesoEO
                self.pesos[1] += novoPesoOS
                print(f'PesosEO novos: {self.pesos[0]}')
                print(f'PesosOS novos: {self.pesos[1]}')
                print('--------------------------------------------------')

                i += 1
                countIter += 1
            if(perda < 0.22):
                print('Fora do loop.')
                print(f'count iterações = {countIter}')
                print(f'epoca: {epoca}')
                print(f'perda: {perda}')
                break
    #}

    def erroQuadratico(self, esperado, obtido):
        return (0.5*(esperado - obtido)*(esperado - obtido))/2

    def sigmoide(self, valor): #{
        return 1 / (1 + 1/np.exp(valor))
    #}

    def derivadaSig(self, valor): #{
        return valor * (1 - valor)
    #}
#}
if (__name__ == "__main__"):
    main()