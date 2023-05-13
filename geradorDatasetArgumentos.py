import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

modelo = ""
model_checkpoint = os.path.join(os.getcwd(), modelo)
#tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, add_prefix_space=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

def extrai_argumentos(texto):
    #print(texto)
    #print(len(texto.split()))
    tokenized = tokenizer(texto, return_tensors='pt', truncation=True)
    #print(tokenized['input_ids'])
    predictions = model(tokenized['input_ids'])
    predictions = np.argmax(predictions[0].detach().numpy(), axis=-1)
    #print(predictions)
    inicio = fim = num = former_label = 0
    componente = []
    for i, label in enumerate(predictions[0]):
        #print(i, label)
        if label == former_label:
            continue
        elif label == 1 and former_label == 0:
            inicio = i
        elif label == 1 and former_label == 2:
            #print("aqui 2")
            fim = i
            #print("Vou fazer append da região: ", inicio, fim)
            componente.append(tokenizer.decode(tokenized['input_ids'][0][inicio:fim]))
            inicio = i
        elif label == 0 and former_label in (1,2):
            #print("aqui")
            fim = i
            #component = [i, start, end]
            #print("Vou fazer append da região: ", inicio, fim)
            componente.append(tokenizer.decode(tokenized['input_ids'][0][inicio:fim]))
            start = -1
            end = -1
        elif label == 2 and former_label == 0:
            inicio = i
        former_label = label
    return componente

from build_dataset import Corpus
import numpy as np


class Dataset:
    def __init__(self, competence):
        self.c = Corpus()
        #self.c.read_corpus().shape
        self.train, self.valid, self.test = self.c.read_splits()
        #print(self.train.shape)
        #self.train.loc[1:5, ['essay', 'score', 'competence']] 
        #print(self.valid.shape)
        #self.valid.loc[1:5, ['essay', 'score', 'competence']]
        #print(self.test.shape)
        #self.test.loc[1:5, ['essay', 'score', 'competence']]
        self.competence = 'c'+str(competence)
    def UnirListas(self, lista):
        if len(lista) < 1:
            return ""
        unificado = ""
        unificado = unificado+lista[0]+'\n'
        for t in range(1,len(lista)):
            unificado += lista[t]+'\n'
        return unificado
        
    def gerarTreinamento(self):
        """
        retorna um conjunto (texto, nota_competencia1). Um texto eh composto de varios paragrafos, cada paragrafo eh uma lista 
        """
        textos = []
        notas = []
        print(self.competence)
        for index, row in self.train.iterrows():
            texto = self.UnirListas(row['essay'])
            #notas.append( float(row['competence'][self.competence] / 40))
            notas.append( float(row[self.competence] / 200))
            textos.append(texto)
        return textos, notas
    def gerarTeste(self):
        textos = []
        notas = []
        for index, row in self.test.iterrows():
            texto = self.UnirListas(row['essay'])
            notas.append( float(row[self.competence] / 200))
            textos.append(texto)
        return textos, notas
    def gerarValidacao(self):
        textos = []
        notas = []
        for index, row in self.valid.iterrows():
            texto = self.UnirListas(row['essay'])
            notas.append( float(row[self.competence] / 200))
            textos.append(texto)
        return textos, notas

def TransformarTextoEmInput(textos):
    tokenizados = []
    for indice in range(len(textos)):
            tokens = tokenizer.encode_plus(textos[indice], add_special_tokens=True, truncation=True, return_tensors='pt')
            tokens = tokens['input_ids']
            #tokens = tokens.type(torch.int64).to(device)
            tokenizados.append(tokens)
    return tokenizados

def TransformarNotasEmVetor(textos, notas):
    novas_notas = []
    for indice in range(len(textos)):
            novas_notas.append(torch.tensor(notas[indice]).unsqueeze(0))
    return novas_notas

def cria_lista_argumentos(textos):
    lista_argumentos = []
    for t in textos:
        lista_argumentos.append(extrai_argumentos(t))
    return lista_argumentos    

def salva_argumentos(lista_argumentos, lista_notas, pasta):
    num = 0
    for argumentos,nota in zip(lista_argumentos, lista_notas):
        nome = os.path.join(os.path.join(os.getcwd(), "Argumentos-e-notas"), pasta)
        file_arg = open(nome+str(num)+"A.txt", "w",encoding='utf-8')
        file_nota = open(nome+str(num)+"N.txt", "w")
        file_nota.write(str(nota))
        for arg in argumentos:
            #print(arg)
            file_arg.write(arg+"\n")
        num += 1
        file_arg.close()

def main():
    ds = Dataset(3)
    texto_treinamento, nota_treinamento = ds.gerarTreinamento()
    texto_teste, nota_teste = ds.gerarTeste()
    texto_valid, nota_valid = ds.gerarValidacao()
    argumentos_treinamento = cria_lista_argumentos(texto_treinamento)
    argumentos_validacao = cria_lista_argumentos(texto_valid)
    salva_argumentos(argumentos_treinamento, nota_treinamento, "Treinamento")
    salva_argumentos(argumentos_validacao, nota_valid, "Validacao")

if __name__ == "__name__":
    main()