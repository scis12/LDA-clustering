
import stanza

import re

from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

from collections import Counter


def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ä", "a"),
        ("ë", "e"),
        ("ï", "i"),
        ("ö", "o"),
        ("ü", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s




def lematizar(raw_text, nlp):
    palabras = nlp(raw_text)
    tokens = []
    for sentence in palabras.sentences:
        for word in sentence.words:
            tokens.append(word.lemma)
    return tokens


#Funcion para remover stop words de una lista de tokens
def rem_stop_words(tokens, language):
    stop_words = set(stopwords.words(language))
   
    

    filtered_sentence = [w for w in tokens if not w in stop_words]  
    return filtered_sentence



def tokenize(raw_text, nlp, language, lemmatize, rem_stop):
    """ Tokenizes and lemmatizes a raw text """
    #Tokenizador: hace algunas modificaciones (replace) que hace un mini preprocesamiento para q funciona bien
    
    raw_text = raw_text.replace("\'", "'")
    raw_text = raw_text.replace("/", " / ")
    raw_text = raw_text.replace("<br /><br />", "\n")
    raw_text = normalize(raw_text) #Quito tildes
    
    #Lematizo, devuelve el texto tokenizado si es True         
    
    if lemmatize:
        tokens = lematizar(raw_text, nlp)
    
    #SI se apaga el lematizador, tokenizo manualmente
    else:
        sentences = sent_tokenize(raw_text) #Tokenizador
        tokens = [e2 for e1 in sentences for e2 in word_tokenize(e1)]  # Nested list comprehension. Para cada palabra, mete el tokenizador
        
    tokens = [e for e in tokens if re.compile("[A-Za-z]").search(e[0])] #Se queda con todas las palabras q tienen caracteres alfanumericos (vuela comas)
    
    tokens = [e.lower() for e in tokens]

    if rem_stop:
        #Quita stopwords
        tokens = rem_stop_words(tokens, language)
    
    
    return(tokens)


def procesar_articulos(articulos, nlp, language='english', lemmatize = True, rem_stop = True ):
    articulos_procesados = list()
    for idx in range(len(articulos)):
        if idx%100==0:
            print(idx)
        art = articulos[idx]
        articulo_tokenizado = tokenize(art, nlp, language, lemmatize = lemmatize, rem_stop = rem_stop )
        articulos_procesados.append(" ".join(articulo_tokenizado))
    return articulos_procesados

