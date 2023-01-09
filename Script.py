
# coding: utf-8

# In[85]:

import re,json,glob
import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.tokenize import word_tokenize
def lire_fichier(chemin):
    f = open(chemin , "r", encoding="utf-8")
    chaine = f.read()
    f.close()
    return chaine
def ecrire_json(chemin , contenu):
    w = open(chemin , "w", encoding="utf-8")
    w.write(json.dumps(contenu , indent=2, ensure_ascii=False))
    w.close()
def suppGuillemets(liste):
    L = []
    ponctuation = ["«", "»", "»","*"]
    for mot in liste:
        for i in ponctuation:
            mot = re.sub(i, "", mot)
        L.append(mot)
    return L
def enlever_nombres(texte):
    texte = re.sub("[0-9\.]*", "", texte)
    return texte

def get_vocabulaire(texte):
    vocabulaire = []
    texte = enlever_nombres(texte)
    mots = word_tokenize(texte, language = "english")
    for mot in mots:
        vocabulaire.append(mot)
    return vocabulaire


# In[88]:

vocabulaire = []
for path in glob.glob("Corporatxt/*"):
    print(path)
    dossier, fichier = re.split("/", path)
    html = lire_fichier(path)
    doc = nlp(html)
vocabulaire += get_vocabulaire(html)

    



    


# In[54]:

fic=lire_fichier("termsfiltre.txt")
doc=nlp(fic)
#for token in doc:
    #print(token,token.lemma_, token.tag_)


# In[55]:

listeterm=[ "Multilingual","Interlingual",
  "Semantic","Representations","Natural","Language",
  "Processing","Computational","Linguistics","articles","context","fast-changing","field","motivation","project",
  "work","semantic","representations","Motivation","observation","language","technologies","research",
  "community","text summarization","speech recognition","translation","globalized word","connected world",
  "speakers","lingua franca","multilinguality","community’s research","Anthology","Papers","term","multilingual",
  "crosslingual","hyphenated","specific languages","spoken languages","computational tools","conceptual definitions",
  "language","meaning""linguistic abstractions","annotated","data sets","application","representation",
  "learning","methods","text","corpora","contextual","word","vectors","linguistic resources","multilingual setting",
  "sentence","transfer","language","technologies","phenomena","abstractions","theories""annotation","schemes","databases",
  "research","interlingual","machine translation","sentence","interlingual","representations","evaluation","Multilinguality",
  "systems","low-resource","crosslingual","challenges", "lexical representation","unsupervised representation",
  "linguistic knowledge","representations","Encoder","Crosslingual","word embeddings","isomorphism","analysis",
  "results","components","authors","combination","The","approach","proposition","senses","vectors","linguistic features",
  "classification","morphological","tag","count","pseudoword","identification","results","downstream","neural",
  "encoder-decoder","architecture","frameworks","Multilingual","Neural","Machine","Translation","downstream","tasks",
  "sentence","representations","classification",
  "tasks","accuracy","similarity","information","inflectional morphology","derivational morphology","data","limitations","domains","semi-automatic","Translation","Autoencoder","Crosslingual word","monolingual embedding","transfer","low-resource","performance","adversarial approaches","non-adversarial approaches","supervised system","knowledge","named entity recognition","part-of-speech","corpora","demands","methods","isomorphic assumption","crosslingual","syntax"]


# In[56]:

doc=nlp(str(listeterm))
for token in doc:
    print(token,token.tag_)


# In[58]:

l = set(listeterm)
print(l)


# In[59]:

#ficj=lire_fichier("termsFiltre.json")
for path in glob.glob("Corporatxt/*"):
    dossier, fichier = re.split("/", path)
    html = lire_fichier(path)
    #print(html)
    for i in l:
        for el in re.finditer(r"(%s)(\W)" % i, html):
            #print(el.group(1))
            html = re.sub(r"%s\W" % el.group(1), "%s/B%s" % (el.group(1), el.group(2)), html)
            break
    print(html)
    1/0


# In[60]:

grandeliste = get_vocabulaire(html)
print(grandeliste[:10])


# In[61]:

listeTermeRemplace = []
for i in listeterm:
    element = i.split(" ")
    if len(element) > 1:
        new = element[0] + "/B"
        for j in range(1, len(element)):
            new += " " + element[j] + "/I"
        listeTermeRemplace.append(new)
    else:
        listeTermeRemplace.append(element[0]+"/B")
        
for el in listeTermeRemplace:
    print(el)


# In[62]:

for path in glob.glob("Corporatxt/*"):
    dossier, fichier = re.split("/", path)
    html = lire_fichier(path)
    voca_annote = []
    grandeliste = get_vocabulaire(html)
    for i in grandeliste:
        if i+"/B" in set(listeTermeRemplace):
            #print(len(i.split(" ")))
            voca_annote.append(i+"/B")
        else:
            voca_annote.append(i+"/O")
    print(voca_annote)
    break


# In[42]:

print(voca_annote[:500])


# In[63]:

StrA = "".join(voca_annote)


# In[36]:

fichier = open("CorpusAnnote.txt", "a")
fichier.write(StrA)
fichier.close()


# In[64]:

pos =nltk.pos_tag(grandeliste)
print(pos) #pos in the corpus


# In[83]:

for path in glob.glob("Corporatxt/*"):
    dossier, fichier = re.split("/", path)
    html = lire_fichier(path)
    listecount=[]
    for cpt, i in enumerate(path):
        for m in pos:
            listecount.append(m[1])
print("adj" ,listecount.count('JJ')) #statical about pos
print("nom ", listecount.count('NN'))
print("verbe", listecount.count('VB'))


# In[7]:


import sys
get_ipython().system('{sys.executable} -m pip install flair')



# In[8]:

get_ipython().system('pip install --upgrade pip')


# In[9]:


import sys
get_ipython().system('{sys.executable} -m pip install flair')


# In[10]:

from flair.data import Corpus


# In[45]:


from flair.data import Corpus
from flair.datasets import ColumnCorpus
columns = {0: 'text', 1: 'iob'}
data_folder = './Corpus/'
corpus: Corpus = ColumnCorpus(data_folder, columns,
train_file='train.txt',
test_file='test.txt',
dev_file='dev.txt')



# In[46]:

len(corpus.train)


# In[49]:

from flair.data import Corpus

from flair.datasets import UD_ENGLISH

from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings


# In[50]:

from flair.trainers import ModelTrainer


# In[84]:

from flair.data import Corpus
from flair.datasets import UD_ENGLISH
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings

corpus: Corpus = UD_ENGLISH().downsample(0.1)
print(corpus)

tag_type = 'B'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

embedding_types = [

    WordEmbeddings('glove'),


embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)


# In[ ]:

from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-pos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)

model = SequenceTagger.load('resources/taggers/example-pos/final-model.pt')

sentence = Sentence('I love Berlin')

model.predict(sentence)

print(sentence.to_tagged_string())

