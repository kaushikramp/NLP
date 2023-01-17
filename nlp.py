#Kaushik Ram P_20BCE1652
#NLP-DA1
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')


#1. Utilize Python NLTK (Natural Language Tool Kit) Platform and do the following. Install relevant Packages and Libraries
    #• Explore Brown Corpus and find the size, tokens, categories,
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize,word_tokenize
from collections import defaultdict
from nltk.probability import FreqDist
print(brown.categories())  #categories

print(brown.fileids())  

text = brown.words(fileids=['cg22'])

print(len(text))    #size

text1 = " ".join(brown.words ())
print(word_tokenize(text1))   #tokenization 

    #• Find the size of word tokens?
length = len(word_tokenize(text1))
print(length)

    #• Find the size of word types?
types = set(text1)
print(types)
print(len(types))

brown_tagged = brown.tagged_words(tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_tagged)
tag_fd.most_common()

    #• Find the size of the category “government”
gov = brown.words(categories='government')
print(len(gov))

    #• List the most frequent tokens
temp = defaultdict(int)
for sub in text1:
    for word in sub.split():
        temp[word] +=1
res = max(temp, key=temp.get)
print(str(res))

count = FreqDist(text1)
count.most_common()

    #• Count the number of sentences
sentences= nltk.sent_tokenize(text1)
length= len(sentences)
length


#2. Explore the corpora available in NLTK (any two)
    #• Raw corpus
from urllib import request
url = "http://www.gutenberg.org/files/2554/2554-0.txt"

response = request.urlopen(url)

raw = response.read().decode('utf8')

type(raw) #type
len(raw)  #length
raw[:50]  #text

tokens = word_tokenize(raw) #tokenization
type(tokens)
len(tokens)
tokens[:10]

raw.find("PART I") #find
raw.rfind("End of Project Gutenberg's Crime")
raw = raw[5338:1157743] 
raw.find("PART III")

    #• POS tagged
brown_fic_tagged = brown.tagged_sents(categories='fiction', tagset='universal')
brown_fic_words = brown.tagged_words(categories='fiction',  tagset='universal')

distw = FreqDist([w for (w, t) in brown_fic_words])
distw.N()  #no. of words

len(distw) #no. of distinct words
distw.max() #word with most occurence
distw['an'] #frequency of word 'an'

toke = word_tokenize(text1)
nltk.pos_tag(toke)  #pos tag


#3. Create a text corpus with a minimum of 200 words (unique content). Implement the following text processing
    #• Word segmentation
f = open('lorem.txt')
raw1 = f.read()
token1 = word_tokenize(raw1)
print(token1)

    #• Sentence segmentation
sent = sent_tokenize(raw1)
print(sent)

    #• Convert to Lowercase
raw1.lower()

    #• Stop words removal
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
line = f.read()
words = line.split() 
for r in words: 
    if not r in stop_words: 
        appendFile = open('filter.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close()

    #• Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
f = open('lorem.txt')
raw2 = f.read()
token2 = word_tokenize(raw2)
for x in token1:
    print(x,":",ps.stem(x))

    #• Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
f = open('lorem.txt')
raw3 = f.read()
token3 = word_tokenize(raw3)
for w in token3:
		print("Lemma for {} is {}".format(w, lemmatizer.lemmatize(w)))

    #• Part of speech tagger
stop_words = set(stopwords.words('english'))
f = open('lorem.txt')
raw4 = f.read()
token4 = sent_tokenize(raw4)
for i in token4:
    wlist = word_tokenize(i)
    wlist = [w for w in wlist if not w in stop_words]
    tag1 = nltk.pos_tag(wlist)
    print(tag1)


