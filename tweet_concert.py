#Import Libraries
import nltk
import operator
import datetime
import itertools
import en_core_web_sm
from pyspark import SparkContext
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Initialize the spark and read the text file
sc = SparkContext("local", "first app")
train = sc.textFile('training_set_tweets.txt')

#Intialize parameters
sid = SentimentIntensityAnalyzer() # sentiment Decoder
sent_key = {'pos':'positive','neg':'negative','neu':'neutral','compound':'compound'} #sentiment values to be displayed

#Singer list made of 2010 pop singers
def singer_list(textfile):
    lines_list = open(textfile).read().splitlines()
    singers = []
    for line in lines_list:
        singers.append(line.lower())
        lin_spl = line.lower().split()
        if len(lin_spl)>1:
            singers.append(lin_spl[0])
            singers.append(" ".join(lin_spl[:2]))
    return singers
singers = singer_list('singers.txt')

#Making a city list
city = sc.textFile('training_set_users.txt').map(lambda x: x.split('\t')).collectAsMap() 

#Initilizing the Named-Entity Recognition tagger
nlp = en_core_web_sm.load()

#Input the number of spark tweets to be analyzed
data = train.take(100000) #100,000 samples

#Initial operations
data = [line.split('\t') for line in data]

#Funtion for Date parsing
def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False

#Function for relevant tweet stage 1
def booler(ent,text):
    if ('PERSON' in ent.values() and ('EVENT' in ent.values() or 'DATE' in ent.values() or 'TIME' in ent.values())):
        return True
    elif 'concert' in text.lower():
        return True
    else: 
        return False
    
#Function for key with the maximum value in a dictionary
def keywithmaxval(d,sent_key):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return sent_key[k[v.index(max(v))]]

#isolating the tweets for nlp functions
iso_text = [[l for l in line if (l.isdigit()==False and validate(l)==False)] for line in data]
iso_text = list(itertools.chain.from_iterable(iso_text))

#NER tagged
processed_text = [nlp((line.lower())) for line in iso_text]


tagged_text = [{X.text: X.label_ for X in item.ents} for item in processed_text]
#initialize a boolen variable for the first stage of sorting
boo = []
for i in range(len(iso_text)):
    boo.append(booler(tagged_text[i],iso_text[i]))

#Applying the initial changes
data_first = list(compress(data,boo))
tagged_text = list(compress(tagged_text,boo))
iso_text = list(compress(iso_text,boo))

# this function is responsible for generating the dictionaries as mentioned earlier in the notebook. It is encoded with the same functionality
def final(dent,singers,ent,text):
    d = []
    boo = []
    i = 0
    while i < len(text):
        dictor = {}
        #print(text[i])
        if 'PERSON' in ent[i].values():
            person = list(ent[i].keys())[list(ent[i].values()).index('PERSON')]
            if person in singers:
                dictor['who'] = person
                boo.append(True)
            else:
                dictor['who'] = 'null'
                boo.append(False)
        else: 
            dictor['who'] = 'null'
            boo.append(False)
        for j in range(len(dent[i])):
            #print(len(dent[i][1]))
            if len(dent[i][0])<9 and dent[i][0].isdigit()==True:
                #print(city[(dent[i][j])])
                dictor['where'] = city[(dent[i][0])]
            elif 'youtube' in text[i].lower():
                dictor['where'] = 'YouTube'
            if validate(dent[i][j]):
                date = datetime.datetime.strptime(dent[i][j], '%Y-%m-%d %H:%M:%S')
                dictor['when'] = date.strftime("%m/%d/%Y")
        sentiment = sid.polarity_scores(text[i])
        #print(dent[i])
        dictor["sentiment"] = keywithmaxval(sentiment,sent_key)
        dictor['audience'] = 'null'
        d.append(dictor)
        
        i+=1
    return list(compress(d,boo))
    
print(final(data_first,singers,tagged_text,iso_text))