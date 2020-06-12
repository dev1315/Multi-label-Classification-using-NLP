#!/usr/bin/env python
# coding: utf-8

# # Toxic Comment Classification
# >Access dataset from [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


train=pd.read_csv('Data.csv')


# In[4]:


train.head()


# **This is a problem of multi-label classification many times people confuse in multi-label and multi-class.**
# * The main difference between the two is lying in the concept of being mutually exclusive
# * **Multi-class classifications** problems are those where each sample belongs to atmost one class only. Eg: In a coin toss the result can either be a heads or tails
# * Whereas in case of **Multi-label classifications** each sample may belong to more than one class. Eg: A news article may belong to sports as well as politics.

# In[5]:


#creating a new column by summing all the target columns
al=train[train.columns[2:]].sum(axis=1)


# In[6]:


info=[]
y=train.shape[0]
info.append(['with no label',len(al[al==0]),(len(al[al==0])/y)*100])
for i in train.columns[2:]:
    x=len(train[train[i]==1][i])
    info.append([i,x,(x/y)*100])


# In[7]:


#Created a dataframe that shows the count and percentage of comments in different categories of target column
pd.DataFrame(info,columns=['Comment type','Count','%'])


# In[8]:


#Getting the count for comments that belong to multiple categories in target columns
multi=al.value_counts()[1:]
multi


# In[9]:


plt.figure(figsize=(7,5))
sns.barplot(x=multi.index,y=multi)
plt.title('Indivisual comments that belong to multiple classes',fontsize=15)


# In[10]:


#creating a copy of dataframe
df=train.copy()


# **Calculating word count**

# In[11]:


#you might need to first install this library
import textstat


# In[12]:


#using textstat.lexicon_count to grab the no of words in each comment
df['word_count']=[textstat.lexicon_count(i,removepunct=True) for i in df['comment_text'].values]


# In[13]:


for i in train.columns[2:]:
    print(df[['word_count',i]].groupby(i).mean())
    print('-----------------------------------')


# It can be observed that except for severe_toxic all the comment belonging to non clean category have a lower average word count

# **Symbols like "!*@#$*&" are also used when abusive words are used**

# In[14]:


from nltk.tokenize import regexp_tokenize


# In[15]:


df['sp_char']=[len(regexp_tokenize(i,"[!@#$&]")) for i in df['comment_text'].values]


# In[16]:


for i in train.columns[2:]:
    print(df[['sp_char',i]].groupby(i).mean())
    print('-----------------------------------')


# It is clearly evident that there is a high probability that a comment with symbols like this !@#$& will be a non clean comment

# **Calculating Unique word count**
# * There are chances that abusive comments use words repeatively

# In[17]:


from nltk.tokenize import word_tokenize


# In[18]:


#using set to create a sequence of only unique words
df['unique_w_count']=[len(set(word_tokenize(i))) for i in df['comment_text'].values]


# In[19]:


for i in train.columns[2:]:
    print(df[['unique_w_count',i]].groupby(i).mean())
    print('-----------------------------------')


# It is also clear that non clean comments use less unique words

# # Lets explore through some of the comments

# In[20]:


train.comment_text[5]


# In[21]:


train.comment_text[1151]


# In[22]:


train.comment_text[8523]


# # Text Cleaning
# Most of the comments have **\n, punctuations, numbers, extra whitespaces, contractions and lots of stopwords**  lets remove that first 

# In[23]:


import re
import spacy
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob


# Function for removing **\n**

# In[24]:


def slash_n(text):
    #removing \n
    text=re.sub('\n',' ',text)
    #converting whole string into lowercase
    text=text.lower()
    return text


# In[25]:


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


# <font size ='4'>**Dealing with contractions**</font>
# * A contraction is an abbreviation for a sequence of words
# * eg: "you'll" is a contraction of "you will"

# In[26]:


def contraction(text):
    """
    This function will return the text in  an expanded form which is in common English. It also helps in generalising the tokens
    """
    tokens=word_tokenize(text)
    tok=[]
    for i in tokens:
        if i in CONTRACTION_MAP.keys():
            tok.append(CONTRACTION_MAP[i])
        else:
            tok.append(i)
    return ' '.join(tok)


# **Created a combined list of Stopwords taken form nltk corpus and spacy library**

# In[27]:


#using stopwords from both libraries
nltk_sw=stopwords.words('english')
spacy_sw=list(STOP_WORDS)
#removing duplicates by converting list into set
stopword=list(set(nltk_sw+spacy_sw))


# **Removing** Punctuations, Stopwords, Whitespaces and Non Alphabatics.

# In[28]:


nlp=spacy.load('en_core_web_md')


# In[29]:


def sw(text):
    text=textstat.remove_punctuation(text) #removing punctuations
    tokens=text.split()
    tok=[]
    for i in tokens:
        if i not in stopword:              #removing stopwords
            i=i.strip()                    #removing all leading and trailing whitespaces as they will create bais in next step
            if i.isalpha():                #removing non alphabatics
                tok.append(i)
    return ' '.join(tok)


# <font size ='4'>**Performing Lemmatizatin**</font>
# * We have used lemmatization instead of stemming because we are going to create vectors for each comment so we want the words in each comment to make some sense (whereas in case of stemming the word is broken to its stem length that may or may not make sense).
# * Their are 2 methods for lemmatization using spacy's .lemma_ or WordNetLemmatizer
# * For this problem we are using spacy's .lemma_ because its more afficient and advanced.

# In[30]:


#Using Spacy's Lemmatization
def lemma(text):
    doc=nlp(text)
    tok=[i.lemma_ for i in doc]
    return ' '.join(tok)


# **Applying all the functions**

# In[31]:


train['comment_text']=train['comment_text'].apply(slash_n)


# In[32]:


train['comment_text']=train['comment_text'].apply(contraction)


# In[33]:


train['comment_text']=train['comment_text'].apply(sw)


# In[34]:


train['comment_text']=train['comment_text'].apply(lemma) #takes 25 mins to process


# # Generating Wordclouds for all categories

# In[36]:


from wordcloud import WordCloud


# In[37]:


def wordcloud_gen(i):
    texts=train[train[i]==1]['comment_text'].values
    wordcloud = WordCloud(background_color='black',width=700,height=500,colormap='viridis').generate(" ".join(texts))
    plt.figure(figsize=(7, 5))
    plt.imshow(wordcloud)
    plt.title(str(i).upper(),fontsize=15)
    plt.grid(False)
    plt.axis(False)


# In[38]:


wordcloud_gen('toxic')


# In[39]:


wordcloud_gen('severe_toxic')


# In[40]:


wordcloud_gen('obscene')


# In[41]:


wordcloud_gen('threat')


# In[42]:


wordcloud_gen('insult')


# In[43]:


wordcloud_gen('identity_hate')


# These wordclouds also indicate that an indivisual comment may belongs to multiple classes as there are many

# In[44]:


texts=train['comment_text'].values
wordcloud = WordCloud(background_color='black',width=700,height=500,colormap='viridis').generate(" ".join(texts))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud)
plt.title('ALL COMMENTS',fontsize=15)
plt.grid(False)
plt.axis(False)


# **Lets Split the data to perform further operations**

# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


training, testing=train_test_split(train, random_state=42, test_size=0.30, shuffle=True)


# In[37]:


training.shape,testing.shape


# Now We'll convert these strings into vectors using **`TfidfVectorizer`**
# * We need to find the most frequently occurring terms i.e. words with high term frequency or **tf**
# * We also want a measure of how unique a word is i.e. how infrequently the word occurs across all documents i.e. inverse document frequency or **idf**
# **Hence Tf-idf**

# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer


# **Choosing an optimum value for `max_features` is very necessary especially when you are dealing with such a big dataset.**
# * If the value is set too high the memory of your system will *fail to handle the amount of data*
# * And if the value is set too low then you'd be *under-utilizing your data*
# * Also cases like over-fitiing and under-fitting may also happen.

# In[39]:


unique_words=len(set((' '.join(training['comment_text'].values)).split()))
print('Total unique words in the training corpus is ',unique_words,', this means the vectorizer can create at max ',unique_words," vectors & definitely while creating so many vectors we'll run out of memory")


# In[40]:


#I have used 2000 as max_features but a higher value could also be tried as per system configuration
tfidf=TfidfVectorizer(strip_accents='unicode',max_features=2000,min_df=2,max_df=0.9)


# Creating a seperate dataframe for `'unique_w_count','sp_char','word_count'` named `feature1`

# In[41]:


feature1=df[['unique_w_count','sp_char','word_count']]
feature1.head()


# In[42]:


from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import MinMaxScaler


# In[43]:


scaler1=MinMaxScaler(feature_range=(0,0.5))


# **Scaling down above 3 features in range of `0,0.5`**

# In[44]:


feature1_t=scaler1.fit_transform(feature1)
feature1_t=pd.DataFrame(feature1_t)
feature1_t.head()


# Splitting these features according to train_test_split and then converting these `3 features` into sparse matrix

# In[45]:


train_feat=csr_matrix(feature1_t.iloc[training.index,:])
test_feat=csr_matrix(feature1_t.iloc[testing.index,:])


# In[46]:


train_feat.shape


# **Generated vectors for each comment that resulting in `300 more features` for classification**

# In[47]:


get_ipython().run_cell_magic('time', '', "#took 37 minutes to complete\nvec=[]\nfor i in range(train.shape[0]):\n    text=train['comment_text'][i]\n    print(i) #just to know the progress as there are 159571 entries\n    vec.append(nlp(text).vector)")


# In[48]:


#creating a dataframe of these vectors
vectors=pd.DataFrame(vec)


# In[49]:


#Splitting these feature vectors according to train_test_split
vectrain=vectors.iloc[training.index,:]
vectest=vectors.iloc[testing.index,:]


# In[52]:


#converting into sparse matrix
train_vec=csr_matrix(vectrain)
test_vec=csr_matrix(vectest)


# ***Fitting TfidfVectorizer only on the training set.***

# In[53]:


tfidf.fit(training['comment_text'])


# In[54]:


#tfidf vectorization on training set
train_tfidf=tfidf.transform(training['comment_text'])


# Using `hstack` to stack sparse matrixes along axis=1

# In[55]:


x_train=hstack((train_tfidf,train_vec,train_feat))


# In[56]:


x_train.shape


# We can clearly see total columns i.e. `features =2303`
# * 2000 tfidf_vectors
# * 300 Vector representation on each comment
# * unique_word_count
# * word_count
# * Count of special characters

# In[57]:


y_train=training.drop(labels = ['id','comment_text'], axis=1)


# In[58]:


#tfidf vectorization on testing set
test_tfidf=tfidf.transform(testing['comment_text'])


# In[59]:


x_test=hstack((test_tfidf,test_vec,test_feat))


# In[60]:


x_test.shape


# In[61]:


y_test=testing.drop(labels = ['id','comment_text'], axis=1)


# # Model Building
# Reference for [Model Building](https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/)

# In[62]:


from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score


# # OneVsRest
# * Here the multi-label problem is decomposed into multiple independent binary classification problems
# * Here each accuracy_score depicts the accuracy of model in predicting whether the comment is say toxic or not.

# In[63]:


pred={}
log=LogisticRegression()
for i in y_train.columns:
    log.fit(x_train, y_train[i])
    print('For',i,'accuracy_score',round(accuracy_score(y_test[i],log.predict(x_test))*100,1),'%')
    print('-------------------------------------------------')
    pred[i]=log.predict_proba(x_test)[:,1]


# In[64]:


print('roc_auc_score using OneVSRest is ',roc_auc_score(y_test,pd.DataFrame(pred)))


# # BinaryRelevance
# * This one is an ensemble of single-class (Yes/No) binary classifier
# * If there are n number of different labels it will create n datasets and train for each label and will result the union of all predicted labels.
# * Here the correlation b/w the labels is not taken into account

# In[65]:


classifier = BinaryRelevance(LogisticRegression())


# In[66]:


classifier.fit(x_train, y_train)
print('Accuracy_score using BinaryRelevance is ',round(accuracy_score(y_test,classifier.predict(x_test))*100,1),'%')
print('-------------------------------------------------')
print('roc_auc_score using BinaryRelevance is ',roc_auc_score(y_test,classifier.predict_proba(x_test).toarray()))


# # Label Powerset
# * Label Powerset creates a unique class for every possible label combination that is present in the training set, this way it makes use of label correlation
# * Only problem with this method is as the no of classes increases its computational complexity also increases.

# In[67]:


log_classifier=LabelPowerset(LogisticRegression())


# In[68]:


log_classifier.fit(x_train, y_train)
print('Accuracy_score using LabelPowerset is ',round(accuracy_score(y_test,log_classifier.predict(x_test))*100,1),'%')
print('-------------------------------------------------')
print('roc_auc_score using LabelPowerset is ',roc_auc_score(y_test,log_classifier.predict_proba(x_test).toarray()))


# # ClassifierChain
# * This method uses a chain of binary classifiers
# * Each new Classifier uses the predictions of all previous classifiers
# * This was the correlation b/w labels is taken into account

# In[69]:


chain=ClassifierChain(LogisticRegression())


# In[70]:


chain.fit(x_train, y_train)
print('Accuracy_score using ClassifierChain is ',round(accuracy_score(y_test,chain.predict(x_test))*100,1),'%')
print('-------------------------------------------------')
print('roc_auc_score using ClassifierChain is ',roc_auc_score(y_test,chain.predict_proba(x_test).toarray()))

