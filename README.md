# Multi-label Toxic Comment Classification
**This is a problem of multi-label classification many times people confuse in multi-label and multi-class.**
* The main difference between the two is lying in the concept of being mutually exclusive
* **Multi-class classifications** problems are those where each sample belongs to atmost one class only. Eg: In a coin toss the result can either be a heads or tails
* Whereas in case of **Multi-label classifications** each sample may belong to more than one class. Eg: A news article may belong to sports as well as politics.

* This project includes feature engineering like finding pattern for assesing the labels like Word count, Specific symbols, no of unique words used.
* It also includes text cleaning like removing \n ,expanding contractions, punctuations, numbers, extra whitespaces and stop words.
<font size ='4'>**Performing Stemming**</font>
* We could also have used lemmatization but SnowballStemmer generalises words more efficiently by transforming words with roughly the same semantics to one standard form.
* Their are 2 methods for lemmatization using spacy's .lemma_ or WordNetLemmatizer
* For this problem we are using SnowballStemmer because its more advanced and takes less computational time as compared to PorterStemmer 
* 
* I have tried using different libraries
