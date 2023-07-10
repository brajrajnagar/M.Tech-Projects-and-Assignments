from operator import xor
import math
import os
import sys
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer


script_name, train_path, test_path = sys.argv


stopwords = {'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'have', 'has', 'had',
     'may', 'can', 'must', 'might', 'shall', 'will', 'should', 'would', 'could', 'a', 'an', 'the', 'br', 'i',
     'we', 'you', 'he', 'she', 'it', 'they', 'her', 'hers', 'herself', 'him', 'himself', 'his', 'it', 'its', 'itself', 
     'me', 'mine', 'my', 'myself', 'that', 'thee', 'their', 'theirs', 'theirself', 'theirselves', 'there', 'these', 'this',
     'those', 'us', 'your', 'yours', 'yourself', 'yourselves', 'at', 'by', 'for', 'from', 'in', 'on', 'of', 'to', 'as', 'with', 'because', 'into', 'onto', 
     'under', 'over', 'and', 'or', 'if', 'but', 'so', 'What', 'When', 'Where', 'Whose', 'Who', 'How', 'Whom', 'Why', 'Which',
     'movie', 'movies', 'film', 'films'}


train_neg_path = train_path+'/neg'
train_pos_path = train_path+'/pos'



all_reviews = []
all_reviews_bin = []
neg_reviews = []
pos_reviews = []

all_neg_files = os.listdir(train_neg_path)

for fil in all_neg_files:
    with open(train_neg_path+'/'+fil, 'r+',encoding='utf-8') as f:
        rew = f.read()
        rew = rew.translate(str.maketrans('','',string.punctuation)).lower()
        neg_reviews.append(rew)
        all_reviews.append(rew)
        all_reviews_bin.append(0)


all_pos_files = os.listdir(train_pos_path)
for fil in all_pos_files:
    with open(train_pos_path+'/'+fil, 'r+',encoding='utf-8') as f:
        rew = f.read()
        rew = rew.translate(str.maketrans('','',string.punctuation)).lower()
        pos_reviews.append(rew)
        all_reviews.append(rew)
        all_reviews_bin.append(1)


neg_reviews_count = len(neg_reviews)
pos_reviews_count = len(pos_reviews)








neg_word_dict = {}
pos_word_dict = {}
# len_neg_review = []
# len_pos_review = []

for rew in neg_reviews:
    rew =  rew.split()
    # len_neg_review.append(len(rew))
    for word in rew:        
        if word in neg_word_dict:
            neg_word_dict[word] += 1
        else:
            neg_word_dict[word] = 1



for rew in pos_reviews:
    rew =  rew.split()
    # len_pos_review.append(len(rew))
    for word in rew:        
        if word in pos_word_dict:
            pos_word_dict[word] += 1
        else:
            pos_word_dict[word] = 1


print("Performing stemming and removing the stop-words....")
for word in neg_word_dict.copy():
    if word in stopwords:
        del neg_word_dict[word]
        
for word in pos_word_dict.copy():
    if word in stopwords:
        del pos_word_dict[word]
        




neg_word_dict_temp = {}
pos_word_dict_temp = {}


lemmatizer = WordNetLemmatizer()

for word in neg_word_dict:
        neg_val = neg_word_dict[word]
        word = lemmatizer.lemmatize(word, pos='n')
        word = lemmatizer.lemmatize(word, pos='v')
        word = lemmatizer.lemmatize(word, pos='a')
        word = lemmatizer.lemmatize(word, pos='r')
        word = lemmatizer.lemmatize(word, pos='s')
        if word in neg_word_dict_temp:
            neg_word_dict_temp[word] += neg_val
        else:
            neg_word_dict_temp[word] = neg_val

for word in pos_word_dict:
        pos_val = pos_word_dict[word]
        word = lemmatizer.lemmatize(word, pos='n')
        word = lemmatizer.lemmatize(word, pos='v')
        word = lemmatizer.lemmatize(word, pos='a')
        word = lemmatizer.lemmatize(word, pos='r')
        word = lemmatizer.lemmatize(word, pos='s')
        if word in pos_word_dict_temp:
            pos_word_dict_temp[word] += pos_val
        else:
            pos_word_dict_temp[word] = pos_val


neg_word_dict = neg_word_dict_temp.copy()
pos_word_dict = pos_word_dict_temp.copy()

word_dict = neg_word_dict_temp.copy()
for word in pos_word_dict_temp:
    if word in word_dict:
        word_dict[word] += pos_word_dict_temp[word]
    else:
        word_dict[word] = pos_word_dict_temp[word]





def naivebayes(word_dict, neg_word_dict, pos_word_dict):
    neg_review_wordcount = 0
    for z in neg_word_dict:
        neg_review_wordcount += neg_word_dict[z]

    pos_review_wordcount = 0
    for z in pos_word_dict:
        pos_review_wordcount += pos_word_dict[z]

    L = len(word_dict)
    alpha = 1.0
    prob_neg_word = {}
    prob_pos_word = {}
    
    for w in word_dict:
        if w in neg_word_dict:
            prob_neg_word[w] = (neg_word_dict[w] + alpha) / (neg_review_wordcount + L*alpha)
        else:
            prob_neg_word[w] = (alpha) / (neg_review_wordcount + L*alpha)

        if w in pos_word_dict:
            prob_pos_word[w] = (pos_word_dict[w] + alpha) / (pos_review_wordcount + L*alpha)
        else:
            prob_pos_word[w] = (alpha) / (pos_review_wordcount + L*alpha)

    return prob_neg_word, prob_pos_word




prob_neg_word, prob_pos_word = naivebayes(word_dict, neg_word_dict, pos_word_dict)

pY0 = neg_reviews_count/(neg_reviews_count+pos_reviews_count)
pY1 = pos_reviews_count/(neg_reviews_count+pos_reviews_count)




def review_sentiment(all_reviews, prob_neg_word, prob_pos_word):
    pred_bin = []
    # neg_prob = []
    # pos_prob = []
    pY0_x = 0
    pY1_x = 0
    for rev in all_reviews:
        words = rev.split()
        
        for w in words:
            w = lemmatizer.lemmatize(w, pos='n')
            w = lemmatizer.lemmatize(w, pos='v')
            w = lemmatizer.lemmatize(w, pos='a')
            w = lemmatizer.lemmatize(w, pos='r')
            w = lemmatizer.lemmatize(w, pos='s')
            if w in prob_neg_word:
                pY0_x += math.log(prob_neg_word[w])
            else:
                continue
            if w in prob_pos_word:
                pY1_x += math.log(prob_pos_word[w])
            else:
                continue
        pY0_x += math.log(pY0)
        pY1_x += math.log(pY1)
        # neg_prob.append(pY0_x)
        # pos_prob.append(pY1_x)
        if pY0_x > pY1_x:
            pred_bin.append(0)
        else:
            pred_bin.append(1)

        pY0_x = 0
        pY1_x = 0

    return pred_bin



pred_bin = review_sentiment(all_reviews, prob_neg_word, prob_pos_word)

err_bin = list(xor(x,y) for x,y in zip(all_reviews_bin,pred_bin))
per_accuracy = ((len(err_bin)-sum(err_bin))/len(err_bin))*100
print(f'Train data accuracy:{per_accuracy}%')




test_neg_path = test_path+'/neg'
test_pos_path = test_path+'/pos'


all_reviews = []
all_reviews_bin = []
neg_test_reviews = []
pos_test_reviews = []


all_neg_files = os.listdir(test_neg_path)

for fil in all_neg_files:
    with open(test_neg_path+'/'+fil, 'r+',encoding='utf-8') as f:
        rew = f.read()
        rew = rew.translate(str.maketrans('','',string.punctuation)).lower()
        neg_test_reviews.append(rew)
        all_reviews.append(rew)
        all_reviews_bin.append(0)



all_pos_files = os.listdir(test_pos_path)
for fil in all_pos_files:
    with open(test_pos_path+'/'+fil, 'r+',encoding='utf-8') as f:
        rew = f.read()
        rew = rew.translate(str.maketrans('','',string.punctuation)).lower()
        pos_test_reviews.append(rew)
        all_reviews.append(rew)
        all_reviews_bin.append(1)


pred_bin = review_sentiment(all_reviews, prob_neg_word, prob_pos_word)

err_bin = list(xor(x,y) for x,y in zip(all_reviews_bin,pred_bin))
per_accuracy = ((len(err_bin)-sum(err_bin))/len(err_bin))*100
print(f'Test data accuracy:{per_accuracy}%')




wordcloud = WordCloud(width = 1000, height = 500, background_color='white', ).generate_from_frequencies(neg_word_dict)
plt.figure(figsize=(8,4))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neg_wordcloud.png')




wordcloud = WordCloud(width = 1000, height = 500, background_color='white', ).generate_from_frequencies(pos_word_dict)
plt.figure(figsize=(8,4))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('pos_wordcloud.png')



TP = 0
FP = 0
TN = 0
FN = 0

for k in range(len(all_reviews_bin)):
    i = all_reviews_bin[k]
    j = pred_bin[k]
    if i==j:
        if j==1:
            TP+=1
        else:
            TN+=1
    else:
        if j == 1:
            FP+=1
        else:
            FN+=1

print("Confusion Matrix for Part(d)")
lst = ['', 'pos', 'neg']
lst2 = ['pos', TP, FP]
lst3 = ['neg', FN, TN]

for i in lst:
    print(i,"\t", end='')
print("\n")

for i in lst2:
    print(i,"\t", end='')
print("\n")

for i in lst3:
    print(i,"\t", end='')
print("\n")