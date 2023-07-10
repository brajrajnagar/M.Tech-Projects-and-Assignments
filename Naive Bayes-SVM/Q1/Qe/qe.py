from operator import xor
import math
import os
import sys
import string
import nltk
from nltk.stem import WordNetLemmatizer


script_name, train_path, test_path = sys.argv


stopwords = {'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'have', 'has', 'had',
     'may', 'can', 'must', 'might', 'shall', 'will', 'should', 'would', 'could', 'a', 'an', 'the', 'br', 'i',
     'we', 'you', 'he', 'she', 'it', 'they', 'her', 'hers', 'herself', 'him', 'himself', 'his', 'it', 'its', 'itself', 
     'me', 'mine', 'my', 'myself', 'that', 'thee', 'their', 'theirs', 'theirself', 'theirselves', 'there', 'these', 'this',
     'those', 'us', 'your', 'yours', 'yourself', 'yourselves', 'at', 'by', 'for', 'from', 'in', 'on', 'of', 'to', 'as', 'with', 'because', 'into', 'onto', 
     'under', 'over', 'and', 'or', 'if', 'but', 'so', 'What', 'When', 'Where', 'Whose', 'Who', 'How', 'Whom', 'Why', 'Which',}


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


all_reviews_tgrm = all_reviews.copy()
all_reviews_bin_tgrm = all_reviews_bin.copy()
neg_reviews_tgrm = neg_reviews.copy()
pos_reviews_tgrm = pos_reviews.copy()



neg_reviews_count = len(neg_reviews)
pos_reviews_count = len(pos_reviews)




word_dict = {}
neg_word_dict = {}
pos_word_dict = {}

for rew in neg_reviews:
    rew =  rew.split()
    for w1 in stopwords:
        for w2 in rew:
            if w1 == w2:
                rew.remove(w2)
    
    lemmatizer = WordNetLemmatizer()
    for i in range(len(rew)):
        rew[i] = lemmatizer.lemmatize(rew[i], pos='n')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='v')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='a')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='r')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='s')

    bgrm_words = list(nltk.bigrams(rew))

    for item in bgrm_words:
	    rew.append(' '.join(item))

    for word in rew:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
            
            if word in neg_word_dict:
                neg_word_dict[word] += 1
            else:
                neg_word_dict[word] = 1


for rew in pos_reviews:
    rew =  rew.split()
    for w1 in stopwords:
        for w2 in rew:
            if w1 == w2:
                rew.remove(w2)
    
    for i in range(len(rew)):
        rew[i] = lemmatizer.lemmatize(rew[i], pos='n')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='v')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='a')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='r')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='s')

    bgrm_words = list(nltk.bigrams(rew))

    for item in bgrm_words:
	    rew.append(' '.join(item))

    for word in rew:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
            
            if word in pos_word_dict:
                pos_word_dict[word] += 1
            else:
                pos_word_dict[word] = 1



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

        for w1 in stopwords:
            for w2 in words:
                if w1 == w2:
                    words.remove(w2)
    
        for i in range(len(words)):
            words[i] = lemmatizer.lemmatize(words[i], pos='n')
            words[i] = lemmatizer.lemmatize(words[i], pos='v')
            words[i] = lemmatizer.lemmatize(words[i], pos='a')
            words[i] = lemmatizer.lemmatize(words[i], pos='r')
            words[i] = lemmatizer.lemmatize(words[i], pos='s')

        bgrm_words = list(nltk.bigrams(words))

        for item in bgrm_words:
            words.append(' '.join(item))

        for w in words:
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
print(f'Train data accuracy on Bi-grams model:{per_accuracy}%')




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



all_reviews_tgrm_test = all_reviews.copy()
all_reviews_bin_tgrm_test = all_reviews_bin.copy()
neg_test_reviews_tgrm_test = neg_test_reviews.copy()
pos_test_reviews_tgrm_test = pos_test_reviews.copy()




pred_bin = review_sentiment(all_reviews, prob_neg_word, prob_pos_word)

err_bin = list(xor(x,y) for x,y in zip(all_reviews_bin,pred_bin))
per_accuracy = ((len(err_bin)-sum(err_bin))/len(err_bin))*100
print(f'Test data accuracy on Bi-grams model:{per_accuracy}%')




##########  Trigram ###########

all_reviews = all_reviews_tgrm
all_reviews_bin = all_reviews_bin_tgrm
neg_reviews = neg_reviews_tgrm
pos_reviews = pos_reviews_tgrm


neg_reviews_count = len(neg_reviews)
pos_reviews_count = len(pos_reviews)


word_dict = {}
word_tgrm_dict = {}
neg_word_dict = {}
neg_word_tgrm_dict = {}
pos_word_dict = {}
pos_word_tgrm_dict = {}

for rew in neg_reviews:
    rew =  rew.split()
    for w1 in stopwords:
        for w2 in rew:
            if w1 == w2:
                rew.remove(w2)
    
    lemmatizer = WordNetLemmatizer()
    for i in range(len(rew)):
        rew[i] = lemmatizer.lemmatize(rew[i], pos='n')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='v')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='a')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='r')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='s')

    tgrm_words = list(nltk.trigrams(rew))

    tgrm_rew = []
    for item in tgrm_words:
	    tgrm_rew.append(' '.join(item))

    for word in rew:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
            
            if word in neg_word_dict:
                neg_word_dict[word] += 1
            else:
                neg_word_dict[word] = 1

    for word in tgrm_rew:
            if word in word_tgrm_dict:
                word_tgrm_dict[word] += 1
            else:
                word_tgrm_dict[word] = 1
            
            if word in neg_word_tgrm_dict:
                neg_word_tgrm_dict[word] += 1
            else:
                neg_word_tgrm_dict[word] = 1


for rew in pos_reviews:
    rew =  rew.split()
    for w1 in stopwords:
        for w2 in rew:
            if w1 == w2:
                rew.remove(w2)
    
    for i in range(len(rew)):
        rew[i] = lemmatizer.lemmatize(rew[i], pos='n')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='v')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='a')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='r')
        rew[i] = lemmatizer.lemmatize(rew[i], pos='s')

    tgrm_words = list(nltk.trigrams(rew))

    tgrm_rew = []
    for item in tgrm_words:
	    tgrm_rew.append(' '.join(item))

    for word in rew:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
            
            if word in pos_word_dict:
                pos_word_dict[word] += 1
            else:
                pos_word_dict[word] = 1

    for word in tgrm_rew:
            if word in word_tgrm_dict:
                word_tgrm_dict[word] += 1
            else:
                word_tgrm_dict[word] = 1
            
            if word in pos_word_tgrm_dict:
                pos_word_tgrm_dict[word] += 1
            else:
                pos_word_tgrm_dict[word] = 1


def naivebayes_tgrm(word_dict, neg_word_dict, pos_word_dict, word_tgrm_dict, neg_word_tgrm_dict, pos_word_tgrm_dict):
    neg_review_wordcount = 0
    for z in neg_word_dict:
        neg_review_wordcount += neg_word_dict[z]

    pos_review_wordcount = 0
    for z in pos_word_dict:
        pos_review_wordcount += pos_word_dict[z]

    neg_review_tgrm_wordcount = 0
    for z in neg_word_tgrm_dict:
        neg_review_tgrm_wordcount += neg_word_tgrm_dict[z]

    pos_review_tgrm_wordcount = 0
    for z in pos_word_tgrm_dict:
        pos_review_tgrm_wordcount += pos_word_tgrm_dict[z]

    L = len(word_dict)
    L2 = len(word_tgrm_dict)
    alpha = 1.0
    prob_neg_word = {}
    prob_pos_word = {}
    prob_neg_tgrm_word = {}
    prob_pos_tgrm_word = {}
    
    for w in word_dict:
        if w in neg_word_dict:
            prob_neg_word[w] = (neg_word_dict[w] + alpha) / (neg_review_wordcount + L*alpha)
        else:
            prob_neg_word[w] = (alpha) / (neg_review_wordcount + L*alpha)

        if w in pos_word_dict:
            prob_pos_word[w] = (pos_word_dict[w] + alpha) / (pos_review_wordcount + L*alpha)
        else:
            prob_pos_word[w] = (alpha) / (pos_review_wordcount + L*alpha)


    for w in word_tgrm_dict:
        if w in neg_word_tgrm_dict:
            prob_neg_tgrm_word[w] = (neg_word_tgrm_dict[w] + alpha) / (neg_review_tgrm_wordcount + L2*alpha)
        else:
            prob_neg_tgrm_word[w] = (alpha) / (neg_review_tgrm_wordcount + L2*alpha)

        if w in pos_word_tgrm_dict:
            prob_pos_tgrm_word[w] = (pos_word_tgrm_dict[w] + alpha) / (pos_review_tgrm_wordcount + L2*alpha)
        else:
            prob_pos_tgrm_word[w] = (alpha) / (pos_review_tgrm_wordcount + L2*alpha)

    return prob_neg_word, prob_pos_word, prob_neg_tgrm_word, prob_pos_tgrm_word




prob_neg_word, prob_pos_word, prob_neg_tgrm_word, prob_pos_tgrm_word = naivebayes_tgrm(word_dict, neg_word_dict, pos_word_dict, word_tgrm_dict, neg_word_tgrm_dict, pos_word_tgrm_dict)

prob_neg_word.update(prob_neg_tgrm_word)
for i in prob_neg_word.copy():
    prob_neg_word[i] = prob_neg_word[i]/2

prob_pos_word.update(prob_pos_tgrm_word)
for i in prob_pos_word.copy():
    prob_pos_word[i] = prob_pos_word[i]/2


pY0 = neg_reviews_count/(neg_reviews_count+pos_reviews_count)
pY1 = pos_reviews_count/(neg_reviews_count+pos_reviews_count)



def review_sentiment_tgrm(all_reviews, prob_neg_word, prob_pos_word):
    pred_bin = []
    # neg_prob = []
    # pos_prob = []
    pY0_x = 0
    pY1_x = 0
    for rev in all_reviews:
        words = rev.split()

        for w1 in stopwords:
            for w2 in words:
                if w1 == w2:
                    words.remove(w2)
    
        for i in range(len(words)):
            words[i] = lemmatizer.lemmatize(words[i], pos='n')
            words[i] = lemmatizer.lemmatize(words[i], pos='v')
            words[i] = lemmatizer.lemmatize(words[i], pos='a')
            words[i] = lemmatizer.lemmatize(words[i], pos='r')
            words[i] = lemmatizer.lemmatize(words[i], pos='s')

        tgrm_words = list(nltk.trigrams(words))

        for item in tgrm_words:
            words.append(' '.join(item))

        for w in words:
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



pred_bin = review_sentiment_tgrm(all_reviews, prob_neg_word, prob_pos_word)

err_bin = list(xor(x,y) for x,y in zip(all_reviews_bin,pred_bin))
per_accuracy = ((len(err_bin)-sum(err_bin))/len(err_bin))*100
print(f'Train data accuracy on modified tri-grams model:{per_accuracy}%')


all_reviews = all_reviews_tgrm_test
all_reviews_bin = all_reviews_bin_tgrm_test
neg_test_reviews = neg_test_reviews_tgrm_test
pos_test_reviews = pos_test_reviews_tgrm_test



pred_bin = review_sentiment_tgrm(all_reviews, prob_neg_word, prob_pos_word)

err_bin = list(xor(x,y) for x,y in zip(all_reviews_bin,pred_bin))
per_accuracy = ((len(err_bin)-sum(err_bin))/len(err_bin))*100
print(f'Test data accuracy on modified tri-grams model:{per_accuracy}%')