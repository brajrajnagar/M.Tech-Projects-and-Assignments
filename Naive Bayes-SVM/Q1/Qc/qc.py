import math
import os
import sys
import string
import random


script_name, train_path, test_path = sys.argv

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



word_dict = {}
neg_word_dict = {}
pos_word_dict = {}
# len_neg_review = []
# len_pos_review = []

for rew in neg_reviews:
    rew =  rew.split()
    # len_neg_review.append(len(rew))
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
    # len_pos_review.append(len(rew))
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



rand_rev_choice = [0,1]
rand_pred_bin = []
for i in range(len(all_reviews)):
    rand_pred_bin.append(random.choice(rand_rev_choice))


pos_pred_bin = [1]*len(all_reviews)




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

print("Confusion Matrix for my algorithm prediction in Part(a)")
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

TP = 0
FP = 0
TN = 0
FN = 0

for k in range(len(all_reviews_bin)):
    i = all_reviews_bin[k]
    j = rand_pred_bin[k]
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

print("Confusion Matrix for Random prediction in Part(b)")
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


TP = 0
FP = 0
TN = 0
FN = 0

for k in range(len(all_reviews_bin)):
    i = all_reviews_bin[k]
    j = pos_pred_bin[k]
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

print("Confusion Matrix for positive prediction in Part(b)")
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