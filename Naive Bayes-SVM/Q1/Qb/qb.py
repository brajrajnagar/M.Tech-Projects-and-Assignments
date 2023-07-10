from operator import xor
import os
import sys
import random


script_name, train_path, test_path = sys.argv

test_neg_path = test_path+'/neg'
test_pos_path = test_path+'/pos'

all_reviews_bin = []
all_neg_files = os.listdir(test_neg_path)

for fil in range(len(all_neg_files)):
        all_reviews_bin.append(0)

all_pos_files = os.listdir(test_pos_path)
for fil in range(len(all_pos_files)):
        all_reviews_bin.append(1)

rand_rev_choice = [0,1]
rand_pred_bin = []
for i in range(len(all_reviews_bin)):
    rand_pred_bin.append(random.choice(rand_rev_choice))

err_bin = list(xor(x,y) for x,y in zip(all_reviews_bin,rand_pred_bin))
rand_per_accuracy = ((len(err_bin)-sum(err_bin))/len(err_bin))*100

print(f'test set accuracy obtained by randomly guessing one of the categories:{rand_per_accuracy}%')

pos_pred_bin = [1]*len(all_reviews_bin)
err_bin = list(xor(x,y) for x,y in zip(all_reviews_bin,pos_pred_bin))
pos_per_accuracy = ((len(err_bin)-sum(err_bin))/len(err_bin))*100

print(f'accuracy when each sample predicted as positive:{pos_per_accuracy}%')