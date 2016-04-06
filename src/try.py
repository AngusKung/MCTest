#import logging
#logging.basicConfig()
import pdb
import numpy as np
import cPickle as pickle
from gensim.models import Word2Vec
from cut import readTXT
from genans import parseAnsOneHot

wordvec_file = '../GloVe/glove.6B.300d.txt'
ansFilename='../Data/mc500.train.ans'
txtFilename='../Data/mc500.train.txt'
#'../Data/mc160.train.txt'
#"../Data/mc500.test.txt"
#"../Data/mc160.test.txt"
#"../Data/mc500.val.txt"
#"../Data/mc160.val.txt"
dataPickle_name = "../Pickle/"+txtFilename.split('/')[-1].split('.')[0]+"."+txtFilename.split('/')[-1].split('.')[1]+".mod2.pickle"
print dataPickle_name
print "Loading word2vec..."
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)
#word_vec ={}
data = []

ans = parseAnsOneHot(ansFilename)

print "Loading",txtFilename.split('/')[-1],"..."
txtList = readTXT(txtFilename)
print np.shape(txtList)
idxCounter = 0
for one in txtList:
    oneQ = []
    counter = []
    for passage in one:
	counter.append(float(len(passage)))
    count = 0
    for entry in one:
	temp_vector = np.zeros(300,dtype='float32')
	for word in entry:
	    word = word.lower()
	    if word not in word_vec:
		if '\'s' in word:
		    word = word.split('\'')[0]
		elif 'n\'t' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('n')[0]])/counter[count])
		    word = 'not'
		elif '\'d' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('\'')[0]])/counter[count])
		    word = 'would'
		elif 'i\'m' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec['i'])/counter[count])
		    word = 'am'
		elif '\'ll' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('\'')[0]])/counter[count])
		    word = 'will'
		elif '\'ve' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('\'')[0]])/counter[count])
		    word = 'have'
		elif '\'re' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('\'')[0]])/counter[count])
		    word = 'are'
		elif '(' in word:
		    word = word.split('(')[1]
		elif ')' in word:
		    word = word.split(')')[0]
		elif '.'  in word:
		    for oneword in word.split('.'):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword])/counter[count])
		    continue
		elif ';' in word:
		    for oneword in word.split(';'):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword])/counter[count])
		    continue
		elif ':' in word:
		    for oneword in word.split(':'):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword])/counter[count])
		    continue
		elif '\'' in word:
		    for oneword in word.split('\''):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword])/counter[count])
		    continue
		elif '-'  in word:
		    for oneword in word.split('-'):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword])/counter[count])
		    continue
	    try:
		temp_vector = np.add(temp_vector,np.asarray(word_vec[word])/counter[count])
	    except:
		print word
	oneQ.append(temp_vector)
	count += 1
    # === TODO:  Insert each LABLE here to retain the order ===
    # onoQ.append(label in numpy array)
    data.append( [np.hstack(tuple(oneQ)),ans[idxCounter]] )
    #data.append(np.hstack(tuple(oneQ)) )
    idxCounter += 1
#pdb.set_trace()
#print "Pickling..."
#fh =open(dataPickle_name,'wb')
#pickle.dump(data,fh,pickle.HIGHEST_PROTOCOL)
#fh.close()
