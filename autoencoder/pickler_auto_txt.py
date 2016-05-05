import pdb
import numpy as np
import cPickle as pickle
from gensim.models import Word2Vec
from cut1 import readTXT
from genans import parseAnsOneHot

wordvec_file = '../GloVe/glove.6B.300d.txt'
folder = "embs.mc500.dev.txt/"
ansFilename='../Data/mc500.dev.ans'
txtFilename = '../Data/mc500.dev.txt'
stopWordFile = '../Data/stopwords.txt'

dataPickle_name = "../Pickle/"+"mc500.dev.auto.txt"+".pickle"
print "pickling to ... ",dataPickle_name

print "Loading wor2vec..."
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)
ans = parseAnsOneHot(ansFilename)

print "Loading",txtFilename.split('/')[-1],"..."
txtList = readTXT(txtFilename)

data = []

for q_id in range(len(ans)):
    fout = open(folder+str(q_id)+'_txt','r')
    one_Q = []
    for line in fout:
	for word in line.split():
	    one_Q.append(word)
    Q = np.asarray(one_Q,dtype='float32')
    oneQ = []
    for entry in txtList[q_id][1:]:
	count = 0.
	temp_vector = np.zeros(300,dtype='float32')
	for word in entry:
	    word = word.lower()
	    if word not in word_vec:
		if '\'s' in word:
		    word = word.split('\'')[0]
		elif 'n\'t' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('n')[0]]))
		    count += 1.
		    word = 'not'
		elif '\'d' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('\'')[0]]))
		    count += 1.
		    word = 'would'
		elif 'i\'m' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec['i']))
		    count += 1.
		    word = 'am'
		elif '\'ll' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('\'')[0]]))
		    count += 1.
		    word = 'will'
		elif '\'ve' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('\'')[0]]))
		    count += 1.
		    word = 'have'
		elif '\'re' in word:
		    temp_vector = np.add(temp_vector,np.asarray(word_vec[word.split('\'')[0]]))
		    count += 1.
		    word = 'are'
		elif '(' in word:
		    word = word.split('(')[1]
		elif ')' in word:
		    word = word.split(')')[0]
		elif '.'  in word:
		    for oneword in word.split('.'):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword]))
			    count+=1.
		    continue
		elif ';' in word:
		    for oneword in word.split(';'):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword]))
			    count+=1.
		    continue
		elif ':' in word:
		    for oneword in word.split(':'):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword]))
			    count+=1.
		    continue
		elif '\'' in word:
		    for oneword in word.split('\''):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword]))
			    count+=1.
		    continue
		elif '-'  in word:
		    for oneword in word.split('-'):
			if oneword and oneword in word_vec:
			    temp_vector = np.add(temp_vector,np.asarray(word_vec[oneword]))
			    count+=1.
		    continue
	    try:
		temp_vector = np.add(temp_vector,np.asarray(word_vec[word]))
	        count += 1.
	    except:
		print word
	if count == 0:
	    oneQ.append(temp_vector)
	else:
	    oneQ.append(np.divide(temp_vector,count))
    data.append([np.hstack([Q]+oneQ),ans[q_id]])

pdb.set_trace()
print "Pickling..."
fh =open(dataPickle_name,'wb')
pickle.dump(data,fh,pickle.HIGHEST_PROTOCOL)
fh.close()
