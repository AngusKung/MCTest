import pdb
import numpy as np
import cPickle as pickle
from cut1 import readTXT
from genans import parseAnsOneHot

folder = "embs.mc500.dev.full/"
ansFilename='../Data/mc500.dev.ans'

dataPickle_name = "../Pickle/"+"mc500.dev.auto"+".pickle"
print "pickling to ... ",dataPickle_name

ans = parseAnsOneHot(ansFilename)
data = []

for q_id in range(len(ans)):
    fout = open(folder+str(q_id),'r')
    one_Q = []
    for line in fout:
	for word in line.split():
	    one_Q.append(word)
    Q = np.asarray(one_Q,dtype='float32')
    data.append([Q,ans[q_id]])

pdb.set_trace()
print "Pickling..."
fh =open(dataPickle_name,'wb')
pickle.dump(data,fh,pickle.HIGHEST_PROTOCOL)
fh.close()
