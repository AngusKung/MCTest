from sklearn.preprocessing import OneHotEncoder
import numpy as np
def parseAnsOneHot(ansFilename):
    label_all = np.array([])
    enc = OneHotEncoder(sparse=False)
    with open(ansFilename,'r') as f:
        count = 0
        for line in f:
            count=count+1
            line_as_int = np.array(line.split( )).view(np.uint8)
            #print count
            #print line_as_int
            label_all = np.append(label_all,line_as_int)
    label_all = label_all.reshape((count,-1))
    ggwp = enc.fit_transform(label_all)
    ggwp2 = ggwp.reshape((count*4,-1))
    print 'The dim of OneHotLabel is '+str(np.shape(ggwp2))
    return ggwp2
