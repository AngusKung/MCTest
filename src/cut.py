#import os
import copy
#import pdb
#os.chdir("/home/MLDickS/MCTest/Data")
#import numpy as np

#f=open ("mc500.train.txt")
#f.readlines()

#initial

def readTXT(txtFilename,splitNum = 10):
    f=[]
    temp=[]
    final=[]
    count=0
    qcount=0
    paragraph=''
    story=0
    with open(txtFilename,'r') as f:
    	for line0 in f:
            #if (temp != [] and qcount>0 and count>qcount and (count-qcount)%5==0):

       	    #line=line0[:-2]
            #line.replace('\n','')
            line=line0.replace('\n','').replace(',','').replace('"','').replace('!','').replace('?','')
        
            if (len(line)<5):
                continue
            #count
            if (line.find('*')==0 and line.find(') ')==-1):
                #print(story)
                story+=1
                count=0
                qcount=0
                paragraph=''
            else:
                count+=1
            #manage
	    #        if (count>0 and count<=4):
	    #            words=line.split()
	    #            temp.append(words[len(words)-1])
	    #            print(temp)
            if (qcount==0 and count>5):
                if (line.find('1: ') == -1):
                    paragraph+=' '
                    paragraph+=line
                else:
                    qcount=count
                    combine=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
                    linenum=len(paragraph.split())/10
                    cut=paragraph.split('. ')
                    wordnum=-1
                    for sen in cut:
                        sensplit=sen.split()
                        if (wordnum>0 and (int(wordnum/linenum) != int((wordnum+len(sensplit))/linenum))):
                            k=int(wordnum/linenum)
                            for i in sensplit:
                                combine[k].append(i)
                        wordnum+=len(sensplit)
                        k=int(wordnum/linenum)
                        for i in sensplit:
                            combine[k].append(i)
            if (qcount != 0):
                line=line.replace('.','')
                if ((count-qcount)%5==0):
                    temp=copy.deepcopy(combine)
                    sen=(line.split(': '))[2]
                    sensplit=sen.split()
                    for i in sensplit:
                        temp[10].append(i)
                else:
                    sen=(line.split(') '))[1]
                    sensplit=sen.split()
                    for i in sensplit:
                        temp[10+(count-qcount)%5].append(i)
                if ((count-qcount)%5==4):
                    final.append(temp)
    f.close()
    return final

#final = readTXT('../Data/mc500.train.txt',splitNum = 10)
#print np.shape(final)
#pdb.set_trace()
