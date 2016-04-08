from cut1 import readTXT
import sys
import numpy as np
import collections

a = readTXT('../Data/mc500.train.txt')

count = collections.Counter()
maxlen = 0
for i,aaa in enumerate(a):
  if i%4 == 0:
    count.update(aaa[0])
    if maxlen < len(aaa[0]):
      maxlen = len(aaa[0])
  for j in range(1,len(aaa)):
    count.update(aaa[j])
