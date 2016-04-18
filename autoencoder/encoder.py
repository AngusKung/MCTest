import pdb
from cut1 import readTXT
from sys import argv


if len(argv) != 4:
    print "Usage: python encoder.py <input(../Data/mc500.train.txt)> <outputdir(mc500.train)> <dict>"
    exit(1)

input_ = argv[1]
output = argv[2]
dictionary = argv[3]

OOV = 1
d_arr = {}
with open(dictionary, 'r') as d:
    word_id = 1
    for line in d:
        word = line.rstrip().lower()
        d_arr[word] = word_id
        word_id += 1

txt = readTXT(input_)
#with open(output, 'w') as fout:
for q_id in range(len(txt)):
    print q_id
    for oneQ in txt:
	fout = open(output+str(q_id)+"_txt","w")
        for word in oneQ[0]:
            word = word.lower()
            if word in d_arr:
                token = d_arr[word]
	    elif word.split(".")[0] in d_arr:
		token = d_arr[word.split(".")[0]]
	    elif word.split("'s")[0] in d_arr:
		token = d_arr[word.split("'s")[0]]
	    elif word.split(":")[0] in d_arr:
		token = d_arr[word.split(":")[0]]
            else:
                token = OOV

            fout.write(str(token))
            fout.write(' ')

        fout.write('\n')
	fout.close()
	fout = open(output+str(q_id)+"_q","w")
        for word in oneQ[1]:
            word = word.lower()
            if word in d_arr:
                token = d_arr[word]
	    elif word.split(".")[0] in d_arr:
		token = d_arr[word.split(".")[0]]
	    elif word.split("'s")[0] in d_arr:
		token = d_arr[word.split("'s")[0]]
	    elif word.split(":")[0] in d_arr:
		token = d_arr[word.split(":")[0]]
            else:
                token = OOV

            fout.write(str(token))
            fout.write(' ')

        fout.write('\n')
	fout.close()
