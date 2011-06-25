#!/usr/bin/python

import nltk
import sys
import getopt 

stemmer = nltk.stem.RSLPStemmer()
stpwrds = nltk.corpus.stopwords.words('portuguese')

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def stopwords(data):
    ret = []
    for a in data:
        if a not in stpwrds:
           ret.append(a)
    return "".join(ret)

def stemo(data):
    ret = []
    for a in data:
        ret.append(stem(a))
    return ret

def main(argv=None):
    if argv is None:
        argv = sys.argv
        try:
            opts, args = getopt.getopt(argv[1:], "sta", ["stemmer", "stopwords", "all"])
        except getopt.error, msg:
            raise Usage(msg)

        stem = stopw = False

        for option, value in opts:
            if option in ("-s", "--stemmer"):
                stem = True
            if option in ("-t", "--stopwords"):
                stopw = True
            if option in ("-a", "--all"):
                stem = stopw = True

    data = sys.stdin.read()        
    
    if stemo == True:
        data = stem(data)
    
    if stopw == True:
        data = stopwords(data)

    sys.stdout.write(str(data))

if __name__ == '__main__':
    main()

