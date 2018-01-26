'''
wordInfo = [index,word,POS,BIO,sentenceStart]

'''

import sys

def getWord(wordInfo):
    return wordInfo[1]
    
def getPOS(wordInfo):
    return wordInfo[2]
    
def isCap(wordInfo):
    return wordInfo[1][0].isupper()

def sentenceStart(wordInfo):
    return wordInfo[4]   

if __name__ == '__main__':
    
    
    allWords = []
    sentenceStart = 1
    
    valFuncList = [
              ('prevWord',getWord,-1),
              ('nextWord',getWord,1),
              ('thisPOS',getPOS,0),
              ('prevPOS',getPOS,-1),
              ('nextPOS',getPOS,1)
              ]
    
    boolFuncList = [
              ('thisCap',isCap,0),
              ('nextCap',isCap,1),
              ('prevCap',isCap,-1),
              ('sentenceStart',sentenceStart,0)
              ]
    
    
    with open(sys.argv[1],'r') as train:
        line = train.readline()
        
        while line != '':
            
            if line == '\n':
                sentenceStart = 1
                line = train.readline()
                continue
                
            wordInfo = line.strip().split('\t')
            wordInfo.append(sentenceStart)
            if sentenceStart:
                sentenceStart = 0
            
            
            allWords.append(wordInfo)
            
            line = train.readline()
            
    with open(sys.argv[2],'w') as output:
        for i in range(len(allWords)):
            thisWordInfo = allWords[i]
            line = thisWordInfo[1] #always start with word
            
            #value features
            for label,func,relInd in valFuncList:
                ind = i + relInd
                if ind < 0 or ind >= len(allWords):
                    continue
                line += '\t'
                line += (label + '=')
                line += func(allWords[ind])
            
            #boolean features
            for label,func,relInd in boolFuncList:
                ind = i + relInd
                if ind < 0 or ind >= len(allWords):
                    continue
                boolVal = func(allWords[ind])
                if boolVal:
                    line += '\t'
                    line += label
            
            line += '\t' + thisWordInfo[3] #add label
                
            
            output.write(line+'\n')
                