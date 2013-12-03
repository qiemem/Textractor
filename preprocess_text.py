import PorterStemmer
import argparse
theStopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
theStopWordDict = {}
for aWord in theStopWords:
	theStopWordDict[aWord] = True
theStemmer = PorterStemmer.PorterStemmer()
def preprocess(aFileName, aNewFileName):
	myCurrentLine=0
	with open(aFileName) as myFile:
		with open(aNewFileName,'w') as myNewFile:
			for aLine in myFile:
				myWordList = aLine.split(" ")
				#myFilteredWords = [stem(aWord) for aWord in myWordList if not theStopWordDict.get(aWord,False)]
				myFilteredWords = [aWord for aWord in myWordList if not theStopWordDict.get(aWord,False)]
				myNewFile.write(' '.join(myFilteredWords))
				myCurrentLine = myCurrentLine + 1
				if (myCurrentLine %1000 == 0): print myCurrentLine #a little over 2872000 lines
def stem(aWord):
	return theStemmer.stem(aWord,0,len(aWord)-1)
def isStopWord(aWord):
	return theStopWordDict.get(aWord,False)

if __name__ == '__main__':
	theArgParser = argparse.ArgumentParser(description='Given a bunch of sentences, outputs feature vectors of the words')
	theArgParser.add_argument('-n', default='new_corpus.txt', type=str, metavar='filename', help='Name of new file')
	theArgParser.add_argument('-f', default='acl_07_corpus.txt', type=str, metavar='filename', help='Name of corpus file')
	theArgs = theArgParser.parse_args()
	preprocess(theArgs.f, theArgs.n)