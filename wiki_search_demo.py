import wikipedia
from SPARQLWrapper import SPARQLWrapper,JSON
#pip install {wikipedia,SPARQLWrapper}

TRAIN_SOURCE = '../project1-train-dev/train.gold'

def search_wiki(word):
	results = wikipedia.search(word,results=2) 

	page = None
	i = 0
	while i<len(results): #.page() fails if the first result is ambiguous or a disambiguation page, so this ignores that and moves to the next best thing
		try:
			page = wikipedia.page(results[i])
			break
		except:
			i+=1

	if page is not None:
		title = page.title.replace(' ','_')
		sparql = SPARQLWrapper("http://dbpedia.org/sparql")
		sparql.setReturnFormat(JSON)

		query = r"SELECT ?hasValue WHERE { <http://dbpedia.org/resource/"+title+r"> ?property ?hasValue FILTER(?hasValue=<http://dbpedia.org/ontology/Agent>) }"

		sparql.setQuery(query) 

		result = sparql.query().convert()

		if result['results']['bindings'] != []:
			return True

	return False

if __name__ == '__main__':
	sent = "President xzuqa travels to see Obama".split() #get the actual data at some point
	tags = []
	found = {}
	for word in sent:
		if word in found: #wikipedia.page() isn't super fast, so skipping redundant searches for speed.
			tags.append(found[word])
		else:
			tags.append(search_wiki(word))
	print(tags)
