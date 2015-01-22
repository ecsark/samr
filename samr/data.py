from collections import namedtuple


Datapoint = namedtuple("Datapoint", "phraseid sentenceid phrase sentiment")
Lexpoint = namedtuple("Lexpoint", "word phrase pos")
Splitpoint = namedtuple("Splitpoint", "lphrase rphrase matched")
Sentimentpoint = namedtuple("Sentimentpoint", "lphrase rphrase lsentiment rsentiment")
Phrasepair = namedtuple("Phrasepair", "lphrase rphrase")