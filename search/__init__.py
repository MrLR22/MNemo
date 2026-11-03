# search/: finds the moments in the reference diary that answer a question. Its own part of ENGRAM: it only READS
# the diary file (modules/storage.py) and never calls the camera, the trigger or the diary writer.
#
#   rank.py    the three searches (keyword, meaning, image) and how their rankings are combined
#   engine.py  Searcher: runs them over a diary file for a time range (what the demo and the app use)
#
# Answering on top of the search: query/locate.py (which moment?) and query/llm_qa.py (a sentence answer).
