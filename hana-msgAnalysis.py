'''
TO RETRIEVE MSG FROM iMessage:

sqlite> .output msg.csv
sqlite> SELECT
   ...>     message.text
   ...> FROM
   ...>     chat
   ...>     JOIN chat_message_join ON chat. "ROWID" = chat_message_join.chat_id
   ...>     JOIN message ON chat_message_join.message_id = message. "ROWID"
   ...> WHERE
   ...>     message.is_from_me = 0
   ...> ORDER BY
   ...>     message_date ASC;
sqlite>
'''

from nltk.sentiment import SentimentIntensityAnalyzer
import demoji
from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load("en-sentiment")

sia = SentimentIntensityAnalyzer()

msg = open("msg.csv", "r").read().split("\n")

count = 0
msgSentimentTotal = 0
msgSentimentPosTotal = 0
msgSentimentNegTotal = 0
msgFlairSentimentTotal = 0

for line in msg:
    count += 1
    msg = demoji.replace_with_desc(line, " ")
    msgSentiment = sia.polarity_scores(msg)["compound"]
    msgSentimentPos = sia.polarity_scores(msg)["pos"]
    msgSentimentNeg = sia.polarity_scores(msg)["neg"]
    sentence = Sentence(msg)
    msgSentiment = (msgSentiment+1)/2*100
    msgSentimentTotal += msgSentiment
    msgSentimentAvg = msgSentimentTotal/count
    msgSentimentPosTotal += msgSentimentPos
    msgSentimentNegTotal += msgSentimentNeg
    classifier.predict(sentence)
    msgFlairSentimentTotal += sentence.labels[0].to_dict()["confidence"]

print("Total message sentiment: ", msgSentimentTotal)
print("Message count:", count)
print("Average message sentiment: ", msgSentimentAvg)

print("Avg positive sentiment:", msgSentimentPosTotal/count)
print("Avg neg sentiment:", msgSentimentNegTotal/count)

print("Total sentiment as predicted by flair: ", msgFlairSentimentTotal)