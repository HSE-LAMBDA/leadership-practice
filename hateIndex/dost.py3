from dostoevsky.tokenization import UDBaselineTokenizer, RegexTokenizer
from dostoevsky.embeddings import SocialNetworkEmbeddings
from dostoevsky.models import SocialNetworkModel
from time import time

startTime = time()

tokenizer = UDBaselineTokenizer() or RegexTokenizer()
embeddings_container = SocialNetworkEmbeddings()
model = SocialNetworkModel(tokenizer=tokenizer, embeddings_container=embeddings_container, lemmatize=False)

print(time() - startTime)

messages = [
    '😞',
    'наступили на ногу',
    'всё суперски',
    'я тебя ненавижу'
]

predictTime = time()

results = model.predict(messages)

print(time() - predictTime)

for message, sentiment in zip(messages, results):
    print(message, '->', sentiment)  # наступили на ногу -> negative

print(results)