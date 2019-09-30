#!/usr/bin/env python
# coding: utf-8

# ## Kindness index 

# - Help (сколько раз кому помог)
# - Advice (дал совет)
# - Smile (улыбнулся)
# - Like
# - Congrats (поздравил)
# - Cоотношение активности на своей странице и странице других пользователей
# ----------
# 
# 
# Анализируем поведение пользователей на предмет их помогающей активности: сколько раз кому-то помог, дал совет, улыбнулся, поздравил. Анализируем embeddingи на предмет схожести с embeddingами, построенными по набору текстов, которые считаем добрыми. Если есть фотографии - анализируем на каком количестве улыбается пользователь. Анализируем по ключевым словам (некоторые из которых считаем добрыми. а некоторые нет) темы, которые человек постит. Можно выделять часто употребляемые в темах слова всякими методами типа латентного размещения Дирихле (Latent Dirichlet allocation) 
# 
# 
# ---------
# Help, advice определяются путем анализа текста и выявления его
# тем. Latent Dirichlet allocation (латентные размещения Дирихле). На
# выходе из этого метода будут слова, которые являются набором
# тем поста. Дальше эти слова по словарю (который нужно будет
# вручную составить) соотносятся со словарем, характеризующим
# помогательную активность.
# 
# Например: в помогательную активность записали набор {H}, потом
# выяснили что в постах пользователя столько-то слов, полученных
# из латентных размещений Дирихле, совпадает со словами из {H}.
# Вводим метрику для оценки того, насколько сильна помогательная
# активность пользователя

# In[ ]:


import psycopg2 as pc
import pandas as pd
import numpy as np 
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    remove_stopwords,
)
from nltk.stem.snowball import RussianStemmer
import emoji
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath

import pickle

# In[ ]:


from sqlalchemy import create_engine
import re
POSTGRE_STR = ('postgresql://vk_parser:WegEWWXfedLf2YbS@13.84.188.132:0/vk_data')


# #### Генератор для доставания коментариев из БД

# In[ ]:


def get_comments(connection, uid):
    return connection.execute("SELECT text, owner_id, from_id FROM posts WHERE owner_id = %d AND text != ''" % uid)

def data_generator(max_iter=2, postgre_str=POSTGRE_STR):
    cnx = create_engine(postgre_str)
    connection = cnx.connect()
    id_list = connection.execute('SELECT DISTINCT owner_id FROM posts')
    cur_iter = 0
    for cur_id in id_list:
        if cur_iter == max_iter:
            break
        cur_iter += 1
        res = get_comments(connection, cur_id[0])
        for row in res:
            yield row
    connection.close()


# In[ ]:


def data_from_list(data):
    for row in data:
        yield row[0]


# #### Генератор для доставания очищенных комментариев. Возвращает список слов, список корней слов и список эмоджи из комментария

# In[ ]:


def data_preprocessor(max_iter=20, from_db=True, get_data_func=None):
    def extract_emojis(text):
        return [c for c in ''.join(text) if c in emoji.UNICODE_EMOJI]
    if from_db:
        data_gen = data_generator(max_iter=max_iter)
    else:
        data_gen = get_data_func
    EN_REGEXP = re.compile("[a-zA-Z]", flags=re.UNICODE)
    EMOJI_REGEXP = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    filters = [lambda x: x.lower(), lambda x: EN_REGEXP.sub('', x), lambda x: EMOJI_REGEXP.sub('', x), strip_tags, strip_punctuation, 
           strip_multiple_whitespaces, strip_numeric, strip_punctuation]
    stemmer = RussianStemmer()
    for text in data_gen:
        emojis = extract_emojis(text)
        filtered_text = preprocess_string(text, filters)
        stemmed_text = list(map(stemmer.stem, filtered_text))
        yield filtered_text, stemmed_text, emojis


# #### Функция для создания корпуса слов, которые использовались в комментариях

# In[ ]:


def create_comment_dictionary(filter_every_n=1000, max_iter=20, from_db=True, get_data_func=None):
    comment_dict = Dictionary()
    text_gen = data_preprocessor(max_iter=max_iter, from_db=from_db, get_data_func=get_data_func)
    n_iter = 0
    for _, stemmed_text, _ in text_gen:
        comment_dict.add_documents([stemmed_text])
        n_iter += 1
        if n_iter % filter_every_n == 0:
            comment_dict.filter_extremes()
    return comment_dict


# #### Создание LDA-модели на комментариях 

# In[ ]:


def create_LDA(comment_dict, num_topics=20, chunk_size=50, max_iter=20, from_db=True, get_data_func=None):
    lda = None
    text_gen = data_preprocessor(max_iter=max_iter, from_db=from_db, get_data_func=get_data_func)
    corpus = []
    for _, stemmed_text, _ in text_gen:
        if len(stemmed_text) != 0:
            corpus.append(comment_dict.doc2bow(stemmed_text))
        if len(corpus) == chunk_size:
            if lda is None:
                lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=comment_dict, per_word_topics=1, passes=10)
            else:
                lda.update(corpus=corpus)
            corpus = []
    return lda


# In[ ]:
def get_lda_and_dict():
    with open('../storage/lda_7', "rb") as f:
        lda = pickle.load(f)
    with open('../storage/dict', "rb") as f:
        d = pickle.load(f)
    return lda, d


# In[ ]:


def count_kindness(lda, dictionary, kind_set, from_id=None, data_df=None, comments=None, treshold=3.2):
    def get_one_comment(text):
        yield text
    if comments is None:
        comments = data_df[data_df['from_id'] == from_id]
        comments = list(comments['text'])
    count_list = []
    count_kind = 0
    for comment in comments:
        cnt = 0
        _, stemmed_text, _ = next(data_preprocessor(from_db=False, get_data_func=get_one_comment(comment)))
        if len(stemmed_text) == 0:
            continue
        topics = lda.get_document_topics(dictionary.doc2bow(stemmed_text))
        topics.sort(key=lambda x: -x[-1])
        for topic, prob in topics[:3]:
            words = set(map(lambda x: x[0], lda.show_topic(topic, 30)))
            cnt += len(kind_set & words) * prob
        #count_list.append(cnt)
        if cnt > treshold:
            #print(comment, cnt)
            count_kind += 1
    return count_kind / len(comments)


# ### Emoji

# In[ ]:


def count_kind_emoji(emoji_list):
    kind_str = ":smiley:, :kissing_closed_eyes:, :wink:, :kissing_smiling_eyes:, :purple_heart:, :heartpulse:,     :sparkles:, :+1:, :clap:, :couplekiss:, :smile:, :relaxed:, :flushed:, :stuck_out_tongue_winking_eye:, :stuck_out_tongue:,     :heart:, :two_hearts:, :thumbsup:, :simple_smile:, :smirk:, :relieved:, :stuck_out_tongue_closed_eyes:, :green_heart:,     :revolving_hearts:, :laughing:, :heart_eyes:, :satisfied:, :grinning:, :yellow_heart:, :yum:, :blush:, :kissing_heart:,     :kissing:, :blue_heart:, :heartbeat:, :sparkling_heart:, :thumbs_up:"
    kind_emoji = set(map(lambda x: x.strip(), kind_str.split(', ')))
    emoji_list = list(map(lambda x: emoji.demojize(x), emoji_list))
    cnt = 0
    for emo in emoji_list:
        cnt += emo in kind_emoji
    return cnt / len(emoji_list)



# ### Smile

# In[ ]:


import cv2
import matplotlib.pyplot as plt
import urllib
from PIL import Image


# In[ ]:


def check_smile(url, face_classifier, smile_classifier):
    f = urllib.request.urlopen(url)
    Image.open(f).save("../storage/photo.png")
    image = cv2.imread("../storage/photo.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        smile = smile_classifier.detectMultiScale(roi_gray, scaleFactor=1.1)
        if len(smile) > 0:
            return 1
    return 0


# In[ ]:


def get_photo(connection, personal_id, postgre_str=POSTGRE_STR):
    pic = connection.execute('SELECT photo_200_orig FROM users WHERE uid = %s' % str(personal_id))
    return pic


# ### Активность

# In[ ]:


def count_activities(user_id, connection):
    inner_act = connection.execute('SELECT COUNT(DISTINCT id) FROM posts WHERE from_id = %d AND owner_id = from_id' % user_id)
    outer_act = connection.execute('SELECT COUNT(DISTINCT id) FROM posts WHERE from_id = %d AND owner_id != from_id' % user_id)
    return next(outer_act)[0], next(inner_act)[0]


# In[ ]:


def extract_emojis(text):
    return [c for c in ''.join(text) if c in emoji.UNICODE_EMOJI]


# ## Kindess index

# In[ ]:


def compute_kindness_index(uid, debug=False):
    connection = create_engine(POSTGRE_STR)
    lda, dictionary = get_lda_and_dict()
    stemmer = RussianStemmer()
    congrats_set = ['поздравляю праздник прошедшим наступающим грядущим новым годом светлой пасхи днем рождения рождеством восьмым марта                всего самого лучшего счастья здоровья успехов в личной жизни февраля женским днем пусть                сбудутся мечты желаю свершений радости пожелания умной хорошей доброй                ']
    congrats_set = set(list(map(lambda x: stemmer.stem(x), congrats_set[0].split())))
    like_set = [' нравится супер очень здорово круто обалдеть ничего себе вау молодец восхитительно                 замечательно классно чудесно']
    like_set = set(list(map(lambda x: stemmer.stem(x), like_set[0].split())))
    kind_sets = [(like_set, 'like', 0.4), (congrats_set, 'congrats', 2.)]
    
    # смотрим, улыбается ли пользователь на аватарке
    face_classifier = cv2.CascadeClassifier('../storage/haarcascade_frontalface_default.xml')
    smile_classifier = cv2.CascadeClassifier('../storage/haarcascade_smile.xml')
        
    
    if debug:
        print('----------%d----------' % uid)
    
    pics = get_photo(connection, uid)
    smiles = []
    for p in pics:
        url = p[0][:-6]
        smiles.append(check_smile(url, face_classifier, smile_classifier))
    if debug:
        print('smile', smiles)
    
    comments = [text[0] for text in get_comments(connection, uid)]
    if len(comments) == 0:
        index_list = [-1 for item in kind_sets]
        emo = -1
    else:
        # вычисляем, сколько хороших комментариев написал пользователь (%)
        index_list = []
        for kind_set, name, tres in kind_sets:
            index = count_kindness(lda, dictionary=dictionary, kind_set=kind_set, comments=comments, treshold=tres)
            index_list.append(index)
            if debug:
                print(name, index)
    
        # считаем, сколько хороших эмоджи пользователь употребил (%)
        emo = []
        for comment in comments:
            emo += extract_emojis(comment)
        if len(emo) != 0:
            emoji_index = count_kind_emoji(emo)
        else:
            emoji_index = 0

        if debug:
            print('emo', emoji_index)
    
    act = count_activities(uid, connection)
    act_ind = act[0] / (act[1] + 1)
    if debug:
        print('activity', act_ind)
    connection.close()
    return sum(index_list) + emoji_index + smiles[0] + act_ind