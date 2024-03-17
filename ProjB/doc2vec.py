from google.colab import drive
drive.mount('/content/drive')

##################################################################################################################
pip install nltk

nltk.download('stopwords')

import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as PS
from gensim.models import Word2Vec

#データセット読み込み
df = pd.read_csv('/content/drive/MyDrive/train_4class.csv')
df1 = pd.read_csv('/content/drive/MyDrive/test_4class.csv')
# l = []
# for i in df.itertuples():
#     l.append([i.label, i.Name1+i.Description1, i.Name2+i.Description2])

# df = pd.DataFrame(l,columns=['label','Description1','Description2'])
# df.to_csv('./train.csv')

# l = []
# for i in df1.itertuples():
#     l.append([i.label, i.Name1+i.Description1, i.Name2+i.Description2])

# df = pd.DataFrame(l,columns=['label','Description1','Description2'])
# df.to_csv('./test.csv')



#分かち書き
# trainings = [TaggedDocument(words = ".".join(body).split(), tags = [i])
#              for i, body in enumerate(df['Description1'].values.tolist() + df['Description2'].values.tolist())]
stopwords.words('english').append('attacker')

stopwords.words('english').append('attack')
stopwords.words('english').append('attacks')
stopwords.words('english').append('adversary')
stopwords.words('english').append('the')

trainings1 = [TaggedDocument(words = [word for word in body.split(' ') if word not in stopwords.words('english') ], tags = [i]) for i, body in enumerate(df['Description1'])]
trainings2 = [TaggedDocument(words = [word for word in body.split(' ') if word not in stopwords.words('english') ], tags = [i+2581]) for i, body in enumerate(df['Description2'])]

trainings = trainings1+trainings2
print(trainings)
# for i, body in enumerate(df['Description1']):
#     print(body)
#     a = body.split(' ')
#     s = [word for word in body.split(' ') if word not in stopwords.words('english')]
#     print(s)
#     print(a)
#     print(len(s), len(a))
#     break

# モデルの学習(dmpvで文書のベクトル化)
model = Doc2Vec(documents=trainings, dm=1, vector_size=300, window=5, min_count=1,epochs=150)
# model = Word2Vec(sentences=trainings, vector_size=300, window=5, min_count=5, workers=4)

# モデルの保存
model.save('/content/drive/MyDrive/doc2vec_4class.model')

#類似度判定
# m = Doc2Vec.load('/content/drive/MyDrive/dataset/doc2vec.model')

#引数ID文書と高類似度上位10件
# print(m.docvecs.most_similar(32))

#引数1と引数2文書の類似度
#print(m.docvecs.similarity(1, 365)) 

##################################################################################################################
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import ensemble, model_selection
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('/content/drive/MyDrive/train_4class.csv')
df_validation = pd.read_csv('/content/drive/MyDrive/test_4class.csv')

def set_d2v_vector(dataframe, d2v_instance, dim):
    
    df_tmp = dataframe.copy()
    # d2v_instance.random.seed(0)

    stopwords.words('english').append('attacker')

    stopwords.words('english').append('attack')
    stopwords.words('english').append('attacks')
    stopwords.words('english').append('adversary')
    stopwords.words('english').append('the')

    # doc2vec でベクトル化するには文書を単語のリストとして保つ必要があるので、変形する
    df_tmp['bind1'] = df_tmp['Description1']
    v1=[]
    df_tmp['doc_words1'] = [[word for word in body.split(' ') if word not in stopwords.words('english')] for body in df_tmp['bind1']]
    # v1 = [d2v_instance.infer_vector(doc_words) for doc_words in df_tmp['doc_words1']]
    for doc_words in df_tmp['doc_words1']:
        d2v_instance.random.seed(0)
        v1.append(d2v_instance.infer_vector(doc_words))
    
    v2 = []
    df_tmp['bind2'] = df_tmp['Description2']
    df_tmp['doc_words2'] = [[word for word in body.split(' ') if word not in stopwords.words('english')] for body in df_tmp['bind2']]
    for doc_words in df_tmp['doc_words2']:
        d2v_instance.random.seed(0)
        v2.append(d2v_instance.infer_vector(doc_words))

    # 文書ベクトル作成
    df_tmp['vector'] = [np.concatenate([doc_words1, doc_words2], axis=0) for doc_words1, doc_words2 in zip(v1,v2)]
    # ベクトルの次元を圧縮
    print(df_tmp['vector'])
    # df_tmp = dimension_reduction(df_tmp, dim)

    # 不要なカラムを削除
    # del df_tmp['bind']
    del df_tmp['label']

    df_vecadd = pd.merge(dataframe, df_tmp, how='left', left_index=True, right_index=True)

    return df_vecadd

############################################################################
from sklearn.decomposition import PCA

def dimension_reduction(data, pca_dimension):
    
    # 文章ベクトルの次元圧縮
    pca_data = data.copy()
    pca = PCA(n_components=pca_dimension)
    vector = np.array([np.array(v) for v in pca_data['vector']])
    pca_vector = pca.fit_transform(vector)
    pca_data['pca_vector'] = [v for v in pca_vector]
    del pca_data['vector']
    pca_data.rename(columns={'pca_vector':'vector'}, inplace=True)

    return pca_data


############################################################################
from gensim.models.doc2vec import Doc2Vec

# モデルのロード
d2v_model_path = '/content/drive/MyDrive/doc2vec_4class.model'
d2v = Doc2Vec.load(d2v_model_path)

# 圧縮後の文書ベクトルの次元数
vector_dim = 20

# ベクトルデータ作成
train_data = set_d2v_vector(df_train, d2v, dim=vector_dim)
validation_data = set_d2v_vector(df_validation, d2v, dim=vector_dim)
print(train_data['vector'])
print(validation_data['vector'])

#############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from gensim.models import Word2Vec



# RandomForest のグリッドサーチで探索するパラメータ
# params = {
#     'criterion'   : ['gini', 'entropy'],
#     'n_estimators': [10, 100, 300, 500, 1000, 1500, 2000],
#     'max_depth'   : [3, 5, 7, 9, 11]
# }

def randomforest_calassifier(train_data, validation_data):

    X_train = np.array([np.array(v) for v in train_data['vector']])
    X_validation = np.array([np.array(v) for v in validation_data['vector']])
    y_train = np.array([np.array(i) for i in train_data['label']])
    y_validation = np.array([np.array(i) for i in validation_data['label']])
    print(X_train)
    print(y_train)
    print(X_validation)
    print(y_validation)

    # clf = ensemble.RandomForestClassifier()
    # grid_search = GridSearchCV(clf, param_grid=params, cv=4)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_score_)
    # print(grid_search.best_params_)

    # RandomForest モデル学習
    mod = RandomForestClassifier(n_estimators=1500, max_depth=11)
    mod.fit(X_train, y_train)

    # 予測, 正解率算出
    y_pred = mod.predict(X_validation)
    acc = accuracy_score(y_validation, y_pred)
    print("accuracy : ", end="")
    print(acc)
    print(confusion_matrix(y_validation, y_pred))

    print(y_validation)
    print(y_pred)


answer = randomforest_calassifier(train_data, validation_data)
