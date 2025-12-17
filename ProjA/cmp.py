#ユークリッド距離を用いた類似度判定
import numpy as np

#オリジナル1
o1 = open("original.xyz", "r").readlines()    # 1行毎にリストを作成
O1 = o1[0]                            
xo1, yo1, zo1 = [], [], []
for l in o1:
    data = l.split()                            # スペース区切りで読み込み
    xo1.append(float(data[0]))                  # x座標
    yo1.append(float(data[1]))                  # y座標
    zo1.append(float(data[2]))                  # z座標

#モデルA1
a1 = open("A1.xyz", "r").readlines()    # 1行毎にリストを作成
A1 = a1[0]                            
xa1, ya1, za1 = [], [], []
for l in a1:
    data = l.split()                          # スペース区切りで読み込み 
    xa1.append(float(data[0]))                  # x座標
    ya1.append(float(data[1]))                  # y座標
    za1.append(float(data[2]))                  # z座標

#モデルB1
b1 = open("B1.xyz", "r").readlines()    # 1行毎にリストを作成
B1 = b1[0]                            
xb1, yb1, zb1 = [], [], []
for l in m1:
    data = l.split()                          # スペース区切りで読み込み 
    xb1.append(float(data[0]))                  # x座標
    yb1.append(float(data[1]))                  # y座標
    zb1.append(float(data[2]))                  # z座標





if(xa1[0] != xo1[0]):
    print("wwwwwwwww")  #検証用




o = open("comparison.csv", "w", encoding="cp932")  # 書き込み用の記述子作成, csvはcp932で作成
for a,b,c,d,e,f in zip(xo1, yo1, zo1,xa1,ya1,za1):
    fmt = "{},{},{},{},{},{}\n"
    o.write(fmt.format(a,b,c,d,e,f))           # .write()で書き込み
print("original1.csv was created.")               # 書き込み終了の表示

