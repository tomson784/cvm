# 確率分布のパラメータ推定

## 最尤推定(Maximum likelyhood estimation)

パラメータ<img src="https://latex.codecogs.com/gif.latex?\theta" />という条件において<img src="https://latex.codecogs.com/gif.latex?\mbox{\boldmath $x$}" />が生成される確率

確率が最も大きくなる時の<img src="https://latex.codecogs.com/gif.latex?\theta" />を導出することで<img src="https://latex.codecogs.com/gif.latex?\mbox{\boldmath $x$}" />の生成ルールを推論する

## MAP推定(Maximum a posteriori)

パラメータ<img src="https://latex.codecogs.com/gif.latex?\theta" />の事前分布を考慮し、尤度との積により事後確率を導出。

参考書の例では事前分布が逆ガンマ分布、尤度がガウス分布なので、事後分布はガウス分布となる

事後確率が最も大きくなる時の<img src="https://latex.codecogs.com/gif.latex?\theta" />を導出する


## ベイズ推定(Bayesian Approach)