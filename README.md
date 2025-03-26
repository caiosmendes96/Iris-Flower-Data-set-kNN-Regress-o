# Iris-Flower-Data-set-kNN-Regressão
Utilização do banco de dados “Iris Flower Data Set” para abordar o problema de regressão utilizando o algoritmo k-NN em Python


Importação do banco de dados "winequality" para a regressão utilizando o algoritmo KNN euclidiano.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from statistics import mean
from sklearn.metrics import accuracy_score 
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import mean_squared_error
dadosVinho = pd.read_csv('./winequality-red.csv')

print("Quantidade de objetos do banco de dados:", dadosVinho.shape[0])
print("Quantidade de atributos do banco de dados:", dadosVinho.shape[1])

dadosVinho.sample(5)
```

    Quantidade de objetos do banco de dados: 1599
    Quantidade de atributos do banco de dados: 12
    




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>269</th>
      <td>11.5</td>
      <td>0.180</td>
      <td>0.51</td>
      <td>4.0</td>
      <td>0.104</td>
      <td>4.0</td>
      <td>23.0</td>
      <td>0.99960</td>
      <td>3.28</td>
      <td>0.97</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>878</th>
      <td>8.8</td>
      <td>0.610</td>
      <td>0.19</td>
      <td>4.0</td>
      <td>0.094</td>
      <td>30.0</td>
      <td>69.0</td>
      <td>0.99787</td>
      <td>3.22</td>
      <td>0.50</td>
      <td>10.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>63</th>
      <td>7.0</td>
      <td>0.735</td>
      <td>0.05</td>
      <td>2.0</td>
      <td>0.081</td>
      <td>13.0</td>
      <td>54.0</td>
      <td>0.99660</td>
      <td>3.39</td>
      <td>0.57</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1317</th>
      <td>9.9</td>
      <td>0.440</td>
      <td>0.46</td>
      <td>2.2</td>
      <td>0.091</td>
      <td>10.0</td>
      <td>41.0</td>
      <td>0.99638</td>
      <td>3.18</td>
      <td>0.69</td>
      <td>11.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1034</th>
      <td>8.9</td>
      <td>0.745</td>
      <td>0.18</td>
      <td>2.5</td>
      <td>0.077</td>
      <td>15.0</td>
      <td>48.0</td>
      <td>0.99739</td>
      <td>3.20</td>
      <td>0.47</td>
      <td>9.7</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



<b> Atributo alvo: quality </b>

Para o treinamento do KNN, utilizamos 80% objetos para o treinamento e 20% para teste. </br></br>
O parâmetro "random_state" em train_test_split  é responsável por randomizar a seleção dos objetos escolhidos para o treinamento e teste. </br></br>
Ao alterar o valor do "random_state", os objetos de treinamento e teste serão diferentes. Com isso, os valores de erro e o melhor K poderão mudar. </br></br>
Para esse experimento, utilizamos o "random_state" igual a 0. </br>

Vamos procurar o melhor valor de K (hiperparâmetro) com os objetos de treinamento ao encontrar o menor erro quadrático médio (EQM) variando o valor de vizinhos entre 1 e 20.  </br>


```python
x = dadosVinho.drop(['quality'], axis = 1)
y = dadosVinho['quality']

train_X, test_X, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("Quantidade de objetos para o treino: ", len(train_X))
print("Quantidade de objetos para o teste: ", len(test_X))
print("-------------------------------------")

listaErros = []
for n in range(1,20):
    
    knn = neighbors.KNeighborsRegressor(n_neighbors = n)
    knn.fit(train_X, train_y)
    predicaoTeste = knn.predict(test_X)
    
    listaErros.append(mean_squared_error(test_y, predicaoTeste))
    

melhorK = listaErros.index(min(listaErros)) + 1

print("Menor erro quadrático médio:", min(listaErros))
print("Valor de k:", melhorK)
print("-------------------------------------")

```

    Quantidade de objetos para o treino:  1279
    Quantidade de objetos para o teste:  320
    -------------------------------------
    Menor erro quadrático médio: 0.47311288088642656
    Valor de k: 19
    -------------------------------------
    

<b>Apresentação de um exemplo da regressão: </b>


```python
melhorKnn = neighbors.KNeighborsRegressor(n_neighbors = melhorK)
melhorKnn.fit(train_X, train_y)

#Variáveis para uma predição utilizando regressão

#Dados do objeto nº 269

fixedAcidity = 11.5
volatileAcidity = 0.180
citricAcid = 0.51
residualSugar = 4.0
chlorides = 0.104
freeSulfurDioxide = 4.0
totalSulfurDioxide = 23.0
density = 0.99960
pH = 3.28
sulphates = 0.97
alcohol = 10.1

predicao = melhorKnn.predict([[fixedAcidity,volatileAcidity,citricAcid,residualSugar,
                                 chlorides,freeSulfurDioxide,totalSulfurDioxide,
                                 density,pH,sulphates,alcohol],])
print("Regressão com o melhor valor de K: ", melhorK)
print("Qualidade do vinho (tabela): ", y[269])
print("Qualidade do vinho (resultado regressão KNN): ", predicao[0])

```

    Regressão com o melhor valor de K:  19
    Qualidade do vinho (tabela):  6
    Qualidade do vinho (resultado regressão KNN):  6.052631578947368
    

    C:\Users\caios\anaconda3\Lib\site-packages\sklearn\base.py:464: UserWarning: X does not have valid feature names, but KNeighborsRegressor was fitted with feature names
      warnings.warn(
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
