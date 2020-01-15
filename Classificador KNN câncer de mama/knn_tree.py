
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn import tree
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#carregamos o dataset
breast_cancer = load_breast_cancer()
# print(breast_cancer.feature_names)


#definimos como x os dados e como y o alvo

X = breast_cancer.data
y = breast_cancer.target
  
#aqui definimos a porcentagem que teremos de teste das amostras
#e consequentemente, a porcentagem de treino

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

#definimos uma variavel para o knn, e treinamos os dados de treino de acordo com o knn

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)

#calculamos sua acuracia

print("\nAcuracia KNN    = "+str(accuracy_score(y_test, clf_knn.predict(X_test))))

#definimos uma variavel para a arvore, e treinamos os dados de treino de acordo com a arvore

# knn = tree.DecisionTreeClassifier()
# knn.fit(X_train, y_train)

#calculamos sua acuracia

# print("Acuracia Arvore = "+str(accuracy_score(y_test, knn.predict(X_test))) + "\n")

error = []

# Calculating error for K values between 1 and 5 com passo 2
for i in range(1, 6, 2):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    classifier = KNeighborsClassifier(n_neighbors=i)  
    classifier.fit(X_train, y_train)

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
            metric_params=None, n_jobs=None, n_neighbors=i, p=2,
            weights='uniform')
    #   manhattan; hamming; "euclidean" ?

    # Aplicando os valores de teste novamente
    y_pred = classifier.predict(X_test)

    # Importando métricas para validação do modelo
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    # Imprimindo o relatório de classificação
    print("Relatório de classificação com K = ", i, ":\n", classification_report(y_test, y_pred))  

    # Imprimindo o quão acurado foi o modelo
    print('Acurácia do modelo: ' , accuracy_score(y_test, y_pred))