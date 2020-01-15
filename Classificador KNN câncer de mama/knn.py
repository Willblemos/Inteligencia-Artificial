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

#carregando do dataset
breast_cancer = load_breast_cancer()

#definimos como x os dados e como y o alvo
X = breast_cancer.data
y = breast_cancer.target

# Calculando erro para K valores entre 1 e 5 pulando 2 (ou seja, 1,3,5). 
for i in range(1, 6, 2):
    #Definição da porcentagem para treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
            metric_params=None, n_jobs=None, n_neighbors=i, p=2,
            weights='uniform')
    #manhattan; euclidean;
    error = []

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    
    # Aplicando os valores de teste novamente
    y_pred = classifier.predict(X_test)

    # Importando métricas para validação do modelo
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    # Imprimindo o relatório de classificação
    print("\nRelatório de classificação com K = ", i, ":\n", classification_report(y_test, y_pred))
    print("\nAcurácia KNN com K =",i, "= "+str(accuracy_score(y_test, knn.predict(X_test))))
 