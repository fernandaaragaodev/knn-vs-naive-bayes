from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


iris = load_iris()
features = iris.data
classe = iris.target


print("--- Avaliando o KNN ---")
melhor_k = 0
melhor_media_knn = 0

for n in range(1, 6):
    classificador_knn = KNeighborsClassifier(n_neighbors=n)
   
    resultados_knn = cross_val_score(estimator = classificador_knn,
                                     X = features, y = classe,
                                     cv = 5, scoring = 'accuracy')
   
    media = resultados_knn.mean()
    desvio = resultados_knn.std()
   
    print(f"K = {n} | Acurácia Média: {media:.4f} (Desvio: {desvio:.4f})")
   
    if media > melhor_media_knn:
        melhor_media_knn = media
        melhor_k = n


print("\n-|- Avaliando o Naive Bayes -|-")
classificador_nb = GaussianNB()

resultados_nb = cross_val_score(estimator = classificador_nb,
                                X = features, y = classe,
                                cv = 5, scoring = 'accuracy')

media_nb = resultados_nb.mean()
desvio_nb = resultados_nb.std()

print(f"Naive Bayes | Acurácia Média: {media_nb:.4f} (Desvio: {desvio_nb:.4f})")


print("\n=|= Veredito Final =|=")
if melhor_media_knn > media_nb:
    print(f"🏆 O KNN venceu usando K={melhor_k} com uma acurácia de {melhor_media_knn:.2%}!")
elif media_nb > melhor_media_knn:
    print(f"O Naive Bayes venceu com uma acurácia de {media_nb:.2%}!")
else:
    print(f"Empate! Ambos alcançaram uma acurácia de {media_nb:.2%}.")