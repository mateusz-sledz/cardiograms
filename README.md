# Kardiogramy


Zbiór danych

Kardiotokografia to badanie monitorowania akcji serca płodu.
Automatycznie przetworzono 2126 kardiotokogramów płodowych (CTG) i zmierzono odpowiednie cechy diagnostyczne w liczbie 23. CTG zostały również sklasyfikowane przez trzech ekspertów położników i każdemu z nich przypisano etykietę konsensusu. Klasyfikacja dotyczyła zarówno wzoru morfologicznego (A, B, C...), jak i stanu płodowego (N, S, P).
My zajmujemy się tylko stanem płodowym.


Wśród naszych danych liczba płodów w poszczególnych grupach przedstawia się następująco:
- 1655 w grupie „normal” (77,8%)
- 295 w grupie „suspect” (13,9%)
- 176 w grupie „pathologic” (8,3%)


Las losowy 
Parametry zostały dobrane metodą random search z walidacją k-krzyżową. Lista strojonych parametrów:
- n_estimators
- max_features
- max_depth
- min_samples_split
- min_samples_leaf


Regresja logistyczna
Parametry zostały dobrane metodą grid search z walidacją k-krzyżową. Lista strojonych parametrów:
- parametr C
- solver
- max_iter
- class_weight

Podział danych w przypadku poszukiwań optymalnych hiperparametrów wynosił: 80% dane treningowe, 20% dane testowe.
