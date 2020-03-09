from loadData import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import random
import statistics


def compare(features, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=120, bootstrap=False)

    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)

    predictions = predictions.round()
    predictions = predictions.astype(int)
    num_false = (predictions == test_labels).sum()
    rf_acc = 100 * round(num_false / len(predictions), 4)
    print("Precyzja lasu: ", rf_acc, " %")

    model = LogisticRegression(solver='newton-cg', class_weight=None, C=10, multi_class='auto', random_state=42,
                               max_iter=400)

    model.fit(train_features, train_labels)

    predictions = model.predict(test_features)

    predictions = predictions.round()
    predictions = predictions.astype(int)

    num_false = (predictions == test_labels).sum()
    lr_acc = 100 * round(num_false / len(predictions), 4)

    print("Precyzja regresji: ", lr_acc, " %")

    print('Różnica: ', round(rf_acc - lr_acc, 2), '%')


def forest_search(features, labels):

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)
    n_estimators = [int(x) for x in np.linspace(start=100, stop=900, num=7)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(30, 150, num=7)]
    max_depth.append(None)
    min_samples_split = [2, 5]
    min_samples_leaf = [1, 2, 4]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf
                   }

    rf = RandomForestRegressor()

    model = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, random_state=42, n_jobs=-1)

    model = model.fit(train_features, train_labels)

    print('Najlepsze parametry: ', model.best_params_)

    predictions = model.predict(test_features)
    predictions = predictions.round()
    predictions = predictions.astype(int)
    num_false = (predictions == test_labels).sum()

    print("Precyzja: ", 100 * round(num_false / len(predictions), 4), " %")


def regression_search(features, labels):

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)
    C = [0.01, 0.1, 1, 10, 100]
    solver = ['liblinear', 'newton-cg']
    max_iterations = [200, 400, 600, 800, 1000, 2000]
    class_weight = ['balanced', None]
    param_grid = dict(
        C=C,
        solver=solver,
        max_iter=max_iterations,
        class_weight=class_weight
    )

    grid = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring='accuracy', n_jobs=-1)

    model = grid.fit(train_features, train_labels)

    print('Najlepsze parametry: ', model.best_params_)

    print('Precyzja: ', round(model.score(test_features, test_labels), 4)*100, '%')


def multiple(features, labels):
    x = input('Ile razy: ')
    forest = []
    regression = []
    for p in range(int(x)):
        rand = random.randint(177, 10000)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                    random_state=rand)

        rf = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_split=2, min_samples_leaf=1,
                                   max_features='sqrt', max_depth=120, bootstrap=False)

        rf.fit(train_features, train_labels)
        predictions = rf.predict(test_features)

        predictions = predictions.round()
        predictions = predictions.astype(int)
        num_false = (predictions == test_labels).sum()
        rf_acc = 100 * round(num_false / len(predictions), 6)
        forest.append(rf_acc)

        model = LogisticRegression(solver='newton-cg', class_weight=None, C=10, multi_class='auto', random_state=rand,
                                   max_iter=400)

        model.fit(train_features, train_labels)

        predictions = model.predict(test_features)

        predictions = predictions.round()
        predictions = predictions.astype(int)

        num_false = (predictions == test_labels).sum()
        lr_acc = 100 * round(num_false / len(predictions), 4)
        regression.append(lr_acc)

    print('Las losowy, średnia: ', statistics.mean(forest), '\n', 'Regresja logistyczna, średnia: ', statistics.mean(regression))


while True:
    print('Wybierz jedną z funkcjonalności:\n1. Porównanie lasu losowego oraz regresji logistycznej\n2.'
          ' Szukanie najlepszych parametrów lasu\n3. Szukanie najlepszych parametrów regresji logistycznej '
          '\n4. Średnia skuteczność z n wykonań \n5. Zakończ')
    choice = input('Wybór:')
    choice = int(choice)

    features, labels = load_data()

    if choice == 1:
        compare(features, labels)
    elif choice == 2:
        forest_search(features, labels)
    elif choice == 3:
        regression_search(features, labels)
    elif choice == 5:
        break
    elif choice == 4:
        multiple(features, labels)
    else:
        continue
