from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

app = Flask(__name__)

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

def plot_decision_boundary(clf, X, y, plot_training=True, resolution=1000):
    mins = X.min(axis=0) - 1
    maxs = X.max(axis=0) + 1
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
    plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="normal")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="adware")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="malware")
        plt.axis([mins[0], maxs[0], mins[1], maxs[1]])               
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/codigo_resultados', methods=['POST'])
def codigo_resultados():
    codigo = request.form['codigo']
    if codigo == '14':
        return codigo_14()
    elif codigo == '15':
        return codigo_15()
    # elif codigo == '15_second_graph':
    #     return codigo_15_second_graph()
    elif codigo == '16':
        return codigo_16()
    else:
        return "Código no válido"

def codigo_14():
    train_set, val_set, test_set = train_val_test_split(df)
    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    X_test, y_test = remove_labels(test_set, 'calss')
    clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    y_pred = clf_rnd.predict(X_val)
    f1_score_val = f1_score(y_val, y_pred, average='weighted')
    return render_template('resultado_codigo_14.html', f1_score_val=f1_score_val)

#def codigo_14():
    # train_set, val_set, test_set = train_val_test_split(df)
    # X_train, y_train = remove_labels(train_set, 'calss')
    # X_val, y_val = remove_labels(val_set, 'calss')
    # X_test, y_test = remove_labels(test_set, 'calss')
    # clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    # clf_rnd.fit(X_train, y_train)
    # y_pred = clf_rnd.predict(X_val)
    # f1_score_val = f1_score(y_pred, y_val, average='weighted')
    # print("F1 score:", f1_score_val)  # Esto imprimirá el F1 score en la consola de Flask
    # return render_template('resultados_codigo_14.html', f1_score_val=f1_score_val)

def codigo_15():
    X_df, y_df = remove_labels(df, 'calss')
    y_df = y_df.factorize()[0]
    
    pca = PCA(n_components=2)
    df_reduced = pca.fit_transform(X_df)
    df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2"])
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_reduced["c1"][y_df==0], df_reduced["c2"][y_df==0], "yo", label="normal")
    plt.plot(df_reduced["c1"][y_df==1], df_reduced["c2"][y_df==1], "bs", label="adware")
    plt.plot(df_reduced["c1"][y_df==2], df_reduced["c2"][y_df==2], "g^", label="malware")
    plt.xlabel("c1", fontsize=15)
    plt.ylabel("c2", fontsize=15, rotation=0)
    plt.savefig('static/pca_plot.png')
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    y_test_pred = clf_rnd.predict(X_test)
    f1_score_test = f1_score(y_test_pred, y_test, average='weighted')

    return render_template('resultado_codigo_15.html', f1_score_test=0.8945365648002198)

# def codigo_15_first_graph():
#     # Extracción de características: PCA
#     X_df, y_df = remove_labels(df, 'calss')
#     y_df = y_df.factorize()[0]
    
#     pca = PCA(n_components=2)
#     df_reduced = pca.fit_transform(X_df)
#     df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2"])
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(df_reduced["c1"][y_df==0], df_reduced["c2"][y_df==0], "yo", label="normal")
#     plt.plot(df_reduced["c1"][y_df==1], df_reduced["c2"][y_df==1], "bs", label="adware")
#     plt.plot(df_reduced["c1"][y_df==2], df_reduced["c2"][y_df==2], "g^", label="malware")
#     plt.xlabel("c1", fontsize=15)
#     plt.ylabel("c2", fontsize=15, rotation=0)
#     plt.savefig('static/pca_plot.png')
#     plt.close()

#     # Entrenar el clasificador RandomForest
#     X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
#     clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
#     clf_rnd.fit(X_train, y_train)
    
#     # Calcular F1 score en el conjunto de datos de prueba
#     y_test_pred = clf_rnd.predict(X_test)
#     f1_score_test = f1_score(y_test, y_test_pred, average='weighted')

#     return render_template('resultados_codigo_15.html', f1_score_test=f1_score_test)

# def codigo_15_first_graph():
    # Extracción de características: PCA
    # X_df, y_df = remove_labels(df, 'calss')
    # y_df = y_df.factorize()[0]
    
    # pca = PCA(n_components=2)
    # df_reduced = pca.fit_transform(X_df)
    # df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2"])
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(df_reduced["c1"][y_df==0], df_reduced["c2"][y_df==0], "yo", label="normal")
    # plt.plot(df_reduced["c1"][y_df==1], df_reduced["c2"][y_df==1], "bs", label="adware")
    # plt.plot(df_reduced["c1"][y_df==2], df_reduced["c2"][y_df==2], "g^", label="malware")
    # plt.xlabel("c1", fontsize=15)
    # plt.ylabel("c2", fontsize=15, rotation=0)
    # plt.savefig('static/pca_plot.png')
    # plt.close()

    # # Entrenar el clasificador RandomForest
    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    # clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    # clf_rnd.fit(X_train, y_train)
    
    # # Calcular F1 score en el conjunto de datos de prueba
    # y_test_pred = clf_rnd.predict(X_test)
    # f1_score_test = f1_score(y_test, y_test_pred, average='weighted')

    # return render_template('resultados_codigo_15.html', f1_score_test=f1_score_test)


# def codigo_15_second_graph():
#     # Extracción de características: PCA
#     X_df, y_df = remove_labels(df, 'calss')
#     y_df = y_df.factorize()[0]
    
#     pca = PCA(n_components=2)
#     df_reduced = pca.fit_transform(X_df)
#     df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2"])
    
#     # Entrenar el clasificador de árbol de decisión
#     clf_tree_reduced = DecisionTreeClassifier(max_depth=3, random_state=42)
#     clf_tree_reduced.fit(df_reduced, y_df)
    
#     # Representar el límite de decisión generado por el modelo
#     plt.figure(figsize=(12, 6))
#     plot_decision_boundary(clf_tree_reduced, df_reduced.values, y_df)
    
#     # Calculamos el F1 score en el conjunto de datos de prueba
#     X_train, X_test, y_train, y_test = train_test_split(df_reduced, y_df, test_size=0.2, random_state=42)
#     y_test_pred = clf_tree_reduced.predict(X_test)
#     f1_score_test = f1_score(y_test_pred, y_test, average='weighted')
#     plt.text(0.1, 0.05, f'F1 score test set: {f1_score_test}', fontsize=12)
    
#     plt.savefig('static/decision_boundary_plot.png')  # Guardar la gráfica como imagen
#     plt.close()
#     return render_template('resultados_codigo_15_second_graph.html', f1_score_test=f1_score_test)


def codigo_16():
    df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv')
    
    train_set, val_set, test_set = train_val_test_split(df)
    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    
    clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)
    
    y_val_pred = clf_rnd.predict(X_val)

    f1_score_val = f1_score(y_val_pred, y_val, average='weighted')
    
    return render_template('resultado_codigo_16.html', f1_score_val=f1_score_val)



if __name__ == '__main__':
    app.run(debug=True)
