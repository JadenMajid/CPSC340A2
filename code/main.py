#!/usr/bin/env python
import argparse
from operator import indexOf
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree
import sklearn 


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    k_vals = [1,3,10]
    for k in k_vals:
        model = KNN(k)
        print("k:",k)
        model.fit(X,y)
        y_hat_train = model.predict(X)
        train_error=np.mean(y_hat_train!=model.y)

        print("train error:", train_error)
        y_hat_test = model.predict(X_test)
        test_error=np.mean(y_hat_test!=y_test)
        print("test_error:",test_error)
        
        plot_classifier(model, X, y)
        plt.title(f"knn @ k={k}")
        plt.savefig(f"../figs/q1_knn_{k}.pdf")
        plt.savefig(f"../figs/q1_knn_{k}.png")
        print()
        
        # SKLEARN KNN
        sk_model = KNeighborsClassifier(n_neighbors=k)
        sk_model.fit(X, y)
        y_hat_train_sk = sk_model.predict(X)
        train_error_sk = np.mean(y_hat_train_sk != y)
        print(f"[sklearn] train error: {train_error_sk}")
        y_hat_test_sk = sk_model.predict(X_test)
        test_error_sk = np.mean(y_hat_test_sk != y_test)
        print(f"[sklearn] test error: {test_error_sk}")
        # Plot sklearn classifier
        def sk_predict(xx):
            return sk_model.predict(xx)
        plot_classifier(sk_model, X, y)
        plt.title(f"sklearn knn @ k={k}")
        plt.savefig(f"../figs/q1_knn_{k}_sklearn.pdf")
        plt.savefig(f"../figs/q1_knn_{k}_sklearn.png")
        print()


    
    


@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    errors_validation = []
    errors_training = []
    

    for k in ks:
        n_folds = 10
        total_error_validation = 0;
        total_error_training = 0;
        for n in range(n_folds):
            train_mask = np.ones(len(X), dtype=bool) 
            train_mask[n*len(X)//n_folds:(n+1)*len(X)//n_folds] = 0
            validation_mask = ~train_mask
            
            model = KNN(k)
            model.fit(X[train_mask],y[train_mask])
            y_hat_train = model.predict(X[train_mask])
            train_error=np.mean(y_hat_train!=y[train_mask])
            total_error_training+=train_error
            
            y_hat_test = model.predict(X[validation_mask])
            test_error=np.mean(y_hat_test!=y[validation_mask])
            total_error_validation+=test_error
            
        avg_test_error = total_error_validation/n_folds
        avg_training_error = total_error_training/n_folds
        errors_validation.append(avg_test_error)
        errors_training.append(avg_training_error)
        print("k:",k)
        print("average training error:",avg_training_error)
        print("average validation error:",avg_test_error)
        print()
    
    errors_validation = np.array(errors_validation)
    plt.plot(ks, errors_validation)
    plt.xlabel("k")
    plt.ylabel("average error")
    plt.title(f"k vs average testing error in {n_folds}-folds")
    plt.savefig("../figs/q2_k_errors.pdf")
    plt.savefig("../figs/q2_k_errors.png")
    
    plt.cla()
    
    plt.plot(ks, -1*errors_validation+1)
    plt.xlabel("k")
    plt.ylabel("average accuracy")
    plt.title(f"k vs average testing accuracy in {n_folds}-folds")
    plt.savefig("../figs/q2_k_accuracy.pdf")
    plt.savefig("../figs/q2_k_accuracy.png")
    
    plt.cla()    
    
    errors_training = np.array(errors_training)
    plt.plot(ks, -1*errors_training+1)
    plt.xlabel("k")
    plt.ylabel("average accuracy")
    plt.title(f"k vs average training accuracy in {n_folds}-folds")
    plt.savefig("../figs/q2_k_accuracy.pdf")
    plt.savefig("../figs/q2_k_accuracy.png")
    
    
        



@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    print("word 73: ", wordlist[72])
    print("training example 803: ", wordlist[np.where(X[802])])
    print("newsgroup name:", groupnames[y[802]])



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""
    raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    raise NotImplementedError()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    raise NotImplementedError()



if __name__ == "__main__":
    main()
