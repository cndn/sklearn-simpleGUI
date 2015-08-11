"""
=====================
Simple GUI of sklearn
=====================

"""
from __future__ import division

print(__doc__)

# Author: Peter Prettenhoer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.contour import ContourSet

import Tkinter as Tk
import tkFileDialog
import ttk
import sys
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score



class Model(object):
    """
    The model which holds the data
    """

    def __init__(self):
        self.train = []
        self.test = []
        self.CVsize = 0.2
        self.clf = None

    def fit(self):
        pass

    

class Model_SVM(object):
    def __init__(self,model,parameter = {"kernel" :"rbf", "C" : 5}):
        self.train = model.train
        self.test = model.test
        self.CVsize = model.CVsize
        self.clf = SVC(kernel=parameter["kernel"], C=parameter["C"])

    def fit(self):
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.clf.fit(self.X_train,self.y_train)
        print("fitted")

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score: " + str(self.clf.score(self.X_train,truth)))
        print ("f1: " + str(f1_score(truth,pre, average=None)))
        print ("AUC score: " + str(roc_auc_score(truth,pre)))

class Model_Adaboost(object):
    def __init__(self,model,parameter = {"n_estimators" : 50}):
        self.train = model.train
        self.test = model.test
        self.CVsize = model.CVsize
        self.clf = AdaBoostClassifier(n_estimators = parameter["n_estimators"])

    def fit(self):
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.clf.fit(self.X_train,self.y_train)
        print("fitted")

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score: " + str(self.clf.score(self.X_train,truth)))
        print ("f1: " + str(f1_score(truth,pre, average=None)))
        print ("AUC score: " + str(roc_auc_score(truth,pre)))

class Model_RF(object):
    def __init__(self,model,parameter = {"n_estimators" :30, "max_depth" :5, "max_features":10}):
        self.train = model.train
        self.test = model.test
        self.CVsize = model.CVsize
        self.clf = RandomForestClassifier(max_depth= 5, n_estimators=30, max_features=min(10,model.train.shape[1] - 1))

    def fit(self):
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.clf.fit(self.X_train,self.y_train)
        print("fitted")

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score: " + str(self.clf.score(self.X_train,truth)))
        print ("f1: " + str(f1_score(truth,pre, average=None)))
        print ("AUC score: " + str(roc_auc_score(truth,pre)))

class Model_KNN(object):
    def __init__(self,model,parameter = {"K":5}):
        self.train = model.train
        self.test = model.test
        self.CVsize = model.CVsize
        self.clf = KNeighborsClassifier(parameter["K"])

    def fit(self):
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.clf.fit(self.X_train,self.y_train)
        print("fitted")

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score: " + str(self.clf.score(self.X_train,truth)))
        print ("f1: " + str(f1_score(truth,pre, average=None)))
        print ("AUC score: " + str(roc_auc_score(truth,pre)))

class Controller(object):
    def __init__(self, model):
        self.model = model
        self.modelType = Tk.IntVar()
        self.parameter = {}
    def showFrame(self):
        frame = Tk.Frame(width=100,height=100,bg="")
        frame.tkraise()

    def fit(self):
        model_map = {0:"SVM", 1:"Adaboost", 2:"Random Forest"}
        # output_map = {0:"0/1 classification", 1:"probability", 2:"regression"}
        print(self.modelType.get())
        if self.modelType.get() == 0:
            if self.parameter != {}:
                self.model = Model_SVM(self.model,self.parameter)
            else:
                self.model = Model_SVM(self.model)

        elif self.modelType.get() == 1:
            if self.parameter != {}:
                self.model = Model_Adaboost(self.model,self.parameter)
            else:
                self.model = Model_Adaboost(self.model)
            
        elif self.modelType.get() == 2:
            if self.parameter != {}:
                self.model = Model_RF(self.model,self.parameter)
            else:
                self.model = Model_RF(self.model)

        elif self.modelType.get() == 3:
            if self.parameter != {}:
                self.model = Model_KNN(self.model,self.parameter)
            else:
                self.model = Model_KNN(self.model)
        self.model.fit()
        self.model.score()

    def clear_data(self):
        self.model.data = []
        self.fitted = False
        self.model.changed("clear")

    def save_results(self):

        pre = self.model.clf.predict(self.model.test)
        df = pd.DataFrame({"predict":pre})
        fileName = tkFileDialog.asksaveasfilename()
        df.to_csv(fileName)

    def loadTrainData(self):
        fileName = tkFileDialog.askopenfilename()
        self.model.train = pd.read_csv(str(fileName))
        print("Train data has been loaded")
        print("Shape: " + str(self.model.train.shape))

    def loadTestData(self):
        fileName = tkFileDialog.askopenfilename()
        self.model.test = pd.read_csv(str(fileName))
        print("Test data has been loaded")
        print("Shape: " + str(self.model.test.shape))

class View(object):
    """Test docstring. """
    def __init__(self, root, controller):
        f = Figure()
        self.controllbar = ControllBar(root, controller)
        self.f = f
        self.controller = controller
        self.contours = []
        self.c_labels = None




    def update(self, event, model):
        self.canvas.draw()




class ControllBar(object):
    def __init__(self, root, controller):
        fm = Tk.Frame(root)
        
        file_group = Tk.Frame(fm)
        Tk.Button(file_group, text="train",
                       command=controller.loadTrainData).pack(anchor=Tk.W)
        Tk.Button(file_group, text="test",
                       command=controller.loadTestData).pack(anchor=Tk.W)
        file_group.pack(side=Tk.LEFT)

        model_group = Tk.Frame(fm)
        # self.box = ttk.Combobox(model_group, textvariable = Tk.StringVar(), values = ["SVM","Adaboost"])
        # self.box.bind("SVM",controller.showFrame)
        # self.box.pack()
        Tk.Radiobutton(model_group, text="SVM(0/1)", variable=controller.modelType,
                       value=0,command = controller.showFrame).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="Adaboost(0/1)", variable=controller.modelType,
                       value=1).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="Random Forest(0/1)", variable=controller.modelType,
                       value=2).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="KNN(0/1)", variable=controller.modelType,
                       value=3).pack(anchor=Tk.W)
        model_group.pack(side=Tk.LEFT)

        # output_group = Tk.Frame(fm)
        # Tk.Radiobutton(output_group, text="0/1 classification",
        #                variable=controller.model.output, value=0).pack(anchor=Tk.W)
        # Tk.Radiobutton(output_group, text="Probability",
        #                variable=controller.model.output, value=1).pack(anchor=Tk.W)
        # Tk.Radiobutton(output_group, text="Regression",
        #                variable=controller.model.output, value=2).pack(anchor=Tk.W)
        # output_group.pack(side=Tk.LEFT)

        Tk.Button(fm, text='Fit', width=5, command=controller.fit).pack()
        Tk.Button(fm, text='Save Results', width=10, command=controller.save_results).pack()
        fm.pack(side=Tk.LEFT)




def main(argv):
    
    root = Tk.Tk()
    model = Model()
    controller = Controller(model)
    root.wm_title("Scikit-learn GUI")
    view = View(root, controller)
    Tk.mainloop()


if __name__ == "__main__":
    main(sys.argv)
