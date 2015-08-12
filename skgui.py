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
        self.CVsize = Tk.StringVar()
        self.clf = None

    def fit(self):
        pass

    

class Model_SVM(object):
    def __init__(self,model,parameter = {"kernel" :"rbf", "C" : 5, "gamma": 1, "poly degree": 3, "CV_size": 0}):
        self.train = model.train
        self.test = model.test
        self.CVsize = float(parameter["CV_size"].get())
        self.clf = SVC(kernel=parameter["kernel"].get(), C = float(parameter["C"].get()), gamma = float(parameter["gamma"].get()))
        self.model = model


    def fit(self):

        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.X_train,self.X_CV,self.y_train,self.y_CV = train_test_split(self.X_train, self.y_train, test_size=self.CVsize)
        self.clf.fit(self.X_train,self.y_train)

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score on training set: " + str(self.clf.score(self.X_train,truth)))
        print ("f1 on training set: " + str(f1_score(truth,pre, average=None)))
        print ("AUC score on training set: " + str(roc_auc_score(truth,pre)))
        if self.CVsize != 0:
            pre = self.clf.predict(self.X_CV)
            truth = self.y_CV
            print ("score on Cross Validation set: " + str(self.clf.score(self.X_CV,truth)))
            print ("f1 on Cross Validation set: " + str(f1_score(truth,pre, average=None)))
            print ("AUC score on Cross Validation set: " + str(roc_auc_score(truth,pre)))

class Model_Adaboost(object):
    def __init__(self,model,parameter = {"n_estimators" : 50, "CV_size": 0}):
        self.train = model.train
        self.test = model.test
        self.clf = AdaBoostClassifier(n_estimators = int(parameter["n_estimators"].get()))

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
    def __init__(self,model,parameter = {"n_estimators" :30, "max_depth" :5, "max_features":10, "CV_size": 0}):
        self.train = model.train
        self.test = model.test
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
        self.clf = KNeighborsClassifier(int(parameter["K"].get()))

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
        self.isShown = False
        self.frame = Tk.Toplevel()
        self.frame.wm_title("Parameter")

    def showFrameHelper(self):
        if self.isShown == False:
            self.isShown = True
            self.showFrame()
        else:
            self.param_group.pack_forget()
            self.showFrame()

    def showFrame(self):
        
        self.parameter["CV_size"] = Tk.StringVar()
        if self.modelType.get() == 0:
            
            self.parameter["kernel"] = Tk.StringVar()
            self.parameter["C"] = Tk.StringVar()
            self.parameter["gamma"] = Tk.StringVar()
            self.parameter["degree"] = Tk.StringVar()
            self.param_group = Tk.Frame(self.frame)
            Tk.Radiobutton(self.param_group, text="linear", variable=self.parameter["kernel"],
                           value="linear").pack(anchor=Tk.W)
            Tk.Radiobutton(self.param_group, text="rbf", variable=self.parameter["kernel"],
                           value="rbf").pack(anchor=Tk.W)
            Tk.Radiobutton(self.param_group, text="poly", variable=self.parameter["kernel"],
                           value="poly").pack(anchor=Tk.W)
            Tk.Label(self.param_group, text = "C").pack()
            Tk.Entry(self.param_group, textvariable = self.parameter["C"]).pack()
            Tk.Label(self.param_group, text = "gamma").pack()
            Tk.Entry(self.param_group, textvariable = self.parameter["gamma"]).pack()
            Tk.Label(self.param_group, text = "degree").pack()
            Tk.Entry(self.param_group, textvariable = self.parameter["degree"]).pack()
            

        if self.modelType.get() == 1:

            self.parameter["n_estimators"] = Tk.StringVar()
            self.param_group = Tk.Frame(self.frame)
            Tk.Label(self.param_group, text = "n_estimators").pack()
            Tk.Entry(self.param_group, textvariable = self.parameter["n_estimators"]).pack()
            

        if self.modelType.get() == 2:

            self.parameter["n_estimators"] = Tk.StringVar()
            self.parameter["max_depth"] = Tk.StringVar()
            self.parameter["max_features"] = Tk.StringVar()
            self.param_group = Tk.Frame(self.frame)
            Tk.Label(self.param_group, text = "n_estimators").pack()
            Tk.Entry(self.param_group, textvariable = self.parameter["n_estimators"]).pack()
            Tk.Label(self.param_group, text = "max_depth").pack()
            Tk.Entry(self.param_group, textvariable = self.parameter["max_depth"]).pack()
            Tk.Label(self.param_group, text = "max_features").pack()
            Tk.Entry(self.param_group, textvariable = self.parameter["max_features"]).pack()
            

        if self.modelType.get() == 3:
            self.parameter["K"] = Tk.StringVar()
            self.param_group = Tk.Frame(self.frame)
            Tk.Label(self.param_group, text = "K").pack()
            Tk.Entry(self.param_group, textvariable = self.parameter["K"]).pack()

        
        Tk.Label(self.param_group, text = "Cross Validation Size").pack()
        Tk.Label(self.param_group, text = "Set it to 0 if no need").pack()
        Tk.Entry(self.param_group, textvariable = self.parameter["CV_size"]).pack()
        self.param_group.pack(side=Tk.LEFT)

    def fit(self):
        model_map = {0:"SVM", 1:"Adaboost", 2:"Random Forest", 3:"KNN"}
        # output_map = {0:"0/1 classification", 1:"probability", 2:"regression"}
        print(self.modelType.get())
        if self.modelType.get() == 0:
            self.model = Model_SVM(self.model,self.parameter)

        elif self.modelType.get() == 1:
            self.model = Model_Adaboost(self.model,self.parameter)
            
        elif self.modelType.get() == 2:
            self.model = Model_RF(self.model,self.parameter)

        elif self.modelType.get() == 3:
            self.model = Model_KNN(self.model,self.parameter)
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

    def __init__(self, root, controller):
        f = Figure()
        self.controllbar = ControllBar(root, controller)
        # self.f = f
        # self.controller = controller




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
                       value=0,command = controller.showFrameHelper).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="Adaboost(0/1)", variable=controller.modelType,
                       value=1,command = controller.showFrameHelper).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="Random Forest(0/1)", variable=controller.modelType,
                       value=2,command = controller.showFrameHelper).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="KNN(0/1)", variable=controller.modelType,
                       value=3,command = controller.showFrameHelper).pack(anchor=Tk.W)
        model_group.pack(side=Tk.LEFT)

        # output_group = Tk.Frame(fm)
        # Tk.Radiobutton(output_group, text="0/1 classification",
        #                variable=controller.model.output, value=0).pack(anchor=Tk.W)
        # Tk.Radiobutton(output_group, text="Probability",
        #                variable=controller.model.output, value=1).pack(anchor=Tk.W)
        # Tk.Radiobutton(output_group, text="Regression",
        #                variable=controller.model.output, value=2).pack(anchor=Tk.W)
        # output_group.pack(side=Tk.LEFT)
        output_group = Tk.Frame(fm)
        Tk.Button(output_group, text='Fit', width=5, command=controller.fit).pack()
        Tk.Button(output_group, text='Save Results', width=10, command=controller.save_results).pack()
        output_group.pack(side=Tk.LEFT)

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
