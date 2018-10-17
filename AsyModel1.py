import math, numpy as np
from random import randint
import random
import csv
from Classification5 import Classification
import time
from Ensemble1 import Ensemble
from sklearn.decomposition import KernelPCA
import os
#from pelts import pelt
#from costs import normal_mean

BASE_URL = '/Users/shihab/Documents/Big Data Lab/game_data/'
SRC_FILE_PATH = os.path.join(BASE_URL, 'norm15group1playertrain.csv')
TGT_FILE_PATH = os.path.join(BASE_URL, 'norm15group1playertest.csv')

class Update(object):

    def __init__(self):
        self.Ensemble=Ensemble(1)
        super(Update, self).__init__()

    @staticmethod
    def readdata(sourcex_matrix=None, sourcey_matrix=None,targetx_matrix=None, targety_matrix=None,src_path=SRC_FILE_PATH,
                   tgt_path=TGT_FILE_PATH, src_size=None, tgt_size=None):
        """ 
        input is: source dataset with y, here we assume it is a list of list, the name is source, target dataset with yhat, 
        here we assume it is a list of list, the name is target 
        """
        if sourcex_matrix is None:
            sourcex_matrix_, sourcey_matrix = Classification.read_csv(src_path, None)   # matrix_ is source data
        else:
            sourcex_matrix_ = sourcex_matrix
            sourcey_matrix_ = sourcey_matrix
        matrix_ = sourcex_matrix_[:src_size, :]

        if targetx_matrix is None:
            targetx_ ,targety_= Classification.read_csv(tgt_path, size=None)
        else:
            targetx_ = targetx_matrix
            targety_ = targety_matrix
        labellist = []
        for i in range(0, len(targety_)):
            if targety_[i] not in labellist:
                labellist.append(targety_[i])
        print("labellistlen",len(labellist))
        sourcey_label = []
        for i in range(0, len(sourcey_matrix)):
            sourcey_label.append(labellist.index(sourcey_matrix[i]))

        for i in range(0, len(targety_)):
            if targety_[i] not in labellist:
                labellist.append(targety_[i])
        targety_label = []
        for i in range(0, len(targety_)):
            targety_label.append(labellist.index(targety_[i]))
        return sourcex_matrix_,sourcey_label, targetx_, targety_label

    def Process(self, sourcex,sourcey, targetx,targety,subsize):
        # fixed size windows for source stream and target stream
        #kpca = KernelPCA(n_components=3, kernel = 'rbf',gamma = 15)
        #print("sourcex.shape", sourcex.shape)
        #x_kpca = kpca.fit_transform(sourcex)
        #print("x_kpca.shape", x_kpca.shape)
        #sourcex = np.matrix(x_kpca)
        #y_kpca = kpca.fit_transform(targetx)
        #targetx = np.matrix(y_kpca)
        src_size, _ = sourcex.shape
        tgt_size, _ = targetx.shape
        #true_label = []

        ### get the initial model by using the first source and target windows
        alpha = 0.5
        b = targetx.T.shape[1];
        fold = 5
        sigma_list = Classification.sigma_list(np.array(targetx.T),
                                               np.array(sourcex.T));
        lambda_list = Classification.lambda_list();
        srcx_array = np.array(sourcex.T);
        trgx_array = np.array(targetx.T);
        #(thetah_old, w, sce_old, sigma_old) = Classification.R_ULSIF(trgx_array, srcx_array, alpha, sigma_list, lambda_list, b, fold)

        self.Ensemble.generateNewModelRULSIF(targetx, sourcex, sourcey, alpha, sigma_list,
                                             lambda_list, b, fold,subsize)
        # print "update model", src_size, source.shape
        truelablecount = 0.0
        totalcount = 0.0
        for i in range(0, len(targety)):
            #print("targetx[i]", targetx[i])
            instanceresult = self.Ensemble.evaluateEnsembleRULSIF(targetx[i])
            #print("instanceresult", instanceresult)
            #print("instanceresult[0]", instanceresult[0])
            #print("truelabel[i]", targety[i])
            if instanceresult[0] == targety[i]:
                truelablecount += 1.0
            totalcount += 1.0
            if i%100 ==0:
                print("truelabelcount", truelablecount)
                print("totalcount", totalcount)
                print("tmp acc", truelablecount/totalcount)
        with open('accuracynormgroup3datafilegame0726.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([len(targety), truelablecount, totalcount, truelablecount / totalcount])



sourcex_matrix_,sourcey_matrix, targetx_, targety_ = Update.readdata(src_size=None)
print("done reading data")
start_time = time.time()
up = Update()
up.Process(sourcex_matrix_,sourcey_matrix, targetx_, targety_,1)
#resTarget = ensemble.evaluateEnsembleRULSIF(targetx_[0])
#print "resTarget", resTarget
#true_y = Classification.get_true_label(targety_)
#truetrg_labellist = []

#error = Classification.get_prediction_error(updateYhat,truetrg_labellist)
end_time = time.time()

print("Execution time for %d iterations is: %s min" % (
1000, (end_time-start_time)/60.0))
#print "update y hat pm2.5", updateYhat
print("errorsyn002alpha10 0522")