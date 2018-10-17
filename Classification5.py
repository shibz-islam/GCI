import csv
import numpy as np
from pylab import *
from scipy import linalg
from scipy.stats import norm
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import time
import logging
from model import Model

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=None,
                            level=logging.INFO)
logger = logging.getLogger('Baseline')
class Classification(object):
    def __init__(self):
        super(Classification, self).__init__()

    @staticmethod
    def compmedDist(X):
        size1 = X.shape[0];
        Xmed = X;

        G = sum((Xmed * Xmed), 1);
        Q = tile(G[:, newaxis], (1, size1));
        R = tile(G, (size1, 1));

        dists = Q + R - 2 * dot(Xmed, Xmed.T);
        dists = dists - tril(dists);
        dists = dists.reshape(size1 ** 2, 1, order='F').copy();
        return sqrt(0.5 * median(dists[dists > 0]));

    @staticmethod
    def kernel_Gaussian(x, c, sigma):
        (d, nx) = x.shape
        (d, nc) = c.shape
        x2 = sum(x ** 2, 0)
        c2 = sum(c ** 2, 0)

        distance2 = tile(c2, (nx, 1)) + \
                    tile(x2[:, newaxis], (1, nc)) \
                    - 2 * dot(x.T, c)

        return exp(-distance2 / (2 * (sigma ** 2)));

    @staticmethod
    def R_ULSIF(x_nu, x_de, alpha, sigma_list, lambda_list, b, fold):
        # x_nu: samples from numerator
        # x_de: samples from denominator
        # x_re: reference sample
        # alpha: alpha defined in relative density ratio
        # sigma_list, lambda_list: parameters for model selection
        # b: number of kernel basis
        # fold: number of fold for cross validation





        (d, n_nu) = x_nu.shape;
        (d, n_de) = x_de.shape;
        # print "n_nu", d, n_nu;
        rand_index = np.random.permutation(n_nu);
        # print "rand_index", rand_index;
        b = min(b, n_nu);
        # print "b, n_nu", b, n_nu
        # x_ce = x_nu[:,rand_index[0:b]]
        # print "r_[0:b]", r_[0:b]
        #x_ce = x_nu
        x_ce = x_nu[:,np.r_[0:b]]
        #print("x_ce", x_ce)
        #print("x_nu", x_nu)

        #score_cv = zeros((size(sigma_list), \
        #                  size(lambda_list)));
        score_cv = np.zeros((len(sigma_list), len(lambda_list)))
        # print "score_cv", score_cv;
        # print "size(sigma_list)", size(sigma_list);
        # print "size(lambda_list)", size(lambda_list);
        #cv_index_nu = permutation(n_nu)
        cv_index_nu = np.random.permutation(n_nu)
        # cv_index_nu = r_[0:n_nu]
        cv_split_nu = np.floor(np.r_[0:n_nu] * fold / n_nu)
        # print "cv_split_nu", cv_split_nu
        cv_index_de = np.random.permutation(n_de)
        # cv_index_de = r_[0:n_de]
        cv_split_de = np.floor(np.r_[0:n_de] * fold / n_de)
        # print "cv_split_de", cv_split_de
        iter= 1000
        count=0
        mu = 0.1
        k1=0.1
        loss_old = float('inf')
        for i in range(0, iter):

            count=count+1
            for sigma_index in r_[0:size(sigma_list)]:
                sigma = sigma_list[sigma_index];
                K_de = Classification.kernel_Gaussian(x_de, x_ce, sigma).T;
                K_nu = Classification.kernel_Gaussian(x_nu, x_ce, sigma).T;
                # print "len x de", x_de.shape
                # print "len x ce", x_ce.shape
                # print "len x nu", x_nu.shape
                # print "len K de", K_de.shape
                # print "len K nu", K_nu.shape
                # print "K_de", K_de
                # print "K-nu", K_nu


                score_tmp = zeros((fold, size(lambda_list)));
                # print "score_tmp", score_tmp

                for k in np.r_[0:fold]:
                    Ktmp1 = K_de[:, cv_index_de[cv_split_de != k]];
                    Ktmp2 = K_nu[:, cv_index_nu[cv_split_nu != k]];
                    #print "lenKtmp1",Ktmp1.shape
                    #print "K-de", K_de.shape
                    #print "K-nu", K_nu.shape
                    #print "lenKtmp2", Ktmp2.shape

                    Ktmp = alpha / Ktmp2.shape[1] * np.dot(Ktmp2, Ktmp2.T) + (1 - alpha) / Ktmp1.shape[1] * np.dot(Ktmp1, Ktmp1.T);

                    mKtmp = np.mean(K_nu[:, cv_index_nu[cv_split_nu != k]], 1);

                    for lambda_index in np.r_[0:size(lambda_list)]:
                        lbd = lambda_list[lambda_index];

                        thetat_cv = linalg.solve(Ktmp + (lbd * eye(b)), mKtmp);
                        thetah_cv = thetat_cv;

                        score_tmp[k, lambda_index] = alpha * np.mean(
                            np.dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv) ** 2) / 2. \
                                                    + (1 - alpha) * np.mean(
                            np.dot(K_de[:, cv_index_de[cv_split_de == k]].T, thetah_cv) ** 2) / 2. \
                                                    - np.mean(dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv));

                    score_cv[sigma_index, :] = mean(score_tmp, 0);

            score_cv_tmp = score_cv.min(1);
            # print "score_cv", score_cv
            # print "score_cv.min(1)", score_cv.min(1)
            lambda_chosen_index = score_cv.argmin(1);

            score = score_cv_tmp.min();
            sigma_chosen_index = score_cv_tmp.argmin();

            lambda_chosen = lambda_list[lambda_chosen_index[sigma_chosen_index]];
            #print "lambda_chosen", lambda_chosen;
            sigma_chosen = sigma_list[sigma_chosen_index];
            #print "sigma_chosen", sigma_chosen;
            K_de = Classification.kernel_Gaussian(x_de, x_ce, sigma_chosen).T;
            K_nu = Classification.kernel_Gaussian(x_nu, x_ce, sigma_chosen).T;
            #print "len K_de", len(K_de)
            #print "len K_nu", len(K_nu)

            coe = alpha * np.dot(K_nu, K_nu.T) / n_nu + \
                (1 - alpha) * np.dot(K_de, K_de.T) / n_de + \
                lambda_chosen * np.eye(b)
            var = np.mean(K_nu, 1)
            #thetatold = thetat
            #print("thetaold", thetatold)
            thetat = linalg.solve(coe, var);
            #print("thetat", thetat)
            #changetheta = thetat-thetatold
            #print("changetheta", changetheta)
            thetattranspose = thetat
            thetattranspose = thetat.transpose()
            loss1 = dot(thetat.T,(alpha*dot(K_nu, K_nu.T) / n_nu - (1-alpha)*dot(K_de, K_de.T) / n_de))
            loss_bias = 0.5*dot(loss1, thetat)-dot(var.T, thetat)+0.5*lambda_chosen*dot(thetat.T, thetat)
            print("part 2 loss", loss_bias)
            if alpha<0:
                lossalpha = -1
            if alpha>1:
                lossalpha = 1
            if alpha>=0 and alpha<=1:
                lossalpha = 0
            print("alpha loss", lossalpha)

            loss_new = lossalpha + 1.0 * loss_bias

            # gradient
            result = dot((thetattranspose),(dot(K_nu, K_nu.T) / n_nu - dot(K_de, K_de.T) / n_de))
            loss2change = 0.5*dot(result,thetat)
            print("loss bias change", loss2change)

            # update alpha
            alphaold = alpha
            print("alphaold", alphaold)

            while True:
                mu = mu * exp(-k1 * i)
                if alpha<0:
                    alpha = alpha - mu*(-1+loss2change)
                if alpha>1:
                    alpha = alpha - mu*(1+loss2change)
                if alpha>= 0 and alpha<=1:
                    alpha = alpha - mu*loss2change

                if (alpha < 0 or alpha >=1):
                    k1 = 2*k1
                    alpha = alphaold
                else:
                    k1 = 0.1
                    break

            print("mu", mu)
            if abs(loss_old-loss_new) < 1e-7:
                break

            print("Old loss: ", loss_old, "new loss", loss_new)
            loss_old = loss_new
            print("alphaupdated", alpha)
            print("alphachange", alphaold-alpha)
            print("count", count)

        thetah = thetat;
        wh_x_de = np.dot(K_de.T, thetah).T;
        wh_x_nu = np.dot(K_nu.T, thetah).T;
        print("wh_x_de", wh_x_de)
        print("whxde len",len(wh_x_de))
        print("wh_x_nu", wh_x_nu)
        print("wh_x_nu len", len(wh_x_nu))


        wh_x_de[wh_x_de < 0] = 0

        return (thetah, wh_x_de, x_ce, sigma_chosen)

    @staticmethod
    def compute_target_weight(thetah, x_ce, sigma, x_nu):
        K_nu = Classification.kernel_Gaussian(x_nu, x_ce, sigma).T;
        wh_x_nu = dot(K_nu.T, thetah).T;
        wh_x_nu[wh_x_nu < 0.00000001] = 0.00000001
        return wh_x_nu


    @staticmethod
    def sigma_list(x_nu, x_de):
        x = c_[x_nu, x_de];
        # print "x", x
        med = Classification.compmedDist(x.T);
        # print "med", med
        return med * array([0.6, 0.8, 1, 1.2, 1.4]);

    @staticmethod
    def lambda_list():
        return  array([10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1)]);


    @staticmethod
    def read_csv(path, size=None, delimiter=",", header=None):
        data = None
        data_label = None
        with open(path) as csvfile:
            count = 0
            if not header:
                reader = csv.reader(csvfile, delimiter=delimiter)
            else:
                reader = csv.DictReader(csvfile, fieldnames=header, delimiter=delimiter)
            for row in reader:
                #print("x[1]", x[1:1])
                #tmp1 = [float(x[1:-1]) for x in row[0:0]]
                #tmp = [float(x[2:-1]) for x in row[:-1]]
                #print("row", row)
                tmp = [float(x) for x in row[:-1]]
                #tmp = tmp1.append(tmp)
                #print("temp", tmp)
                #print("lentemp", len(tmp))
                label = row[-1]
                if data is None:
                    data = np.array(tmp)
                else:
                    data = np.vstack((data, tmp))
                count += 1
                if data_label is None:
                    data_label = np.array(label)
                else:
                    data_label = np.vstack((data_label, label))
                if size and count > size:
                    break
            data = np.matrix(data, dtype=np.float64)
            return data, data_label

    @staticmethod
    def get_model(trgx_matrix, srcx_matrix, srcy_, alpha, sigma_list, lambda_list, b, fold, subsize):
        """
        :param m: a row matrix contains current parameter values corresponding to [Y, x1,x2,...,xn, 1]
        :param data_matrix: M*(N+1) matrix, M - number of instances, N - number of features, 
        the last column is target value
        :return: mu - MLE estimate value of y for each data point.
        """
        #model = Model()
        srcx_array = np.array(srcx_matrix.T);
        srcy_array = np.array(srcy_);
        srcy_labellist = srcy_array
        #for i in range(len(srcy_array[0])):
            #if srcy_array[0][i] == 'class1':
                #srcy_labellist.append(1)
            #if srcy_array[0][i] == 'class2':
                #srcy_labellist.append(2)
            #if srcy_array[0][i] == 'class3':
                #srcy_labellist.append(3)
            #if srcy_array[0][i] == 'class4':
                #srcy_labellist.append(4)
            #if srcy_array[0][i] == 'class5':
                #srcy_labellist.append(5)
            #if srcy_array[0][i] == 'class6':
                #srcy_labellist.append(6)
            #if srcy_array[0][i] == 'class7':
                #srcy_labellist.append(7)
        #subsize = 2
        subwindowsize = len(trgx_matrix)/subsize
        #print "subwindowsize", subwindowsize
        #print "srcx_array.shape", srcx_array.shape
        #print "srcy_array", srcy_array.shape
        avgw = []
        for i in range(0,subsize):
            trgx_array = np.array(trgx_matrix[i*int(subwindowsize):(i+1)*int(subwindowsize)].T); #shihab: changed for python 3
            (thetah, w, x_ce, sigma) = Classification.R_ULSIF(trgx_array, srcx_array, alpha, sigma_list, lambda_list, b, fold)
            avgw.append(w)
            break
        avg = np.average(avgw, axis=0)
        for i in range(0, len(avg)):
            with open('weightnormgroup10726.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow([avg[i]])
        params = {'gamma': [2 ** 2, 2 ** -16], 'C': [2 ** -6, 2 ** 15]}
        svr = svm.SVC()
        opt = GridSearchCV(svr, params)
        opt.fit(X = np.array(srcx_matrix), y = np.array(srcy_labellist))
        optParams = opt.best_params_
        #model = linear_model.RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None,
         #                                    tol=0.001, class_weight=None, solver='auto', random_state=None)

        model = svm.SVC(decision_function_shape='ovr', probability=True, C=optParams['C'], gamma=optParams['gamma'])
        #model1 = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        #src_y = src_matrix[:, -1];

        # design a loss function to choosing the model in hypothesis set.
        print(srcx_array.shape, np.array(srcy_labellist).shape, len(avg))
        model.fit(np.array(srcx_matrix), np.array(srcy_labellist), avg);
        #model1.fit(srcx_matrix,src_y);
        #predictions = model.predict(np.array(trgx_matrix));
        #confidence = model.decision_function(np.array(trgx_matrix));
        #print "predictions", predictions.transpose().tolist()[0]
        #pred_lr = model1.predict(trgx_matrix);
        return model;

    @staticmethod
    def get_predictlabel(trgx_matrix, model):

        predictions = model.predict(np.array(trgx_matrix));
        confidence = model.decision_function(np.array(trgx_matrix));
        #print "predictions", predictions.transpose().tolist()[0]
        #pred_lr = model1.predict(trgx_matrix);
        return predictions,confidence;


    @staticmethod
    def get_true_label(target_label):
        y = target_label
        return y.transpose().tolist()[0]

    @staticmethod
    def get_prediction_error(prediction_label, true_label):
        error = 0.00
        for i in range(len(prediction_label)):
            if prediction_label[i] != true_label[i]:
                error=error+1.00;
            #error += abs(prediction_value[i]-true_value[i])
        return error/len(prediction_label)

# if __name__ == '__main__':
#     classification = Classification()
#     srcx_matrix, srcy_matrix = Classification.read_csv('datasets/syndata_002_normalized_no_novel_class_source_stream.csv', size=500)
#     # find execution time
#     start_time = time.clock()
#     trgx_matrix, trgy_matrix = Classification.read_csv('datasets/syndata_002_normalized_no_novel_class_target_stream.csv', size=500)
#     alpha = 0.0;
#     b = trgx_matrix.T.shape[1];
#     print "b:",b
#     fold = 5;
#     sigma_list = Classification.sigma_list(np.array(trgx_matrix.T), np.array(srcx_matrix.T));
#     lambda_list = Classification.lambda_list();
#     #srcx_matrix = matrix_[:, :-1];
#     #trgx_matrix = target_[:, :-1];
#     srcx_array = np.array(srcx_matrix.T);
#     trgx_array = np.array(trgx_matrix.T);
#     #(PE, w, s) = Classification.R_ULSIF(trgx_array, srcx_array, alpha, sigma_list, lambda_list,
#                                     #b, fold)
#     # w_de = w[0:srcx_array.shape[1]]
#     model = Classification.get_model(trgx_matrix, srcx_matrix, srcy_matrix, alpha, sigma_list, lambda_list, b, fold,2)
#     predict_label, confidence = Classification.get_predictlabel(trgx_matrix, model)
#     print "predict label",predict_label;
#     print "confidence", confidence;
#     true_label = Classification.get_true_label(trgy_matrix);
#     truetrg_labellist = []
#     #print "trgy_array", np.array(trgy_matrix)
#     for i in range(len(np.array(trgy_matrix))):
#         if np.array(trgy_matrix)[i] == 'class1':
#             truetrg_labellist.append(1)
#         if np.array(trgy_matrix)[i] == 'class2':
#             truetrg_labellist.append(2)
#         if np.array(trgy_matrix)[i] == 'class3':
#             truetrg_labellist.append(3)
#         if np.array(trgy_matrix)[i] == 'class4':
#             truetrg_labellist.append(4)
#         if np.array(trgy_matrix)[i] == 'class5':
#             truetrg_labellist.append(5)
#         if np.array(trgy_matrix)[i] == 'class6':
#             truetrg_labellist.append(6)
#         if np.array(trgy_matrix)[i] == 'class7':
#             truetrg_labellist.append(7)
#     print "true_label", truetrg_labellist
#     print "len true_label", len(truetrg_labellist)
#     print "predict_label", predict_label
#     print "len predict_label", len(predict_label)
#     err = Classification.get_prediction_error(predict_label, truetrg_labellist)
#
#     print "the err:", err;