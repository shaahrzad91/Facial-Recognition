from scipy.io import loadmat
#from sklearn import preprocessing
import numpy as np
import numpy.matlib
import math
from scipy.stats import invgamma
from scipy.special import gammaln
#from scipy.special import gamma
from sklearn.metrics import confusion_matrix
#import sys

#np.set_printoptions(threshold=sys.maxsize) #to print whole matrices
#----------------------------------------------------Initialization---------------------------------------

#dataset = loadmat(r'C:\Users\stm\Desktop/shahrzou/ck_500.mat')
dataset = loadmat(r'C:\Users\SHAZ\Desktop/ck_1k.mat')
X=dataset['data']
Y= dataset['test_labels']# 1X695 (number = correct emotion from 1 to 6)
[N, D] = X.shape #(N=695, D=500)
M = 6

alpha_l = np.random.rand(M, D)
Beta_l = np.random.rand(M, D)
Nu = np.ones((M, D))
Lambda = np.random.rand(M,D)

#-----------------------------------
# k-means used to initialize weights
from sklearn.cluster import KMeans
P = KMeans(n_clusters=6, random_state=0).fit(X)
P = P.labels_ + 1
P.shape
#-----------------------------------


#Lambda = np.random.rand(M, D)#now using method of moments

#----------Incomplete MATLAB code from Nuha (what is the function dirichlet_moment_match()----------
#function a = polya_moment_match(data)
#% DATA is a matrix of count vectors (rows)
#[N,D]=size(data);
#sdata = sum(data,2);
#p = data ./ repmat(sdata+eps,1,D);
#a = dirichlet_moment_match(p);
#-----------------------------

#---------- MATLAB code received for dirichlet_moment_match()
# function a = dirichlet_moment_match(p)
# % Each row of p is a multivariate observation on the probability simplex.
# a = mean(p);
# m2 = mean(p.*p);
# ok = (a > 0);
# s = (a(ok) - m2(ok)) ./ (m2(ok) - a(ok).^2);
# % each dimension of p gives an independent estimate of s, so take the median.
# s = median(s);
# a = a*s;
#-----------------------------

#
#def dirichlet_moment_match(proportions):
#    a = np.array(np.average(proportions, axis=0))
#    m2 = np.array(np.average(np.multiply(proportions, proportions), axis=0))
#    nz = (a > 0)
#    aok = a[nz]
#    m2ok = m2[nz]
#    s = np.median((aok - m2ok) / (m2ok - (aok * aok)), axis=0)
#    return np.matrix(a * s)
#
#def polya_moment_match(X):
#    sdata= np.sum(X,axis=1)
#    sdata.shape
#    [N,D] = X.shape
#    proportions = X / np.transpose(np.matlib.repmat(sdata+np.finfo(float).eps,D,1))
#    a = dirichlet_moment_match(proportions)
#    return a
#
#X1=X[P==1]
#X2=X[P==2]
#X3=X[P==3]
#X4=X[P==4]
#X5=X[P==5]
#X6=X[P==6]
#
#mom1 = polya_moment_match(X1)
#mom2 = polya_moment_match(X2)
#mom3 = polya_moment_match(X3)
#mom4 = polya_moment_match(X4)
#mom5 = polya_moment_match(X5)
#mom6 = polya_moment_match(X6)
#
#mom1.shape #gives a 1X500 matrix
#
#Lambda=np.concatenate((mom1, mom2, mom3, mom4, mom5, mom6))#,axis=0)
#Lambda=np.array(Lambda)#to convert to numpy array
#Lambda.shape #gives a 6X500 matrix
##next: getting rid of 0s since taking log of lambda
#Lambda[Lambda==0]=np.nextafter(0,1)*2 #replacing 0 with smallest positive number representable by float64 multiplied by 2
#Lambda[Lambda > 1] = 0.54
#Lambda[Lambda > 2] = 0.65
#**************************************************************************************************************
#---------------------------------------------------Likelihood_Lambda--------------------------------------

def function_Likelihood_lambda(alpha_l,Beta_l,Lambda,Nu ):#want to return a 6x695 matrix
    pdfmatrix = np.zeros((M,N))
    err=0 #for debug
    for i in range(N): # for all pictures
        sumx = 0
        lnsumx = 0
        for k in range(D):#for all features (not dependent on classes)
            sumx = sumx + X[i][k]
        sumxtot = sumx
        while sumx > 0:
            lnsumx = lnsumx + math.log(sumx)
            sumx = sumx -1
        for j in range(M):#for all classes
            sumlambda = 0
            for k in range(D):#for all features (computation dependent on classes)
                sumlambda = sumlambda + Lambda[j][k]
            sumif = 0
            sumelse = 0
            for k in range(D):#for all features 
                if (X[i][k]>= 1):
                    sumif = sumif  + math.log(Lambda[j][k]) - (X[i][k] * math.log(Nu[j][k])) - math.log(X[i][k])
#                    try:
#                        sumif = sumif  + math.log(Lambda[j][k]) - (X[i][k] * math.log(Nu[j][k])) - math.log(X[i][k])
#                    except:
#                        err=1
                else:
                    sumelse = sumelse #- (X[i][k] * math.log(Nu[j][k])) #last part commented out since Nuha told Shahrzad althought mathematically indicator does not multiply last term, it should be 0
#                    try:
#                        sumelse = sumelse #- (X[i][k] * math.log(Nu[j][k])) #last part commented out since Nuha told Shahrzad althought mathematically indicator does not multiply last term, it should be 0
#                    except:
#                        err=1
            pdfmatrix[j][i] = lnsumx + gammaln(sumlambda) - gammaln(sumxtot + sumlambda) + sumif + sumelse
#            if pdfmatrix[j][i] == 0: #nuber too small to be represented
#                pdfmatrix[j][i] = np.nextafter(0,1) #smallest positive number representable by float64       
    if err==1:
        print("There was an error computing the pdf!")
    pdfmatrix1= np.exp(pdfmatrix)#take exponential since took log
#    for i in range(N): # for all pictures
#        for j in range(M):#for all classes
#            if pdfmatrix1[j][i] == 0: #nuber too small to be represented
#                pdfmatrix1[j][i] = np.nextafter(0,1)*2 #smallest positive number representable by float64--- multiplied by 2         
    return pdfmatrix1


#------------------------------------------------result matrix-------------------------------------------
def fn_result_matrix(pdfmatrix):#want to return a 1x695 matrix
    result_matrix = np.zeros((1,N))
    result_matrix=np.argmax(pdfmatrix,axis=0)
    result_matrix=result_matrix+1
    return result_matrix

#------------------------------------------------weight vector-------------------------------------------
def fn_weight_vector(result_matrix):#want to return a 1X6 vector
    weight_vector= np.zeros((1,M))
    for i in range(6):
        weight_vector[0,i]=np.count_nonzero(result_matrix == (i+1))
    return weight_vector

#------------------------------------------------weighted pdf-------------------------------------------
def fn_weighted_pdf(pdfmatrix, weight_vector):#want to return a 6x695 matrix
    weighted_pdf = np.zeros((M,N))
    weight_vectorT=np.transpose(weight_vector)
    weighted_pdf = np.multiply(pdfmatrix,weight_vectorT)
    return weighted_pdf

#------------------------------------------------convert proper pdf-------------------------------------------
#divide by sum --> so sum of pdf (for one picture) = 1
def fn_proper_pdf(pdfmatrix):#want to return a 6x695 matrix
    proper_pdfmatrix= np.zeros((M,N))
    vector_sum= np.sum(pdfmatrix,axis=0)
    vector_sum[vector_sum == 0] = 1#np.nextafter(0,1)*2 #smallest positive number representable by float64--- multiplied by 2
    for i in range(N):
        proper_pdfmatrix[:,i]= np.divide(pdfmatrix[:,i],vector_sum[i])#here!!!!!!!!!
    return proper_pdfmatrix
    

#**************************************************************************************************************
#------------------------------------------------generate proper pdf-------------------------------------------
def generate_proper_pdf(alpha_l,Beta_l,Lambda,Nu):
    pdfmatrix=function_Likelihood_lambda(alpha_l,Beta_l,Lambda,Nu)
    result_matrix= fn_result_matrix(pdfmatrix)
    weight_vector= fn_weight_vector(result_matrix)
    weighted_pdf= fn_weighted_pdf(pdfmatrix, weight_vector)
    proper_pdf=fn_proper_pdf(weighted_pdf)
    return proper_pdf

def generate_proper_pdf_initialisation(alpha_l,Beta_l,Lambda,Nu,P):#using k-means weights (only use this function for first iteration)
    pdfmatrix=function_Likelihood_lambda(alpha_l,Beta_l,Lambda,Nu)
#    result_matrix= fn_result_matrix(pdfmatrix)
    weight_vector= fn_weight_vector(P)
    weighted_pdf= fn_weighted_pdf(pdfmatrix, weight_vector)
    proper_pdf=fn_proper_pdf(weighted_pdf)
    return proper_pdf


#**************************************************************************************************************
#**************************************************************************************************************
    

#----------------------------------------------------Prior_Lambda------------------------------------------

def function_prior_lambda(alpha_l,Beta_l,Lambda):
    [M, D] = Lambda.shape
    prior_lambda = np.random.beta(alpha_l, Beta_l, size=(M,D))
    prior_lambda[prior_lambda == 0] = np.nextafter(0,1) * 2
    return prior_lambda

#-----------------------------------------------Prior_Nu----------------------------------------------------
def function_prior_Nu(alpha_l,Beta_l,Nu):
    [M, D] = Nu.shape
    prior_nu = invgamma.rvs(alpha_l, scale=Beta_l, size=(M,D))
    prior_nu[prior_nu == 0] = np.nextafter(0,1) * 2
    prior_nu[prior_nu >= np.inf] = 4.58967761e+100#a large number far from inf
    return prior_nu


#------------------------------------------------Posterior-------------------------------------------
def function_posterior(pdf,prior):
    likelihood = np.transpose(np.resize(np.transpose(pdf), (500,6))) #force it to 6X500 not 6X695
    posterior = np.multiply(prior,likelihood)
    return posterior


#------------------------------------------------r ratio-------------------------------------------   
def function_acc_ratio(prior_old, prior_new, posterior_old, posterior_new):
    numerator = np.multiply(posterior_new, prior_old)
    denominator = np.multiply(posterior_old, prior_new)
    denominator[denominator == 0] = np.nextafter(0,1) * 2
    return numerator/denominator#np.divide(numerator,denominator)

#--------------------
def fn_update_parameter(r, prior_old, prior_new):  
    uniform_mat=np.random.uniform(size=(6,500))
    updated_prior=prior_new
    for i in range(M):
        for j in range(D):
            if r[i,j] >= uniform_mat[i,j]:
                updated_prior[i,j] = prior_old[i,j]
    return updated_prior



#---------------------------------------------------------Main_Algorithm----------------------------------------    



pdf= generate_proper_pdf_initialisation(alpha_l,Beta_l,Lambda,Nu,P)#using P the weights initialized using k-means
posterior_Lambda=function_posterior(pdf,Lambda)
posterior_Nu=function_posterior(pdf,Nu)

i=0
while i < 10:
    print("\n")
    print(i)
    
    print('updating lambda')
    Lambda_new = function_prior_lambda(alpha_l,Beta_l,Lambda)

#    np.amax(Lambda)
#    np.amin(Lambda)
#    np.amax(Lambda_new)
#    np.amin(Lambda_new)
#    
    pdf_new_Lambda= generate_proper_pdf(alpha_l,Beta_l,Lambda_new,Nu)
    posterior_Lambda_new=function_posterior(pdf_new_Lambda,Lambda_new)
    r_Lambda=function_acc_ratio(Lambda, Lambda_new, posterior_Lambda, posterior_Lambda_new)
    Lambda=fn_update_parameter(r_Lambda, Lambda, Lambda_new)
    print(Lambda)
 
    print('updating nu')
    Nu_new = function_prior_Nu(alpha_l,Beta_l,Nu)
    
#    np.amax(Nu)
#    np.amin(Nu)
#    np.amax(Nu_new)
#    np.amin(Nu_new)
    
    pdf_new_Nu= generate_proper_pdf(alpha_l,Beta_l,Lambda,Nu_new)
    posterior_Nu_new=function_posterior(pdf_new_Nu,Nu_new)
    r_Nu=function_acc_ratio(Nu, Nu_new, posterior_Nu, posterior_Nu_new)
    Nu=fn_update_parameter(r_Nu, Nu, Nu_new)
    print(Nu)
    
#    print("Lambda" , Lambda)
#    print("Nu", Nu)
    i+=1

pdfmatrix=function_Likelihood_lambda(alpha_l,Beta_l,Lambda,Nu)
result_matrix= fn_result_matrix(pdfmatrix)#to get weights
weight_vector= fn_weight_vector(result_matrix)
weighted_pdf= fn_weighted_pdf(pdfmatrix, weight_vector)
proper_pdf=fn_proper_pdf(weighted_pdf)
result_matrix= fn_result_matrix(proper_pdf)#to get results

#print("weight", result_matrix)


#---------------------------------------------------Confusion Matrix--------------------------------------------

#print(Y.shape)
#print(result_matrix.shape)

y_true = []
for i in range(N):
    y_true=y_true+[Y[0,i]]

y_predicted = []
for i in range(N):
    y_predicted=y_predicted+[result_matrix[i]]

print(confusion_matrix(y_true, y_predicted))

#using new values of lambda when updating nu (in same iteration since nu after labda in code)



#***************************************************************************************************************

#w= [[6]]    
#df_confusion = pd.crosstab(YY,)
#TP = df_confusion[1][1]
#TN = df_confusion[0][0]
#FP = df_confusion[0][1]
#FN = df_confusion[1][0]
#print('True Positives:', TP)
#print('True Negatives:', TN)
#print('False Positives:', FP)
#print('False Negatives:', FN)
#
## calculate accuracy
#conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN))
#print("confusion matrix :", df_confusion)
#print("confusion accuracy is :", conf_accuracy)
#
#print("vv")
#print(w)
#print("Y")
#print(Y)
#
#a = confusion_matrix(Y,w)
#print(a)








#------------------------------------
#Finding the probability for step 3 in gibbs
#def probability(eta, nt ):
#    f= math.lgamma(eta.sum())
#    for j in range(M):
#        g= gamma(eta).T / ( (eta.T -1 + nt) * np.log(P))
#        gg= preprocessing.normalize(g)
#    return  f * gg
#
