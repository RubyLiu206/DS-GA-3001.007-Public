# %%

## Why should I scale my data?

# Common wisdom has it that data should be centered and scaled, but why is that?
# In this part of the tutorial we will look at a dataset which is initially badly scaled.
# [Example from "A Practical Guide to Support Vector Classification" by Hsu, Chang, Lin http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf]

# %%

#Start with imports
import warnings
import exceptions
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV


# %%

#read in the data

#it is in SVMlight format - so we use the corresponding dataset loader from sklearn
warnings.simplefilter("ignore", exceptions.DeprecationWarning) #the function gives warnings, going to ignore them here.
X_train, y_train = load_svmlight_file("data/svmguide1")
X_train = X_train.todense()


X_test, y_test = load_svmlight_file("data/svmguide1.t")
X_test = X_test.todense()

#doing something sneaky here: mis-scaling my data
X_train[:,3] = (X_train[:,3] - np.mean(X_train[:,3]))*100
X_test[:,3] = (X_test[:,3] - np.mean(X_train[:,3]))*100


# %%

#show the first few rows of X and y
print X_train[:5, :]
print y_train[:5]


# Out[3]:

#     [[  2.61730000e+01   5.88670000e+01  -1.89469700e-01   9.45733547e+02]
#      [  5.70739700e+01   2.21404000e+02   8.60795900e-02   7.24623547e+02]
#      [  1.72590000e+01   1.73436000e+02  -1.29805300e-01   9.36663547e+02]
#      [  2.17794000e+01   1.24953100e+02   1.53885300e-01   3.70498355e+03]
#      [  9.13399700e+01   2.93569900e+02   1.42391800e-01   4.48750355e+03]]
#     [ 1.  1.  1.  1.  1.]
# 

# %%

#learn a classifier to predict X from y
#using default settings
#kernel=rbf
#C=1
#gamma=0.1
classifier = SVC()


# %%

classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)


# Out[5]:

#     0.5

# Ooh- that's pretty bad. Maybe I can get it to be better by doing cross validation on the C and gamma parameters?

# %%

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 
                     'gamma': [1e-1, 1e-2,1e-3, 1e-4, 1e-5],
                     'C': [0.01, 0.1, 1, 10, 100]}]
        
clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
clf.fit(X_train, y_train)


# Out[6]:

#     GridSearchCV(cv=5,
#            estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#       kernel='rbf', max_iter=-1, probability=False, random_state=None,
#       shrinking=True, tol=0.001, verbose=False),
#            fit_params={}, iid=True, loss_func=None, n_jobs=1,
#            param_grid=[{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001, 1e-05]}],
#            pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring=None,
#            verbose=0)

# %%

clf.best_estimator_


# Out[7]:

#     SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#       gamma=1e-05, kernel='rbf', max_iter=-1, probability=False,
#       random_state=None, shrinking=True, tol=0.001, verbose=False)

# %%

clf.score(X_test, y_test)


# Out[8]:

#     0.46024999999999999

# nope, not much better at all!
# What about the scaling of my parameters? What does it look like?

# %%

plt.bar(xrange(4), [np.mean(X_train[:, i]) for i in xrange(4)], yerr=[np.std(X_train[:, i]) for i in xrange(4)])
#plt.yscale('log')
plt.show()


# Out[9]:

# image file:

# It looks like the 3rd dimension would completely dominate any kernel calculation:
# K(x1, x2) = e^{-(x1 dot x2)/gamma} \approx e^{-(x1[3]*x2[3]} since X[3] is so much bigger
# What would happen if I just used the third column of my data

# %%

X_train_trunc = X_train[:,3]
X_test_trunc = X_test[:,3]

clf.fit(X_train_trunc, y_train)
clf.score(X_test_trunc, y_test)


# Out[10]:

#     0.5

# It's about the same- they're both pretty bad. Let's try scaling the data.

# %%

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)


# %%

plt.bar(xrange(4), [np.mean(X_train[:, i]) for i in xrange(4)], yerr=[np.std(X_train[:, i]) for i in xrange(4)])
#plt.yscale('log')
plt.show()


# Out[12]:

# image file:

# %%

clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
clf.fit(X_train, y_train)


# Out[13]:

#     GridSearchCV(cv=5,
#            estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#       kernel='rbf', max_iter=-1, probability=False, random_state=None,
#       shrinking=True, tol=0.001, verbose=False),
#            fit_params={}, iid=True, loss_func=None, n_jobs=1,
#            param_grid=[{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001, 1e-05]}],
#            pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring=None,
#            verbose=0)

# %%

clf.best_estimator_


# Out[14]:

#     SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.1,
#       kernel='rbf', max_iter=-1, probability=False, random_state=None,
#       shrinking=True, tol=0.001, verbose=False)

# %%

clf.score(X_test, y_test)


# Out[15]:

#     0.5

# Nothing changed!! what happened?
# Well, I need to scale my training set too!

# %%

X_test = scaler.transform(X_test) #note - scale with the same transform function used to transform the train data


# %%

clf.score(X_test, y_test) 


# Out[17]:

#     0.75475000000000003

# Better - and maybe with a bit more searching over parameters I could do even better.

# %%




# Out[55]:

#     GridSearchCV(cv=5,
#            estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#       kernel='rbf', max_iter=-1, probability=False, random_state=None,
#       shrinking=True, tol=0.001, verbose=False),
#            fit_params={}, iid=True, loss_func=None, n_jobs=1,
#            param_grid=[{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001]}],
#            pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring=None,
#            verbose=0)
