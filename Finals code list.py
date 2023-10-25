import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import math

#   My matrix operations                |  MATLAB
#                                       | 
#   Transpose:         M.T              |  transpose(M)
#   Dot product:       dot(x,y)         |  dot(x,y)
#   Matrix-Vector:     M @ v            |  M * v
#   Vector-Matrix:     v @ M            |  v * M
#   Matrix-Matrix:     X @ Y            |  X * Y
#   Inverse:           inv(M)           |  inv(M)
#   Rank:              rank(M)          |  rank(M)
#   Determinant:       det(M)           |  det(M)
#   Cofactor:          cofactor(M)      |  transpose(adjoint(M))
#   Adjugate:          adjoint(M)       |  adjoint(M)

# Functions work with both list and np.array,
# but .T and @ only works with np.array

def dot(x,y):
    if type(x)==list:
        x = np.array(x)
    if type(y)==list:
        y = np.array(y)
    return x.T @ y

def inv(M):
    from numpy.linalg import inv as inverse
    if type(M)==list:
        M = np.array(M)
    return inverse(M)

def det(M):
    from numpy.linalg import det as determinant
    if type(M)==list:
        M = np.array(M)
    return determinant(M)

def rank(M):
    from numpy.linalg import matrix_rank
    if type(M)==list:
        M = np.array(M)
    return matrix_rank(M)

def row_echelon(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

def cofactor(M):
    lst = []
    for i in range(len(M)):
        row = []
        for j in range(len(M)):
            minor_matrix = []
            for r in range(len(M)):
                if r != i:
                    the_row = []
                    for c in range(len(M)):
                        if c != j:
                            the_row.append(M[r][c])
                    minor_matrix.append(the_row)
            minor = det(minor_matrix)
            row.append((-1)**(i+j)*minor)
        lst.append(row)
    return np.array(lst)

def adjoint(A):
    if type(A) == list:
        A = np.array(A)
    print(cofactor(A).T)
    return cofactor(A).T

def onehot(A):
    if type(A) == list:
        A = np.array(A)
    onehot_encoder = OneHotEncoder(sparse=False)
    reshaped = A.reshape(len(A),1)
    Ytr_onehot = onehot_encoder.fit_transform(reshaped)
    return Ytr_onehot

###################################################################################
#reg_L = 0.0001*np.identity(P.shape[1]) --- making lamba ridge 
#w = inv(P.T @ P + reg_L) @ P.T @ y  --- for primal ridge regression (over_D)
#w = P.T @ inv(P @ P.T + reg_L) @ P.T @ y  --- for dual ridge regression (under_D)
def primal_ridge(X, y, alpha):
    reg_L = alpha*np.identity(X.shape[1])
    w = inv(X.T @ X + reg_L) @ X.T @ y
    return w

def dual_ridge(X, y, alpha):
    reg_L = alpha*np.identity(X.shape[1])
    w = X.T @ inv(X @ X.T + reg_L) @ y
    return w
###################################################################################
def left_inverse(X,y):   #left inverse (over-determined/ x is tall)
    if type(X) == list:
        X = np.array(X)
        #X = X.T (add this line if dealing with w^T)
    if type(y) == list:
        y = np.array(y)
    w = inv(X.T @ X) @ X.T @ y
    return w

def right_inverse(X,y):  #right inverse (under-determined/ x is wide)
    if type(X) == list:
        X = np.array(X)
        #X = X.T (add this line if dealing with w^T)
    if type(y) == list:
        y = np.array(y)
    w = X.T @ inv(X @ X.T) @ y
    return np.array(w)
###################################################################################

def getW(X,y):  #auto detects whether to do right or left inverse
    X = np.array(X)
    #X = X.T (add this line if dealing with w^T)
    y = np.array(y)
    Sizecheck = list(X.shape)
    if Sizecheck[0] > Sizecheck[1]:
        w = inv(X.T @ X) @ X.T @ y
    elif Sizecheck[0] == Sizecheck[1]:
        w = inv(X) @ y
    else:
        w = X.T @ inv(X @ X.T) @ y
    return w


def nobias(X,y,xtest):  #no offset regression
    X = np.array(X)
    y = np.array(y)
    Sizecheck = list(X.shape)
    if Sizecheck[0] > Sizecheck[1]:
        w = inv(X.T @ X) @ X.T @ y
    elif Sizecheck[0] == Sizecheck[1]:
        w = inv(X) @ y
    else:
        w = X.T @ inv(X @ X.T) @ y
    ytest = xtest@w
    print("w = " + str(w))
    print(w)
    print("newY = " + str(ytest))
    return ytest

def LGT(X,y,xtest): #adds bias for X and xtest
    new_xtest = []
    new_X = []
    #add bias into xtest set
    for i in xtest:
        i.insert(0,1)
        new_xtest.append(i)
    new_xtest = np.array(new_xtest)
    #add bias into X matrix
    for i in X:
        i.insert(0,1)
        new_X.append(i)
    new_X = np.array(new_X)
    y = np.array(y)
    #check if tall or wide, and to use left or right inverse
    Sizecheck = list(new_X.shape)
    print('Sizecheck = ' + str(Sizecheck))
    if Sizecheck[0] > Sizecheck[1]:
        w = left_inverse(new_X,y)
        print('left inverse')
    elif Sizecheck[0] == Sizecheck[1]:
        w = inv(X) @ y
        print('even')
    else:
        w = right_inverse(new_X,y)
        print('right inverse')
    print("w = " + str(w))
    #finding ytest
    Ytest = new_xtest@w
    print("newY =")
    print(Ytest)
    return Ytest


def argmax(x):
    Class = []
    if type(x) == list:
        x = np.array(x)
    output = np.argmax(x, axis = 1)
    for i in output:
        Class.append(i + 1)
    return Class

def makeP(x, order):
    if type(x) == list:
        x = np.array(x)
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(x)
    return P

def matrix(m,n):
    rows = m
    columns = n
    matrix = []
    for i in range(0,rows):
        row_list = []
        print("Row " + str(i + 1))
        for j in range(0,columns):
            number_to_add = float(input("Row " + str(i + 1) + " Column " + str(j + 1) + ": "))
            row_list.append(number_to_add)
        matrix.append(row_list)
    print("rmb to np.array if need be, as now it is not an array")
    print(matrix)
    return matrix

def array(x):
    if type(x)==list:
        x = np.array(x)
    return x


def poly(X,y,xtest,order): #poly must be done before LGT as LGT alters matrix X
    print("Poly must be done before LGT")
    X = np.array(X)
    y = np.array(y)
    XP = makeP(X, order)
    xtestP = makeP(xtest, order)
    Sizecheck = list(XP.shape)
    if Sizecheck[0] > Sizecheck[1]:
        w = left_inverse(XP,y)
    elif Sizecheck[0] == Sizecheck[1]:
        w = inv(X) @ y
    else:
        w = right_inverse(XP,y)
    print("w = " + str(w))
    ynew = xtestP@w
    print("rmb to sign(answer) to get the classes")
    print("ynew = " + str(ynew))
    return ynew


# impurity = 'gini' or 'entropy' or 'misc'as in misclassification
# list of classnum [numberofclass1,numberofclass2,numberofclass3.....]

def impurity(listofclassnum,impurity=None):
    if impurity==None:
        print("PUT AN IMPURITY TYPE!!")
    total_cnt=sum(listofclassnum)
    
    if impurity=='gini':
        Q_node=1
        for clsnum in listofclassnum:
            Q_node-=((clsnum/total_cnt)**2)
        print(f'GINI IMPURITY FOR THIS NODE:{Q_node}')
        return Q_node
    
    if impurity=='entropy':
        Q_node=0
        for clsnum in listofclassnum:
            if clsnum==0:
                Q_node-=0
            else:
                Q_node-=((clsnum/total_cnt)*math.log2((clsnum/total_cnt)))
        print(f'ENTROPY IMPURITY FOR THIS NODE:{Q_node}')
        return Q_node
    
    if impurity=='misc':
        Q_node=0
        max_cls=max(listofclassnum)
        Q_node=1-max_cls/total_cnt
        print(f'MISCLASSIFICATION IMPURITY FOR THIS NODE:{Q_node}')
        return Q_node
    
def MSE(y_true,pred):
    MSE = mean_squared_error(y_true,pred)
    print('MSE Found')
    return MSE

def Kmeans(X):
    # e.g. X = np.array([[0,0],[0,1],[1,1],[1,0],[3,0],[3,1],[4,0],[4,1]]) - Tutorial 11 Qn 5
    plt.scatter(X[:,0],X[:,1], label = 'True Position')
    plt.show()
    # creating kmeans cluster with n clusters
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    print(kmeans.cluster_centers_)
    # to see labels for the data points 
    print(kmeans.labels_)
    plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.show()
    
def grad_descent(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters

    """
    
    # initialize output lists 
    # add more lists if there is more than 1 variable
    a_out = []
    f1_out = []

    # initial values and gradients (TO BE INPUT AND UPDATED BETWEEN QUESTIONS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
    a = 3
    gradient_a = 40 * (a**4)
    
    # gradient descent algorithm 
    for i in range(num_iters):
        
        # part(a)
        a = a - learning_rate * gradient_a
        f1 = a**4
        a_out.append(a)
        f1_out.append(f1)
        gradient_a = 4 * (a**3)
    
    # converting to numpy arrays
    a_out = np.array(a_out)
    f1_out = np.array(f1_out)

    # return in this order
    return a_out, f1_out


#lOSS FUNCTIONS error_type = sq_loss, binary_loss, hinge_loss, exp_loss
def lossfunc(true,pred,error_type=None):
    pdt=true*pred
    if error_type==None:
        print('Please input an error')
        return None
    elif error_type=="sq_loss":
        error=np.sum((pred-true)**2)
        return error
    elif error_type=='binary_loss':
        ans=[]
        for i in pdt:
            if i>0:
                ans.append(0)
            elif i<=0:
                ans.append(1)
        error=sum(ans)
        return error
    elif error_type=='hinge_loss':
        ans=[]
        for i in pdt:
            val=max(0,1-i)
            ans.append(val)
        error=sum(ans)
        return error
    elif error_type=='exp_loss':
        error_array=np.exp(-pdt)
        error=np.sum(error_array)
        return error

