import numpy as np
import scipy.sparse as sp
import keras.backend as K

def adjacency_matrix(n):
    S = np.zeros((n,n))
    for i in range(len(S)):
        if i==0:
            S[0][1] =1
            S[0][n-1] = 1
        elif i==(n-1):
            S[n-1][n-2] = 1
            S[n-1][0] = 1
        else:
            S[i][i-1]=1
            S[i][i+1]=1
    return S

def normalized_laplacian(adj):
    D = 1/adj.sum(axis=1)
    D = np.diag(D)
    return adj+D

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(axis = 1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_adjacency_k(L, k):
    dim = L.shape[0]
    if k > 0:
        output = L
        for i in range(k-1):
            output = np.matmul(output,L)
        out = np.sign(output)
    elif k == 0:
        out = np.eye(dim)
    return out

def A_k(L, k):
    output = L
    for i in range(k-1):
        output = K.dot(output,L)
    return np.sign(output)

def compute_threshold(batch, k=4):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return K.cast(k / (k + K.exp(batch / k)), 'float32')

    
def rmse(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def RMSE(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))
    
def MAPE(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true)/y_true))
def MAE(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

############################################
def Speed_k_correlation(ground_truth, weights, k):
    
    nb_samples = ground_truth.shape[0]
    time_steps = ground_truth.shape[1]
    N = ground_truth.shape[2]
    
    speed_vector = [i for i in range(115, -1, -5)]
    print(speed_vector)
    counts = []
    f_value = np.zeros(len(speed_vector))
    importances = np.zeros((len(speed_vector), 2*k+1))
    statistics = []
    for i in range(len(speed_vector)):
        statistics.append([])
    
    for i in range(len(speed_vector)):
        ground_truth[ground_truth >= speed_vector[i]] = -i-1
        counts.append(np.count_nonzero(ground_truth == -i-1))
    counts = np.array(counts)
    print(counts)
    
    W_head = weights[...,:k]
    W_end = weights[...,-k:]
    kernels =  np.concatenate([W_end, weights, W_head], axis=-1)
    
    mask = np.array([i for i in range(-k,k+1)])
    
    for i in range(nb_samples-1):
        for j in range(time_steps):
            for l in range(N):
                ref = ground_truth[i+1][j][l]
                f = 0.
                for p in range(-k,k+1):
                    f += mask[p+k]*kernels[i][j][l][l+p+k]
                    importances[int(-ref-1)][p+k] += kernels[i][j][l][l+p+k]
                    #f += mask[p+k]*kernels[i][j][l][l+p]
                    #importances[int(-ref-1)][p+k] += kernels[i][j][l][l+p]
                f_value[int(-ref-1)] += f
                statistics[int(-ref-1)].append(f)
            
    f_out = f_value/counts
    
    importance = importances.transpose()
    sp = []
    for i in range(2*k+1):
        sp.append(importance[i]/counts)
    
    return f_out, np.array(sp), statistics

def STcorrelation(ground_truth, weights, k):
    
    nb_samples = ground_truth.shape[0]
    time_steps = ground_truth.shape[1]
    N = ground_truth.shape[2]
    
    speed_vector = [i for i in range(115, -1, -5)]
    print(speed_vector)
    
    counts = np.zeros((N, len(speed_vector)))
    f_value = np.zeros((N, len(speed_vector)))
    J = np.zeros((N, len(speed_vector), 2*k+1))
    
    for i in range(len(speed_vector)):
        ground_truth[ground_truth >= speed_vector[i]] = -i-1
    
    W_head = weights[...,:k]
    W_end = weights[...,-k:]
    kernels =  np.concatenate([W_end, weights, W_head], axis=-1)
    print(kernels.shape)
    
    mask = np.array([i for i in range(-k,k+1)])
    
    for i in range(nb_samples-1):
        for j in range(time_steps):
            filters = kernels[i][j]
            for l in range(N):
                ref = ground_truth[i+1][j][l]
                counts[l][int(-ref-1)] += 1
                f = 0
                for p in range(-k,k+1):
                    f += mask[p+k]*filters[l][l+p+k]
                    J[l][int(-ref-1)][p+k] += filters[l][l+p+k]
                f_value[l][int(-ref-1)] += f
          
    f_out = f_value/counts
    sp = []
    for i in range(2*k+1):
        sp.append(J[...,i]/counts)
    st_out = np.array(sp)
    print(st_out.shape)
    
    return f_out, st_out

def Compare(ground_truth, weights, k):
    
    nb_samples = ground_truth.shape[0]
    N = ground_truth.shape[1]
    
    f_value = np.zeros((nb_samples, N))
    
    W_head = weights[...,:k]
    W_end = weights[...,-k:]
    kernels =  np.concatenate([W_end, weights, W_head], axis=-1)
    print(kernels.shape)
    
    mask = np.array([i for i in range(-k,k+1)])
    
    for i in range(nb_samples):
        filters = kernels[i]
        for l in range(N):
            f1 = 0
            for p in range(-k,k+1):
                f1 += mask[p+k]*filters[l][l+p+k]
            f_value[i][l] = f1
    print(f_value.shape)
    
    return f_value


def z_score(x, mean, std):
    return (x - mean) / std


def z_inverse(x, mean, std):
    return x * std + mean