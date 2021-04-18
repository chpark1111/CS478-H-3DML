import numpy as np

def to_onehot(x, n):
    '''
    :params x: batch_size, pg_len + 1
    '''
    sz = x.shape
    ret = np.zeros((sz[0], sz[1]+1, n+1))
    ret[:, 0, n] = 1
    for i in range(sz[0]):
        for j in range(sz[1]):
            ret[i][j+1][sz[i][j]]=1
    return ret