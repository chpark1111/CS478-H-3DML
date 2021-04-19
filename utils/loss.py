import torch
import torch.nn
import numpy as np
import cv2

from sklearn.preprocessing import normalize

def cosine_similarity(arr1, arr2):
    arr1 = np.reshape(arr1, (arr1.shape[0], -1))
    arr2 = np.reshape(arr2, (arr2.shape[0], -1))
    arr1 = normalize(arr1, "l2", 1)
    arr2 = normalize(arr2, "l2", 1)

    similarity = np.multiply(arr1, arr2)
    similarity = np.sum(similarity, 1)
    
    return similarity

def sup_loss(pred, labels, time_steps):
    '''
    Supervised loss
    :params pred: (pg_len+1, batch_size, num_draws)
    :params lables: (batch_size, pg_len+1)
    '''
    nllloss = torch.nn.NLLLoss()
    loss = torch.zeros((1)).cuda()
    for i in range(time_steps):
        loss += nllloss(pred[i], labels[:, i])
    return loss

'''
Chamfer_distance by
https://github.com/Hippogriff/CSGNet
'''
def chamfer(images1, images2):
    """
    Chamfer distance on a minibatch, pairwise.
    :param images1: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :param images2: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :return: pairwise chamfer distance
    """
    # Convert in the opencv data format
    images1 = images1.astype(np.uint8)
    images1 = images1 * 255
    images2 = images2.astype(np.uint8)
    images2 = images2 * 255
    N = images1.shape[0]
    size = images1.shape[-1]

    D1 = np.zeros((N, size, size))
    E1 = np.zeros((N, size, size))

    D2 = np.zeros((N, size, size))
    E2 = np.zeros((N, size, size))
    summ1 = np.sum(images1, (1, 2))
    summ2 = np.sum(images2, (1, 2))

    # sum of completely filled image pixels
    filled_value = int(255 * size**2)
    defaulter_list = []
    for i in range(N):
        img1 = images1[i, :, :]
        img2 = images2[i, :, :]

        if (summ1[i] == 0) or (summ2[i] == 0) or (summ1[i] == filled_value) or (summ2[\
                i] == filled_value):
            # just to check whether any image is blank or completely filled
            defaulter_list.append(i)
            continue
        edges1 = cv2.Canny(img1, 1, 3)
        sum_edges = np.sum(edges1)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue
        dst1 = cv2.distanceTransform(
            ~edges1, distanceType=cv2.DIST_L2, maskSize=3)

        edges2 = cv2.Canny(img2, 1, 3)
        sum_edges = np.sum(edges2)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue

        dst2 = cv2.distanceTransform(
            ~edges2, distanceType=cv2.DIST_L2, maskSize=3)
        D1[i, :, :] = dst1
        D2[i, :, :] = dst2
        E1[i, :, :] = edges1
        E2[i, :, :] = edges2
    distances = np.sum(D1 * E2, (1, 2)) / (
        np.sum(E2, (1, 2)) + 1) + np.sum(D2 * E1, (1, 2)) / (np.sum(E1, (1, 2)) + 1)
    
    distances = distances / 2.0
    # This is a fixed penalty for wrong programs
    distances[defaulter_list] = 16
    return distances