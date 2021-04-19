import numpy as np
import torch
from typing import List

def to_onehot(x, n):
    '''
    Changes input sequence to one-hot vector
    :params x: batch_size, pg_len + 1
    :params n: num_draws
    '''
    sz = x.shape
    ret = np.zeros((sz[0], sz[1]+1, n+1))
    ret[:, 0, n] = 1
    for i in range(sz[0]):
        for j in range(sz[1]):
            ret[i][j+1][x[i][j]]=1

    return ret

def validity(program: List, max_time: int, timestep: int):
    """
    Checks the validity of the program. In short implements a pushdown automaton that accepts valid strings.
    :param program: List of dictionary containing program type and elements
    :param max_time: Max allowed length of program
    :param timestep: Current timestep of the program, or in a sense length of
    program
    # at evey index
    :return:
    """
    num_draws = 0
    num_ops = 0
    for i, p in enumerate(program):
        if p["type"] == "draw":
            # draw a shape on canvas kind of operation
            num_draws += 1
        elif p["type"] == "op":
            # +, *, - kind of operation
            num_ops += 1
        elif p["type"] == "stop":
            # Stop symbol, no need to process further
            if num_draws > ((len(program) - 1) // 2 + 1):
                return False
            if not (num_draws > num_ops):
                return False
            return (num_draws - 1) == num_ops

        if num_draws <= num_ops:
            # condition where number of operands are lesser than 2
            return False
        if num_draws > (max_time // 2 + 1):
            # condition for stack over flow
            return False
    if (max_time - 1) == timestep:
        return (num_draws - 1) == num_ops
    return True