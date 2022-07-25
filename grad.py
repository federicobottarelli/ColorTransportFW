import numpy as np


## define gradient and partial gradient function for the different algorithms

# gradient
def get_grad(T, C, a, lam=1e-6):
    # calculate second term of the gradient
    Id = np.ones(T.shape[1])
    p2 = np.dot(T, Id)-a

    # stack second term n times to fit the dimension of C
    for i in range(C.shape[1]):
        if not i:
            V = p2
        else:
            V = np.column_stack((V, p2))

    # calculate final gradient
    out = C + 1/lam*V
    return out

# partial gradient
def grad_part(T, C, a, i, lam=1e-6):
    # calculate the partial gradient for block i
    Id = np.ones(T.shape[1])
    out = C[:, i] + 1/lam*(np.dot(T, Id)-a)
    return out