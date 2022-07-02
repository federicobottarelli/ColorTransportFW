import numpy as np

from time import process_time_ns


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

## define the fw algorithm
# define the fw step
def fw_update(var, grad, a, b, gamma):
    # get steepest gradient descent direction for each of the columns
    grad_min_index = np.argmin(grad, axis = 0)
    var_update = var.copy()

    # initiate mask to update variable; fill mask at the positions given by
    # the gradient with the correct atom b[i]
    s = np.zeros(var.shape)
    for i, g in enumerate(grad_min_index):
        s[g, i] = b[i]

    # calculate direction
    d_fw = s - var

    # update variable
    var_update = (1-gamma)*var + gamma*s
    return var_update, d_fw

# define the whole fw algorithm calling the fw step iteratively
def min_fw(var, a, b, C, epoch, gamma = 0.8, threshold = 10 ** (-7)):
    # initiate variables
    start_time = process_time_ns()
    it = 0
    errors = []
    gradient_norms = []

    # loop over epochs
    for k in range(epoch):

        # update iteration and stepsize
        it += 1
        gamma = 2/(k+2)

        # calculate gradient and perform the FW step
        grad = get_grad(var, C, a)
        new_var, d_fw = fw_update(var, grad, a, b, gamma)

        # calculate the error
        err = (-grad*d_fw).sum()
        errors.append(err)
        gradient_norms.append(np.linalg.norm(grad))
        # preliminary stopping condition
        if (err < threshold):
            t = (process_time_ns() - start_time) / 1e9
            return var, it, errors, gradient_norms, t

        # update variable
        var = new_var
        t = (process_time_ns() - start_time) / 1e9
    return var, it, errors, gradient_norms, t
