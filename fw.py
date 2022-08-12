import numpy as np
from grad import get_grad
from time import process_time_ns





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
