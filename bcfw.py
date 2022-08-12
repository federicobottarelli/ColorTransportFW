import numpy as np
from time import process_time_ns

from grad import get_grad
from grad import grad_part

from fw import fw_update

#define bcfw step 
def block_fw_update(var, grad, a, b, gamma, rand_dim):
    # get steepest gradient descent direction for the selected block
    grad_min_index = np.argmin(grad)
    var_update = var.copy()
    
    # initiate mask to update variable; 
    s_i = np.zeros(var.shape[1])
    s_i[grad_min_index] = b[rand_dim]
    d_fw = s_i - var[:, rand_dim]
    
    # update block with new direction 
    var_update[:, rand_dim] = (1-gamma)*var[:, rand_dim] + gamma*s_i
    
    return s_i, var_update, d_fw

# define bcfw algorithm
def min_block_fw(var, a, b, C, epoch, gamma = 0.8, threshold = 10 ** (-7)):
    # initiate variables
    start_time = process_time_ns()
    it = 0
    errors = []
    gradient_norms = []
    for k in range(epoch):
        it += 1
        gamma = (2 * b.size)/(k + 2 * b.size)
        rand_dim = np.random.randint(0, b.size)
        grad = grad_part(var, C, a, rand_dim)
        s, new_var, d_fw = block_fw_update(var, grad, a, b, gamma, rand_dim)
        # err = np.abs(new_var - var).sum()
        if (k%var.shape[0] == 0):
            # calculate full gradient and FW direction to evaluate the dual gap
            g = get_grad(var, C, a)
            _, d = fw_update(var, g, a, b, gamma)

            # calculate the error
            err = (-g*d).sum()
            errors.append(err)
            gradient_norms.append(np.linalg.norm(g))
            if (err < threshold):
                t = (process_time_ns() - start_time) / 1e9
                return var, it, errors, gradient_norms, t
        var = new_var     
        t = (process_time_ns() - start_time) / 1e9
    return var, it, errors, gradient_norms, t