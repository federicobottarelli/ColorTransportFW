import numpy as np
from time import process_time_ns

from grad import get_grad
from grad import grad_part

from fw import fw_update

#  define bcafw step with line search
def block_away_fw_update_ELS(var, grad, a, b, C, rand_dim, S, lam):
    away = 0
    drop = 0
    
    # find frank-wolfe direction
    grad_min_index = np.argmin(grad)
    s_fw = np.zeros(var.shape[1])
    s_fw[grad_min_index] = b[rand_dim]
    d_fw = s_fw - var[:, rand_dim]
    
    # find away-step direction
    if S[:, rand_dim].sum() > 0:
        grad_max_index = np.where(grad == np.max(grad[S[:, rand_dim] == 1]))
        s_aw = np.zeros(var.shape[1])
        s_aw[grad_max_index] = b[rand_dim]
        d_aw = var[:, rand_dim] - s_aw
    
        # decide between fw and aw
        if np.dot(grad, d_fw) <= np.dot(grad, d_aw):
            # choose frank-wolfe step
            s_i = s_fw
            d = d_fw
            S[grad_min_index, rand_dim] = 1
            gamma_max = 1
        else:
            # choose away step
            s_i = s_aw
            d = d_aw
            away = 1
            alpha_i = var[grad_max_index, rand_dim]
            gamma_max = alpha_i/(1-alpha_i)
    else:
        s_i = s_fw
        d = d_fw
        S[grad_min_index, rand_dim] = 1
        gamma_max = 1
    
    # update iterate
    var_update = var.copy()
    
    # compute gamma by line search
    Id = np.ones(var.shape[1])
    d_norm2 = np.linalg.norm(d)**2
    gamma = -(lam*np.dot(d, C[:, rand_dim])+np.dot(d, (np.dot(var, Id)-a)))/d_norm2 # i or rand_dim
        
    # detect if gamma if above gamma_max
    if (gamma>gamma_max):
        # if we do an away-step with gamma_max we remove an atom and do a drop-step
        if away:
            drop = 1
            S[grad_max_index, rand_dim] = 0
        
        # set gamma to gamma max
        gamma = gamma_max
        
    var_update[:, rand_dim] = (1-gamma)*var[:, rand_dim] + gamma*s_i  
    
    if drop:
        var_update[grad_max_index, rand_dim] = 0
    
    return s_i, var_update, away, d, S, drop

def min_block_away_fw_ELS(var, a, b, C, epoch, gamma = 0.8, threshold = 10 ** (-7)):
    # initiate variables
    start_time = process_time_ns()
    it = 0
    aw = 0
    drop_steps = 0
    S = np.zeros(var.shape)
    S[:, 0] = np.ones(var.shape[1])
    errors = []
    gradient_norms = []
    gammas = []
    gammas_max = []
    aways = []
    
    # loop over epochs
    for k in range(epoch):
        # update iterator, stepsize and block
        it += 1
        
        # gamma = (2 * b.size)/(k + 2 * b.size)
        rand_dim = np.random.randint(0, b.size)
        
        # prevent updating the same column twice in a row
        if (k!=0):
            while rand_dim == old_dim:
                rand_dim = np.random.randint(0, b.size)
        # calculate partial gradient for the selected block
        grad = grad_part(var, C, a, rand_dim)
        old_dim = rand_dim
        # perfrom fw or away step
        s, new_var, away, d, S, drop = block_away_fw_update_ELS(var, grad, a, b, C, rand_dim, S, lam=1e-6)
        aw += away
        drop_steps += drop
        
        # calculate err
        if (k%var.shape[0] == 0):
            # calculate full gradient and FW direction to evaluate the dual gap
            gam = 2/(k+2)
            g = get_grad(var, C, a)
            _, d = fw_update(var, g, a, b, gam)

            # calculate the error
            err = (-g*d).sum()
            errors.append(err)
            gradient_norms.append(np.linalg.norm(g))
            aways.append(away)
            
            if err < threshold:
                t = (process_time_ns() - start_time) / 1e9
                return var, it, errors, aw, S, gradient_norms, drop_steps, aways, t
            
        var = new_var
        t = (process_time_ns() - start_time) / 1e9
    return var, it, errors, aw, S, gradient_norms, drop_steps, aways, t