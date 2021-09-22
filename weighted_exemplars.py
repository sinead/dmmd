import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import pdb
from scipy.special import logsumexp
import pdb
import time


def calculate_mmd(K, X_inds, Y_inds, Y_weights=None, XX_part=None):
    # Calculate (biased estimate of) MMD between X_inds and Y_inds
    # Y_weights contains optional weights for Y_inds
    # XX_part optionally pre-calculates XX_part
    # kernel must contain all indices
    
    Kxx = K[X_inds, :][:, X_inds]
    Kyy = K[Y_inds, :][:, Y_inds]
    Kxy = K[X_inds, :][:, Y_inds]
    
    N = len(X_inds)
    if XX_part is None:
        XX_part = np.sum(Kxx)/(N**2)
    else:
        XX_part_tmp = np.sum(Kxx)/(N**2)
        if XX_part != XX_part_tmp:
            pdb.set_trace()
    if Y_weights is None:
        M = len(Y_inds)
        YY_part = np.sum(Kyy)/(M**2)
        XY_part = np.sum(Kxy)/(M*N)
    else:
        YY_part = np.dot(Y_weights, np.dot(Kyy, Y_weights))
        XY_part = np.sum(np.dot(Kxy, Y_weights))/N
    
    mmd = XX_part + YY_part - 2*XY_part
    return mmd


def get_scores(K, target_indices, other_exemplars, candidates, colsum=None):
    # calculate contributions to MMD
    # based on code from https://github.com/BeenKim/MMD-critic
    if colsum is None:
        n = len(target_indices)
        colsum = 2*np.sum(K[target_indices, :], axis=0) / n
    s1array =  colsum[candidates]
    if len(other_exemplars) > 0:
        temp = K[other_exemplars, :][:, candidates]
        s2array = (np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates])
        s2array = s2array/(len(other_exemplars))
        s1array = s1array - s2array
    else:
        s1array = s1array - (np.abs(np.diagonal(K)[candidates]))
    return s1array

def basic_exemplars(K, target_indices, M=None, candidate_indices = None, colsum=None):
    '''
    MMD-critic algorithm for generating exemplars
    modified from https://github.com/BeenKim/MMD-critic
    K = square kernel matrix, containing at a minimum rows corresponding to target_indices and candidate_indices
    target_indices: indices of K corresponding to our target empirical distribution
    candidate indices: indices of K corresponding to candidate exemplars (if None, use all indices)
    colsum: (optional) pre-calculation of 2 * np.sum(K[target_indices, :], axis=0) / len(target_indices)
    '''
    if candidate_indices is None:
        candidate_indices = np.arange(K.shape[0])
   
    if colsum is None:
        n = len(target_indices)
        colsum = 2*np.sum(K[target_indices, :], axis=0) / n
    selected = np.array([], dtype=int)
    for i in range(M):
        candidates = np.setdiff1d(candidate_indices, selected)

        scores = get_scores(K, target_indices, selected, candidates, colsum=colsum)
        argmax = candidates[np.argmax(scores)]
        selected = np.append(selected, argmax)
    for i in range(M):
        selected = selected[1:]
        candidates = np.setdiff1d(candidate_indices, selected)
        scores = get_scores(K, target_indices, selected, candidates, colsum=colsum)
        argmax = candidates[np.argmax(scores)]
        selected = np.append(selected, argmax)

    return selected

def weighted_exemplars(K, targets, M=None, time_limit=None, MMD_threshold=None, candidate_indices=None, relearn_weights=False, return_mmds = False):
    '''
    Selects exemplars and weights
    K:  square kernel matrix, containing at a minimum rows corresponding to candidate_indices and all the indices in targets
    targets: list of target_indices for each target
    M: Number of exemplars to use. If None, MMD_threshold must be set.
    time_limit: max run time in seconds (actually, latest time to start last iteration)
    MMD_threshold: target MMD between weighted exemplars and targets
    NOTE: If both M and MMD_threshold are specified, M will be used
    candidate indices: indices of K corresponding to candidate exemplars (if None, use all indices)
    relearn_weights: If False, the relative weights of previously observed exemplar are never re-learned. If True, we re-learn after each exemplar.
    return_mmds: If true, return MMDs and times at each step
    '''
    if M is None:
        assert MMD_threshold is not None
    else:
        if MMD_threshold is not None:
            print('Both M and MMD_threshold specified; using M and ignoring MMD_threshold')
            MMD_threshold = None
    kernels = [K[target, :][:, target] for target in targets]
    nn = [len(target) for target in targets]
    num_targets = len(targets)

    weights = {}
    for i in range(num_targets):
        weights[i] = np.ones(1)

    if candidate_indices is None:
        candidate_indices = np.arange(K.shape[0])

    colsums = [np.sum(K[target, :], axis=0) for target in targets]

    if (MMD_threshold is not None) or (return_mmds):
        
        XX_part = [np.sum(kernels[i])/(nn[i] ** 2) for i in range(num_targets)]
    else:
        XX_part = None
        
    ## get the initial exemplar
    if candidate_indices is None:
        candidate_indices = np.arange(K.shape[0])


    tic = time.time()
    times = []
    scores = np.sum(np.array([cs[candidate_indices] for cs in colsums]), axis=0)  # m=1, so no need to do anything beyond mean distance
    exemplars = np.array([candidate_indices[np.argmax(scores)]])

    exemplars_complete = False

    targets_to_include = np.ones(num_targets)
    if return_mmds:
        try:
            mmds = [[calculate_mmd(K, targets[t], exemplars, Y_weights=weights[t], XX_part=XX_part[t]) for t in range(num_targets)]]# could speed up more to include colsum for XY part
            times.append(time.time() - tic)
        except TypeError:
            pdb.set_trace()
                    
    while exemplars_complete is False:
        print('adding exemplar {}'.format(len(exemplars) + 1))
        candidates = np.setdiff1d(candidate_indices, exemplars)

        exemplar_kernel = K[exemplars, :][:, exemplars]
        temp = [np.expand_dims(weights[i], 1) * K[exemplars, :] for i in range(num_targets)]
        M1 = [np.sum(exemplar_kernel * np.outer(weights[i], weights[i])) for i in range(num_targets)]
        M2 = [2*np.sum(temp[i][:, candidates], axis=0) for i in range(num_targets)]
        M3 = [-2*np.sum(temp[i][:, targets[i]])/nn[i] for i in range(num_targets)]
        M4 = [-2*colsums[i][candidates]/nn[i] for i in range(num_targets)]
            
        ## calculate optimal w...

        ww = {}
        scores = 0
        
        for t in range(len(targets)):
            ww[t] = [np.maximum(0, (-2*M1[t] + M2[t][j] - M3[t] + M4[t][j]) / (M2[t][j] - 2 + M3[t] - M4[t][j])) for j in range(len(candidates))]
            
            ww[t] = [0 if np.isnan(x) else x for x in ww[t]]
            
            if targets_to_include[t]:
                tmp = [M1[t]/((1+ww[t][j])**2) + M2[t][j]*ww[t][j]/((1+ww[t][j])**2) + (ww[t][j]/(1+ww[t][j]))**2 + M3[t]/(1+ww[t][j]) + ww[t][j]*M4[t][j]/(1+ww[t][j]) for j in range(len(candidates))]
                scores += np.array(tmp)

                
        assert len(scores) == len(ww[0])

        
        new_ind = np.argmin(scores)
        exemplars = np.append(exemplars, candidates[new_ind])
        
        for t in range(len(targets)):
            if optimal_weights:
                if np.isnan(ww[t][new_ind]):
                    pdb.set_trace()
                weights[t] = np.append(weights[t], ww[t][new_ind])
                weights[t] = weights[t]/ np.sum(weights[t])
            else:
                weights[t] = np.append(weights[t], 1)
                weights[t] = weights[t]/np.sum(weights[t])
                relearn_weights=True

        if relearn_weights:
            for t in range(len(targets)):
                weights[t] = np.maximum(weights[t], 0.0001)
                if np.any(np.isnan(weights[t])):
                    pdb.set_trace()
                weights[t] = improve_weights(K, targets[t], exemplars, weights[t], normalize=normalize)
        if MMD_threshold is not None:
            for t in range(num_targets):
                if targets_to_include[t]:
                    MMD = calculate_mmd(K, targets[t], exemplars, Y_weights=weights[t], XX_part=XX_part[t]) # could speed up more to include colsum for XY part
                    
                        
                    if MMD < MMD_threshold:
                        targets_to_include[t] = 0
                else: # sanity check, remove later
                    MMD = calculate_mmd(K, targets[t], exemplars, Y_weights=weights[t])#, XX_part=XX_part[t])
                    if MMD > MMD_threshold:
                        pdb.set_trace()
            if np.sum(targets_to_include) == 0:
                exemplars_complete = True
        else:                      
            if len(exemplars) == M:
                exemplars_complete = True
        if time_limit is not None:
            print(time.time() - tic)
            if time.time() - tic > time_limit:
                exemplars_complete = True
        if return_mmds:
            mmds.append([calculate_mmd(K, targets[t], exemplars, Y_weights=weights[t], XX_part=XX_part[t]) for t in range(num_targets)])
            times.append(time.time() - tic)
    
    if return_mmds:
        return exemplars, weights, mmds, times
    return exemplars, weights



def calculate_partial_mmd_weights(K, X_inds, Y_inds, Y_weights, colsum_X_S):
    # Calculate (biased estimate of) MMD between X_inds and Y_inds
    # kernel must contain all indices

    k2 = K[Y_inds, :][:, Y_inds]
    part_b= np.dot(Y_weights, np.dot(k2, Y_weights))
    part_c = np.dot(colsum_X_S, Y_weights) / X_inds.shape[0]
    
    mmd_part = part_b - 2*part_c
    return mmd_part

def improve_weights(K, targets, exemplars, initial_weights, normalize=True):
    k1 = K[targets, :][:, targets]
    mmd_offset = np.sum(k1)/(len(targets)**2)
    colsum = np.sum(K[targets, :], axis=0)
    initial_scores = np.log(initial_weights)
    def loss_wrapper(scores):
        weights = np.exp(scores)
        
        if normalize:
            weights = weights /np.sum(weights)
        out = mmd_offset + calculate_partial_mmd_weights(K, targets, exemplars, weights, colsum[exemplars])
        return out
    loss_grad = grad(loss_wrapper)
    res = minimize(loss_wrapper, initial_scores, method='BFGS', jac=loss_grad,
                   options={'gtol': 1e-6, 'disp': False})
    new_weights = np.exp(res.x - logsumexp(res.x))
    
    new_weights = new_weights/np.sum(new_weights)
    if np.any(np.isnan(new_weights)):
        pdb.set_trace()
    return new_weights


def protodash(K, targets, time_limit=None,  M=None, MMD_threshold=None, candidate_indices=None, return_mmds = False):
    '''
    dependent version of the protodash algorithm
    params and outputs as per weighted_exemplars
    '''
    if M is None:
        assert MMD_threshold is not None
    else:
        if MMD_threshold is not None:
            print('Both M and MMD_threshold specified; using M and ignoring MMD_threshold')
            MMD_threshold = None

    kernels = [K[target, :][:, target] for target in targets]
    nn = [len(target) for target in targets]
    num_targets = len(targets)

    if candidate_indices is None:
        candidate_indices = np.arange(K.shape[0])

    mean_embeddings = np.array([np.sum(K[targets[t], :], axis=0) / nn[t] for t in range(num_targets)])
    tic = time.time()
    weights = {}
    for i in range(num_targets):
        weights[i] = np.array([])

    g = 0.
    for t in range(num_targets):
        g += mean_embeddings[t][candidate_indices]
    
    exemplars = np.array([]).astype(int)
    if return_mmds:
        mmds = []
        times = []
        XX_part = [np.sum(kernels[i])/(nn[i] ** 2) for i in range(num_targets)]
    elif MMD_threshold is not None:
        XX_part = [np.sum(kernels[i])/(nn[i] ** 2) for i in range(num_targets)]

    exemplars_complete = False

    targets_to_include = np.ones(num_targets)
    
    while exemplars_complete is False:
        try:
            new_exemplar = candidate_indices[np.argmax(g)]
        except IndexError:
            pdb.set_trace()
        print('adding exemplar {}'.format(len(exemplars)+1))
        exemplars = np.append(exemplars, new_exemplar)
        for t in range(num_targets):
            weights[t] = np.ones(len(exemplars))/len(exemplars)
            try:
                weights[t] = improve_weights(K, targets[t], exemplars, weights[t], normalize=False)
            except IndexError:
                pdb.set_trace()
        candidate_indices = np.setdiff1d(candidate_indices, new_exemplar)


        g = 0.
        for t in range(num_targets):
            if targets_to_include[t]:
                g += mean_embeddings[t][candidate_indices] - np.dot(K[candidate_indices, :][:, exemplars], weights[t])

        if MMD_threshold is not None:
            for t in range(num_targets):
                if targets_to_include[t]:
                    MMD = calculate_mmd(K, targets[t], exemplars, Y_weights=weights[t], XX_part=XX_part[t]) # could speed up more to include colsum for XY part
                    
                    if MMD < MMD_threshold:
                        targets_to_include[t] = 0
                else: # sanity check, remove later
                    MMD = calculate_mmd(K, targets[t], exemplars, Y_weights=weights[t], XX_part=XX_part[t])
                    if MMD > MMD_threshold:
                        pdb.set_trace()
            if np.sum(targets_to_include) == 0:
                exemplars_complete = True
        else:                      
            if len(exemplars) == M:
                exemplars_complete = True

        if time_limit is not None:
            if time.time() - tic > time_limit:
                exemplars_complete = True
        print(time.time() - tic)
        if return_mmds:
            mmds.append([[calculate_mmd(K, targets[t], exemplars, Y_weights=weights[t], XX_part=XX_part[t]) for t in range(num_targets)]])
            times.append(time.time()-tic)
            #print(times[-1])
    if return_mmds:
        return exemplars, weights, mmds, times
    return exemplars, weights
    
