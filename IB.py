import numpy as np
import pandas as pd
import time
import math
vlog = np.vectorize(math.log)
vexp = np.vectorize(math.exp)

# TODOS
# add normalization check to entropy and KL
# set max number of steps/time for single beta?
# make work with sparse data
# improve parallelization
# make IB init that perfectly interpolates between wavy and uni

# A word on notation: for probability variables, an underscore here means a
# conditioning, so read _ as |.

def verify_inputs(pxy,beta,alpha,Tmax,p0,ctol_abs,ctol_rel,ptol,zeroLtol,clamp,compact,verbose):
    """Helper function for IB which checks for validity of input parameters."""
    if not(isinstance(pxy,np.ndarray)):
        raise ValueError('pxy must be a numpy array')
    if np.any(pxy<0) or np.any(pxy>1):
        raise ValueError('entries of pxy must be between 0 and 1')
    if abs(np.sum(pxy)-1)>ptol:
        raise ValueError('pxy must be normalized')
    if not(beta>0) or not(isinstance(beta,(int,float))):
        raise ValueError('beta must be a positive scalar')
    if alpha<0 or not(isinstance(alpha,(int,float))):
        raise ValueError('alpha must be a non-negative scalar')
    if Tmax<1:
        raise ValueError('Tmax must be a positive integer (or infinity)')
    if p0<-1 or p0>1 or not(isinstance(p0,(int,float))):
        raise ValueError('p0 must be a float/int between -1 and 1')
    if not(ctol_abs>=0) or not(isinstance(ctol_abs,float)):
        raise ValueError('ctol_abs must be a non-negative float')        
    if not(ctol_rel>=0) or not(isinstance(ctol_rel,float)):
        raise ValueError('ctol_rel must be a non-negative float')
    if (ctol_rel==0) and (ctol_abs==0):
        raise ValueError('One of ctol_rel and ctol_abs must be postive')        
    if not(ptol>0) or not(isinstance(ptol,float)):
        raise ValueError('ptol must be a positive float')
    if zeroLtol<0:
        raise ValueError('zeroLtol must be positive')
    if not(isinstance(clamp,bool)):
        raise ValueError('clamp must be a boolean')
    if not(verbose in (0,1,2)):
        raise ValueError('verbose should be 0, 1, or 2')
    if not(compact in (0,1,2)):
        raise ValueError('compact should be 0, 1, or 2')
    return 0
    
def entropy_term(x):
    """Helper function for entropy_single: calculates one term in the sum."""
    if x==0:
        return 0.0
    else:
        return -x*math.log2(x)

def entropy_single(p):
    """Returns entropy of p: H(p)=-sum(p*log(p)). (in bits)"""
    ventropy_term = np.vectorize(entropy_term)
    vec = ventropy_term(p)
    h = np.sum(vec)
    return h

def entropy(P):
    """Returns entropy of a distribution, or series of distributions.
    
    For the input array P [=] M x N, treats each col as a prob distribution
    (over M elements), and thus returns N entropies. If P is a vector, treats
    P as a single distribution and returns its entropy."""
    if P.ndim==1:
        return entropy_single(P)
    else:
        M,N = P.shape
        H = np.zeros(N)
        for n in range(N):
            H[n] = entropy_single(P[:,n])
        return H

def process_pxy(pxy,verbose=1):
    """Helper function for IB that preprocesses p(x,y) and computes metrics."""
    if pxy.dtype!='float':
        pxy = pxy.astype(float)
    Xorig = pxy.shape[0]
    Yorig = pxy.shape[1]
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nzx = px>0 # find nonzero-prob entries
    nzy = py>0
    zx = np.where(px<=0)
    zx = zx[0]
    zy = np.where(py<=0)
    zy = zy[0]
    px = px[nzx] # drop zero-prob entries
    py = py[nzy]
    X = len(px)
    Y = len(py)
    if verbose>0 and (Xorig-X)>0:
        print('%i of %i Xs dropped due to zero prob; size now %i. Dropped IDs:' % (Xorig-X,Xorig,X))
        print(zx)
    if verbose>0 and (Yorig-Y)>0:
        print('%i of %i Ys dropped due to zero prob; size now %i. Dropped IDs:' % (Yorig-Y,Yorig,Y))
        print(zy)
    pxy_orig = pxy
    tmp = pxy_orig[nzx,:]
    pxy = tmp[:,nzy] # pxy_orig with zero-prob x,y removed
    py_x = np.multiply(pxy.T,np.tile(1./px,(Y,1)))
    hx = entropy(px)
    hy = entropy(py)
    hy_x = np.dot(px,entropy(py_x))
    ixy = hy-hy_x
    return pxy, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy

def kl_term(x,y):
    """Helper function for kl: calculates one term in the sum."""
    if x>0 and y>0:
        return x*math.log2(x/y)
    elif x==0:
        return 0.0
    else:
        return math.inf
    
def kl_single(p,q):
    """Returns KL divergence of p and q: KL(p,q)=sum(p*log(p/q)). (in bits)"""
    vkl_term = np.vectorize(kl_term)
    vec = vkl_term(p,q)
    dkl = np.sum(vec)
    return dkl
    
def kl(P,Q):
    """Returns KL divergence of one or more pairs of distributions.
    
    For the input arrays P [=] M x N and Q [=] M x L, calculates KL of each col
    of P with each col of Q, yielding the KL matrix DKL [=] N x L. If P=Q=1,
    returns a single KL divergence."""
    if P.ndim==1 and Q.ndim==1:
        return kl_single(P,Q)
    elif P.ndim==1 and Q.ndim!=1: # handle vector P case
        M = len(P)
        N = 1
        M2,L = Q.shape
        if M!=M2:
            raise ValueError("P and Q must have same number of columns")
        DKL = np.zeros((1,L))
        for l in range(L):
            DKL[0,l] = kl_single(P,Q[:,l])
    elif P.ndim!=1 and Q.ndim==1: # handle vector Q case
        M,N = P.shape
        M2 = len(Q)
        L = 1
        if M!=M2:
            raise ValueError("P and Q must have same number of columns")
        DKL = np.zeros((N,1))
        for n in range(N):
            DKL[n,0] = kl_single(P[:,n],Q)
    else:
        M,N = P.shape
        M2,L = Q.shape        
        if M!=M2:
            raise ValueError("P and Q must have same number of columns")    
        DKL = np.zeros((N,L))
        for n in range(N):
            for l in range(L):
                DKL[n,l] = kl_single(P[:,n],Q[:,l])
    return DKL

def qt_step(qt_x,px,ptol,verbose):
    """Peforms q(t) update step for generalized Information Bottleneck."""
    T, X = qt_x.shape   
    qt = np.dot(qt_x,px)
    dropped = qt<=ptol # clusters to drop due to near-zero prob
    if any(dropped):
        qt = qt[~dropped] # drop ununsed clusters
        qt_x = qt_x[~dropped,:]
        T = len(qt) # update number of clusters
        qt_x = np.multiply(qt_x,np.tile(1./np.sum(qt_x,axis=0),(T,1))) # renormalize
        qt = np.dot(qt_x,px)
        if verbose==2:
            print('%i cluster(s) dropped. Down to %i cluster(s).' % (np.sum(dropped),T)) 
    return qt_x,qt,T
    
def qy_t_step(qt_x,qt,px,py_x):
    """Peforms q(y|t) update step for generalized Information Bottleneck."""
    qy_t = np.dot(py_x,np.multiply(qt_x,np.outer(1./qt,px)).T)
    return qy_t

def qt_x_step(qt,py_x,qy_t,T,X,alpha,beta,verbose):
    """Peforms q(t|x) update step for generalized Information Bottleneck."""
    if T==1: # no need for computation
        qt_x = np.ones((1,X))
    else:
        qt_x = np.zeros((T,X))
        for x in range(X):
            l = vlog(qt)-beta*kl(py_x[:,x],qy_t) # [=] T x 1 # scales like X*Y*T
            if alpha==0: # DIB
                qt_x[np.argmax(l),x] = 1
            else:  # IB and interpolations
                qt_x[:,x] = vexp(l/alpha)/np.sum(vexp(l/alpha)) # note: l/alpha<-745 is where underflow creeps in
    return qt_x
    
def init_qt_x(alpha,X,T,p0):
    """Initializes q(t|x) for generalized Information Bottleneck."""
    if alpha==0: # DIB: spread points evenly across clusters
        n = math.ceil(float(X)/float(T)) # approx number points per cluster
        I = np.repeat(np.arange(0,T),n).astype("int") # data-to-cluster assignment vector
        np.random.shuffle(I)
        qt_x = np.zeros((T,X))
        for i in range(X):
            qt_x[I[i],i] = 1
    else: # not DIB
        if p0==0: # normalized uniform random vector
            qt_x = np.random.rand(T,X)
            qt_x = np.multiply(qt_x,np.tile(1./np.sum(qt_x,axis=0),(T,1))) # renormalize
        # others are spike plus wavy: if pos, spread peaks around like DIB init;
                                    # if neg, put all peaks on same cluster
        elif p0>0: # noisy approx to DIB init
            wavy = False
            if wavy:
                # insert wavy noise part
                f = .25; # max percent variation of flat part of dist around mean           
                qt_x = np.ones((T,X))+2*(np.random.rand(T,X)-.5)*f # 1+-f%
                # choose clusters for each x to get spikes
                n = math.ceil(float(X)/float(T)) # approx number points per cluster
                I = np.repeat(np.arange(0,T),n).astype("int") # data-to-cluster assignment vector
                np.random.shuffle(I)
                for i in range(X):
                    qt_x[I[i],i] = 0 # zero out that cluster
                    qt_x[:,i] = (1-p0)*qt_x[:,i]/np.sum(qt_x[:,i]) # normalize others to 1-p0
                    qt_x[I[i],i] = p0 # insert p0 spike
            else: # uniform random vector instead of wavy
                qt_x = np.zeros((T,X))
                # choose clusters for each x to get spikes
                n = math.ceil(float(X)/float(T)) # approx number points per cluster
                I = np.repeat(np.arange(0,T),n).astype("int") # data-to-cluster assignment vector
                np.random.shuffle(I)
                for i in range(X):
                    u = np.random.rand(T)
                    u[I[i]] = 0
                    u = (1-p0)*u/np.sum(u)
                    u[I[i]] = p0
                    qt_x[:,i] = u
        else: # same as above, but all x assigned same cluster for spike
            wavy = False
            if wavy:
                p0 = -p0
                # p0 = prob on spike, mean prob elsewhere = (1-p0)/(T-1)
                f = .25; # max percent variation of flat part of dist around mean           
                qt_x = np.ones((T,X))+2*(np.random.rand(T,X)-.5)*f # 1+-f%
                t = np.random.randint(T) # pick cluster to get delta spike
                qt_x[t,:] = np.zeros((1,X)) # zero out that cluster
                qt_x = np.multiply(qt_x,np.tile(1./np.sum(qt_x,axis=0),(T,1))) # normalize the rest...
                qt_x = (1-p0)*qt_x # ...to 1-p0
                qt_x[t,:] = p0*np.ones((1,X)) # put in delta spike
            else: # uniform random vector instead of wavy
                p0 = -p0
                qt_x = np.zeros((T,X))
                # choose clusters for each x to get spikes
                t = np.random.randint(T) # pick cluster to get delta spike
                for i in range(X):
                    u = np.random.rand(T)
                    u[t] = 0
                    u = (1-p0)*u/np.sum(u)
                    u[t] = p0
                    qt_x[:,i] = u                
    return qt_x
    
def calc_IB_metrics(qt_x,qt,qy_t,px,hy,alpha,beta):
    """Calculates IB performance metrics."""
    ht = entropy(qt)
    hy_t = np.dot(qt,entropy(qy_t))
    iyt = hy-hy_t
    ht_x = np.dot(px,entropy(qt_x))
    ixt = ht-ht_x
    L = ht-alpha*ht_x-beta*iyt
    return ht, hy_t, iyt, ht_x, ixt, L 

def IB_single(pxy,beta,alpha,Tmax,p0,ctol_abs,ctol_rel,ptol,zeroLtol,clamp,compact,verbose):
    """Performs the generalized Information Bottleneck on the joint p(x,y).
    
    Note: fixed distributions denoted by p; optimized ones by q.
    
    INPUTS
    pxy = p(x,y) [=] X x Y
    beta = tradeoff param between compression/relevance [=] pos scalar
    alpha = tradeoff param between DIB and IB [=] non-neg scalar
    Tmax = max alphabet size for compressed representation, aka max number of
            clusters; is set to min(X,Tmax)
    p0 = determines initialization if alpha>0 (i.e. non-DIB); see README [=] scalar in -1 to +1
    ctol_abs = absolute convergence tolerance [=] non-neg scalar
    ctol_rel = relative convergence tolerance [=] non-neg scalar
    ptol = error tolerance for probabilities; distributions considered
        normalized within ptol of 1, and clusters considered unused within
        ptol of 0 [=] pos scalar
    zeroLtol = if algorithm converges with L>zeroLtol, revert to trivial L=0
        solution with all x assigned to single cluster [=] non-neg scalar
    compact = integer indicating how much data to save [=] 0/1/2
    verbose = integer indicating verbosity of updates [=] 0/1/2
    
    OUTPUTS
    metrics_stepwise = dataframe of scalar metrics for each fit step:
        L = objective function value [=] scalar
        ixt = I(X,T) [=] scalar
        iyt = I(Y,T) [=] scalar
        ht = H(T) [=] scalar
        T = number of clusters used [=] pos integer
        ht_x = H(T|X) [=] scalar
        hy_t = H(Y|T) [=] scalar
        hx = H(X) [=] scalar
        ixy = I(X,Y) [=] scalar
        step = index of fit step [=] pos integer
        step_time = time to complete this step (in s) [=] pos scalar
        hx = H(X) [=] scalar
        ixy = I(X,Y) [=] scalar
        Tmax
        beta
        alpha
        p0
        ctol
        ptol
    distributions_stepwise = dataframe of optimized distributions for each step:
        qt_x = q(t|x) = [=] T x X (note: size T changes during iterations)
        qt = q(t) [=] T x 1 (note: size T changes during iterations)
        qy_t = q(y|t) [=] Y x T (note: size T changes during iterations)
        step = index of fit step [=] pos integer 
        Tmax
        beta
        alpha
        p0
        ctol
        ptol
    metrics_converged = dataframe of last (converged) step for each Tmax/fit above:
        step_time -> conv_time = time to run all steps (in s)
        step -> conv_steps = number of steps to converge
        conv_condition = string indicating reason for convergence [=] {cost_func_inc,small_changes,single_cluster,cost_func_NaN}
    distributions_converged = dataframe of last (converged) step for each Tmax/fit above:
        step_time -> conv_time = time to run all steps (in s)
        step -> conv_steps = number of steps to converge
        conv_condition = string indicating reason for convergence"""
    
    verify_inputs(pxy,beta,alpha,Tmax,p0,ctol_abs,ctol_rel,ptol,zeroLtol,clamp,compact,verbose)
    
    conv_thresh = 1 # steps in a row of small change to consider converged
    
    # process inputs
    if isinstance(alpha,int):
        alpha = float(alpha)
    if isinstance(beta,int):
        beta = float(beta)        
    pxy, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy = process_pxy(pxy,verbose)
    if Tmax==math.inf:
        Tmax = X
        if verbose==2:
            print('Tmax set to %i based on X' % Tmax)
    elif Tmax>X:
        if verbose==2:
            print('Reduced Tmax from %i to %i based on X' % (Tmax,X))
        Tmax = X
    else:
        Tmax = int(Tmax)
        
    # initialize dataframes
    metrics_stepwise = pd.DataFrame(columns=['L','ixt','iyt','ht','T','ht_x',
                                             'hy_t','step','step_time'])
    if compact>1:
        distributions_stepwise = pd.DataFrame(columns=['qt_x','qt','qy_t','step'])

    # initialize other stuff
    T = Tmax
    step_start_time = time.time()
    # STEP 0: INITIALIZE
    # initialize q(t|x)
    qt_x = init_qt_x(alpha,X,T,p0)
    # initialize q(t) given q(t|x)
    qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)
    # initialize q(y|t) given q(t|x) and q(t)
    qy_t = qy_t_step(qt_x,qt,px,py_x)
    # calculate and print metrics
    if verbose==2:
        print('IB initialized')
    ht, hy_t, iyt, ht_x, ixt, L = calc_IB_metrics(qt_x,qt,qy_t,px,hy,alpha,beta)
    if verbose==2:
        print('I(X,T) = %.6f, H(T) = %.6f, H(X) = %.6f, I(Y,T) = %.6f, I(X,Y) = %.6f, L = %.6f' % (ixt,ht,hx,iyt,ixy,L))
    step_time = time.time() - step_start_time
    if len(metrics_stepwise.index)==0:
        this_index = 0
    else:
        this_index = max(metrics_stepwise.index)+1
    metrics_stepwise = metrics_stepwise.append(pd.DataFrame(data={
                                'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht,
                                'T': T, 'ht_x': ht_x, 'hy_t': hy_t,
                                'step_time': step_time, 'step': 0},
                                index=[this_index]))
    if compact>1:
        distributions_stepwise = distributions_stepwise.append(pd.DataFrame(data={
                                    'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                                    'step': 0},
                                    index=[this_index]))
    del this_index
    
    # ITERATE STEPS 1-3 TO CONVERGENCE
    converged = 0
    if T==1:
        converged = conv_thresh
        if verbose>0:
            print('Converged due to reduction to single cluster')
        conv_condition = 'single_cluster'
    else:
        conv_condition = ''
    Nsteps = 0
    L_old = L
    iter_start_time = time.time()
    while converged<conv_thresh: 
        step_start_time = time.time()
        Nsteps += 1
        if verbose==2:
            print('Beginning IB step %i' % Nsteps)            
        # STEP 1: UPDATE Q(T|X)
        qt_x = qt_x_step(qt,py_x,qy_t,T,X,alpha,beta,verbose)
        # STEP 2: UPDATE Q(T)
        qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)
        # STEP 3: UPDATE Q(Y|T)
        qy_t = qy_t_step(qt_x,qt,px,py_x)
        # calculate and print metrics
        ht, hy_t, iyt, ht_x, ixt, L = calc_IB_metrics(qt_x,qt,qy_t,px,hy,alpha,beta)
        if verbose==2:
            print('I(X,T) = %.6f, H(T) = %.6f, H(X) = %.6f, I(Y,T) = %.6f, I(X,Y) = %.6f, L = %.6f' % (ixt,ht,hx,iyt,ixy,L))
        # check for convergence
        L_abs_inc_flag = L>(L_old+ctol_abs)
        L_rel_inc_flag = L>(L_old+(abs(L_old)*ctol_rel))
        if abs(L_old-L)<ctol_abs:
            converged += 1
            if (converged>=conv_thresh):
                conv_condition = 'small_abs_changes'
                if verbose>0:
                    print('Converged due to small absolute changes in objective')
        else:
            converged = 0 # reset counter if change wasn't small
        if (abs(L_old-L)/abs(L_old))<ctol_rel:
            converged = conv_thresh
            if verbose>0:
                print('Converged due to small relative changes in objective')
            if len(conv_condition)==0:
                conv_condition = 'small_rel_changes'
            else:
                conv_condition += '_AND_small_rel_changes'
        if (T==1) and not(L_abs_inc_flag) and not(L_rel_inc_flag):
            converged = conv_thresh
            if verbose>0:
                print('Converged due to reduction to single cluster')
            if len(conv_condition)==0:
                conv_condition = 'single_cluster'
            else:
                conv_condition += '_AND_single_cluster'
        if np.isnan(L):
            converged = conv_thresh
            if verbose>0:
                print('Stopped because objective = NaN')
            if len(conv_condition)==0:
                conv_condition = 'cost_func_NaN'
            else:
                conv_condition += '_AND_cost_func_NaN'
        # check if obj went up by amount above threshold (after 1st step)
        if (L_abs_inc_flag or L_rel_inc_flag) and (Nsteps>1): # if so, don't store or count this step!
            converged = conv_thresh
            if L_abs_inc_flag:
                if verbose>0:
                    print('Converged due to absolute increase in objective value')
                if len(conv_condition)==0:
                    conv_condition = 'cost_func_abs_inc'
                else:
                    conv_condition += '_AND_cost_func_abs_inc'
            if L_rel_inc_flag:
                if verbose>0:
                    print('Converged due to relative increase in objective value')
                if len(conv_condition)==0:
                    conv_condition = 'cost_func_rel_inc'
                else:
                    conv_condition += '_AND_cost_func_rel_inc'
            # revert to metrics/distributions from last step
            L = L_old
            ixt = ixt_old
            iyt = iyt_old
            ht = ht_old
            T = T_old
            ht_x = ht_x_old
            hy_t = hy_t_old
            qt_x = qt_x_old
            qt = qt_old
            qy_t = qy_t_old
        else:
            # store stepwise data
            step_time = time.time() - step_start_time 
            this_index = max(metrics_stepwise.index)+1
            metrics_stepwise = metrics_stepwise.append(pd.DataFrame(data={
                                    'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht,
                                    'T': T, 'ht_x': ht_x, 'hy_t': hy_t,
                                    'step_time': step_time, 'step': Nsteps},
                                    index=[this_index]))
            if compact>1:
                distributions_stepwise = distributions_stepwise.append(pd.DataFrame(data={
                                        'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                                        'step': Nsteps},
                                        index=[this_index]))
            del this_index
            L_old = L
            ixt_old = ixt
            iyt_old = iyt
            ht_old = ht
            T_old = T
            ht_x_old = ht_x
            hy_t_old = hy_t
            qt_x_old = qt_x
            qt_old = qt
            qy_t_old = qy_t
    # end iterative IB steps
            
    # replace converged step with single-cluster map if better
    if T>1:
        if verbose>0:
            print("Trying single-cluster mapping.")
        step_start_time = time.time()
        sqt_x = np.zeros((T,X))
        sqt_x[0,:] = 1.
        sqt_x,sqt,sT = qt_step(sqt_x,px,ptol,verbose)
        sqy_t = qy_t_step(sqt_x,sqt,px,py_x)
        sht, shy_t, siyt, sht_x, sixt, sL = calc_IB_metrics(sqt_x,sqt,sqy_t,px,hy,alpha,beta)
        if sL<(L-zeroLtol): # if better fit...
            conv_condition += '_AND_force_single'
            if verbose>0:
                print("Single-cluster mapping reduces L from %.6f to %.6f; replacing." % (L,sL))
            # replace everything
            qt_x = sqt_x
            qt = sqt
            T = sT
            qy_t = sqy_t
            ht = sht
            hy_t = shy_t
            iyt = siyt
            ht_x = sht_x
            ixt = sixt
            L = sL
            # store stepwise data
            step_time = time.time() - step_start_time 
            this_index = max(metrics_stepwise.index)+1
            metrics_stepwise = metrics_stepwise.append(pd.DataFrame(data={
                                    'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht,
                                    'T': T, 'ht_x': ht_x, 'hy_t': hy_t,
                                    'step_time': step_time, 'step': Nsteps+1},
                                    index=[this_index]))
            if compact>1:
                distributions_stepwise = distributions_stepwise.append(pd.DataFrame(data={
                                        'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                                        'step': Nsteps+1},
                                        index=[this_index]))
            del this_index
            L_old = L
            ixt_old = ixt
            iyt_old = iyt
            ht_old = ht
            T_old = T
            ht_x_old = ht_x
            hy_t_old = hy_t
            qt_x_old = qt_x
            qt_old = qt
            qy_t_old = qy_t
        elif verbose>0:
            print("Single-cluster mapping not better; increases L from %.6f to %.6f." % (L,sL))
    # end single-cluster check
    
    conv_time = time.time() - iter_start_time
    metrics_converged = pd.DataFrame(data={
                            'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht,
                            'T': T, 'ht_x': ht_x, 'hy_t': hy_t,
                            'conv_time': conv_time, 'conv_steps': Nsteps,
                            'hx': hx, 'ixy': ixy, 'Tmax': Tmax,
                            'beta': beta, 'alpha': alpha, 'p0': p0,
                            'ctol_abs': ctol_abs, 'ctol_rel': ctol_rel,
                            'ptol': ptol, 'zeroLtol': zeroLtol,
                            'conv_condition': conv_condition, 'clamp': False},
                            index=[0])
    if compact>0:
        distributions_converged = pd.DataFrame(data={
                                'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                                'Tmax': Tmax, 'beta': beta, 'alpha': alpha,
                                'p0': p0, 'ctol_abs': ctol_abs, 'ctol_rel': ctol_rel,
                                'ptol': ptol, 'zeroLtol': zeroLtol,
                                'conv_condition': conv_condition, 'clamp': False},
                                index=[0])
                                
    # add in stuff that doesn't vary by step
    metrics_stepwise['hx'] = hx
    metrics_stepwise['ixy'] = ixy
    metrics_stepwise['Tmax'] = Tmax
    metrics_stepwise['beta'] = beta
    metrics_stepwise['alpha'] = alpha  
    metrics_stepwise['p0'] = p0
    metrics_stepwise['ctol_abs'] = ctol_abs
    metrics_stepwise['ctol_rel'] = ctol_rel
    metrics_stepwise['ptol'] = ptol 
    metrics_stepwise['zeroLtol'] = zeroLtol
    if compact>1:
        distributions_stepwise['Tmax'] = Tmax
        distributions_stepwise['beta'] = beta
        distributions_stepwise['alpha'] = alpha
        distributions_stepwise['p0'] = p0
        distributions_stepwise['ctol_abs'] = ctol_abs
        distributions_stepwise['ctol_rel'] = ctol_rel
        distributions_stepwise['ptol'] = ptol
        distributions_stepwise['zeroLtol'] = zeroLtol
    
    # optional clamping step (doesn't apply to DIB)
    if (alpha>0) and clamp:
        
        start_time = time.time()
    
        if verbose>0:
            print('****************************** Clamping IB fit with following parameters ******************************')
            if Tmax == math.inf:
                print('alpha = %.6f, beta = %.6f, Tmax = inf, p0 = %.6f, ctol_abs = %.6f, ctol_rel = %.6f, ptol = %.6f'\
                    % (alpha,beta,p0,ctol_abs,ctol_rel,ptol))
            else:
                print('alpha = %.6f, beta = %.6f, Tmax = %i, p0 = %.6f, ctol_abs = %.6f, ctol_rel = %.6f, ptol = %.6f'\
                    % (alpha,beta,Tmax,p0,ctol_abs,ctol_rel,ptol))
            print('**************************************************************************************************')
        
        # STEP 1: CLAMP Q(T|X)
        for x in range(X):
            tstar = np.argmax(qt_x[:,x])
            qt_x[:,x] = 0
            qt_x[tstar,x] = 1
            del tstar
        del x
        
        # STEP 2: UPDATE Q(T)
        qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)
        
        # STEP 3: UPDATE Q(Y|T)
        qy_t = qy_t_step(qt_x,qt,px,py_x)
        
        # calculate and print metrics
        ht, hy_t, iyt, ht_x, ixt, L = calc_IB_metrics(qt_x,qt,qy_t,px,hy,alpha,beta)
        if verbose>0:
            print('***** unclamped fit *****')
            print('I(X,T) = %.6f, H(T) = %.6f, H(X) = %.6f, I(Y,T) = %.6f, I(X,Y) = %.6f, L = %.6f' % (ixt_old,ht_old,hx,iyt_old,ixy,L_old))
            print('***** clamped fit *****')
            print('I(X,T) = %.6f, H(T) = %.6f, H(X) = %.6f, I(Y,T) = %.6f, I(X,Y) = %.6f, L = %.6f' % (ixt,ht,hx,iyt,ixy,L))
            
        # store everything
        this_step_time = time.time()-start_time
        metrics_converged = metrics_converged.append(pd.DataFrame(data={
                        'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht,
                        'T': T, 'ht_x': ht_x, 'hy_t': hy_t,
                        'conv_time': conv_time+this_step_time, 'conv_steps': Nsteps+1,
                        'hx': hx, 'ixy': ixy, 'Tmax': Tmax,
                        'beta': beta, 'alpha': alpha, 'p0': p0, 'zeroLtol': zeroLtol,
                        'ctol_abs': ctol_abs, 'ctol_rel': ctol_rel,
                        'ptol': ptol, 'conv_condition': conv_condition,
                        'clamp': True},
                        index=[1]))
        if compact>0:
            distributions_converged = distributions_converged.append(pd.DataFrame(data={
                            'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                            'Tmax': Tmax, 'beta': beta, 'alpha': alpha,
                            'p0': p0, 'ctol_abs': ctol_abs, 'ctol_rel': ctol_rel,
                            'ptol': ptol, 'zeroLtol': zeroLtol,
                            'conv_condition': conv_condition, 'clamp': True},
                            index=[1]))
    
    # return results
    if compact>1:
        return metrics_stepwise, distributions_stepwise,\
                metrics_converged, distributions_converged
    elif compact>0:
        return metrics_stepwise,\
                metrics_converged, distributions_converged
    else:
        return metrics_stepwise,\
                metrics_converged
            
def refine_beta(metrics_converged,verbose=2):
    """Helper function for IB to refine beta parameter."""
    
    # parameters governing insertion of betas, or when there is a transition to NaNs (due to under/overflow)
    l = 1 # number of betas to insert into gaps
    del_R = .05 # if fractional change in I(Y;T) exceeds this between adjacent betas, insert more betas
    del_C = .05 # if fractional change in H(T) or I(X;T) exceeds this between adjacent betas, insert more betas
    min_abs_res = 1e-1 # if beta diff smaller than this absolute threshold, don't insert; consider as phase transition
    min_rel_res = 2e-2 # if beta diff smaller than this fractional threshold, don't insert
    # parameters governing insertion of betas when I(X;T) doesn't reach zero
    eps0 = 1e-2 # tolerance for considering I(X;T) to be zero
    l0 = 1 # number of betas to insert at low beta end
    f0 = .5 # new betas will be minbeta*f0.^1:l0
    # parameters governing insertion of betas when I(T;Y) doesn't reach I(X;Y)
    eps1 = .99 # tolerance for considering I(T;Y) to be I(X;Y)
    l1 = 1 # number of betas to insert at high beta end
    f1 = 2 # new betas will be maxbeta*f0.^1:l0
    max_beta_allowed = 100 # any proposed betas above this will be filtered out and replaced it max_beta_allowed

    # sort by beta
    metrics_converged = metrics_converged.sort_values(by='beta')
    
    # init
    new_betas = np.array([])
    NaNtran = False
    ixy = metrics_converged['ixy'].iloc[0]
    logT = math.log2(metrics_converged['Tmax'].iloc[0])
    if verbose>0:
        print('-----------------------------------')
    
    # check that smallest beta was small enough
    if metrics_converged['ixt'].min()>eps0:
        minbeta = metrics_converged['beta'].min()
        betas_to_add = np.array([minbeta*(f0**n) for n in range(1,l0+1)])
        new_betas = np.append(new_betas,betas_to_add)
        if verbose==2:
            print('Added %i small betas. %.6f was too large.' % (l0,minbeta))
        del betas_to_add
    
    # check for gaps to fill
    for i in range(metrics_converged.shape[0]-1):
        beta1 = metrics_converged['beta'].iloc[i]
        beta2 = metrics_converged['beta'].iloc[i+1]
        if (beta2-beta1)<min_abs_res or ((beta2-beta1)/beta1)<min_rel_res: # if beta diff small than absolute/relative resolution, just check for NaNtran
            cc1 = metrics_converged['conv_condition'].iloc[i]
            cc2 = metrics_converged['conv_condition'].iloc[i+1]            
            NaNtran = (("cost_func_NaN" not in cc1) and ("cost_func_NaN" in cc2))
        else: # otherwise, do all gap checks
            iyt1 = metrics_converged['iyt'].iloc[i]
            iyt2 = metrics_converged['iyt'].iloc[i+1]
            ixt1 = metrics_converged['ixt'].iloc[i]
            ixt2 = metrics_converged['ixt'].iloc[i+1]
            ht1 = metrics_converged['ht'].iloc[i]
            ht2 = metrics_converged['ht'].iloc[i+1]
            cc1 = metrics_converged['conv_condition'].iloc[i]
            cc2 = metrics_converged['conv_condition'].iloc[i+1]            
            NaNtran = (("cost_func_NaN" not in cc1) and ("cost_func_NaN" in cc2))
            
            if ((abs(iyt1-iyt2)/ixy)>del_R) or\
               ((abs(ixt1-ixt2)/logT)>del_C) or\
               ((abs(ht1-ht2)/logT)>del_C) or\
               NaNtran:
                   betas_to_add = np.linspace(beta1,beta2,l+2)[1:l+1]
                   new_betas = np.append(new_betas,betas_to_add)
                   del betas_to_add
                   if verbose==2:
                       print('Added %i betas between %.3f and %.3f.' % (l,beta1,beta2))
            if NaNtran: # stop search if there was a NaNtran
                if verbose==2:
                    print('(...because there was a transition to NaNs.)')
                break
    
    # check that largest beta was large enough
    if ((metrics_converged['iyt'].max()/ixy)<eps1) and ~NaNtran:
        maxbeta = metrics_converged['beta'].max()
        betas_to_add = np.array([maxbeta*(f1**n) for n in range(1,l1+1)])
        new_betas = np.append(new_betas,betas_to_add)
        if verbose==2:
            print('Added %i large betas. %.3f was too small.' % (l1,maxbeta))
        del betas_to_add
        
    # filter out betas above max_beta_allowed
    too_large_mask = new_betas>max_beta_allowed
    to_keep_mask = new_betas<max_beta_allowed
    max_beta_allowed_used = (metrics_converged['beta'].max()==max_beta_allowed)
    if any(too_large_mask):
        new_betas = new_betas[to_keep_mask]
        if verbose==2:
            print('Filtered out %i betas larger than max_beta_allowed.' % np.sum(too_large_mask))
        if max_beta_allowed_used:
            if verbose==2:
                print('...and not replaced since max_beta_allowed = %i already used.' % max_beta_allowed)
        else:
            new_betas = np.append(new_betas,max_beta_allowed)
            if verbose==2:
                print('And replaced them with max_beta_allowed = %i.' % max_beta_allowed)        
    
    if verbose>0:
        print('Added %i new betas.' % len(new_betas))
        print('-----------------------------------')

    return new_betas
    
def set_param(fit_param,param_name,def_val):
    """Helper function for IB to handle setting of fit parameters."""
    if param_name in fit_param.index.values: # check if param included 
        param_val = fit_param[param_name]
        if np.isnan(param_val) or math.isinf(param_val): # if so, check val
            param = def_val
        else:
            param = param_val
    else: # if not, use default
        param = def_val
    return param

def IB(pxy,fit_param,compact=1,verbose=2):
    """Performs many generalized IB fits to a single p(x,y).
    
    One fit is performed for each row of input dataframe fit_param. Columns
    correspond to parameters of IB_single. See definition of IB_single for
    more details.
    
    REQUIRED INPUTS
    pxy - p(x,y) [=] X x Y
    fit_param = pandas df, with each row specifying a single IB fit; see README for details
            
    OPTIONAL INPUTS
    verbose = integer indicating verbosity of updates [=] 0/1/2
    compact = integer indicating how much data to save [=] 0/1/2
            (0: only save metrics;
             1: also save converged distributions;
             2: also save stepwise distributions)"""
    
    # set defaults
    low = np.array([.1])
    mid = np.array([1,2,3,4,5,7,9])
    high = np.array([10])
    def_betas = np.concatenate((low,mid,high))
    def_Tmax = math.inf
    def_p0 = .75
    def_ctol_abs = 10**-3
    def_ctol_rel = 0.
    def_ptol = 10**-8
    def_max_fits = 100
    def_max_time = 7*24*60*60 # 1 week
    def_repeats = 1
    def_zeroLtol = 1
    def_clamp = True
    
    # initialize dataframes
    metrics_stepwise_allreps = pd.DataFrame(columns=['L','T','ht','ht_x','hy_t',
                                             'ixt','iyt','step','step_time',
                                             'hx','ixy','Tmax','beta','alpha',
                                             'p0','ctol_abs','ctol_rel','ptol',
                                             'zeroLtol','repeat','repeats'])
    if compact>1:
        distributions_stepwise_allreps = pd.DataFrame(columns=['qt','qt_x','qy_t',
                                                      'step','Tmax','beta','alpha',
                                                      'p0','ctol_abs','ctol_rel',
                                                      'ptol','zeroLtol',
                                                      'repeat','repeats'])
                                                                                           
    metrics_converged_allreps = pd.DataFrame(columns=['L','T','conv_steps','conv_time',
                                              'ht','ht_x','hy_t','ixt','iyt',
                                              'hx','ixy','Tmax','beta','alpha',
                                              'p0','ctol_abs','ctol_rel','ptol',
                                              'zeroLtol','conv_condition','clamp',
                                              'repeat','repeats'])
    if compact>0:
        distributions_converged_allreps = pd.DataFrame(columns=['qt','qt_x','qy_t',
                                                        'Tmax','beta','alpha',
                                                        'p0','ctol_abs','ctol_rel',
                                                        'ptol','zeroLtol',
                                                        'conv_condition','clamp',
                                                        'repeat','repeats'])

    metrics_stepwise = pd.DataFrame(columns=['L','T','ht','ht_x','hy_t',
                                             'ixt','iyt','step','step_time',
                                             'hx','ixy','Tmax','beta','alpha',
                                             'p0','ctol_abs','ctol_rel','ptol',
                                             'zeroLtol','repeat','repeats'])
    if compact>1:
        distributions_stepwise = pd.DataFrame(columns=['qt','qt_x','qy_t',
                                                      'step','Tmax','beta','alpha',
                                                      'p0','ctol_abs','ctol_rel',
                                                      'ptol','zeroLtol',
                                                      'repeat','repeats']) 
                                                                                              
    metrics_converged = pd.DataFrame(columns=['L','T','conv_steps','conv_time',
                                              'ht','ht_x','hy_t','ixt','iyt',
                                              'hx','ixy','Tmax','beta','alpha',
                                              'p0','ctol_abs','ctol_rel','ptol',
                                              'zeroLtol','conv_condition','clamp',
                                              'repeat','repeats'])
                                              
    if compact>0:
        distributions_converged = pd.DataFrame(columns=['qt','qt_x','qy_t',
                                                        'Tmax','beta','alpha',
                                                        'p0','ctol_abs','ctol_rel',
                                                        'ptol','zeroLtol',
                                                        'conv_condition','clamp',
                                                        'repeat','repeats'])
    
    # iterate over fit parameters (besides beta, which is done below)                                
    for irow in range(len(fit_param.index)):
        # extract required parameters
        this_fit = fit_param.iloc[irow]
        this_alpha = this_fit['alpha']
        # extract optional parameters            
        this_Tmax = set_param(this_fit,'Tmax',def_Tmax)
        if this_alpha>0:
            this_p0 = set_param(this_fit,'p0',def_p0)
        else:
            this_p0 = 1.
        this_ctol_abs = set_param(this_fit,'ctol_abs',def_ctol_abs)
        this_ctol_rel = set_param(this_fit,'ctol_rel',def_ctol_rel)
        this_ptol = set_param(this_fit,'ptol',def_ptol)
        this_max_fits = set_param(this_fit,'max_fits',def_max_fits)
        this_max_time = set_param(this_fit,'max_time',def_max_time)
        this_repeats = set_param(this_fit,'repeats',def_repeats)
        this_zeroLtol = set_param(this_fit,'zeroLtol',def_zeroLtol)
        this_clamp = set_param(this_fit,'clamp',def_clamp)
        # make pre-fitting initializations
        betas = def_betas # stack of betas
        fit_count = 0
        fit_time = 0
        fit_start_time = time.time()
        these_betas_metrics_converged = pd.DataFrame(columns=['ht','ixt','iyt',
                                                        'hx','ixy','Tmax','beta',
                                                        'conv_condition']) # used for beta refinement
        while (fit_count<=this_max_fits) and (fit_time<=this_max_time) and (betas.size>0):
            this_beta = betas[0] # use beta at front of list
            
            # init data structures that will store the repeated fits for this particular setting of parameters
            these_reps_metrics_stepwise = pd.DataFrame(columns=['L','T','ht','ht_x','hy_t',
                                                     'ixt','iyt','step','step_time',
                                                     'hx','ixy','Tmax','beta','alpha',
                                                     'p0','ctol_abs','ctol_rel','ptol','zeroLtol',
                                                     'repeat','repeats'])
            if compact>1:
                these_reps_distributions_stepwise = pd.DataFrame(columns=['qt','qt_x','qy_t',
                                                              'step','Tmax','beta','alpha',
                                                              'p0','ctol_abs','ctol_rel','ptol',
                                                              'zeroLtol','repeat','repeats'])
                                                                                                                         
            these_reps_metrics_converged = pd.DataFrame(columns=
                                         ['L','T','conv_steps','conv_time',
                                          'ht','ht_x','hy_t','ixt','iyt',
                                          'hx','ixy','Tmax','beta','alpha',
                                          'p0','ctol_abs','ctol_rel','ptol',
                                          'zeroLtol','conv_condition','clamp',
                                          'repeat','repeats'])
            if compact>0:
                these_reps_distributions_converged = pd.DataFrame(columns=
                                                   ['qt','qt_x','qy_t',
                                                    'Tmax','beta','alpha',
                                                    'p0','ctol_abs','ctol_rel',
                                                    'ptol','zeroLtol',
                                                    'conv_condition','clamp',
                                                    'repeat','repeats'])
                                                    
            for repeat in range(this_repeats):
                # do a single fit
                if verbose>0:
                    print('****************************** Running IB with following parameters ******************************')
                    if this_Tmax == math.inf:
                        print('alpha = %.6f, beta = %.6f, Tmax = inf, p0 = %.6f, ctol_abs = %.6f, ctol_rel = %.6f, ptol = %.6f, zeroLtol = %.6f, repeat = %i of %i'\
                            % (this_alpha,this_beta,this_p0,this_ctol_abs,this_ctol_rel,this_ptol,this_zeroLtol,repeat,this_repeats))
                    else:
                        print('alpha = %.6f, beta = %.6f, Tmax = %i, p0 = %.6f, ctol_abs = %.6f, ctol_rel = %.6f, ptol = %.6f, zeroLtol = %.6f, repeat = %i of %i'\
                            % (this_alpha,this_beta,this_Tmax,this_p0,this_ctol_abs,this_ctol_rel,this_ptol,this_zeroLtol,repeat,this_repeats))
                    print('**************************************************************************************************')
                if compact>1:
                    this_metrics_stepwise, this_distributions_stepwise, \
                    this_metrics_converged, this_distributions_converged = \
                    IB_single(pxy,this_beta,this_alpha,this_Tmax,
                              this_p0,this_ctol_abs,this_ctol_rel,this_ptol,this_zeroLtol,this_clamp,compact,verbose)
                elif compact>0:
                    this_metrics_stepwise, \
                    this_metrics_converged, this_distributions_converged = \
                    IB_single(pxy,this_beta,this_alpha,this_Tmax,
                              this_p0,this_ctol_abs,this_ctol_rel,this_ptol,this_zeroLtol,this_clamp,compact,verbose)
                else:
                    this_metrics_stepwise, \
                    this_metrics_converged = \
                    IB_single(pxy,this_beta,this_alpha,this_Tmax,
                              this_p0,this_ctol_abs,this_ctol_rel,this_ptol,this_zeroLtol,this_clamp,compact,verbose)
                # add repeat labels
                this_metrics_stepwise['repeat'] = repeat
                this_metrics_stepwise['repeats'] = this_repeats
                if compact>1:
                    this_distributions_stepwise['repeat'] = repeat
                    this_distributions_stepwise['repeats'] = this_repeats
                this_metrics_converged['repeat'] = repeat
                this_metrics_converged['repeats'] = this_repeats
                if compact>0:
                    this_distributions_converged['repeat'] = repeat
                    this_distributions_converged['repeats'] = this_repeats
                # add this repeat to these repeats
                these_reps_metrics_stepwise = these_reps_metrics_stepwise.append(this_metrics_stepwise) 
                if compact>1:
                    these_reps_distributions_stepwise = these_reps_distributions_stepwise.append(this_distributions_stepwise)  
                these_reps_metrics_converged = these_reps_metrics_converged.append(this_metrics_converged)
                if compact>0:
                    these_reps_distributions_converged = these_reps_distributions_converged.append(this_distributions_converged)  
            # end of repeat fit loop for single beta 
                
            # reindex fits to work with existing dataframe
            if len(metrics_stepwise_allreps.index)>0:
                num_there = max(metrics_stepwise_allreps.index)
            else:
                num_there = 0
            num_added = len(these_reps_metrics_stepwise.index)
            these_reps_metrics_stepwise.index = np.arange(num_there+1,num_there+num_added+1)
            if compact>1:
                these_reps_distributions_stepwise.index = np.arange(num_there+1,num_there+num_added+1)
            del num_there, num_added
            if len(metrics_converged_allreps.index)>0:
                num_there = max(metrics_converged_allreps.index)
            else:
                num_there = 0
            num_added = len(these_reps_metrics_converged.index)
            these_reps_metrics_converged.index = np.arange(num_there+1,num_there+num_added+1)
            if compact>0:
                these_reps_distributions_converged.index = np.arange(num_there+1,num_there+num_added+1)
            del num_there, num_added
            
            # store all repeats
            metrics_stepwise_allreps = metrics_stepwise_allreps.append(these_reps_metrics_stepwise)
            if compact>1:
                distributions_stepwise_allreps = distributions_stepwise_allreps.append(these_reps_distributions_stepwise)
            metrics_converged_allreps = metrics_converged_allreps.append(these_reps_metrics_converged)
            if compact>0:
                distributions_converged_allreps = distributions_converged_allreps.append(these_reps_distributions_converged)
                
            # find best repeat (lowest L)
            these_reps_metrics_converged_unclamped = these_reps_metrics_converged[these_reps_metrics_converged['clamp']==False]
            best_id = these_reps_metrics_converged_unclamped['L'].idxmin()
            if np.isnan(best_id): # if all repeats NaNs, just use first repeat
                best_repeat = 0
            else: # otherwise use best
                best_repeat = these_reps_metrics_converged['repeat'].loc[best_id]
            best_metrics_converged = these_reps_metrics_converged[these_reps_metrics_converged['repeat']==best_repeat]
            if compact>0:            
                best_distributions_converged = these_reps_distributions_converged[these_reps_distributions_converged['repeat']==best_repeat]
            best_metrics_stepwise = these_reps_metrics_stepwise[these_reps_metrics_stepwise['repeat']==best_repeat]
            if compact>1:
                best_distributions_stepwise = these_reps_distributions_stepwise[these_reps_distributions_stepwise['repeat']==best_repeat]
                
            # store in best fits dataframe 
            metrics_stepwise = metrics_stepwise.append(best_metrics_stepwise)
            if compact>1:
                distributions_stepwise = distributions_stepwise.append(best_distributions_stepwise)
            metrics_converged = metrics_converged.append(best_metrics_converged)
            if compact>0:
                distributions_converged = distributions_converged.append(best_distributions_converged)
                
            # store best fits across beta for this set of parameters
            best_metrics_converged_unclamped = best_metrics_converged[best_metrics_converged['clamp']==False]
            these_betas_metrics_converged = these_betas_metrics_converged.append(\
                    best_metrics_converged_unclamped[['ht','ixt','iyt','hx','ixy','Tmax','beta','conv_condition']])
                    
            # advance
            betas = np.delete(betas,0) # toss out the beta used
            fit_count += this_repeats
            fit_time = time.time()-fit_start_time
            
            # refine beta if needed
            if betas.size==0:
                betas = refine_beta(these_betas_metrics_converged,verbose)
                
        if verbose>0:
            if fit_count>=this_max_fits:
                print('Stopped beta refinement because ran over max fit count of %i' % this_max_fits)
            if fit_time>=this_max_time:
                print('Stopped beta refinement because ran over max fit time of %i seconds' % this_max_time)
            if betas.size==0:
                print('Beta refinement complete.')

    # end iteration over fit parameters
    if compact>1:
        return metrics_stepwise, distributions_stepwise,\
               metrics_converged, distributions_converged,\
               metrics_stepwise_allreps, distributions_stepwise_allreps,\
               metrics_converged_allreps, distributions_converged_allreps
    elif compact>0:
        return metrics_stepwise,\
               metrics_converged, distributions_converged,\
               metrics_stepwise_allreps,\
               metrics_converged_allreps, distributions_converged_allreps
    else:
        return metrics_stepwise,\
               metrics_converged,\
               metrics_stepwise_allreps,\
               metrics_converged_allreps

def clamp_IB(metrics_converged,distributions_converged,pxy,verbose=1):
    """Function to 'clamp' IB fits after the fact; IB also has functionality
        for doing this during normal fitting; see above."""
    
    metrics_converged['clamp'] = False
    distributions_converged['clamp'] = False
    
    # process pxy
    pxy, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy = process_pxy(pxy,verbose)
    
    # init data structure of clamped results
    these_metrics_converged = pd.DataFrame(columns=[
                            'L','ixt','iyt','ht','T','ht_x','hy_t',
                            'conv_time','conv_steps','hx','ixy','Tmax','beta',
                            'alpha','p0','ctol_abs','ctol_rel','ptol','conv_condition','clamp'])
    these_distributions_converged = pd.DataFrame(columns=[
                            'qt_x','qt','qy_t','Tmax','beta',
                            'alpha','p0','ctol_abs','ctol_rel','ptol','conv_condition','clamp'])
    num_added = 0
    
    # iterate over converged results
    for irow in range(len(distributions_converged.index)):
        
        alpha = metrics_converged['alpha'].iloc[irow] 
        if alpha>0: # don't clamp DIB fits
        
            start_time = time.time()
            beta = metrics_converged['beta'].iloc[irow]
            Tmax = metrics_converged['Tmax'].iloc[irow]
            p0 = metrics_converged['p0'].iloc[irow]            
            ctol_abs = metrics_converged['ctol_abs'].iloc[irow]
            ctol_rel = metrics_converged['ctol_rel'].iloc[irow]
            ptol = metrics_converged['ptol'].iloc[irow]
            conv_condition = metrics_converged['conv_condition'].iloc[irow]
        
            if verbose>0:
                print('****************************** Clamping IB fit with following parameters ******************************')
                if Tmax == math.inf:
                    print('alpha = %.6f, beta = %.6f, Tmax = inf, p0 = %.6f, ctol_abs = %.6f, ctol_rel = %.6f, ptol = %.6f'\
                        % (alpha,beta,p0,ctol_abs,ctol_rel,ptol))
                else:
                    print('alpha = %.6f, beta = %.6f, Tmax = %i, p0 = %.6f, ctol_abs = %.6f, ctol_rel = %.6f, ptol = %.6f'\
                        % (alpha,beta,Tmax,p0,ctol_abs,ctol_rel,ptol))
                print('**************************************************************************************************')

            qt_x = distributions_converged['qt_x'].iloc[irow]
            
            # STEP 1: CLAMP Q(T|X)
            for x in range(X):
                tstar = np.argmax(qt_x[:,x])
                qt_x[:,x] = 0
                qt_x[tstar,x] = 1
                del tstar
            del x
            
            # STEP 2: UPDATE Q(T)
            qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)
            
            # STEP 3: UPDATE Q(Y|T)
            qy_t = qy_t_step(qt_x,qt,px,py_x)
            
            # calculate and print metrics
            ht, hy_t, iyt, ht_x, ixt, L = calc_IB_metrics(qt_x,qt,qy_t,px,hy,alpha,beta)
            if verbose>0:
                old_ixt = metrics_converged['ixt'].iloc[irow]
                old_ht = metrics_converged['ht'].iloc[irow]
                old_iyt = metrics_converged['iyt'].iloc[irow]
                old_L = metrics_converged['L'].iloc[irow]
                print('***** unclamped fit *****')
                print('I(X,T) = %.6f, H(T) = %.6f, H(X) = %.6f, I(Y,T) = %.6f, I(X,Y) = %.6f, L = %.6f' % (old_ixt,old_ht,hx,old_iyt,ixy,old_L))
                print('***** clamped fit *****')
                print('I(X,T) = %.6f, H(T) = %.6f, H(X) = %.6f, I(Y,T) = %.6f, I(X,Y) = %.6f, L = %.6f' % (ixt,ht,hx,iyt,ixy,L))
                
            # store everything
            this_step_time = time.time()-start_time
            Nsteps = metrics_converged['conv_steps'].iloc[irow]+1
            conv_time = metrics_converged['conv_time'].iloc[irow]+this_step_time
            this_metrics_converged = pd.DataFrame(data={
                            'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht,
                            'T': T, 'ht_x': ht_x, 'hy_t': hy_t,
                            'conv_time': conv_time, 'conv_steps': Nsteps,
                            'hx': hx, 'ixy': ixy, 'Tmax': Tmax,
                            'beta': beta, 'alpha': alpha, 'p0': p0,
                            'ctol_abs': ctol_abs, 'ctol_rel': ctol_rel,
                            'ptol': ptol, 'conv_condition': conv_condition,
                            'clamp': True},
                            index=[num_added])
            this_distributions_converged = pd.DataFrame(data={
                            'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                            'Tmax': Tmax, 'beta': beta, 'alpha': alpha,
                            'p0': p0, 'ctol_abs': ctol_abs, 'ctol_rel': ctol_rel,
                            'ptol': ptol, 'conv_condition': conv_condition, 
                            'clamp': True},
                            index=[num_added])
            num_added += 1
            these_metrics_converged = these_metrics_converged.append(this_metrics_converged)
            these_distributions_converged = these_distributions_converged.append(this_distributions_converged)
    
    # reindex clamped fits
    num_there = max(metrics_converged.index)
    these_metrics_converged.index = np.arange(num_there+1,num_there+num_added+1)
    these_distributions_converged.index = np.arange(num_there+1,num_there+num_added+1)
    metrics_converged = metrics_converged.append(these_metrics_converged)
    distributions_converged = distributions_converged.append(these_distributions_converged)
    
    return metrics_converged, distributions_converged