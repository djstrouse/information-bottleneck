import numpy as np
import pandas as pd
import time
import math
vlog = np.vectorize(math.log)
vexp = np.vectorize(math.exp)

# A word on notation: for probability variables, an underscore here means a
# conditioning, so read _ as |.

def verify_inputs(pxy,beta,alpha,Tmax,p0,waviness,ctol_abs,ctol_rel,ptol,zeroLtol,clamp,compact,verbose):
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
    if Tmax is not None and (not(isinstance(Tmax,int)) or Tmax<1):
        raise ValueError('Tmax must be a positive integer (or None)')
    if p0 is not None and (p0<-1 or p0>1 or not(isinstance(p0,(int,float)))):
        raise ValueError('p0 must be a float/int between -1 and 1')
    if waviness is not None and (waviness<0 or waviness>1 or not(isinstance(waviness,float))):
        raise ValueError('waviness must be a float between 0 and 1')
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
    if x==0: return 0.0
    else: return -x*math.log2(x)

def entropy_single(p):
    """Returns entropy of p: H(p)=-sum(p*log(p)). (in bits)"""
    ventropy_term = np.vectorize(entropy_term)
    return np.sum(ventropy_term(p))

def entropy(P):
    """Returns entropy of a distribution, or series of distributions.
    
    For the input array P [=] M x N, treats each col as a prob distribution
    (over M elements), and thus returns N entropies. If P is a vector, treats
    P as a single distribution and returns its entropy."""
    if P.ndim==1: return entropy_single(P)
    else:
        M,N = P.shape
        H = np.zeros(N)
        for n in range(N):
            H[n] = entropy_single(P[:,n])
        return H

def process_pxy(pxy,verbose=1):
    """Helper function for IB that preprocesses p(x,y) and computes metrics."""
    if pxy.dtype!='float': pxy = pxy.astype(float)
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
    if x>0 and y>0: return x*math.log2(x/y)
    elif x==0: return 0.0
    else: return math.inf
    
def kl_single(p,q):
    """Returns KL divergence of p and q: KL(p,q)=sum(p*log(p/q)). (in bits)"""
    vkl_term = np.vectorize(kl_term)
    return np.sum(vkl_term(p,q))
    
def kl(P,Q):
    """Returns KL divergence of one or more pairs of distributions.
    
    For the input arrays P [=] M x N and Q [=] M x L, calculates KL of each col
    of P with each col of Q, yielding the KL matrix DKL [=] N x L. If P=Q=1,
    returns a single KL divergence."""
    if P.ndim==1 and Q.ndim==1: return kl_single(P,Q)
    elif P.ndim==1 and Q.ndim!=1: # handle vector P case
        M = len(P)
        N = 1
        M2,L = Q.shape
        if M!=M2: raise ValueError("P and Q must have same number of columns")
        DKL = np.zeros((1,L))
        for l in range(L):
            DKL[0,l] = kl_single(P,Q[:,l])
    elif P.ndim!=1 and Q.ndim==1: # handle vector Q case
        M,N = P.shape
        M2 = len(Q)
        L = 1
        if M!=M2: raise ValueError("P and Q must have same number of columns")
        DKL = np.zeros((N,1))
        for n in range(N):
            DKL[n,0] = kl_single(P[:,n],Q)
    else:
        M,N = P.shape
        M2,L = Q.shape        
        if M!=M2: raise ValueError("P and Q must have same number of columns")    
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
        if verbose==2: print('%i cluster(s) dropped. Down to %i cluster(s).' % (np.sum(dropped),T)) 
    return qt_x,qt,T
    
def qy_t_step(qt_x,qt,px,py_x):
    """Peforms q(y|t) update step for generalized Information Bottleneck."""
    return np.dot(py_x,np.multiply(qt_x,np.outer(1./qt,px)).T)

def qt_x_step(qt,py_x,qy_t,T,X,alpha,beta,verbose):
    """Peforms q(t|x) update step for generalized Information Bottleneck."""
    if T==1: qt_x = np.ones((1,X))
    else:
        qt_x = np.zeros((T,X))
        for x in range(X):
            l = vlog(qt)-beta*kl(py_x[:,x],qy_t) # [=] T x 1 # scales like X*Y*T
            if alpha==0: qt_x[np.argmax(l),x] = 1
            else: qt_x[:,x] = vexp(l/alpha)/np.sum(vexp(l/alpha)) # note: l/alpha<-745 is where underflow creeps in
    return qt_x
    
def init_qt_x(alpha,X,T,p0,waviness):
    """Initializes q(t|x) for generalized Information Bottleneck.
    
    For p0 = 0: init is random noise. If waviness = None, normalized uniform
    random vector. Otherwise, uniform over clusters +- uniform noise of
    magnitude waviness.
    
    For p0 positive: attempt to spread points as evenly across clusters as
    possible. Prob mass p0 is given to the "assigned" clusters, and the
    remaining 1-p0 prob mass is randomly assigned. If waviness = None, again
    use a normalized random vector to assign the remaining mass. Otherwise,
    uniform +- waviness again.
    
    For p0 negative: just as above, except that all data points are "assigned"
    to the same cluster (well, at least |p0| of their prob mass).""" 
    if p0==0: # don't insert any peaks; init is all "noise"
        if waviness: # flat + wavy style noise
            qt_x = np.ones((T,X))+2*(np.random.rand(T,X)-.5)*waviness # 1+-waviness%                
            for i in range(X):
                qt_x[:,i] = qt_x[:,i]/np.sum(qt_x[:,i]) # normalize
        else: # uniform random vector
            qt_x = np.random.rand(T,X)
            qt_x = np.multiply(qt_x,np.tile(1./np.sum(qt_x,axis=0),(T,1))) # renormalize
    elif p0>0: # spread points evenly across clusters; "assigned" clusters for each data point get prob mass p0
        if waviness:
            # insert wavy noise part          
            qt_x = np.ones((T,X))+2*(np.random.rand(T,X)-.5)*waviness # 1+-waviness%                
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
    else: # put all points in the same cluster; primary cluster gets prob mass |p0|
        p0 = -p0
        if waviness:      
            qt_x = np.ones((T,X))+2*(np.random.rand(T,X)-.5)*waviness # 1+-waviness%
            t = np.random.randint(T) # pick cluster to get delta spike
            qt_x[t,:] = np.zeros((1,X)) # zero out that cluster
            qt_x = np.multiply(qt_x,np.tile(1./np.sum(qt_x,axis=0),(T,1))) # normalize the rest...
            qt_x = (1-p0)*qt_x # ...to 1-p0
            qt_x[t,:] = p0*np.ones((1,X)) # put in delta spike
        else: # uniform random vector instead of wavy
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
    
def calc_metrics(qt_x,qt,qy_t,px,hy,alpha,beta):
    """Calculates IB performance metrics."""
    ht = entropy(qt)
    hy_t = np.dot(qt,entropy(qy_t))
    iyt = hy-hy_t
    ht_x = np.dot(px,entropy(qt_x))
    ixt = ht-ht_x
    L = ht-alpha*ht_x-beta*iyt
    return ht, hy_t, iyt, ht_x, ixt, L 
    
def store_sw_metrics(metrics_sw,L,ixt,iyt,ht,T,ht_x,hy_t,time,step):
    return metrics_sw.append(pd.DataFrame(data={
                'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht, 'T': T, 'ht_x': ht_x,
                'hy_t': hy_t, 'time': time, 'step': step},
                index = [0]), ignore_index = True)
    
def store_sw_dist(dist_sw,qt_x,qt,qy_t,step):    
    return dist_sw.append(pd.DataFrame(data={
                'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],'step': step},
                index = [0]), ignore_index = True)
    
def metrics_string(ixt,ht,hx,iyt,ixy,L):
    return 'I(X,T) = %.4f, H(T) = %.4f, H(X) = %.4f, I(Y,T) = %.4f, I(X,Y) = %.4f, L = %.4f' % (ixt,ht,hx,iyt,ixy,L)
    
def IB_single(pxy,beta,alpha,Tmax=None,p0=None,waviness=None,ctol_abs=10**-3,ctol_rel=0.,
              ptol=10**-8,zeroLtol=1,clamp=True,compact=1,verbose=2):
    """Performs a single fit of the generalized Information Bottleneck on the
    joint distribution p(x,y) for a given beta and alpha.
    
    See IB.py function documentation below for more details."""
        
    # PRE-IB STEP INIT AND PROCESSING
        
    verify_inputs(pxy,beta,alpha,Tmax,p0,waviness,ctol_abs,ctol_rel,ptol,zeroLtol,clamp,compact,verbose)
    
    conv_thresh = 1 # steps in a row of small change to consider converged
    
    # process inputs
    if isinstance(alpha,int): alpha = float(alpha)
    if isinstance(beta,int): beta = float(beta)
    if p0 is None:
        if alpha==0: p0 = 1. # DIB default: deterministic init that spreads points evenly across clusters
        else: p0 = .75 # non-DIB default: DIB-like init but with only 75% prob mass on "assigned" cluster           
    pxy, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy = process_pxy(pxy,verbose)
    if Tmax is None:
        Tmax = X
        if verbose==2: print('Tmax set to %i based on X' % Tmax)
    elif Tmax>X:
        if verbose==2: print('Reduced Tmax from %i to %i based on X' % (Tmax,X))
        Tmax = X
    else: Tmax = int(Tmax)
        
    # report
    if verbose>0:
        print('****************************************************** Beginning IB fit with the following parameters ******************************************************')
        if Tmax is None: Tmax_str = 'None'
        else: Tmax_str = '%i' % Tmax
        if p0 is None: p0_str = 'None'
        else: p0_str = '%.3f' % p0
        if waviness is None: waviness_str = 'None'
        else: waviness_str = '%.2f' % waviness
        print('alpha = %.2f, beta = %.1f, Tmax = %s, p0 = %s, waviness = %s, ctol_abs = %.1e, ctol_rel = %.1e, ptol = %.1e, zeroLtol = %.1e, clamp = %s' %\
              (alpha, beta, Tmax_str, p0_str, waviness_str, ctol_abs, ctol_rel, ptol, zeroLtol, clamp))
        print('************************************************************************************************************************************************************')
                        
    # initialize dataframes
    metrics_sw = pd.DataFrame(columns=['L','ixt','iyt','ht','T','ht_x','hy_t','step','time'])
    if compact>1: dist_sw = pd.DataFrame(columns=['qt_x','qt','qy_t','step'])
        
    # IB STEP ITERATIONS

    # initialize other stuff
    T = Tmax
    step_start_time = time.time()
    # STEP 0: INITIALIZE
    # initialize q(t|x)
    qt_x = init_qt_x(alpha,X,T,p0,waviness)
    # initialize q(t) given q(t|x)
    qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)
    # initialize q(y|t) given q(t|x) and q(t)
    qy_t = qy_t_step(qt_x,qt,px,py_x)
    # calculate and print metrics
    ht, hy_t, iyt, ht_x, ixt, L = calc_metrics(qt_x,qt,qy_t,px,hy,alpha,beta)
    if verbose==2: print('init: ' + metrics_string(ixt,ht,hx,iyt,ixy,L))
    step_time = time.time() - step_start_time
    metrics_sw = store_sw_metrics(metrics_sw,L,ixt,iyt,ht,T,ht_x,hy_t,step_time,step=0)
    if compact>1: dist_sw = store_sw_dist(dist_sw,qt_x,qt,qy_t,step=0)
    
    # ITERATE STEPS 1-3 TO CONVERGENCE
    converged = 0
    Nsteps = 0
    L_old = L
    iter_start_time = time.time()
    if T==1:
        converged = conv_thresh
        if verbose>0: print('Converged due to reduction to single cluster')
        conv_condition = 'single_cluster'
    else: conv_condition = ''
    while converged<conv_thresh: 
        step_start_time = time.time()
        Nsteps += 1        
        # STEP 1: UPDATE Q(T|X)
        qt_x = qt_x_step(qt,py_x,qy_t,T,X,alpha,beta,verbose)
        # STEP 2: UPDATE Q(T)
        qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)
        # STEP 3: UPDATE Q(Y|T)
        qy_t = qy_t_step(qt_x,qt,px,py_x)
        # calculate and print metrics
        ht, hy_t, iyt, ht_x, ixt, L = calc_metrics(qt_x,qt,qy_t,px,hy,alpha,beta)
        if verbose==2: print('step %i: ' % Nsteps + metrics_string(ixt,ht,hx,iyt,ixy,L))
        # check for convergence
        L_abs_inc_flag = L>(L_old+ctol_abs)
        L_rel_inc_flag = L>(L_old+(abs(L_old)*ctol_rel))
        small_changes = False
        # check for small absolute changes
        if abs(L_old-L)<ctol_abs:
            small_changes = True
            converged += 1
            if (converged>=conv_thresh):
                conv_condition = 'small_abs_changes'
                if verbose>0: print('Converged due to small absolute changes in objective')  
        # check for small relative changes
        if (abs(L_old-L)/abs(L_old))<ctol_rel:
            small_changes = True
            converged += 1
            if (converged>=conv_thresh):
                if len(conv_condition)==0: conv_condition = 'small_rel_changes'
                else: conv_condition += '_AND_small_rel_changes'
                if verbose>0: print('Converged due to small relative changes in objective')
        if not(small_changes): converged = 0 # reset counter of small changes in a row
        # check for reduction to single cluster        
        if (T==1) and not(L_abs_inc_flag) and not(L_rel_inc_flag):
            converged = conv_thresh
            if verbose>0: print('Converged due to reduction to single cluster')
            if len(conv_condition)==0: conv_condition = 'single_cluster'
            else: conv_condition += '_AND_single_cluster'
        # check for objective becoming NaN
        if np.isnan(L):
            converged = conv_thresh
            if verbose>0: print('Stopped because objective = NaN')
            if len(conv_condition)==0: conv_condition = 'cost_func_NaN'
            else: conv_condition += '_AND_cost_func_NaN'
        # check if obj went up by amount above threshold (after 1st step)
        if (L_abs_inc_flag or L_rel_inc_flag) and (Nsteps>1): # if so, don't store or count this step!
            converged = conv_thresh
            if L_abs_inc_flag:
                if verbose>0: print('Converged due to absolute increase in objective value')
                if len(conv_condition)==0: conv_condition = 'cost_func_abs_inc'
                else: conv_condition += '_AND_cost_func_abs_inc'
            if L_rel_inc_flag:
                if verbose>0: print('Converged due to relative increase in objective value')
                if len(conv_condition)==0: conv_condition = 'cost_func_rel_inc'
                else: conv_condition += '_AND_cost_func_rel_inc'
            # revert to metrics/distributions from last step
            L, ixt, iyt, ht, T, ht_x, hy_t, qt_x, qt, qy_t =\
            L_old, ixt_old, iyt_old, ht_old, T_old, ht_x_old, hy_t_old, qt_x_old, qt_old, qy_t_old
        else:
            # store stepwise data
            step_time = time.time() - step_start_time 
            metrics_sw = store_sw_metrics(metrics_sw,L,ixt,iyt,ht,T,ht_x,hy_t,step_time,Nsteps)
            if compact>1: dist_sw = store_sw_dist(dist_sw,qt_x,qt,qy_t,Nsteps)
            # store this step in case need to revert at next step
            L_old, ixt_old, iyt_old, ht_old, T_old, ht_x_old, hy_t_old, qt_x_old, qt_old, qy_t_old =\
            L, ixt, iyt, ht, T, ht_x, hy_t, qt_x, qt, qy_t
    # end iterative IB steps
    
    # report
    if verbose>0: print('converged in %i step(s) to: ' % Nsteps + metrics_string(ixt,ht,hx,iyt,ixy,L))
            
    # replace converged step with single-cluster map if better
    if T>1:
        step_start_time = time.time()
        sqt_x = np.zeros((T,X))
        sqt_x[0,:] = 1.
        sqt_x,sqt,sT = qt_step(sqt_x,px,ptol,verbose)
        sqy_t = qy_t_step(sqt_x,sqt,px,py_x)
        sht, shy_t, siyt, sht_x, sixt, sL = calc_metrics(sqt_x,sqt,sqy_t,px,hy,alpha,beta)
        if sL<(L-zeroLtol): # if better fit...
            conv_condition += '_AND_force_single'
            if verbose>0: print("Single-cluster mapping reduces L from %.4f to %.4f; replacing solution." % (L,sL))
            # replace everything
            L, ixt, iyt, ht, T, ht_x, hy_t, qt_x, qt, qy_t =\
            sL, sixt, siyt, sht, sT, sht_x, shy_t, sqt_x, sqt, sqy_t
            # store stepwise data
            step_time = time.time() - step_start_time 
            metrics_sw = store_sw_metrics(metrics_sw,L,ixt,iyt,ht,T,ht_x,hy_t,step_time,Nsteps+1)
            if compact>1: dist_sw = store_sw_dist(dist_sw,qt_x,qt,qy_t,Nsteps+1)
            L_old, ixt_old, iyt_old, ht_old, T_old, ht_x_old, hy_t_old, qt_x_old, qt_old, qy_t_old =\
            L, ixt, iyt, ht, T, ht_x, hy_t, qt_x, qt, qy_t
            if verbose>0: print('single-cluster solution: ' + metrics_string(ixt,ht,hx,iyt,ixy,L))            
        elif verbose>0: print("Single-cluster mapping not better; changes L from %.4f to %.4f (zeroLtol = %.1e)." % (L,sL,zeroLtol))
    # end single-cluster check
    
    # build converged solution dataframe
    conv_time = time.time() - iter_start_time
    metrics_conv = pd.DataFrame(data={
                        'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht, 'T': T,
                        'ht_x': ht_x, 'hy_t': hy_t, 'hx': hx, 'ixy': ixy,
                        'time': conv_time, 'step': Nsteps, 'Tmax': Tmax,
                        'beta': beta, 'alpha': alpha, 'p0': p0, 'waviness': waviness,
                        'ctol_abs': ctol_abs, 'ctol_rel': ctol_rel,
                        'ptol': ptol, 'zeroLtol': zeroLtol,
                        'conv_condition': conv_condition, 'clamp': False},
                        index=[0])
    if compact>0: dist_conv = pd.DataFrame(data={
                        'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                        'Tmax': Tmax, 'beta': beta, 'alpha': alpha,
                        'p0': p0,  'waviness': waviness, 'ctol_abs': ctol_abs,
                        'ctol_rel': ctol_rel, 'ptol': ptol, 'zeroLtol': zeroLtol,
                        'time': conv_time, 'step': Nsteps,
                        'conv_condition': conv_condition, 'clamp': False},
                        index=[0])
                                
    # add in stuff that doesn't vary by step to the stepwise dataframe
    metrics_sw['hx'] = hx
    metrics_sw['ixy'] = ixy
    metrics_sw['Tmax'] = Tmax
    metrics_sw['beta'] = beta
    metrics_sw['alpha'] = alpha  
    metrics_sw['p0'] = p0
    metrics_sw['waviness'] = waviness
    metrics_sw['ctol_abs'] = ctol_abs
    metrics_sw['ctol_rel'] = ctol_rel
    metrics_sw['ptol'] = ptol 
    metrics_sw['zeroLtol'] = zeroLtol
    if compact>1:
        dist_sw['Tmax'] = Tmax
        dist_sw['beta'] = beta
        dist_sw['alpha'] = alpha
        dist_sw['p0'] = p0
        dist_sw['waviness'] = waviness
        dist_sw['ctol_abs'] = ctol_abs
        dist_sw['ctol_rel'] = ctol_rel
        dist_sw['ptol'] = ptol
        dist_sw['zeroLtol'] = zeroLtol
    
    # optional clamping step (doesn't apply to DIB)
    if alpha>0 and clamp:       
        clamp_start_time = time.time()        
        # STEP 1: CLAMP Q(T|X)
        for x in range(X):
            tstar = np.argmax(qt_x[:,x])
            qt_x[:,x] = 0
            qt_x[tstar,x] = 1        
        # STEP 2: UPDATE Q(T)
        qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)        
        # STEP 3: UPDATE Q(Y|T)
        qy_t = qy_t_step(qt_x,qt,px,py_x)        
        # calculate and print metrics
        ht, hy_t, iyt, ht_x, ixt, L = calc_metrics(qt_x,qt,qy_t,px,hy,alpha,beta)
        if verbose>0: print('clamped: ' + metrics_string(ixt,ht,hx,iyt,ixy,L))            
        # store everything
        clamp_step_time = time.time()-clamp_start_time
        metrics_conv = metrics_conv.append(pd.DataFrame(data={
                        'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht, 'T': T,
                        'ht_x': ht_x, 'hy_t': hy_t, 'hx': hx, 'ixy': ixy,
                        'time': conv_time+clamp_step_time, 'step': Nsteps+1,
                        'Tmax': Tmax, 'beta': beta, 'alpha': alpha,
                        'p0': p0, 'waviness': waviness, 'zeroLtol': zeroLtol,
                        'ctol_abs': ctol_abs, 'ctol_rel': ctol_rel,
                        'ptol': ptol, 'conv_condition': conv_condition,
                        'clamp': True}, index=[0]), ignore_index = True)
        if compact>0: dist_conv = dist_conv.append(pd.DataFrame(data={
                        'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                        'time': conv_time+clamp_step_time, 'step': Nsteps+1,
                        'Tmax': Tmax, 'beta': beta, 'alpha': alpha,
                        'p0': p0, 'waviness': waviness, 'ctol_abs': ctol_abs,
                        'ctol_rel': ctol_rel, 'ptol': ptol, 'zeroLtol': zeroLtol,
                        'conv_condition': conv_condition, 'clamp': True},
                        index=[0]), ignore_index = True)
    
    # return results
    if compact<=1: dist_sw = None
    if compact==0: dist_conv = None
    return metrics_sw, dist_sw, metrics_conv, dist_conv
            
def refine_beta(metrics_conv,verbose=2):
    """Helper function for IB to automate search over parameter beta."""
    
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

    # sort fits by beta
    metrics_conv = metrics_conv.sort_values(by='beta')
    
    # init
    new_betas = []
    NaNtran = False
    ixy = metrics_conv['ixy'].iloc[0]
    logT = math.log2(metrics_conv['Tmax'].iloc[0])
    if verbose>0: print('-----------------------------------')
    
    # check that smallest beta was small enough
    if metrics_conv['ixt'].min()>eps0:
        minbeta = metrics_conv['beta'].min()
        new_betas += [minbeta*(f0**n) for n in range(1,l0+1)]
        if verbose==2: print('Added %i small betas. %.1f was too large.' % (l0,minbeta))
    
    # check for gaps to fill
    for i in range(metrics_conv.shape[0]-1):
        beta1 = metrics_conv['beta'].iloc[i]
        beta2 = metrics_conv['beta'].iloc[i+1]
        cc1 = metrics_conv['conv_condition'].iloc[i]
        cc2 = metrics_conv['conv_condition'].iloc[i+1]
        NaNtran = (("cost_func_NaN" not in cc1) and ("cost_func_NaN" in cc2))
        # if beta gap not too small, do all metric gap checks
        if (beta2-beta1)>min_abs_res and ((beta2-beta1)/beta1)>min_rel_res:
            iyt1 = metrics_conv['iyt'].iloc[i]
            iyt2 = metrics_conv['iyt'].iloc[i+1]
            ixt1 = metrics_conv['ixt'].iloc[i]
            ixt2 = metrics_conv['ixt'].iloc[i+1]
            ht1 = metrics_conv['ht'].iloc[i]
            ht2 = metrics_conv['ht'].iloc[i+1]            
            if ((abs(iyt1-iyt2)/ixy)>del_R) or\
               ((abs(ixt1-ixt2)/logT)>del_C) or\
               ((abs(ht1-ht2)/logT)>del_C) or\
               NaNtran:
                   new_betas += list(np.linspace(beta1,beta2,l+2)[1:l+1])
                   if verbose==2: print('Added %i betas between %.1f and %.1f.' % (l,beta1,beta2))
            if NaNtran: # stop search if there was a NaNtran
                if verbose==2: print('(...because there was a transition to NaNs.)')
                break
    
    # check that largest beta was large enough
    if ((metrics_conv['iyt'].max()/ixy)<eps1) and ~NaNtran:
        maxbeta = metrics_conv['beta'].max()
        new_betas += [maxbeta*(f1**n) for n in range(1,l1+1)]
        if verbose==2: print('Added %i large betas. %.1f was too small.' % (l1,maxbeta))
        
    # filter out betas above max_beta_allowed
    if any([beta>max_beta_allowed for beta in new_betas]):
        if verbose==2: print('Filtered out %i betas larger than max_beta_allowed.' % len([beta for beta in new_betas if beta>max_beta_allowed]))
        new_betas = [beta for beta in new_betas if beta<max_beta_allowed]
        if max_beta_allowed in (list(metrics_conv['beta'].values)+new_betas):
            if verbose==2: print('...and not replaced since max_beta_allowed = %i already used.' % max_beta_allowed)
        else:
            new_betas += [max_beta_allowed]
            if verbose==2: print('And replaced them with max_beta_allowed = %i.' % max_beta_allowed)        
    
    if verbose>0:
        print('Added %i new beta(s).' % len(new_betas))
        print('-----------------------------------')

    return new_betas
    
def set_param(fit_param,param_name,def_val):
    """Helper function for IB.py to handle setting of fit parameters."""
    param = fit_param.get(param_name,def_val) # extract param, def = def_val
    if param is None: param = def_val # if extracted param is None, use def_val
    return param

def make_param_dict(fit_param,*args):
    """Helper function for IB.py to handle setting of keyword arguments for the
    IB_single.py function."""
    param_dict = {}
    for arg in args: # loop over column names passed
        if fit_param.get(arg,None) is not None: # if col specified exists and isn't None...
            param_dict[arg] = fit_param[arg] # ...add to keyword dictionary
    return param_dict

def IB(pxy,fit_param,compact=1,verbose=2):
    """Performs many generalized IB fits to a single p(x,y).
    
    One fit is performed for each row of input dataframe fit_param. Columns
    correspond to fit parameters.
    
    INPUTS
    pxy = input distribution p(x,y) [=] X x Y
    fit_param = pandas df, with each row specifying a single round of IB fits,
            where a "round" means a bunch of fits where the only parameter that
            varies from fit to fit is beta; columns include:
        **** required ****
        alpha = IB parameter interpolating between IB and DIB [=] pos scalar (required)
        **** optional (have defaults) ****
        *** parameters not passed directly to IB_single (defaults set below) ***
        betas = list of initial beta values to run, where beta is an IB parameter
            specifying the "coarse-grainedness" of the solution [=] list of pos scalars
        beta_search = flag indicating whether to perform automatic beta search
            or use *only* initial beta(s) [=] boolean
        max_fits = max number of beta fits allowed for each input row [=] pos integer
        max_time = max time (in seconds) allowed for fitting of each input row [=] pos scalar
        repeats = repeated fits per beta / row, after which fit with best value
            of objective function L is retained [=] pos int
        *** parameters passed to IB_single (defaults set there) ***    
        Tmax = max cardinality of T / max # of clusters; if None, defaults
            to using |X|, which is the most conservative setting [=] pos integer
        p0 = determines initialization of q(t|x) jointly with waviness
            for zero p0, q(t|x) is initialized fully randomly with no structure;
                see waviness below for more
            for pos p0, p0 is prob mass on ~unique cluster for each input x
                (i.e. q(t_i|x_i)=p0 where t_i is unique for each x_i)
            for neg p0, p0 is prob mass on shared cluster for all inputs
                (i.e. q(t*_x_i)=p0 for all x_i)
            what happens to the remaining 1-p0 prob mass is determined by waviness
        waviness = if waviness is None, remaining prob mass is a normalized
            uniform random vector. Otherwise, remaining prob mass is assigned
            uniformly, with uniform noise of magnitude +- waviness [=] 0<=waviness<=1
            (for more on p0 and waviness, see init_qt_x)
        ctol_abs = absolute convergence tolerance; if L is the objective
            function, then if abs(L_old-L)<ctol_abs, converge [=] non-neg scalar
        ctol_rel = relative convergence tolerance; if
            abs(L_old-L)/abs(L_old)<ctol_rel, converge [=] non-neg scalar
        ptol = probalitiy tolerance; x,y,t values dropped if prob<ptol [=] non-neg scalar        
        zeroLtol = if converged solution has L>zeroLtol, revert to solution
            mapping all x to same t (which has L=0) [=] non-neg scalar
        clamp = if true, for all non-DIB fits, insert a clamped version of the
            solution into the results after convergence [=] boolean
    verbose = integer indicating verbosity of updates [=] {0,1,2}
             0: only tell me about errors;
             1: tell me when and why things converge;
             2: tell me about every step of the algorithm
    compact = integer indicating how much data to save [=] {0,1,2}
             0: only save metrics;
             1: also save converged distributions;
             2: also save stepwise distributions
    
    OUTPUTS
    all outputs are pandas dfs. "sw" means stepwise; each row corresponds to a
    step in the IB algorithm. "conv" means converged; each row corresponds to a
    converged solution of the IB algorithm. "metrics" has columns corresponding
    to things like the objective function value L, informations and entropies,
    and the number of clusters used, while "dist" has columns for the actual
    distributions being optimizing, such as the encoder q(t|x). thus, dist dfs
    are much larger than metrics dfs. "allreps" means that all repeats for a
    set of parameters are included, whereas those dfs without this tag retain
    only the 'best' fit, that is the one with lowest L.
    *** columns that all dfs contain ***
    parameters described above: alpha, beta (single value now; not list), Tmax,
                                p0, ctol_abs, ctol_rel, ptol, zeroLtol, repeats
    repeat = id of repeat [=] pos integer in range of 0 to repeats
    step = index of fit step for 'sw', or number of steps to converge for 'conv'
        [=] pos int
    time = time to complete this step for 'sw', or time to converge for 'conv'
        [=] pos scale (in s)
    *** columns that only 'conv' dfs contain ***
    conv_condition
    clamp
    *** columns that only 'metrics' dfs contain ***
    L = objective function value [=] scalar
    T = number of clusters used [=] pos integer
    ht = H(T) [=] pos scalar
    ht_x = H(T|X) [=] pos scalar
    hy_t = H(Y|T) [=] pos scalar
    ixt = I(X,T) [=] pos scalar
    iyt = I(Y,T) [=] pos scalar
    hx = H(X) [=] pos scalar (fixed property of p(x,y); here for comparison)
    ixy = I(X,Y) [=] pos scalar (fixed property of p(x,y); here for comparison)
    *** columns that only 'dist' dfs contain ***
    qt_x = q(t|x) = [=] T x X (note: size T may change during iterations)
    qt = q(t) [=] T x 1 (note: size T may change during iterations)
    qy_t = q(y|t) [=] Y x T (note: size T may change during iterations)"""
    
    # set defaults
    def_betas = [.1,1,2,3,4,5,7,9,10]
    def_betas.reverse()
    def_beta_search = True
    def_max_fits = 100
    def_max_time = 7*24*60*60 # 1 week
    def_repeats = 1
    
    # frequently used column names
    metrics_cols = ['L','T','ht','ht_x','hy_t','ixt','iyt','hx','ixy']
    dist_cols = ['qt_x','qt','qy_t']
    converged_cols = ['conv_condition','clamp']
    all_cols = ['alpha','beta','Tmax','p0','waviness','ctol_abs','ctol_rel',
                'ptol','zeroLtol','step','time','repeat','repeats']
    
    # initialize dataframes
    metrics_sw = pd.DataFrame(columns=all_cols+metrics_cols)
    metrics_sw_allreps = pd.DataFrame(columns=all_cols+metrics_cols)
    metrics_conv = pd.DataFrame(columns=all_cols+metrics_cols+converged_cols)
    metrics_conv_allreps = pd.DataFrame(columns=all_cols+metrics_cols+converged_cols)
    if compact>1: dist_sw = pd.DataFrame(columns=all_cols+dist_cols)
    if compact>1: dist_sw_allreps = pd.DataFrame(columns=all_cols+dist_cols) 
    if compact>0: dist_conv = pd.DataFrame(columns=all_cols+dist_cols+converged_cols)
    if compact>0: dist_conv_allreps = pd.DataFrame(columns=all_cols+dist_cols+converged_cols)                                                                                            
    
    # iterate over fit parameters (besides beta, which is done below)
    fit_param = fit_param.where((pd.notnull(fit_param)), None) # NaN -> None                               
    for irow in range(len(fit_param.index)):
        
        # extract parameters for this fit
        this_fit = fit_param.iloc[irow]
        this_alpha = this_fit['alpha']
        # optional parameters that have defaults set above
        this_betas = set_param(this_fit,'betas',def_betas[:]) # slice here to pass by value, not ref
        if not isinstance(this_betas,list): this_betas = [this_betas]
        this_beta_search = set_param(this_fit,'beta_search',def_beta_search)
        this_max_fits = set_param(this_fit,'max_fits',def_max_fits)
        this_max_time = set_param(this_fit,'max_time',def_max_time)
        this_repeats = int(set_param(this_fit,'repeats',def_repeats))
        # optional parameters that have defaults set by IB_single.py
        param_dict = make_param_dict(this_fit,'Tmax','p0','waviness','ctol_abs','ctol_rel','ptol','zeroLtol','clamp') 
        # make pre-fitting initializations
        betas = this_betas # stack of betas
        fit_count = 0
        fit_time = 0
        fit_start_time = time.time()
        # this df is used for beta refinement
        these_betas_metrics_conv = pd.DataFrame(columns=['ht','ixt','iyt','hx','ixy','Tmax','beta','conv_condition'])
        
        while (fit_count<=this_max_fits) and (fit_time<=this_max_time) and (len(betas)>0):
            # pop beta from stack
            this_beta = betas.pop()            
            # init data structures that will store the repeated fits for this particular setting of parameters
            these_reps_metrics_sw = pd.DataFrame(columns=all_cols+metrics_cols)
            these_reps_metrics_conv = pd.DataFrame(columns=all_cols+metrics_cols+converged_cols)
            if compact>1: these_reps_dist_sw = pd.DataFrame(columns=all_cols+dist_cols)                                                                                                                         
            if compact>0: these_reps_dist_conv = pd.DataFrame(columns=all_cols+dist_cols+converged_cols)
            
            # loop over repeats                                        
            for repeat in range(this_repeats):
                # do a single fit
                if verbose>0: print("+++++++++++ repeat %i of %i +++++++++++" % (repeat+1,this_repeats))
                this_metrics_sw, this_dist_sw, this_metrics_conv, this_dist_conv = \
                    IB_single(pxy,this_beta,this_alpha,compact=compact,verbose=verbose,**param_dict)
                # add repeat labels
                this_metrics_sw['repeat'] = repeat
                this_metrics_sw['repeats'] = this_repeats
                this_metrics_conv['repeat'] = repeat
                this_metrics_conv['repeats'] = this_repeats
                if compact>1: this_dist_sw['repeat'] = repeat
                if compact>1: this_dist_sw['repeats'] = this_repeats
                if compact>0: this_dist_conv['repeat'] = repeat
                if compact>0: this_dist_conv['repeats'] = this_repeats
                # add this repeat to these repeats
                these_reps_metrics_sw = these_reps_metrics_sw.append(this_metrics_sw, ignore_index = True)
                these_reps_metrics_conv = these_reps_metrics_conv.append(this_metrics_conv, ignore_index = True)
                if compact>1: these_reps_dist_sw = these_reps_dist_sw.append(this_dist_sw, ignore_index = True)
                if compact>0: these_reps_dist_conv = these_reps_dist_conv.append(this_dist_conv, ignore_index = True)  
            # end of repeat fit loop for single beta 
            
            # store all repeats
            metrics_sw_allreps = metrics_sw_allreps.append(these_reps_metrics_sw, ignore_index = True)
            metrics_conv_allreps = metrics_conv_allreps.append(these_reps_metrics_conv, ignore_index = True)
            if compact>1: dist_sw_allreps = dist_sw_allreps.append(these_reps_dist_sw, ignore_index = True)
            if compact>0: dist_conv_allreps = dist_conv_allreps.append(these_reps_dist_conv, ignore_index = True)
                
            # find best repeat (lowest L)
            these_reps_metrics_conv_unclamped = these_reps_metrics_conv[these_reps_metrics_conv['clamp']==False]
            best_id = these_reps_metrics_conv_unclamped['L'].idxmin()
            if np.isnan(best_id): # if all repeats NaNs, just use first repeat
                best_repeat = 0
            else: # otherwise use best
                best_repeat = these_reps_metrics_conv['repeat'].loc[best_id]
            best_metrics_conv = these_reps_metrics_conv[these_reps_metrics_conv['repeat']==best_repeat]
            best_metrics_sw = these_reps_metrics_sw[these_reps_metrics_sw['repeat']==best_repeat]
            if compact>1: best_dist_sw = these_reps_dist_sw[these_reps_dist_sw['repeat']==best_repeat]
            if compact>0: best_dist_conv = these_reps_dist_conv[these_reps_dist_conv['repeat']==best_repeat]
                
            # store in best fits dataframe 
            metrics_sw = metrics_sw.append(best_metrics_sw, ignore_index = True)
            metrics_conv = metrics_conv.append(best_metrics_conv, ignore_index = True)
            if compact>1: dist_sw = dist_sw.append(best_dist_sw, ignore_index = True)
            if compact>0: dist_conv = dist_conv.append(best_dist_conv, ignore_index = True)
                
            # store best fits across beta for this set of parameters
            these_betas_metrics_conv = these_betas_metrics_conv.append(best_metrics_conv[best_metrics_conv['clamp']==False], ignore_index = True)
                    
            # advance
            fit_count += this_repeats
            fit_time = time.time()-fit_start_time
            
            # refine beta if needed
            if len(betas)==0 and this_beta_search:
                betas = refine_beta(these_betas_metrics_conv,verbose)
                betas.reverse()
                
        if verbose>0:
            if fit_count>=this_max_fits: print('Stopped beta refinement because ran over max fit count of %i' % this_max_fits)
            if fit_time>=this_max_time: print('Stopped beta refinement because ran over max fit time of %i seconds' % this_max_time)
            if len(betas)==0: print('Beta refinement complete.')

    # end iteration over fit parameters
    if compact<=1: dist_sw, dist_sw_allreps = None, None
    if compact==0: dist_conv, dist_conv_allreps = None, None
    return metrics_sw, dist_sw, metrics_conv, dist_conv,\
           metrics_sw_allreps, dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps

def clamp_IB(metrics_conv,dist_conv,pxy,verbose=1):
    """Function to 'clamp' IB fits after the fact; IB also has functionality
        for doing this during normal fitting; see above. Assumes all fits in
        input are unclamped."""
    
    metrics_conv['clamp'] = False
    dist_conv['clamp'] = False
    
    # process pxy
    pxy, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy = process_pxy(pxy,verbose)
    
    # init data structure of clamped results
    these_metrics_conv = pd.DataFrame(columns=list(metrics_conv.columns.values))
    these_dist_conv = pd.DataFrame(columns=list(dist_conv.columns.values))
    
    # iterate over converged results
    for irow in range(len(dist_conv.index)):
        
        alpha = metrics_conv['alpha'].iloc[irow] 
        if alpha>0: # don't clamp DIB fits
        
            start_time = time.time()
            beta = metrics_conv['beta'].iloc[irow]
            Tmax = metrics_conv['Tmax'].iloc[irow]
            p0 = metrics_conv['p0'].iloc[irow]
            waviness = metrics_conv['waviness'].iloc[irow]          
            ctol_abs = metrics_conv['ctol_abs'].iloc[irow]
            ctol_rel = metrics_conv['ctol_rel'].iloc[irow]
            ptol = metrics_conv['ptol'].iloc[irow]
            zeroLtol = metrics_conv['zeroLtol'].iloc[irow]
            conv_condition = metrics_conv['conv_condition'].iloc[irow]

            if verbose>0:
                print('****************************** Clamping IB fit with following parameters ******************************')
                if Tmax is None:
                    print('alpha = %.2f, beta = %.1f, Tmax = None, p0 = %.3f, waviness = %.2f, ctol_abs = %.1e, ctol_rel = %.1e, ptol = %.1e, zeroLtol = %.1e'\
                        % (alpha,beta,p0,waviness,ctol_abs,ctol_rel,ptol,zeroLtol))
                else:
                    print('alpha = %.2f, beta = %.1f, Tmax = %i, p0 = %.3f, waviness = %.2f, ctol_abs = %.1e, ctol_rel = %.1e, ptol = %.1e, zeroLtol = %.1e'\
                        % (alpha,beta,Tmax,p0,waviness,ctol_abs,ctol_rel,ptol,zeroLtol))
                print('**************************************************************************************************')

            qt_x = dist_conv['qt_x'].iloc[irow]
            
            # STEP 1: CLAMP Q(T|X)
            for x in range(X):
                tstar = np.argmax(qt_x[:,x])
                qt_x[:,x] = 0
                qt_x[tstar,x] = 1            
            # STEP 2: UPDATE Q(T)
            qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)            
            # STEP 3: UPDATE Q(Y|T)
            qy_t = qy_t_step(qt_x,qt,px,py_x)            
            # calculate and print metrics
            ht, hy_t, iyt, ht_x, ixt, L = calc_metrics(qt_x,qt,qy_t,px,hy,alpha,beta)
            if verbose>0:
                old_ixt = metrics_conv['ixt'].iloc[irow]
                old_ht = metrics_conv['ht'].iloc[irow]
                old_iyt = metrics_conv['iyt'].iloc[irow]
                old_L = metrics_conv['L'].iloc[irow]
                print('***** unclamped fit *****')
                print('I(X,T) = %.4f, H(T) = %.4f, H(X) = %.4f, I(Y,T) = %.4f, I(X,Y) = %.4f, L = %.4f' % (old_ixt,old_ht,hx,old_iyt,ixy,old_L))
                print('***** clamped fit *****')
                print('I(X,T) = %.4f, H(T) = %.4f, H(X) = %.4f, I(Y,T) = %.4f, I(X,Y) = %.4f, L = %.4f' % (ixt,ht,hx,iyt,ixy,L))
                
            # store everything
            this_step_time = time.time()-start_time
            Nsteps = metrics_conv['step'].iloc[irow]+1
            conv_time = metrics_conv['time'].iloc[irow]+this_step_time
            these_metrics_conv = these_metrics_conv.append(pd.DataFrame(data={
                            'L': L, 'ixt': ixt, 'iyt': iyt, 'ht': ht, 'T': T,
                            'ht_x': ht_x, 'hy_t': hy_t, 'hx': hx, 'ixy': ixy, 
                            'time': conv_time, 'step': Nsteps, 'Tmax': Tmax,
                            'beta': beta, 'alpha': alpha, 'p0': p0, 'waviness': waviness,
                            'zeroLtol': zeroLtol, 'ctol_abs': ctol_abs,
                            'ctol_rel': ctol_rel, 'ptol': ptol,
                            'conv_condition': conv_condition, 'clamp': True},
                            index=[0]),ignore_index = True)
            these_dist_conv = these_dist_conv.append(pd.DataFrame(data={
                            'qt_x': [qt_x], 'qt': [qt], 'qy_t': [qy_t],
                            'Tmax': Tmax, 'beta': beta, 'alpha': alpha,
                            'p0': p0, 'waviness': waviness, 'ctol_abs': ctol_abs,
                            'ctol_rel': ctol_rel, 'ptol': ptol, 'zeroLtol': zeroLtol,
                            'time': conv_time, 'step': Nsteps,
                            'conv_condition': conv_condition,  'clamp': True},
                            index=[0]),ignore_index = True)
            
    metrics_conv = metrics_conv.append(these_metrics_conv, ignore_index = True)
    dist_conv = dist_conv.append(these_dist_conv, ignore_index = True)
    
    return metrics_conv, dist_conv