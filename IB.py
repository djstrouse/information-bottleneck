import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import pickle
vlog = np.vectorize(math.log)
vexp = np.vectorize(math.exp)

# A word on notation: for probability variables, an underscore here means a
# conditioning, so read _ as |.

# todos:
#   update documentation for OOP version
#   fold smoothing into IB so that s is a parameter
#   implement non-spherical smoothing into dataset class
#   function to plot geometric clustering solutions (i.e. color points in coord plane)
#   allow refine_beta parameters to be set by model.fit()
#   draw IB curve during fits

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

class dataset:
    
    def __init__(self,pxy=None,coord=None,labels=None,gen_param=None,name=None,s=None):
        if pxy is not None:
            if not(isinstance(pxy,np.ndarray)):
                raise ValueError('pxy must be a numpy array')
            if np.any(pxy<0) or np.any(pxy>1):
                raise ValueError('entries of pxy must be between 0 and 1')
            if abs(np.sum(pxy)-1)>10**-8:
                raise ValueError('pxy must be normalized')
        self.pxy = pxy # the distribution that (D)IB acts upon
        if coord is not None:
            if not(isinstance(coord,np.ndarray)):
                raise ValueError('coord must be a numpy array')
        self.coord = coord # locations of data points if geometric, assumed 2D
        if labels is not None:
            if len(labels)!=coord.shape[0]:
                raise ValueError('number of labels must match number of rows in coord')
        self.labels = labels # class labels of data (if synthetic)
        if s is not None:
            if not(isinstance(s,(int,float))) and s>0:
                raise ValueError('s must be a positive scalar')
        self.s = s # smoothing parameter for coord->pxy
        if gen_param is not None:
            if not(isinstance(gen_param,dict)):
                raise ValueError('gen_param must be a dictionary')
        self.gen_param = gen_param # generative parameters of data (if synthetic)
        self.name = name # name of dataset, used for saving
        if self.pxy is not None:
            self.process_pxy()
        elif self.coord is not None:
            self.X = self.coord.shape[0]
        if self.pxy is None and self.coord is not None and self.s is not None:
            self.coord_to_pxy()
        
    def __str__(self):
        return(self.name)
    
    def process_pxy(self):
        """Drops unused x and y, and calculates info-theoretic stats of pxy."""
        Xorig, Yorig = self.pxy.shape
        px = self.pxy.sum(axis=1)
        py = self.pxy.sum(axis=0)
        nzx = px>0 # find nonzero-prob entries
        nzy = py>0
        zx = np.where(px<=0)[0]
        zy = np.where(py<=0)[0]
        self.px = px[nzx] # drop zero-prob entries
        self.py = py[nzy]
        self.X = len(px)
        self.Y = len(py)
        if (Xorig-self.X)>0:
            print('%i of %i Xs dropped due to zero prob; size now %i. Dropped IDs:' % (Xorig-self.X,Xorig,self.X))
            print(zx)
        if (Yorig-self.Y)>0:
            print('%i of %i Ys dropped due to zero prob; size now %i. Dropped IDs:' % (Yorig-self.Y,Yorig,self.Y))
            print(zy)
        pxy_orig = self.pxy
        tmp = pxy_orig[nzx,:]
        self.pxy = tmp[:,nzy] # pxy_orig with zero-prob x,y removed
        self.py_x = np.multiply(self.pxy.T,np.tile(1./self.px,(self.Y,1)))
        self.hx = entropy(self.px)
        self.hy = entropy(self.py)
        self.hy_x = np.dot(self.px,entropy(self.py_x))
        self.ixy = self.hy-self.hy_x

    def coord_to_pxy(self,total_bins=2500):
        """Uses smoothing paramters to transform coord into pxy."""
        # assumes 2D coord, total_bins is approximate
        
        print('Smoothing coordinates with scale %.2f into p(x,y)' % self.s)
        
        pad = 2*self.s # bins further than this from all data points are dropped
        
        # dimensional preprocessing
        min_x1 = np.min(self.coord[:,0])
        min_x2 = np.min(self.coord[:,1])
        max_x1 = np.max(self.coord[:,0])
        max_x2 = np.max(self.coord[:,1])
        range_x1 = max_x1-min_x1
        range_x2 = max_x2-min_x2
        bins1 = int(math.sqrt(total_bins*range_x1/range_x2)) # divy up bins according to spread of data
        bins2 = int(math.sqrt(total_bins*range_x2/range_x1))
        Y = int(bins1*bins2)
        
        # generate bins and construct gaussian-smoothed p(y|x)
        min_y1 = min_x1-pad
        max_y1 = max_x1+pad
        min_y2 = min_x2-pad
        max_y2 = max_x2+pad
        y1 = np.linspace(min_y1,max_y1,bins1)
        y2 = np.linspace(min_y2,max_y2,bins2)
        y1v,y2v = np.meshgrid(y1,y2)
        Ygrid = np.array([np.reshape(y1v,Y),np.reshape(y2v,Y)]).T    
        py_x = np.zeros((Y,self.X))
        y_count_near = np.zeros(Y) # counts data points within pad of each bin
        for x in range(self.X):
            for y in range(Y):
                l = np.linalg.norm(self.coord[x,:]-Ygrid[y,:])
                py_x[y,x] = (1./math.sqrt(2*math.pi*(self.s**(2*2))))*math.exp(-(1./(2.*(self.s**2)))*l)
                if l<pad: y_count_near[y] += 1
            
        # drop ybins that are too far away from data
        ymask = y_count_near>0
        py_x = py_x[ymask,:]
        print("Dropped %i ybins. Y reduced from %i to %i." % (Y-np.sum(ymask),Y,np.sum(ymask)))
        self.Y = np.sum(ymask)
        # normalize p(y|x), since gaussian binned/truncated and bins dropped
        for x in range(self.X): py_x[:,x] = py_x[:,x]/np.sum(py_x[:,x])
        self.py_x = py_x
        # construct p(x,y)
        self.px = (1/self.X)*np.ones(self.X)    
        self.pxy = np.multiply(np.tile(self.px,(self.Y,1)),self.py_x).T
        
        # calc and display I(x,y)
        self.process_pxy()
        print("I(X;Y) = %.3f" % self.ixy)
        
    def plot_coord(self):
        if self.coord is not None:
            plt.scatter(self.coord[:,0],self.coord[:,1])
            plt.axis('scaled')
            plt.show()
        else:
            print("coord not yet defined")
        
    def plot_pxy(self):
        if self.pxy is not None:
            plt.contourf(self.pxy)
            plt.show()
        else:
            print("pxy not yet defined")
            
    def save(self,directory,filename=None):
        """Pickles dataset in directory with filename."""
        if filename is None: filename = self.name+'_dataset'
        with open(directory+filename+'.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            
    def load(self,directory,filename=None):
        """Replaces current content with pickled data in directory with filename."""
        if filename is None: filename = self.name+'_dataset'
        with open(directory+filename+'.pkl', 'rb') as input:
            obj = pickle.load(input)
        self.__init__(pxy = obj.pxy, coord = obj.coord, labels = obj.labels,
                      gen_param = obj.gen_param, name = obj.name, s = obj.s)
            
class model:

    def __init__(self,ds,alpha,beta,Tmax=None,qt_x=None,p0=None,waviness=None,
                 ctol_abs=10**-4,ctol_rel=0.,cthresh=1,ptol=10**-8,zeroLtol=0,
                 geoapprox=False,quiet=False):
        if not(isinstance(ds,dataset)):
            raise ValueError('ds must be a dataset')
        self.ds = ds # dataset
        if alpha<0 or not(isinstance(alpha,(int,float,np.int64))):
            raise ValueError('alpha must be a non-negative scalar')
        else: alpha = float(alpha)
        self.alpha = alpha
        if not(beta>0) or not(isinstance(beta,(int,float,np.int64))):
            raise ValueError('beta must be a positive scalar')
        else: beta = float(beta)
        self.beta = beta
        if Tmax is None:
            Tmax = ds.X
            print('Tmax set to %i based on X' % Tmax)
        elif Tmax<1 or Tmax!=int(Tmax):
            raise ValueError('Tmax must be a positive integer')
        elif Tmax>ds.X:            
            print('Reduced Tmax from %i to %i based on X' % (Tmax,ds.X))
            Tmax = ds.X
        else: Tmax = int(Tmax)
        self.Tmax = Tmax
        self.T = Tmax
        if ctol_rel==0 and ctol_abs==0:
            raise ValueError('One of ctol_rel and ctol_abs must be postive')
        if ctol_abs<0 or not(isinstance(ctol_abs,float)):
            raise ValueError('ctol_abs must be a non-negative float')
        self.ctol_abs = ctol_abs
        if ctol_rel<0 or not(isinstance(ctol_rel,float)):
            raise ValueError('ctol_rel must be a non-negative float')
        self.ctol_rel = ctol_rel
        if cthresh<1 or cthresh!=int(cthresh):
            raise ValueError('cthresh must be a positive integer')
        self.cthresh = cthresh
        if not(ptol>0) or not(isinstance(ptol,float)):
            raise ValueError('ptol must be a positive float')
        self.ptol = ptol
        if zeroLtol<0:
            raise ValueError('zeroLtol must be a non-negative float or integer')
        self.zeroLtol = zeroLtol
        self.geoapprox = geoapprox
        self.quiet = quiet
        self.clamped = False
        self.conv_time = None
        self.conv_condition = None
        self.step = 0
        if p0 is None:
            if alpha==0: p0 = 1. # DIB default: deterministic init that spreads points evenly across clusters
            else: p0 = .75 # non-DIB default: DIB-like init but with only 75% prob mass on "assigned" cluster
        elif p0<-1 or p0>1 or not(isinstance(p0,(int,float))):
            raise ValueError('p0 must be a float/int between -1 and 1')
        else: p0 = float(p0)
        self.p0 = p0
        if waviness is not None and (waviness<0 or waviness>1 or not(isinstance(waviness,float))):
            raise ValueError('waviness must be a float between 0 and 1')
        self.waviness = waviness
        start_time = time.time()
        if qt_x is not None: # use initialization if provided
            if not(isinstance(qt_x,np.ndarray)):
                raise ValueError('qt_x must be a numpy array')
            if isinstance(qt_x,np.ndarray):
                if np.any(qt_x<0) or np.any(qt_x>1):
                    raise ValueError('entries of qt_x must be between 0 and 1')
                if any(abs(np.sum(qt_x,axis=0)-1)>ptol):
                    raise ValueError('columns of qt_x must be normalized')
            self.qt_x = qt_x
            self.T = qt_x.shape[0]
        else: # initialize randomly if not
            self.init_qt_x()
        self.make_step(init=True)
        if self.geoapprox: self.qy_t = None
        else: self.Dxt = None
        self.step_time = time.time()-start_time
        if not(self.quiet): print('init: ' + self.report_metrics())
    
    def init_qt_x(self):
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
        if self.p0==0: # don't insert any peaks; init is all "noise"
            if self.waviness: # flat + wavy style noise
                self.qt_x = np.ones((self.T,self.ds.X))+2*(np.random.rand(self.T,self.ds.X)-.5)*self.waviness # 1+-waviness%                
                for i in range(self.ds.X):
                    self.qt_x[:,i] = self.qt_x[:,i]/np.sum(self.qt_x[:,i]) # normalize
            else: # uniform random vector
                self.qt_x = np.random.rand(self.T,self.ds.X)
                self.qt_x = np.multiply(self.qt_x,np.tile(1./np.sum(self.qt_x,axis=0),(self.T,1))) # renormalize
        elif self.p0>0: # spread points evenly across clusters; "assigned" clusters for each data point get prob mass p0
            if self.waviness:
                # insert wavy noise part          
                self.qt_x = np.ones((self.T,self.ds.X))+2*(np.random.rand(self.T,self.ds.X)-.5)*self.waviness # 1+-waviness%                
                # choose clusters for each x to get spikes
                n = math.ceil(float(self.ds.X)/float(self.T)) # approx number points per cluster
                I = np.repeat(np.arange(0,self.T),n).astype("int") # data-to-cluster assignment vector
                np.random.shuffle(I)
                for i in range(self.ds.X):
                    self.qt_x[I[i],i] = 0 # zero out that cluster
                    self.qt_x[:,i] = (1-self.p0)*self.qt_x[:,i]/np.sum(self.qt_x[:,i]) # normalize others to 1-p0
                    self.qt_x[I[i],i] = self.p0 # insert p0 spike
            else: # uniform random vector instead of wavy
                self.qt_x = np.zeros((self.T,self.ds.X))
                # choose clusters for each x to get spikes
                n = math.ceil(float(self.ds.X)/float(self.T)) # approx number points per cluster
                I = np.repeat(np.arange(0,self.T),n).astype("int") # data-to-cluster assignment vector
                np.random.shuffle(I)
                for i in range(self.ds.X):
                    u = np.random.rand(self.T)
                    u[I[i]] = 0
                    u = (1-self.p0)*u/np.sum(u)
                    u[I[i]] = self.p0
                    self.qt_x[:,i] = u
        else: # put all points in the same cluster; primary cluster gets prob mass |p0|
            p0 = -self.p0
            if self.waviness:      
                self.qt_x = np.ones((self.T,self.ds.X))+2*(np.random.rand(self.T,self.ds.X)-.5)*self.waviness # 1+-waviness%
                t = np.random.randint(self.T) # pick cluster to get delta spike
                self.qt_x[t,:] = np.zeros((1,self.ds.X)) # zero out that cluster
                self.qt_x = np.multiply(self.qt_x,np.tile(1./np.sum(self.qt_x,axis=0),(self.T,1))) # normalize the rest...
                self.qt_x = (1-self.p0)*self.qt_x # ...to 1-p0
                self.qt_x[t,:] = self.p0*np.ones((1,self.ds.X)) # put in delta spike
            else: # uniform random vector instead of wavy
                self.qt_x = np.zeros((self.T,self.ds.X))
                # choose clusters for each x to get spikes
                t = np.random.randint(self.T) # pick cluster to get delta spike
                for i in range(self.ds.X):
                    u = np.random.rand(self.T)
                    u[t] = 0
                    u = (1-self.p0)*u/np.sum(u)
                    u[t] = self.p0
                    self.qt_x[:,i] = u                
    
    def qt_step(self):
        """Peforms q(t) update step for generalized Information Bottleneck."""
        self.qt = np.dot(self.qt_x,self.ds.px)
        dropped = self.qt<=self.ptol # clusters to drop due to near-zero prob
        if any(dropped):
            self.qt = self.qt[~dropped] # drop ununsed clusters
            self.qt_x = self.qt_x[~dropped,:]
            self.T = len(self.qt) # update number of clusters
            self.qt_x = np.multiply(self.qt_x,np.tile(1./np.sum(self.qt_x,axis=0),(self.T,1))) # renormalize
            self.qt = np.dot(self.qt_x,self.ds.px)
            if not(self.quiet): print('%i cluster(s) dropped. Down to %i cluster(s).' % (np.sum(dropped),self.T)) 
        
    def qy_t_step(self):
        """Peforms q(y|t) update step for generalized Information Bottleneck."""        
        self.qy_t = np.dot(self.ds.py_x,np.multiply(self.qt_x,np.outer(1./self.qt,self.ds.px)).T)    
    
    def qt_x_step(self):
        """Peforms q(t|x) update step for generalized Information Bottleneck."""
        if self.T==1: self.qt_x = np.ones((1,self.X))
        else:
            self.qt_x = np.zeros((self.T,self.ds.X))
            for x in range(self.ds.X):
                l = vlog(self.qt)-self.beta*kl(self.ds.py_x[:,x],self.qy_t) # [=] T x 1 # scales like X*Y*T
                if self.alpha==0: self.qt_x[np.argmax(l),x] = 1
                else: self.qt_x[:,x] = vexp(l/self.alpha)/np.sum(vexp(l/self.alpha)) # note: l/alpha<-745 is where underflow creeps in
 
    def build_dist_mat(self):
        """Replaces the qy_t_step whens using geoapprox."""
        self.Dxt = np.zeros((self.ds.X,self.T))
        for x in range(self.ds.X):
            for t in range(self.T):
                for otherx in np.nditer(np.nonzero(self.qt_x[t,:])): # only iterate over x with nonzero involvement
                    self.Dxt[x,t] += self.qt_x[t,otherx]*np.linalg.norm(self.ds.coord[x,:]-self.ds.coord[otherx,:])**2
                self.Dxt[x,t] *= 1/(self.ds.X*self.qt[t])

    def qt_x_step_geoapprox(self):
        """Peforms q(t|x) update step for approximate generalized Information
        Bottleneck, an algorithm for geometric clustering."""
        if self.T==1: self.qt_x = np.ones((1,self.ds.X))
        else:
            self.qt_x = np.zeros((self.T,self.ds.X))
            for x in range(self.ds.X):            
                l = vlog(self.qt)-(self.beta/(2*self.s**2))*self.Dxt[x,:] # only substantive difference from qt_x_step         
                if self.alpha==0: self.qt_x[np.argmax(l),x] = 1
                else: self.qt_x[:,x] = vexp(l/self.alpha)/np.sum(vexp(l/self.alpha)) # note: l/alpha<-745 is where underflow creeps in

    def calc_metrics(self):
        """Calculates IB performance metrics.."""
        self.ht = entropy(self.qt)
        self.hy_t = np.dot(self.qt,entropy(self.qy_t))
        self.iyt = self.ds.hy-self.hy_t
        self.ht_x = np.dot(self.ds.px,entropy(self.qt_x))
        self.ixt = self.ht-self.ht_x
        self.L = self.ht-self.alpha*self.ht_x-self.beta*self.iyt
        
    def report_metrics(self):
        """Returns string of model metrics."""
        self.calc_metrics()
        return 'I(X,T) = %.4f, H(T) = %.4f, H(X) = %.4f, I(Y,T) = %.4f, I(X,Y) = %.4f, L = %.4f' % (self.ixt,self.ht,self.ds.hx,self.iyt,self.ds.ixy,self.L)

    def report_param(self):
        """Returns string of model parameters."""
        if self.p0 is None or self.qt_x is not None: p0_str = 'None'
        else: p0_str = '%.3f' % self.p0
        if self.waviness is None or self.qt_x is not None: waviness_str = 'None'
        else: waviness_str = '%.2f' % self.waviness
        return 'alpha = %.2f, beta = %.1f, Tmax = %i, p0 = %s, wav = %s, geo = %s, ctol_abs = %.0e, ctol_rel = %.0e, cthresh = %i, ptol = %.0e, zeroLtol = %.0e' %\
              (self.alpha, self.beta, self.Tmax, p0_str, waviness_str, self.geoapprox, self.ctol_abs, self.ctol_rel, self.cthresh, self.ptol, self.zeroLtol)

    def make_step(self,init=False):
        """Performs one IB step."""
        if not(init):
            start_time = time.time()
            if self.geoapprox: self.qt_x_step_geoapprox()
            else: self.qt_x_step()
        self.qt_step()
        if self.geoapprox: self.build_dist_mat()
        else: self.qy_t_step()
        self.calc_metrics()
        self.step += 1
        if not(init):
            self.step_time = time.time()-start_time        
              
    def clamp(self):
        """Clamps solution to argmax_t of q(t|x) for each x, i.e. hard clustering."""
        print('before clamp: ' + self.report_metrics())
        if self.alpha==0: print('WARNING: clamping with alpha=0; solution is likely already deterministic.')       
        for x in range(self.ds.X):
            tstar = np.argmax(self.qt_x[:,x])
            self.qt_x[tstar,x] = 1
        self.qt_step()
        if self.geoapprox: self.build_dist_mat()
        else: self.qy_t_step()
        self.clamped = True
        print('after clamp: ' + self.report_metrics())
        
    def panda(self,dist_to_keep=set()):
        """"Return dataframe of model. If dist, include distributions.
        If conv, include converged variables; otherwise include stepwise."""
        df = pd.DataFrame(data={
                'alpha': self.alpha, 'beta': self.beta, 'step': self.step,
                'L': self.L, 'ixt': self.ixt, 'iyt': self.iyt, 'ht': self.ht,
                'T': self.T, 'ht_x': self.ht_x, 'hy_t': self.hy_t,
                'hx': self.ds.hx, 'ixy': self.ds.ixy, 'Tmax': self.Tmax,
                'p0': self.p0, 'waviness': self.waviness,  'ptol': self.ptol,
                'ctol_abs': self.ctol_abs, 'ctol_rel': self.ctol_rel,
                'cthresh': self.cthresh, 'zeroLtol': self.zeroLtol,
                'clamped': self.clamped, 'geoapprox': self.geoapprox,
                'step_time': self.step_time, 'conv_time': self.conv_time,
                'conv_condition': self.conv_condition}, index = [0])
        if 'qt_x' in dist_to_keep:
            df['qt_x'] = [self.qt_x]
        if 'qt' in dist_to_keep:
            df['qt'] = [self.qt]
        if 'qy_t' in dist_to_keep:
            df['qy_t'] = [self.qy_t]
        if 'Dxt' in dist_to_keep:
            df['Dxt'] = [self.Dxt]
        return df

    def depanda(self,df):
        """Replaces current model with one in df."""
        self.alpha = df['alpha'][0]
        self.beta = df['beta'][0]
        self.step = df['step'][0]
        self.L = df['L'][0]
        self.ixt = df['ixt'][0]
        self.iyt = df['iyt'][0]
        self.ht = df['ht'][0]
        self.T = df['T'][0]
        self.ht_x = df['ht_x'][0]
        self.hy_t = df['hy_t'][0]
        self.hx = df['hx'][0]
        self.ixy = df['ixy'][0]
        self.Tmax = df['Tmax'][0]
        self.p0 = df['p0'][0]
        self.waviness = df['waviness'][0]
        self.ptol = df['ptol'][0]
        self.ctol_abs = df['ctol_abs'][0]
        self.ctol_rel = df['ctol_rel'][0]
        self.cthresh = df['cthresh'][0]
        self.zeroLtol = df['zeroLtol'][0]
        self.clamped = df['clamped'][0]
        self.geoapprox = df['geoapprox'][0]        
        self.step_time = df['step_time'][0]
        self.conv_time = df['conv_time'][0]
        self.conv_condition = df['conv_condition'][0]
        self.qt_x = df['qt_x'][0]
        self.qt = df['qt'][0]
        self.qy_t = df['qy_t'][0]
        self.Dxt = df['Dxt'][0]

    def append_conv_condition(self,cond):        
        if self.conv_condition is None: self.conv_condition = cond
        else: self.conv_condition += '_AND_' + cond
                
    def update_sw(self):
        if self.keep_steps:
            # store stepwise data                
            self.metrics_sw = self.metrics_sw.append(self.panda(), ignore_index = True)
            if bool(self.dist_to_keep): self.dist_sw = self.dist_sw.append(self.panda(self.dist_to_keep), ignore_index = True)

    def check_converged(self):
        """Checks if most recent step triggered convergence, and stores step
        if necessary."""
        Lold = self.prev['L'][0] 
        small_changes = False
        
        # check for small changes
        small_abs_changes = abs(Lold-self.L)<self.ctol_abs
        small_rel_changes = (abs(Lold-self.L)/abs(Lold))<self.ctol_rel
        if small_abs_changes or small_rel_changes: self.cstep += 1
        else: self.cstep = 0 # reset counter of small changes in a row
        if small_abs_changes and self.cstep>=self.cthresh:
            self.conv_condition = 'small_abs_changes'
            print('converged due to small absolute changes in objective')  
        if small_rel_changes and self.cstep>=self.cthresh:
            self.append_conv_condition('small_rel_changes')
            print('converged due to small relative changes in objective')
            
        # check for objective becoming NaN
        if np.isnan(self.L):
            self.cstep = self.cthresh            
            self.append_conv_condition('cost_func_NaN')
            print('stopped because objective = NaN')
            
        L_abs_inc_flag = self.L>(Lold+self.ctol_abs)
        L_rel_inc_flag = self.L>(Lold+(abs(Lold)*self.ctol_rel))
        
        # check for reduction to single cluster        
        if self.T==1 and not(L_abs_inc_flag) and not(L_rel_inc_flag):
            self.cstep = self.cthresh
            self.append_conv_condition('single_cluster')
            print('converged due to reduction to single cluster')
            
        # check if obj went up by amount above threshold (after 1st step)
        if (L_abs_inc_flag or L_rel_inc_flag) and self.step>1: # if so, don't store or count this step!
            self.cstep = self.cthresh
            if L_abs_inc_flag:
                self.append_conv_condition('cost_func_abs_inc')
                print('converged due to absolute increase in objective value')
            if L_rel_inc_flag:
                self.append_conv_condition('cost_func_rel_inc')
                print('converged due to relative increase in objective value')
            # revert to metrics/distributions from last step
            self.prev.conv_condition = self.conv_condition
            self.depanda(self.prev)
        #  otherwise, store step
        else: self.update_sw()
                
    def check_single_better(self):
        """ Replace converged step with single-cluster map if better."""
        sqt_x = np.zeros((self.T,self.ds.X))
        sqt_x[0,:] = 1.
        smodel = model(ds=self.ds,alpha=self.alpha,beta=self.beta,Tmax=self.Tmax,
                       qt_x=sqt_x,p0=self.p0,waviness=self.waviness,
                       ctol_abs=self.ctol_abs,ctol_rel=self.ctol_rel,cthresh=self.cthresh,
                       ptol=self.ptol,zeroLtol=self.zeroLtol,geoapprox=self.geoapprox,
                       quiet=True)
        smodel.step = self.step
        smodel.conv_condition = self.conv_condition + '_AND_force_single'
        if smodel.L<(self.L-self.zeroLtol): # if better fit...            
            print("single-cluster mapping reduces L from %.4f to %.4f (zeroLtol = %.1e); replacing solution." % (self.L,smodel.L,self.zeroLtol))
            # replace everything
            self.depanda(smodel.panda(dist_to_keep={'qt_x','qt','qy_t','Dxt'}))
            self.update_sw()
            print('single-cluster solution: ' + self.report_metrics())            
        else: print("single-cluster mapping not better; changes L from %.4f to %.4f (zeroLtol = %.1e)." % (self.L,smodel.L,self.zeroLtol)) 
                
    def fit(self,keep_steps=False,dist_to_keep={'qt_x','qt','qy_t','Dxt'}):
        """Runs generalized IB algorithm to convergence for current model.
        keep_steps determines whether pre-convergence models / statistics about
        them are kept. dist_to_keep is a set with the model distributions to be
        kept for each step."""
        
        fit_start_time = time.time()
                       
        self.keep_steps = keep_steps
        self.dist_to_keep = dist_to_keep
        
        print(20*'*'+' Beginning IB fit with the following parameters '+20*'*')
        print(self.report_param())
        print(88*'*')
        
        # initialize stepwise dataframes, if tracking them             
        if self.keep_steps:
            self.metrics_sw = self.panda()
            if bool(self.dist_to_keep): self.dist_sw = self.panda(self.dist_to_keep)
        
        # check if single cluster init
        if self.T==1:
            self.cstep = self.cthresh
            print('converged due to initialization with single cluster')
            self.conv_condition = 'single_cluster_init'
        else: # init iterative parameters
            self.cstep = 0
            self.conv_condition = None           
        
        # save encoder init
        self.qt_x0 = self.qt_x
        
        # iterate to convergence
        while self.cstep<self.cthresh:
            
            self.prev = self.panda(dist_to_keep={'qt_x','qt','qy_t','Dxt'}) 
            self.make_step()            
            print('step %i: ' % self.step + self.report_metrics())
            self.check_converged()

        # report
        print('converged in %i step(s) to: ' % self.step + self.report_metrics())
        
        # replace converged step with single-cluster map if better
        if self.T>1: self.check_single_better()
        
        # clean up
        self.cstep = None
        self.prev = None
        self.step_time = None

        # record total time to convergence
        self.conv_time = time.time() - fit_start_time

def refine_beta(metrics_conv):
    """Helper function for IB to automate search over parameter beta."""
    
    # parameters governing insertion of betas, or when there is a transition to NaNs (due to under/overflow)
    l = 1 # number of betas to insert into gaps
    del_R = .02 # if fractional change in I(Y;T) exceeds this between adjacent betas, insert more betas
    del_C = .02 # if fractional change in H(T) or I(X;T) exceeds this between adjacent betas, insert more betas
    del_T = 1 # if difference in number of clusters used exceeds this between adjacent betas, insert more betas
    min_abs_res = 1e-4 # if beta diff smaller than this absolute threshold, don't insert; consider as phase transition
    min_rel_res = 1e-4 # if beta diff smaller than this fractional threshold, don't insert
    # parameters governing insertion of betas when I(X;T) doesn't reach zero
    eps0 = 1e-2 # tolerance for considering I(X;T) to be zero
    l0 = 1 # number of betas to insert at low beta end
    f0 = .5 # new betas will be minbeta*f0.^1:l0
    # parameters governing insertion of betas when I(T;Y) doesn't reach I(X;Y)
    eps1 = .999 # tolerance for considering I(T;Y) to be I(X;Y)
    l1 = 1 # number of betas to insert at high beta end
    f1 = 2 # new betas will be maxbeta*f0.^1:l0
    max_beta_allowed = 10000 # any proposed betas above this will be filtered out and replaced it max_beta_allowed

    # sort fits by beta
    metrics_conv = metrics_conv.sort_values(by='beta')
    
    # init
    new_betas = []
    NaNtran = False
    ixy = metrics_conv['ixy'].iloc[0]
    logT = math.log2(metrics_conv['Tmax'].iloc[0])
    print('-----------------------------------')
    
    # check that smallest beta was small enough
    if metrics_conv['ixt'].min()>eps0:
        minbeta = metrics_conv['beta'].min()
        new_betas += [minbeta*(f0**n) for n in range(1,l0+1)]
        print('Added %i small betas. %.1f was too large.' % (l0,minbeta))
    
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
            T1 = metrics_conv['T'].iloc[i]
            T2 = metrics_conv['T'].iloc[i+1]         
            if ((abs(iyt1-iyt2)/ixy)>del_R) or\
               ((abs(ixt1-ixt2)/logT)>del_C) or\
               ((abs(ht1-ht2)/logT)>del_C) or\
               ((T2-T1)>del_T) or\
               NaNtran:
                   new_betas += list(np.linspace(beta1,beta2,l+2)[1:l+1])
                   print('Added %i betas between %.1f and %.1f.' % (l,beta1,beta2))
            if NaNtran: # stop search if there was a NaNtran
                print('(...because there was a transition to NaNs.)')
                break
    
    # check that largest beta was large enough
    if ((metrics_conv['iyt'].max()/ixy)<eps1) and ~NaNtran:
        maxbeta = metrics_conv['beta'].max()
        new_betas += [maxbeta*(f1**n) for n in range(1,l1+1)]
        print('Added %i large betas. %.1f was too small.' % (l1,maxbeta))
        
    # filter out betas above max_beta_allowed
    if any([beta>max_beta_allowed for beta in new_betas]):
        if verbose==2: print('Filtered out %i betas larger than max_beta_allowed.' % len([beta for beta in new_betas if beta>max_beta_allowed]))
        new_betas = [beta for beta in new_betas if beta<max_beta_allowed]
        if max_beta_allowed in (list(metrics_conv['beta'].values)+new_betas):
            print('...and not replaced since max_beta_allowed = %i already used.' % max_beta_allowed)
        else:
            new_betas += [max_beta_allowed]
            print('And replaced them with max_beta_allowed = %i.' % max_beta_allowed)        
    
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

def IB(ds,fit_param,conv_dist_to_keep={'qt_x','qt','qy_t','Dxt'},
       keep_steps=True,sw_dist_to_keep={'qt_x','qt','qy_t','Dxt'}):
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
    def_betas.sort(reverse=True)
    def_beta_search = True
    def_max_fits = 100
    def_max_time = 7*24*60*60 # 1 week
    def_repeats = 1
    def_geoapprox = False
    def_clamp = True
    
    # initialize primary dataframes
    metrics_conv = None
    metrics_sw = None
    dist_conv = None
    dist_sw = None    
    
    # iterate over fit parameters (besides beta, which is done below)
    fit_param = fit_param.where((pd.notnull(fit_param)), None) # NaN -> None
    paramID = 0 # all betas have same
    fitID = 0 # all repeats have same
    fitIDwrep = 0 # repeats get unique       
    for irow in range(len(fit_param.index)):
        
        # tick counter
        paramID += 1
        
        # extract parameters for this fit
        this_fit = fit_param.iloc[irow]
        this_alpha = this_fit['alpha']
        
        # parameters that have defaults set above
        this_betas = set_param(this_fit,'betas',def_betas[:]) # slice here to pass by value, not ref
        if not isinstance(this_betas,list): this_betas = [this_betas]
        this_beta_search = set_param(this_fit,'beta_search',def_beta_search)
        this_max_fits = set_param(this_fit,'max_fits',def_max_fits)
        this_max_time = set_param(this_fit,'max_time',def_max_time)
        this_repeats = int(set_param(this_fit,'repeats',def_repeats))
        this_geoapprox = set_param(this_fit,'geoapprox',def_geoapprox)
        this_clamp = set_param(this_fit,'clamp',def_clamp)
        
        # optional parameters that have defaults set by IB_single.py
        param_dict = make_param_dict(this_fit,'Tmax','p0','waviness','ctol_abs','ctol_rel','cthresh','ptol','zeroLtol')
        
        # make pre-fitting initializations
        betas = this_betas # stack of betas
        fit_count = 0
        fit_time = 0
        fit_start_time = time.time()
        
        # loop over betas
        while fit_count<=this_max_fits and fit_time<=this_max_time and len(betas)>0:
            
            # tick counter
            fitID += 1
            
            # pop beta from stack
            this_beta = betas.pop()  
            
            # init data structures that will store the repeated fits for this particular setting of parameters
            these_metrics_sw = None
            these_metrics_conv = None
            these_dist_sw = None
            these_dist_conv = None  
            
            # loop over repeats                                        
            for repeat in range(this_repeats):
                
                # tick counter
                fitIDwrep += 1
                print("+++++++++++ repeat %i of %i +++++++++++" % (repeat+1,this_repeats))
                
                # do a single fit and extract resulting dataframes
                m = model(ds=ds,alpha=this_alpha,beta=this_beta,**param_dict,geoapprox=False)
                m.fit(keep_steps=keep_steps,dist_to_keep=sw_dist_to_keep)
                this_metrics_conv = m.panda()
                if bool(conv_dist_to_keep): this_dist_conv = m.panda(conv_dist_to_keep)
                
                # once converged model is extracted, clamp if necessary
                if this_clamp and this_alpha!=0:
                    m.clamp()
                    this_metrics_conv = this_metrics_conv.append(m.panda(), ignore_index = True)
                    if bool(conv_dist_to_keep): this_dist_conv = this_dist_conv.append(m.panda(conv_dist_to_keep), ignore_index = True)
                    
                # extract sw models as necessary
                if keep_steps:
                    this_metrics_sw = m.metrics_sw
                    if bool(sw_dist_to_keep): this_dist_sw = m.dist_sw
                    
                # if also running geoapprox...
                if this_geoapprox:
                    
                    # ...then run with approx on same init
                    m2 = model(ds=ds,alpha=this_alpha,beta=this_beta,**param_dict,geoapprox=True,qt_x0=m.qt_x0)
                    m2.fit()
                    
                    # ...and append results
                    this_metrics_conv = this_metrics_conv.append(m2.panda(), ignore_index = True)
                    if bool(conv_dist_to_keep): this_dist_conv = this_dist_conv.append(m2.panda(conv_dist_to_keep), ignore_index = True)
                    
                    # once converged model is extracted, clamp if necessary
                    if this_clamp and this_alpha!=0:
                        m2.clamp()
                        this_metrics_conv = this_metrics_conv.append(m2.panda(), ignore_index = True)
                        if bool(conv_dist_to_keep): this_dist_conv = this_dist_conv.append(m2.panda(conv_dist_to_keep), ignore_index = True)
                        
                    # extract sw models as necessary
                    if keep_steps:
                        this_metrics_sw = this_metrics_sw.append(m2.metrics_sw, ignore_index = True)
                        if bool(sw_dist_to_keep): this_dist_sw = this_dist_sw.append(m2.dist_sw, ignore_index = True)
                        
                # add repeat labels and fit IDs, which are specific to this repeat            
                this_metrics_conv['repeat'] = repeat
                this_metrics_conv['fitIDwrep'] = fitIDwrep
                if keep_steps:
                    this_metrics_sw['repeat'] = repeat
                    this_metrics_sw['fitIDwrep'] = fitIDwrep
                if bool(conv_dist_to_keep):
                    this_dist_conv['repeat'] = repeat
                    this_dist_conv['fitIDwrep'] = fitIDwrep
                if bool(sw_dist_to_keep):
                    this_dist_sw['repeat'] = repeat
                    this_dist_sw['fitIDwrep'] = fitIDwrep
                    
                # add this repeat to these repeats
                if these_metrics_conv is not None: these_metrics_conv = these_metrics_conv.append(this_metrics_conv, ignore_index = True)
                else: these_metrics_conv = this_metrics_conv
                if keep_steps:
                    if these_metrics_sw is not None: these_metrics_sw = these_metrics_sw.append(this_metrics_sw, ignore_index = True)
                    else: these_metrics_sw = this_metrics_sw
                if bool(conv_dist_to_keep):
                    if these_dist_conv is not None: these_dist_conv = these_dist_conv.append(this_dist_conv, ignore_index = True)
                    else: these_dist_conv = this_dist_conv
                if bool(sw_dist_to_keep):
                    if these_dist_sw is not None: these_dist_sw = these_dist_sw.append(this_dist_sw, ignore_index = True)
                    else: these_dist_sw = this_dist_sw
            # end of repeat fit loop for single beta

            # add number of repeats and fit ID (without repeats)
            these_metrics_conv['paramID'] = paramID # this is assigned earlier than expected to help with beta refinement
            these_metrics_conv['fitID'] = fitID
            these_metrics_conv['repeats'] = this_repeats
            if keep_steps:
                these_metrics_sw['paramID'] = paramID
                these_metrics_sw['fitID'] = fitID
                these_metrics_sw['repeats'] = this_repeats
            if bool(conv_dist_to_keep):
                these_dist_conv['paramID'] = paramID
                these_dist_conv['fitID'] = fitID
                these_dist_conv['repeats'] = this_repeats
            if bool(sw_dist_to_keep):
                these_dist_sw['paramID'] = paramID
                these_dist_sw['fitID'] = fitID
                these_dist_sw['repeats'] = this_repeats
            
            # mark best repeat (lowest L): ignored clamped and approx fits
            df = these_metrics_conv[(these_metrics_conv['clamped']==False) & (these_metrics_conv['geoapprox']==False)]
            best_id = df['L'].idxmin()
            if np.isnan(best_id): # if all repeats NaNs, just use first repeat
                best_repeat = 0
            else: # otherwise use best
                best_repeat = these_metrics_conv['repeat'].loc[best_id]
            these_metrics_conv['bestrep'] = False
            these_metrics_conv.loc[these_metrics_conv['repeat'] == best_repeat, 'bestrep'] = True
            if keep_steps:
                these_metrics_sw['bestrep'] = False
                these_metrics_sw.loc[these_metrics_sw['repeat'] == best_repeat, 'bestrep'] = True
            if bool(conv_dist_to_keep):
                these_dist_conv['bestrep'] = False
                these_dist_conv.loc[these_dist_conv['repeat'] == best_repeat, 'bestrep'] = True
            if bool(sw_dist_to_keep):
                these_dist_sw['bestrep'] = False
                these_dist_sw.loc[these_dist_sw['repeat'] == best_repeat, 'bestrep'] = True
                
            # store repeats in primary dataframe
            if metrics_conv is not None: metrics_conv = metrics_conv.append(these_metrics_conv, ignore_index = True)
            else: metrics_conv = these_metrics_conv
            if keep_steps:
                if metrics_sw is not None: metrics_sw = metrics_sw.append(these_metrics_sw, ignore_index = True)
                else: metrics_sw = these_metrics_sw
            if bool(conv_dist_to_keep):
                if dist_conv is not None: dist_conv = dist_conv.append(these_dist_conv, ignore_index = True)
                else: dist_conv = these_dist_conv
            if bool(sw_dist_to_keep):
                if dist_sw is not None: dist_sw = dist_sw.append(these_dist_sw, ignore_index = True)
                else: dist_sw = these_dist_sw

            # advance fit counters
            fit_count += this_repeats
            fit_time = time.time()-fit_start_time
            
            # refine beta if needed
            if len(betas)==0 and this_beta_search:
                betas = refine_beta(metrics_conv[(metrics_conv['paramID']==paramID) & (metrics_conv['bestrep']==True) & (metrics_conv['clamped']==False) & (metrics_conv['geoapprox']==False)])
                if this_geoapprox:
                    print('Now adding betas for geoapprox algorithm...')
                    betas += refine_beta(metrics_conv[(metrics_conv['paramID']==paramID) & (metrics_conv['bestrep']==True) & (metrics_conv['clamped']==False) & (metrics_conv['geoapprox']==True)])
                betas.sort(reverse=True)
                betas = list(set(betas)) # removes duplicates
                
        if fit_count>=this_max_fits: print('Stopped beta refinement because ran over max fit count of %i' % this_max_fits)
        if fit_time>=this_max_time: print('Stopped beta refinement because ran over max fit time of %i seconds' % this_max_time)
        if len(betas)==0: print('Beta refinement complete.')

    # end iteration over fit parameters
    return metrics_conv, dist_conv, metrics_sw, dist_sw