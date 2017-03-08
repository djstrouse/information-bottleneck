from IB import *
import matplotlib.pyplot as plt
import os
import numpy as np

def gen_zipf_pxy():
    X = 1024
    Y = X
    pxy = np.eye(X)
    pxy = pxy/np.sum(pxy[:])
    return pxy
    
def gen_blurred_diag_pxy(s):
    X = 1024
    Y = X

    # generate pdf
    from scipy.stats import multivariate_normal
    pxy = np.zeros((X,Y))
    rv = multivariate_normal(cov=s)
    for x in range(X):        
        pxy[x,:] = np.roll(rv.pdf(np.linspace(-X/2,X/2,X+1)[:-1]),int(X/2+x))
    pxy = pxy/np.sum(pxy)
        
    # plot p(x,y)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(pxy)
    plt.ion()
    plt.title("p(x,y)")
    plt.show()
    
    return pxy

def gen_dir_pxy():
    # param
    X = 128
    Y = 16
    cx = 1000.
    cys = np.logspace(-2.,1.,num=X,base=10)
    # build pxy
    px = np.random.dirichlet(cx*np.ones(X))
    py_x = np.zeros((Y,X))
    for x in range(X):
        py_x[:,x] = np.random.dirichlet(cys[x]*np.ones(Y))
    pxy = np.multiply(np.tile(px,(Y,1)),py_x).T 
    
    # plot p(x,y)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(pxy)
    plt.ion()
    plt.title("p(x,y)")
    plt.show()   
    
    # plot histogram of H(p(y|x)) over x
    plt.hist(entropy(py_x), bins='auto')
    plt.title("entropies of conditionals p(y|x)")
    plt.show()   
    
    # calc ixy
    py = pxy.sum(axis=0)
    hy = entropy(py)
    hy_x = np.dot(px,entropy(py_x))
    ixy = hy-hy_x
    print("I(X;Y) = %.3f" % ixy)
    
    return pxy
    
def gen_gaussian_pxy():
    # param
    cov = np.array([[1.5,1.1],[1.1,1]])
    X = 128
    Y = 128
    xlow = -2
    xhigh = 2
    ylow = -2
    yhigh = 2
    #x, y = np.mgrid[-1.5:1.5:.01, -1.5:1.5:.01]
    x,y = np.meshgrid(np.linspace(xlow,xhigh,X),np.linspace(ylow,yhigh,Y))
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x; pos[:,:,1] = y
    # generate pdf
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt
    rv = multivariate_normal(cov=cov)
    pxy = rv.pdf(pos)
    pxy = pxy/np.sum(pxy)
    # plot to make sure everything looks right
    plt.figure()
    plt.contourf(x, y, rv.pdf(pos))
    plt.ion()
    plt.show()
    # calc ixy analytically and numerically
    cx = abs(cov[0,0])
    cy = abs(cov[1,1])
    c = np.linalg.det(cov)
    ixy_true = .5*math.log2((cx*cy)/c)
    print("I(X;Y) = %.3f (analytical)" % ixy_true) 
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    py_x = np.multiply(pxy.T,np.tile(1./px,(Y,1)))
    hy = entropy(py)
    hy_x = np.dot(px,entropy(py_x))
    ixy_emp = hy-hy_x
    print("I(X;Y) = %.3f (empirical)" % ixy_emp)   
    return pxy
    
def gen_gmm_pxy():
    # param
    mu1 = np.array([1.2,-.5])
    mu2 = np.array([-1,1])
    mu3 = np.array([1,2])
    mu4 = np.array([-2,-2])
    cov1 = np.array([[1.2,1.1],[1,1]])
    cov2 = np.array([[1.05,-.9],[-.95,1.1]])
    cov3 = np.array([[.5,-.35],[-.3,.4]])
    cov4 = np.array([[.6,.4],[.7,1]])
    w1 = .3
    w2 = .27
    w3 = .23
    w4 = .2
    X = 128
    Y = 128
    xlow = -3
    xhigh = 3
    ylow = -3
    yhigh = 3
    x,y = np.meshgrid(np.linspace(xlow,xhigh,X),np.linspace(ylow,yhigh,Y))
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x; pos[:,:,1] = y
    # generate pdf
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt
    rv1 = multivariate_normal(mean=mu1,cov=cov1)
    pxy1 = rv1.pdf(pos)
    rv2 = multivariate_normal(mean=mu2,cov=cov2)
    pxy2 = rv2.pdf(pos)
    rv3 = multivariate_normal(mean=mu3,cov=cov3)
    pxy3 = rv3.pdf(pos)
    rv4 = multivariate_normal(mean=mu4,cov=cov4)
    pxy4 = rv4.pdf(pos)
    pxy = w1*pxy1+w2*pxy2+w3*pxy3+w4*pxy4
    pxy = pxy/np.sum(pxy)
    # plot to make sure everything looks right
    plt.figure()
    plt.contourf(x, y, pxy)
    plt.ion()
    plt.show()
    # calc ixy
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    py_x = np.multiply(pxy.T,np.tile(1./px,(Y,1)))
    hy = entropy(py)
    hy_x = np.dot(px,entropy(py_x))
    ixy_emp = hy-hy_x
    print("I(X;Y) = %.3f (empirical)" % ixy_emp)   
    return pxy
    
def array_split2(x,f,minsize=1):
    """Splits numpy array x into two pieces, with f specifying fraction in 1st."""
    X = len(x)
    if X==1 or X<(2*minsize): # if too few elements, can't split
        return [x]
    i = math.floor(X*f)
    if i<minsize:
        i = minsize
    x_split = [x[0:i],x[i:X]]
    return x_split
    
def gen_hierarchical_pxy(component_type):
    """component_type in {uni,dir}"""
    # todo:
    # could have random num branches per split
    # could bound independent contribution
   
    import matplotlib.pyplot as plt
    # param
    X = 512
    Y = 512
    f = .5 # fraction to scale frozen p added at each depth
    d = 6 # depth of tree
    # total branches = b^d, X per branch = X/(b^d), assume b=2
    if component_type=="dir":
        cw = 1. # concentration for dirichlet over topic/level weights
        cs = np.logspace(0.,-3.,num=d,base=10) # dirichlet concentrations for p(y|x) for each depth
    elif component_type=="uni":
        hmin = .7
        hmax = 1.3
    else:
        raise ValueError("component_type must be uni or dir")    
    
    groups = [[list(range(X))]]
    pxy = np.zeros((X,Y))
    
    # make groups
    for i in range(d):
        newgroups = []
        # loop over groups at this depth
        for g in range(len(groups[i])):
            group = groups[i][g]
            # split group
            minf = .3
            maxf = .7
            s = minf+np.random.rand(1)*(maxf-minf)
            splitgroup = array_split2(group,s)
            # and to new grouping
            newgroups += splitgroup
        groups += [newgroups]
    
    if component_type=="dir":
        # sample level mixture weights for each leaf
        leafs = groups[d]
        G = len(leafs) # number of leaf groups
        W = np.zeros((G,d)) # mixture weights [=] leaves X tree depth
        leaf_groups = np.zeros(X) # id of leaf group for each X
        for g in range(G):
            W[g,:] = np.random.dirichlet(cw*np.ones(d))
            leaf_groups[leafs[g]] = g
        leaf_groups = leaf_groups.astype(int)

    # sample p(y|x) contributions for each depth
    for i in range(d):
        # ADDING TO P(Y|X) FOR EACH NEW GROUP
        for g in range(len(groups[i+1])):
            this_level = groups[i+1]
            group = this_level[g]
            # sample a contribution to p(y|x) for this group and add it to p(x,y)
            x1 = group[0]
            x2 = group[-1]
            if component_type=="dir":
                p = np.random.dirichlet(cs[i]*np.ones(Y)) # "node/topic" distributions
                w_this_depth = W[:,i]
                w_this_group = w_this_depth[leaf_groups[x1:(x2+1)]]
                w = np.tile(w_this_group,(Y,1)).T
                pxy[x1:(x2+1),:] += np.multiply(w,np.tile((f**i)*p,(x2+1-x1,1))) # assumes indices in group are consecutive and ordered!
            elif component_type=="uni":
                y1 = int(round((Y/X)*x1))
                y2 = int(round((Y/X)*x2))
                h = hmin+np.random.rand(1)*(hmin-hmax)
                p = h*np.ones(y2+1-y1)/(y2+1-y1)
                pxy[x1:(x2+1),y1:(y2+1)] += np.tile((f**i)*p,(x2+1-x1,1)) # assumes indices in group are consecutive and ordered!

    # normalize
    pxy = pxy/np.sum(pxy)
    
    pxy2, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy = process_pxy(pxy,0)
    
    # plot p(x,y)
    plt.figure()
    plt.contourf(pxy)
    plt.ion()
    plt.show()
    
    # plot histogram of H(p(y|x)) over x
    print(entropy(py_x))
    plt.hist(entropy(py_x), bins=10)
    plt.title("entropies of conditionals p(y|x)")
    plt.show()    
    
    # calc ixy
    print("I(X;Y) = %.3f" % ixy) 
    
    return pxy, groups
    
def plot_pxy(exp_name):
    
    # load p(x,y)
    cwd = os.getcwd()
    datapath = cwd+'/data/geometric/'+exp_name+'_'
    pxy = np.load(datapath+'pxy.npy') 
    pxy2, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy = process_pxy(pxy,0)
    
    # plot p(x,y)
    plt.figure()
    plt.contourf(pxy)
    plt.ion()
    plt.show()
    
    # plot histogram of H(p(y|x)) over x
    plt.hist(entropy(py_x), bins='auto')
    plt.title("entropies of conditionals p(y|x)")
    plt.show()  
    
    # print ixy
    print("I(X;Y) = %.3f" % ixy)
    
    return 0
    
def coord_to_pxy(coord,s,bins_per_dim=50):
    # assumes 2D coord
    
    print('Smoothing coordinates with scale %.2f into p(x,y)' % s)
    
    pad = 2*s # max distance from data points for bins
    
    # dimensional preprocessing
    X = coord.shape[0]
    Ymax = int(bins_per_dim**2)   
    Y = Ymax
    min_x1 = np.min(coord[:,0])
    min_x2 = np.min(coord[:,1])
    max_x1 = np.max(coord[:,0])
    max_x2 = np.max(coord[:,1])
    
    # generate bins and construct gaussian-smoothed p(y|x)
    min_y1 = min_x1-pad
    max_y1 = max_x1+pad
    min_y2 = min_x2-pad
    max_y2 = max_x2+pad
    y1 = np.linspace(min_y1,max_y1,bins_per_dim)
    y2 = np.linspace(min_y2,max_y2,bins_per_dim)
    y1v,y2v = np.meshgrid(y1,y2)
    Ygrid = np.array([np.reshape(y1v,Y),np.reshape(y2v,Y)]).T    
    py_x = np.zeros((Y,X))
    y_count_near = np.zeros(Y) # counts data points within pad of each bin
    for x in range(X):
        for y in range(Y):
            l = np.linalg.norm(coord[x,:]-Ygrid[y,:])
            py_x[y,x] = (1./math.sqrt(2*math.pi*(s**(2*2))))*math.exp(-(1./(2.*(s**2)))*l)
            if l<pad:
                y_count_near[y] += 1
        
    # drop ybins that are too far away from data
    ymask = y_count_near>0
    py_x = py_x[ymask,:]
    print("Dropped %i ybins. Y reduced from %i to %i." % (Y-np.sum(ymask),Y,np.sum(ymask)))
    Y = np.sum(ymask)
    # normalize p(y|x), since gaussian binned/truncated and bins dropped
    for x in range(X):
        py_x[:,x] = py_x[:,x]/np.sum(py_x[:,x])
    # construct p(x,y)
    px = (1/X)*np.ones(X)    
    pxy = np.multiply(np.tile(px,(Y,1)),py_x).T
    
    # plot p(x,y)
    plt.figure()
    plt.contourf(pxy)
    plt.ion()
    plt.show()
    
    # calc and display I(x,y)
    pxy2, px2, py_x2, hx, hy, hy_x, ixy, X2, Y2, zx, zy = process_pxy(pxy,0)
    print("I(X;Y) = %.3f" % ixy)
    
    return pxy
    
def gen_3_even_sph_wellsep():
            
    # set all parameters    
    mu1 = np.array([0,0])
    sig1 = 1
    mu2 = np.array([8,3])
    sig2 = 1
    mu3 = np.array([0,10])
    sig3 = 1
    samp_per_comp = 30
    labels = np.array([0]*samp_per_comp+[1]*samp_per_comp+[2]*samp_per_comp)
    
    
    # generate coordinates of data points
    coord = np.r_[sig1*np.random.randn(samp_per_comp,2)+mu1,
                  sig2*np.random.randn(samp_per_comp,2)+mu2,
                  sig3*np.random.randn(samp_per_comp,2)+mu3]
    
    # plot coordinates
    plt.scatter(coord[:,0],coord[:,1])
    plt.show()
        
    return {'coord': coord, 'labels': labels}