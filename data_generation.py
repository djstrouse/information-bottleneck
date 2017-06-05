from IB import *
import matplotlib.pyplot as plt
import os
import numpy as np

def gen_easytest(plot=True):
    
    # set name
    name = "easytest"
            
    n = 10
    # set generative parameters  
    mu1 = np.array([0,0])
    sig1 = np.eye(2)
    n1 = n
    mu2 = np.array([math.sqrt(75),5])
    sig2 = np.eye(2)
    n2 = n
    mu3 = np.array([0,10])
    sig3 = np.eye(2)
    n3 = n
    param = {'mu1': mu1, 'sig1': sig1, 'n1': n1,
             'mu2': mu2, 'sig2': sig2, 'n2': n2,
             'mu3': mu3, 'sig3': sig3, 'n3': n3}
    
    # make labels
    labels = np.array([0]*n1+[1]*n2+[2]*n3)
    
    # make coordinates
    coord = np.concatenate((np.random.multivariate_normal(mu1,sig1,n1),
                            np.random.multivariate_normal(mu2,sig2,n2),
                            np.random.multivariate_normal(mu3,sig3,n3)))
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds

def gen_blob(plot=True):

    # set name
    name = "blob"
            
    # set generative parameters  
    mu1 = np.array([0,0])
    sig1 = np.eye(2)
    n1 = 90
    param = {'mu1': mu1, 'sig1': sig1, 'n1': n1}
    
    # make labels
    labels = np.array([0]*n1)
    
    # make coordinates
    coord = np.random.multivariate_normal(mu1,sig1,n1)
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds
     
def gen_3sph_evensamp_evenspacing(plot=True):

    # set name
    name = "3sph_evensamp_evenspacing"
            
    # set generative parameters  
    mu1 = np.array([0,0])
    sig1 = np.eye(2)
    n1 = 30
    mu2 = np.array([math.sqrt(75),5])
    sig2 = np.eye(2)
    n2 = 30
    mu3 = np.array([0,10])
    sig3 = np.eye(2)
    n3 = 30
    param = {'mu1': mu1, 'sig1': sig1, 'n1': n1,
             'mu2': mu2, 'sig2': sig2, 'n2': n2,
             'mu3': mu3, 'sig3': sig3, 'n3': n3}
    
    # make labels
    labels = np.array([0]*n1+[1]*n2+[2]*n3)
    
    # make coordinates
    coord = np.concatenate((np.random.multivariate_normal(mu1,sig1,n1),
                            np.random.multivariate_normal(mu2,sig2,n2),
                            np.random.multivariate_normal(mu3,sig3,n3)))
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds
    
def gen_3sph_unevensamp_evenspacing(plot=True):

    # set name
    name = "3sph_unevensamp_evenspacing"
            
    # set generative parameters  
    mu1 = np.array([0,0])
    sig1 = np.eye(2)
    n1 = 10
    mu2 = np.array([math.sqrt(75),5])
    sig2 = np.eye(2)
    n2 = 30
    mu3 = np.array([0,10])
    sig3 = np.eye(2)
    n3 = 60
    param = {'mu1': mu1, 'sig1': sig1, 'n1': n1,
             'mu2': mu2, 'sig2': sig2, 'n2': n2,
             'mu3': mu3, 'sig3': sig3, 'n3': n3}
    
    # make labels
    labels = np.array([0]*n1+[1]*n2+[2]*n3)
    
    # make coordinates
    coord = np.concatenate((np.random.multivariate_normal(mu1,sig1,n1),
                            np.random.multivariate_normal(mu2,sig2,n2),
                            np.random.multivariate_normal(mu3,sig3,n3)))
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds

def gen_3sph_evensamp_unevenspacing(plot=True):

    # set name
    name = "3sph_evensamp_unevenspacing"
            
    # set generative parameters  
    mu1 = np.array([0,2.5])
    sig1 = np.eye(2)
    n1 = 30
    mu2 = np.array([0,-2.5])
    sig2 = np.eye(2)
    n2 = 30
    mu3 = np.array([15,0])    
    sig3 = np.eye(2)
    n3 = 30
    param = {'mu1': mu1, 'sig1': sig1, 'n1': n1,
             'mu2': mu2, 'sig2': sig2, 'n2': n2,
             'mu3': mu3, 'sig3': sig3, 'n3': n3}
    
    # make labels
    labels = np.array([0]*n1+[1]*n2+[2]*n3)
    
    # make coordinates
    coord = np.concatenate((np.random.multivariate_normal(mu1,sig1,n1),
                            np.random.multivariate_normal(mu2,sig2,n2),
                            np.random.multivariate_normal(mu3,sig3,n3)))
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds

def make_circle(radius,num_points):
    
    count = 0
    points = np.zeros((num_points,2))
    while count<num_points:
        x1 = 2*radius*np.random.rand()-radius
        x2 = 2*radius*np.random.rand()-radius
        x = np.array([x1,x2])
        if np.linalg.norm(x)<radius:
            points[count,:] = x
            count += 1
    return points

def gen_mouse(plot=True):

    # set name
    name = "mouse"
            
    # set generative parameters  
    mu1 = np.array([0,0])
    rad1 = 4
    n1 = 180
    mu2 = np.array([-3.5,5])
    rad2 = 1.4
    n2 = 25
    mu3 = np.array([3.5,5]) 
    rad3 = 1.4
    n3 = 25
    param = {'mu1': mu1, 'rad1': rad1, 'n1': n1,
             'mu2': mu2, 'rad2': rad2, 'n2': n2,
             'mu3': mu3, 'rad3': rad3, 'n3': n3}
    
    # make labels
    labels = np.array([0]*n1+[1]*n2+[2]*n3)
    
    # make coordinates
    coord = np.concatenate((make_circle(rad1,n1)+mu1,
                            make_circle(rad2,n2)+mu2,
                            make_circle(rad3,n3)+mu3))
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds

def gen_circleandcigar(plot=True):

    # set name
    name = "circleandcigar"
            
    # set generative parameters  
    mu1 = np.array([5,0])
    sig1 = np.eye(2)
    n1 = 50
    mu2 = np.array([-5,0])
    sig2 = np.array([[1,0],[0,25]])
    n2 = 50
    param = {'mu1': mu1, 'sig1': sig1, 'n1': n1,
             'mu2': mu2, 'sig2': sig2, 'n2': n2}
    
    # make labels
    labels = np.array([0]*n1+[1]*n2)
    
    # make coordinates
    coord = np.concatenate((np.random.multivariate_normal(mu1,sig1,n1),
                            np.random.multivariate_normal(mu2,sig2,n2)))
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds

def gen_2cigars(plot=True):
    
    # set name
    name = "2cigars"
     
    # set generative parameters   
    mu1 = np.array([0,-4])
    sig1 = np.array([[25,0],[0,1]])
    n1 = 50
    mu2 = np.array([0,4])
    sig2 = np.array([[25,0],[0,1]])
    n2 = 50
    param = {'mu1': mu1, 'sig1': sig1, 'n1': n1,
             'mu2': mu2, 'sig2': sig2, 'n2': n2}
    
    # make labels
    labels = np.array([0]*n1+[1]*n2)
    
    # make coordinates
    coord = np.concatenate((np.random.multivariate_normal(mu1,sig1,n1),
                            np.random.multivariate_normal(mu2,sig2,n2)))
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds
    
def gen_2over3(plot=True):

    # set name
    name = "2over3"
            
    # set generative parameters
    sig = .75
    mu1 = np.array([0,0])
    sig1 = (sig**2)*np.eye(2)
    n1 = 20
    mu2 = np.array([-4,0])
    sig2 = (sig**2)*np.eye(2)
    n2 = 20
    mu3 = np.array([4,0])    
    sig3 = (sig**2)*np.eye(2)
    n3 = 20
    mu4 = np.array([-2,12])
    sig4 = (sig**2)*np.eye(2)
    n4 = 20
    mu5 = np.array([2,12])
    sig5 = (sig**2)*np.eye(2)
    n5 = 20
    param = {'mu1': mu1, 'sig1': sig1, 'n1': n1,
             'mu2': mu2, 'sig2': sig2, 'n2': n2,
             'mu3': mu3, 'sig3': sig3, 'n3': n3,
             'mu4': mu4, 'sig4': sig4, 'n4': n4,
             'mu5': mu5, 'sig5': sig5, 'n5': n5}
    
    # make labels
    labels = np.array([0]*n1+[1]*n2+[2]*n3+[3]*n4+[4]*n5)
    
    # make coordinates
    coord = np.concatenate((np.random.multivariate_normal(mu1,sig1,n1),
                            np.random.multivariate_normal(mu2,sig2,n2),
                            np.random.multivariate_normal(mu3,sig3,n3),
                            np.random.multivariate_normal(mu4,sig4,n4),
                            np.random.multivariate_normal(mu5,sig5,n5)))
    
    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds
    
def gen_halfconcentric(plot=True):
    
    # set name
    name = "halfconcentric"
            
    # set generative parameters
    nt = 80 # number of thetas
    nd = 1 # number of samples per theta
    no = nd*nt # number of samples for outer circle
    ni = 20 # number of samples for inner circle
    r = 5 # radius of outer loop
    so = .25 # gaussian noise variance of outer circle
    si = .25 # gaussian noise variance of inner circle
    thetas = -np.linspace(0,math.pi,nt)
    x = [r*math.cos(theta) for theta in thetas]
    y = [r*math.sin(theta) for theta in thetas]
    param = {'nt': nt, 'nd': nd, 'no': no, 'ni': ni, 'r': r, 'so': so, 'si': si}
         
    # make labels
    labels = np.array([0]*ni+[1]*no)
          
    # make coordinates
    coord = np.random.multivariate_normal(np.array([0,0]),si*np.eye(2),ni)
    for i in range(len(x)):
        coord = np.concatenate((coord,np.random.multivariate_normal(np.array([x[i],y[i]]),so*np.eye(2),nd)))

    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds
    
def gen_concentric(plot=True):
    
    # set name
    name = "concentric"
            
    # set generative parameters
    nt = 80 # number of thetas
    nd = 1 # number of samples per theta
    no = nd*nt # number of samples for outer circle
    ni = 20 # number of samples for inner circle
    r = 8 # radius of outer loop
    so = .25 # gaussian noise variance of outer circle
    si = .25 # gaussian noise variance of inner circle
    thetas = -np.linspace(0,2*math.pi,nt)
    x = [r*math.cos(theta) for theta in thetas]
    y = [r*math.sin(theta) for theta in thetas]
    param = {'nt': nt, 'nd': nd, 'no': no, 'ni': ni, 'r': r, 'so': so, 'si': si}
         
    # make labels
    labels = np.array([0]*ni+[1]*no)
          
    # make coordinates
    coord = np.random.multivariate_normal(np.array([0,0]),si*np.eye(2),ni)
    for i in range(len(x)):
        coord = np.concatenate((coord,np.random.multivariate_normal(np.array([x[i],y[i]]),so*np.eye(2),nd)))

    # make dataset
    ds = dataset(coord = coord, labels = labels, gen_param = param, name = name)
    
    # plot coordinates
    if plot: ds.plot_coord()
    
    # normalize
    ds.normalize_coord()
    if plot: ds.plot_coord()
        
    return ds

# CODE BELOW NOT YET ADAPTED TO USE NEW IB DATASET CLASS
# GENERATE ONLY P(X,Y)

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