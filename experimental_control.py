from IB import *
from data_generation import *
import os

def test_IB(pxy,compact=1):   
    """Just runs IB and DIB with default parameters."""
    # set up fit param
    fit_param = pd.DataFrame(data={'alpha': [0.,1.]})
    fit_param['repeats'] = 2
    
    # do IB
    return IB(pxy,fit_param,compact)
               
def best_beta(pxy,compact=1):   
    """just uses beta=.6*H(X)/I(X,Y) as attempt to hit knee of IB curve"""
    # set up fit param
    pxy, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy = process_pxy(pxy)
    fit_param = pd.DataFrame(data={'alpha': [0.,1.]})
    fit_param['betas'] = .6*hx/ixy # proposed beta that will pick out best number of clusters
    fit_param['beta_search'] = False
    
    # do IB
    return IB(pxy,fit_param,compact)
     
def test_p0_pos(pxy,compact=1): 
    # set up fit param
    fit_param1 = pd.DataFrame(data={'alpha': [0.]})
    fit_param2 = pd.DataFrame(columns=['alpha','p0'])
    fit_param2['p0'] = [.99,.95,.9,.75,.5,.25,.1,.05,0]
    fit_param2['alpha'] = 1.
    fit_param = fit_param1.append(fit_param2)
    
    # do IB
    return IB(pxy,fit_param,compact)
               
def test_p0_neg(pxy,compact=1): 
    # set up fit param
    fit_param1 = pd.DataFrame(data={'alpha': [0.]})
    fit_param2 = pd.DataFrame(columns=['alpha','p0'])
    fit_param2['p0'] = [-.05,-.1,-.25,-.5,-.75,-.9,-.95,-.99]
    fit_param2['alpha'] = 1.
    fit_param = fit_param1.append(fit_param2)
    
    # do IB
    return IB(pxy,fit_param,compact)

def test_ctol(pxy,compact=1):
    # set up fit param
    ctols = np.logspace(-4,-1,num=4)
    fit_param1 = pd.DataFrame(columns=['alpha','ctol_rel'])
    fit_param1['ctol_rel'] = ctols
    fit_param1['alpha'] = 0.
    fit_param2 = pd.DataFrame(columns=['alpha','ctol_rel'])
    fit_param2['ctol_rel'] = ctols
    fit_param2['alpha'] = 1.
    fit_param = fit_param1.append(fit_param2)
    
    # do IB
    return IB(pxy,fit_param,compact)
               
def test_zeroLtol(pxy,compact=1):
    
    # set up fit param
    zeroLtols = np.logspace(-2,1,num=7)
    fit_param1 = pd.DataFrame(columns=['alpha','zeroLtol'])
    fit_param1['zeroLtol'] = zeroLtols
    fit_param1['alpha'] = 0.
    fit_param2 = pd.DataFrame(columns=['alpha','zeroLtol'])
    fit_param2['zeroLtol'] = zeroLtols
    fit_param2['alpha'] = 1.
    fit_param = fit_param1.append(fit_param2)
    
    # do IB
    return IB(pxy,fit_param,compact)
    

def insert_true_clustering(exp_name,verbose=1):
    cwd = os.getcwd()
    datapath = cwd+'/data/geometric/'+exp_name+'_'
    pxy = np.load(datapath+'pxy.npy')
    groups = np.load(datapath+'groups.npy')
    pxy, px, py_x, hx, hy, hy_x, ixy, X, Y, zx, zy = process_pxy(pxy)
    ptol = 10**-8
    id_conversion = False
    
    # hand choose groupings
    #grouping1 = 30*[1]+60*[0]
    #grouping2 = 30*[0]+30*[1]+30*[0]
    #grouping3 = 60*[0]+30*[1]
    #grouping4 = 30*[0]+30*[1]+30*[2]
    #groups = [grouping1,grouping2,grouping3,grouping4]

    # convert grouping by Xids to grouping by groupids
    if id_conversion:
        G = len(groups)    
        groups_id = np.zeros((G,X))
        for g in range(G): # loop over various groupings
            grouping = groups[g]
            for g2 in range(len(grouping)): # loop over groups within a grouping
                group = grouping[g2]   
                group = [x for x in group if x not in zx]                        
                groups_id[g,group] = g2
        groups_id = groups_id.astype(int)
    else:
        G = max(groups)
        groups_id = groups
    
    # init dataframe
    metrics_conv = pd.DataFrame(columns=['T','ht','ht_x','hy_t','ixt','iyt',
                                              'hx','ixy','alpha',
                                              'ptol','conv_condition'])
    if verbose>0:
        print("****************************** Inserting hand-picked solutions ******************************")
    
    # loop over hand-chosen groupings
    for g in range(G):
        grouping = groups_id[g,:]
        # STEP 1: BUILD Q(T|X)
        T = max(grouping)+1
        qt_x = np.zeros((T,X))
        for x in range(X):
            tstar = grouping[x]
            qt_x[tstar,x] = 1
     
        # STEP 2: UPDATE Q(T)
        qt_x,qt,T = qt_step(qt_x,px,ptol,verbose)
            
        # STEP 3: UPDATE Q(Y|T)
        qy_t = qy_t_step(qt_x,qt,px,py_x)
            
        # calculate and print metrics
        ht, hy_t, iyt, ht_x, ixt, ignorethisL = calc_IB_metrics(qt_x,qt,qy_t,px,hy,1,1)
        if verbose>0:
            print('I(X,T) = %.6f, H(T) = %.6f, H(X) = %.6f, I(Y,T) = %.6f, I(X,Y) = %.6f' % (ixt,ht,hx,iyt,ixy))
                
        # store everything
        metrics_conv = metrics_conv.append(pd.DataFrame(data={
                        'ixt': ixt, 'iyt': iyt, 'ht': ht,
                        'T': T, 'ht_x': ht_x, 'hy_t': hy_t,
                        'hx': hx, 'ixy': ixy,
                        'ptol': ptol, 'conv_condition': 'hand_picked'},
                        index=[g]))
    
    # save
    metrics_conv.to_csv(datapath+'metrics_conv_handpicked.csv')
    
    return 0   
    
def run_experiments(data_set="",compact=2,exp_name="",x=""):
    """x should be a string subset of m/r/ip/in/c/z/gb, compact of 0/1/2."""
    cwd = os.getcwd()
    results_path = cwd+'/data/'+exp_name+'_'
    dataset_path = cwd+'/data/'+data_set+'_'
    compact = int(compact)
    if "m" in x:
        # make new pxy
        pxy = gen_dir_pxy()
        #np.save(dataset_path+'Xdata',Xdata)
        #np.save(dataset_path+'groups',groups)
        np.save(dataset_path+'pxy',pxy)
    else:
        # load existing pxy
        pxy = np.load(dataset_path+'pxy.npy')
    if "r" in x: # regular experiments
        metrics_sw, dist_sw, metrics_conv, dist_conv, metrics_sw_allreps,\
            dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps = \
            test_IB(pxy,compact)
        if compact>1:           
            metrics_conv.to_csv(results_path+'metrics_conv.csv')
            metrics_sw.to_csv(results_path+'metrics_sw.csv')
            dist_conv.to_pickle(results_path+'dist_conv.pkl')
            dist_sw.to_pickle(results_path+'dist_sw.pkl')
        elif compact>0:
            metrics_conv.to_csv(results_path+'metrics_conv.csv')
            metrics_sw.to_csv(results_path+'metrics_sw.csv')
            dist_conv.to_pickle(results_path+'dist_conv.pkl')
        else:
            metrics_conv.to_csv(results_path+'metrics_conv.csv')
            metrics_sw.to_csv(results_path+'metrics_sw.csv')
    if "ip" in x: # initialization experiments - positive p0
        metrics_sw, dist_sw, metrics_conv, dist_conv, metrics_sw_allreps,\
            dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps = \
            test_p0_pos(pxy,compact)
        if compact>1:
            metrics_conv.to_csv(results_path+'metrics_conv_p0_pos.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_p0_pos.csv')
            dist_conv.to_pickle(results_path+'dist_conv_p0_pos.pkl')
            dist_sw.to_pickle(results_path+'dist_sw_p0_pos.pkl')
        elif compact>0:
            metrics_conv.to_csv(results_path+'metrics_conv_p0_pos.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_p0_pos.csv')
            dist_conv.to_pickle(results_path+'dist_conv_p0_pos.pkl')
        else:
            metrics_conv.to_csv(results_path+'metrics_conv_p0_pos.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_p0_pos.csv')
    if "in" in x: # initialization experiments - negative p0
        metrics_sw, dist_sw, metrics_conv, dist_conv, metrics_sw_allreps,\
            dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps = \
            test_p0_neg(pxy,compact)
        if compact>1:
            metrics_conv.to_csv(results_path+'metrics_conv_p0_neg.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_p0_neg.csv')
            dist_conv.to_pickle(results_path+'dist_conv_p0_neg.pkl')
            dist_sw.to_pickle(results_path+'dist_sw_p0_neg.pkl')
        elif compact>0:
            metrics_conv.to_csv(results_path+'metrics_conv_p0_neg.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_p0_neg.csv')
            dist_conv.to_pickle(results_path+'dist_conv_p0_neg.pkl')
        else:
            metrics_conv.to_csv(results_path+'metrics_conv_p0_neg.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_p0_neg.csv')
    if "c" in x: # convergence tolerance experiments
        metrics_sw, dist_sw, metrics_conv, dist_conv, metrics_sw_allreps,\
            dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps = \
            test_ctol(pxy,compact)
        if compact>1:
            metrics_conv.to_csv(results_path+'metrics_conv_ctol.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_ctol.csv')
            dist_conv.to_pickle(results_path+'dist_conv_ctol.pkl')
            dist_sw.to_pickle(results_path+'dist_sw_ctol.pkl')
        elif compact>0:
            metrics_conv.to_csv(results_path+'metrics_conv_ctol.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_ctol.csv')
            dist_conv.to_pickle(results_path+'dist_conv_ctol.pkl')
        else:
            metrics_conv.to_csv(results_path+'metrics_conv_ctol.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_ctol.csv')
    if "z" in x: # zeroL tolerance experiments
        metrics_sw, dist_sw, metrics_conv, dist_conv, metrics_sw_allreps,\
            dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps = \
            test_zeroLtol(pxy,compact)
        if compact>1:
            metrics_conv.to_csv(results_path+'metrics_conv_zeroLtol.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_zeroLtol.csv')
            dist_conv.to_pickle(results_path+'dist_conv_zeroLtol.pkl')
            dist_sw.to_pickle(results_path+'dist_sw_zeroLtol.pkl')
        elif compact>0:
            metrics_conv.to_csv(results_path+'metrics_conv_zeroLtol.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_zeroLtol.csv')
            dist_conv.to_pickle(results_path+'dist_conv_zeroLtol.pkl')
        else:
            metrics_conv.to_csv(results_path+'metrics_conv_zeroLtol.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_zeroLtol.csv')
    if "b" in x: # trying proposed optimal beta
        metrics_sw, dist_sw, metrics_conv, dist_conv, metrics_sw_allreps,\
            dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps = \
            best_beta(pxy,compact)
        if compact>1:
            metrics_conv.to_csv(results_path+'metrics_conv_bestbeta.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_bestbeta.csv')
            dist_conv.to_pickle(results_path+'dist_conv_bestbeta.pkl')
            dist_sw.to_pickle(results_path+'dist_sw_bestbeta.pkl')
        elif compact>0:
            metrics_conv.to_csv(results_path+'metrics_conv_bestbeta.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_bestbeta.csv')
            dist_conv.to_pickle(results_path+'dist_conv_bestbeta.pkl')
        else:
            metrics_conv.to_csv(results_path+'metrics_conv_bestbeta.csv')
            metrics_sw.to_csv(results_path+'metrics_sw_bestbeta.csv')
    return 0