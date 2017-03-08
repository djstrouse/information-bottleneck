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

def test_geo(pxy,compact=1):
    
    # set up fit param
    X = pxy['pxy'].shape[0]
    fit_param = pd.DataFrame(data={'alpha': [0.,1.], 'Tmax': [X/2,None]})
    fit_param['geoapprox'] = True
    
    return IB(pxy,fit_param,compact)
    
def impute_coordinates(dist_conv,pxy):
    
    print('Imputing coordinates...')
    
    # load data
    df = dist_conv[dist_conv['alpha']==1] # only process (a)DIB fits
    dfc = pd.DataFrame(columns=list((set(df.columns.values)-{'qt_x','qt','qy_t','Dxt'}))+['x1','x2','cluster'])
    coord = pxy['coord']

    # iterate over fits
    for irow in range(len(df.index)):
        
        print('Working on fit %i of %i' % (irow,len(df.index)))
        
        # extract results for this fit
        this_fit = df.iloc[irow]
        qt_x = this_fit['qt_x']
        del this_fit['qt_x']
        del this_fit['qt']
        del this_fit['qy_t']
        del this_fit['Dxt']
        T,X = qt_x.shape
        
        # iterate over datapoints
        for x in range(X):
            print('Working on data point %i of %i' % (x,X))
            this_fit['x1'] = coord[x,0]
            this_fit['x2'] = coord[x,1]
            this_fit['cluster'] = np.nonzero(qt_x[:,x])[0][0]
            dfc = dfc.append(this_fit,ignore_index = True)
            
    return dfc
    
def test_zipf(pxy,compact=1):
    # dumb experiment to remake fig2
    lambdas = list(np.linspace(0,1,52))[1:-1]
    betas = [l/(1-l) for l in lambdas]
    fit_param = pd.DataFrame(data={'alpha': 0.,
                                   'betas': [betas],
                                   'beta_search': False,
                                   'zeroLtol': 0})
    
    # experiment focused on midpoint lambda=1/2 aka beta=1
    #fit_param2 = pd.DataFrame(data={'waviness': [.01, .1, .25, .5, .7, .9, None]})
    #fit_param2['alpha'] = 0.
    #fit_param2['betas'] = 1.
    #fit_param2['p0'] = 0.
    #fit_param2['beta_search'] = False
    #fit_param2['repeats'] = 30

    # combine
    #fit_param = fit_param1.append(fit_param2, ignore_index = True)

    #return fit_param
    return IB(pxy,fit_param,compact)  
    
def run_experiments(compact=2,folder="",exp_type="",dataset_name="",exp_name="",s=None):
    """data_set indicates p(x,y) loaded, compact what gets saved,
    exp_name the name of the saved results, exp_type the fit_param run,
    and folder the directory within /data/ where the data/results live"""
    cwd = os.getcwd()
    results_path = cwd+'/data/'+folder+'/'+exp_name+'_'
    dataset_path = cwd+'/data/'+folder+'/'+dataset_name+'_'
    compact = int(compact)
    if exp_type == "m":
        # make new pxy
        dataset = gen_3_even_sph_wellsep() # contains coord and label
        np.save(dataset_path+'coord',dataset)
        return
    if exp_type == "zipf": # zipf experiments
        print('Running zipf experiments...')
        metrics_sw, dist_sw, metrics_conv, dist_conv, metrics_sw_allreps,\
            dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps = \
            test_zipf(pxy,compact)
    if exp_type == "geo": # geometric experiments
        print('Running geometric experiments...')
        dataset = np.load(dataset_path+'coord.npy').item()
        dataset['s'] = s
        dataset['pxy'] = coord_to_pxy(dataset['coord'],s)
        np.save(dataset_path+'dataset_s'+str(s).replace('.','p'),dataset)
        metrics_sw, dist_sw, metrics_conv, dist_conv, metrics_sw_allreps,\
            dist_sw_allreps, metrics_conv_allreps, dist_conv_allreps = \
            test_geo(dataset,compact)
        clusters_conv = impute_coordinates(dist_conv,pxy)
        clusters_conv.to_pickle(results_path+'clusters_conv.pkl')
    if compact>1:
        metrics_conv_allreps.to_csv(results_path+'metrics_conv.csv')
        metrics_sw_allreps.to_csv(results_path+'metrics_sw.csv')
        dist_conv_allreps.to_pickle(results_path+'dist_conv.pkl')
        dist_sw_allreps.to_pickle(results_path+'dist_sw.pkl')
    elif compact>0:
        metrics_conv_allreps.to_csv(results_path+'metrics_conv.csv')
        metrics_sw_allreps.to_csv(results_path+'metrics_sw.csv')
        dist_conv_allreps.to_pickle(results_path+'dist_conv.pkl')
    else:
        metrics_conv_allreps.to_csv(results_path+'metrics_conv.csv')
        metrics_sw_allreps.to_csv(results_path+'metrics_sw.csv')
    return