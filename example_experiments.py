from IB import *
from data_generation import *
import os

def test_IB_single():   
    
    # generate and visualize dataset
    ds = gen_easytest()
    ds.s = 1.
    ds.coord_to_pxy()
    ds.plot_pxy()

    # init model
    m = model(ds=ds,alpha=1,beta=5)
    
    # fit model
    m.fit(keep_steps=True)
    
    return m

def test_IB():
    
    # generate and visualize dataset
    ds = gen_easytest()
    ds.s = 1.
    ds.coord_to_pxy()
    ds.plot_pxy()
    
    # set up fit param
    fit_param = pd.DataFrame(data={'alpha': [1,0]})
    fit_param['repeats'] = 3
    
    # fit models
    metrics_conv, dist_conv, metrics_sw, dist_sw = IB(ds,fit_param)
    
    return metrics_conv, dist_conv, metrics_sw, dist_sw