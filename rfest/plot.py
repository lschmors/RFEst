import numpy as np
import matplotlib.pyplot as plt

from .utils import get_n_samples, uvec, get_spatial_and_temporal_filters

def plot1d(models, X_test, y_test, model_names=None, figsize=None, vmax=0.5, response_type='spike', dt=None, len_time=None):
    
    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    plot_w_spl = any([hasattr(model, 'w_spl') for model in models])
    plot_w_opt = any([hasattr(model, 'w_opt') for model in models])
    plot_nl = any([hasattr(model, 'fnl_fitted') for model in models])
    plot_h_opt = any([hasattr(model, 'h_opt') for model in models])
    
    nrows = len(models) + 1 # add row for prediction
    ncols = 3
    figsize = figsize if figsize is not None else (3 * ncols, 2 * nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)   
    
    ax_pred = fig.add_subplot(spec[nrows-1, :])
    dt = models[0].dt if dt is None else dt
    if len_time is not None: 
        n = get_n_samples(len_time / 60, dt)
    else:
        n = y_test.shape[0]

    t_pred = np.arange(n)
    
    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.set_xlabel('Time (s)')
    
    for idx, model in enumerate(models):
                
        dims = model.dims
        ax_w_rf = fig.add_subplot(spec[idx, 0])
        if idx == 0:
            ax_w_rf.set_title('RF', fontsize=14)

        w_sta = uvec(model.w_sta.reshape(dims))
        ax_w_rf.plot(w_sta, color='C0', label='STA')
        ax_w_rf.spines['top'].set_visible(False)
        ax_w_rf.spines['right'].set_visible(False)
        ax_w_rf.set_ylabel(model_names[idx], fontsize=14)
        
        if hasattr(model, 'w_spl'):
            w_spl = uvec(model.w_spl.reshape(dims))
            ax_w_rf.plot(w_spl, color='C1', label='SPL')
                        
        if hasattr(model, 'w_opt'):
            w_opt = uvec(model.w_opt.reshape(dims))
            ax_w_rf.plot(w_opt, color='C2', label='OPT')
        
        if plot_h_opt:
            ax_h_opt = fig.add_subplot(spec[idx, 1])
            
            if hasattr(model, 'h_opt'):
                
                h_opt = model.h_opt
                ax_h_opt.plot(h_opt, color='C2')
                ax_h_opt.spines['top'].set_visible(False)
                ax_h_opt.spines['right'].set_visible(False)
            else:
                ax_h_opt.axis('off')
                
            if idx == 0:
                ax_h_opt.set_title('History Filter')
                
        if plot_nl:
            if plot_h_opt:
                ax_nl = fig.add_subplot(spec[idx, 2])
            else:
                ax_nl = fig.add_subplot(spec[idx, 1])

            if hasattr(model, 'fnl_fitted'):
                
                nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)                
                nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
                xrng = model.nl_xrange
                ax_nl.plot(xrng, nl0, color='black', label='init')
                ax_nl.plot(xrng, nl_opt, color='red', label='fitted')
                ax_nl.spines['top'].set_visible(False)
                ax_nl.spines['right'].set_visible(False)
            else:
                ax_nl.axis('off')
                
            if idx == 0:
                ax_nl.set_title('Fitted nonlinearity')
                
        y_pred = model.predict(X_test, y_test)
        pred_score = model.score(X_test, y_test)
    
        ax_pred.plot(t_pred * dt, y_pred[t_pred], color=f'C{idx}', linewidth=2,
            label=f'{model_names[idx]} = {pred_score:.3f}')
        ax_pred.legend(frameon=False)
        ax_w_rf.legend(frameon=False)
        ax_nl.legend(frameon=False)

    fig.tight_layout()   

def plot2d(models, X_test, y_test, model_names=None, figsize=None, vmax=0.5, response_type='spike', dt=None, len_time=None):
    
    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    plot_w_spl = any([hasattr(model, 'w_spl') for model in models])
    plot_w_opt = any([hasattr(model, 'w_opt') for model in models])
    plot_nl = any([hasattr(model, 'fnl_fitted') for model in models])
    plot_h_opt = any([hasattr(model, 'h_opt') for model in models])

    
    nrows = len(models) + 1 # add row for prediction
    ncols = 1 + sum([plot_w_spl, plot_w_opt, plot_nl, plot_h_opt])
    figsize = figsize if figsize is not None else (2 * ncols, nrows + 2)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)   
    
    ax_pred = fig.add_subplot(spec[nrows-1, :])
    dt = models[0].dt if dt is None else dt
    if len_time is not None: 
        n = get_n_samples(len_time / 60, dt)
    else:
        n = y_test.shape[0]

    t_pred = np.arange(n)
    
    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.set_xlabel('Time (s)')
    
    for idx, model in enumerate(models):
                
        dims = model.dims
        ax_w_sta = fig.add_subplot(spec[idx, 0])
        w_sta = uvec(model.w_sta.reshape(dims))
        ax_w_sta.imshow(w_sta, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
        ax_w_sta.set_xticks([])
        ax_w_sta.set_yticks([])
        ax_w_sta.set_ylabel(model_names[idx], fontsize=14)
        
        ax_w_spl = fig.add_subplot(spec[idx, 1])
        w_spl = uvec(model.w_spl.reshape(dims))
        ax_w_spl.imshow(w_spl, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
        ax_w_spl.set_xticks([])
        ax_w_spl.set_yticks([])
        
        if idx == 0:
            ax_w_sta.set_title('STA', fontsize=14)
            ax_w_spl.set_title('SPL', fontsize=14)    
                
        if hasattr(model, 'w_opt'):
            ax_w_opt = fig.add_subplot(spec[idx, 2])
            w_opt = uvec(model.w_opt.reshape(dims))
            ax_w_opt.imshow(w_opt, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
            ax_w_opt.set_xticks([])
            ax_w_opt.set_yticks([])
            if idx == 0:
                ax_w_opt.set_title('OPT', fontsize=14)
        
        if plot_h_opt:
            ax_h_opt = fig.add_subplot(spec[idx, 3])
            
            if hasattr(model, 'h_opt'):
                
                h_opt = model.h_opt
                ax_h_opt.plot(h_opt)
                ax_h_opt.spines['top'].set_visible(False)
                ax_h_opt.spines['right'].set_visible(False)
            else:
                ax_h_opt.axis('off')
                
            if idx == 0:
                ax_h_opt.set_title('History Filter')
                
        if plot_nl:
            if plot_h_opt:
                ax_nl = fig.add_subplot(spec[idx, 4])
            else:
                ax_nl = fig.add_subplot(spec[idx, 3])

            if hasattr(model, 'fnl_fitted'):
                
                nl = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
                xrng = model.nl_xrange
                ax_nl.plot(xrng, nl)
                ax_nl.spines['top'].set_visible(False)
                ax_nl.spines['right'].set_visible(False)
            else:
                ax_nl.axis('off')
                
            if idx == 0:
                ax_nl.set_title('Fitted nonlinearity')
                
        y_pred = model.predict(X_test, y_test)
        pred_score = model.score(X_test, y_test)
    
        ax_pred.plot(t_pred * dt, y_pred[t_pred], color=f'C{idx}', linewidth=2,
            label=f'{model_names[idx]} = {pred_score:.3f}')
        ax_pred.legend(frameon=False)

    fig.tight_layout()
    
def plot3d(model, X_test, y_test, dt=None,
        shift=None, model_name=None, response_type='spike'):
        
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name 

    dims = model.dims
    dt = model.dt if dt is None else dt
    shift = 0 if shift is None else -shift

    w = uvec(model.w_opt.reshape(dims))
    sRF, tRF = get_spatial_and_temporal_filters(w, dims)
    ref = [sRF.max(), sRF.min()][np.argmax([np.abs(sRF.max()), np.abs(sRF.min())])]
    max_coord = np.where(sRF == ref)
    tRF = w[:,max_coord[0], max_coord[1]].flatten()
    t_tRF = np.linspace(-(dims[0]-shift)*dt, shift*dt, dims[0]+1)[1:]
    t_hRF = np.linspace(-(dims[0]+1)*dt, -1*dt, dims[0]+1)[1:]
    
    fig = plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(ncols=8, nrows=3, figure=fig)
    
    ax_sRF_min = fig.add_subplot(spec[0, 0:2])
    ax_sRF_max = fig.add_subplot(spec[0, 2:4])
    ax_tRF = fig.add_subplot(spec[0, 4:6])
    ax_hRF = fig.add_subplot(spec[0, 6:])

    vmax = np.max([np.abs(sRF.max()), np.abs(sRF.min())])
    tRF_max = np.argmax(tRF)
    sRF_max = w[tRF_max]
    tRF_min = np.argmin(tRF)
    sRF_min = w[tRF_min]

    ax_sRF_max.imshow(sRF_max, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
    ax_sRF_min.imshow(sRF_min, cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
    ax_sRF_min.set_title('Spatial (min)')
    ax_sRF_max.set_title('Spatial (max)')

    ax_tRF.plot(t_tRF, tRF, color='black')
    ax_tRF.axvline(t_tRF[tRF_max], color='C3', linestyle='--', alpha=0.6)
    ax_tRF.axvline(t_tRF[tRF_min], color='C0', linestyle='--', alpha=0.6)
    ax_tRF.set_title('Temporal (center)')
    
    if hasattr(model, 'h_opt'):
        ax_hRF.plot(t_hRF, model.h_opt, color='black')
        ax_hRF.set_title('post-spike filter')
    else:
        ax_hRF.axis('off')
        
    ax_pred = fig.add_subplot(spec[1, :])

    y_pred = model.predict(X_test, y_test)
    t_pred = np.arange(300)

    pred_score = model.score(X_test, y_test)

    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')
    
    ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'SPL LG={pred_score:.3f}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left" , frameon=False)
    ax_pred.set_title('Prediction performance')
    
    ax_cost = fig.add_subplot(spec[2, :4])
    ax_metric = fig.add_subplot(spec[2, 4:])
    
    ax_cost.plot(model.cost_train, color='black', label='train')
    ax_cost.plot(model.cost_dev, color='red', label='dev')
    ax_cost.set_title('cost')
    ax_cost.set_ylabel('MSE')
    ax_cost.legend(frameon=False)

    ax_metric.plot(model.metric_train, color='black', label='train')
    ax_metric.plot(model.metric_dev, color='red', label='dev')
    ax_metric.set_title('metric')
    ax_metric.set_ylabel('corrcoef')
    
    for ax in [ax_sRF_min, ax_sRF_max]:
        ax.set_xticks([])
        ax.set_yticks([])
        
    for ax in [ax_tRF, ax_hRF, ax_metric, ax_cost]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(model_name, fontsize=14)

def plot3d_frames(model, shift=None):
    
    dims = model.dims
    
    dt = model.dt
    nt = dims[0] # number of time frames
    ns = model.n_s if hasattr(model, 'n_s') else 1# number of subunits
    shift = 0 if shift is None else -shift
    t_tRF = np.linspace(-(nt-shift)*dt, shift*dt, nt+1)[1:]    
    
    fig, ax = plt.subplots(ns, nt, figsize=(1.5 * nt, 2*ns))
    if ns == 1:
        w = uvec(model.w_opt.reshape(dims))
        vmax = np.max([np.abs(w.max()), np.abs(w.min())])


        for i in range(nt):
            ax[i].imshow(w[i], cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f'{t_tRF[i]:.3f} s', fontsize=18)
    else:
        for k in range(ns):
            w = uvec(model.w_opt[:, k].reshape(dims))
            vmax = np.max([np.abs(w.max()), np.abs(w.min())])
            for i in range(nt):
                ax[k, i].imshow(w[i], cmap=plt.cm.bwr, vmin=-vmax, vmax=vmax)
                ax[k, i].set_xticks([])
                ax[k, i].set_yticks([]) 
                ax[0, i].set_title(f'{t_tRF[i]:.3f} s', fontsize=18)
            ax[k, 0].set_ylabel(f'S{k}', fontsize=18)

    fig.tight_layout()
        
def plot_prediction(models, X_test, y_test, dt=None, len_time=None, 
                    response_type='spike', model_names=None):
    
    """
    Parameters
    ==========

    models : a model object or a list of models
        List of models. 

    X_test, y_test : array_likes
        Test set.

    dt : float
        Stimulus frame rate in second.

    length : None or float
        Length of y_test to display (in second). 
        If it's None, then use the whole y_test.

    response_type : str
        Plot y_test as `spike` or others.
    """

    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]

    dt = models[0].dt if dt is None else dt
    if len_time is not None: 
        n = get_n_samples(len_time / 60, dt)
    else:
        n = y_test.shape[0]

    trange = np.arange(n)

    fig, ax = plt.subplots(figsize=(12,3))

    if response_type == 'spike':
        markerline, stemlines, baseline = ax.stem(trange * dt, y_test[trange], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax.plot(trange * dt, y_test[trange], color='black', label=f'{response_type}')
    
    for i, model in enumerate(models):
        
        y_pred = model.predict(X_test, y_test)
        pred_score = model.score(X_test, y_test)
    
        ax.plot(trange * dt, y_pred[trange], color=f'C{i}', linewidth=2,
            label=f'{model_names[i]} = {pred_score:.3f}')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax.tick_params(axis='y', colors='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.legend(loc="upper right" , frameon=False, bbox_to_anchor=(1., 1.))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'Prediction - Pearson\'s r', fontsize=14)

def plot_learning_curves(models, model_names=None):

    if type(models) is not list:
        models = [models]

    if model_names is not None:
        if len(model_names) != len(models):
            raise ValueError('`model_names` and `models` must be of same length.')
    else:
        model_names = [str(type(model)).split('.')[-1][:-2] for model in models]    

    len_iters = []
    fig, ax = plt.subplots(len(models),2, figsize=(8, len(models)*2))
    ax = ax.reshape(len(models), 2)

    for i, model in enumerate(models):
        
        ax[i, 0].plot(model.cost_train, label='train', color='black', linewidth=3)
        ax[i, 0].plot(model.cost_dev, label='dev', color='red', linewidth=3)
        

        ax[i, 1].plot(model.metric_train, label='train', color='black', linewidth=3)
        ax[i, 1].plot(model.metric_dev, label='dev', color='red', linewidth=3)
        if i < len(models)-1:
            ax[i, 0].set_xticks([])
            ax[i, 1].set_xticks([])

        len_iters.append(len(model.metric_train))
            

        ax[i, 1].set_ylim(0, 1) 

        ax[i, 0].set_ylabel(f'{model_names[i]}', fontsize=14)
        ax[i, 0].set_yticks([])

        ax[i, 0].spines['top'].set_visible(False)
        ax[i, 0].spines['right'].set_visible(False)
        ax[i, 1].spines['top'].set_visible(False)
        ax[i, 1].spines['right'].set_visible(False)

    for i, model in enumerate(models):
        ax[i, 0].set_xlim(-100, max(len_iters))
        ax[i, 1].set_xlim(-100, max(len_iters))
        
    ax[0, 0].set_title('Cost')
    ax[0, 1].set_title('Performance')
        
    ax[-1, 0].set_xlabel('Iteration')
    ax[-1, 1].set_xlabel('Iteration')
    
    ax[0, 0].legend(frameon=False)
    
    fig.tight_layout()

def plot_subunits2d(model, X_test, y_test, dt=None, shift=None, model_name=None, response_type='spike', len_time=30, ncols=5):
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name 
    
    ws = uvec(model.w_opt)
    dims = model.dims
    num_subunits = ws.shape[1]
    
    vmax = np.max([np.abs(ws.max()), np.abs(ws.min())])
    t_hRF = np.linspace(-(dims[0]+1)*dt, -1*dt, dims[0]+1)[1:]
    
    fig = plt.figure(figsize=(8, 4))
    
    nrows = np.ceil(num_subunits/ncols).astype(int)
    num_left = ncols - num_subunits % ncols
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows+1, figure=fig)  
    axs = []
    
    for j in range(nrows):
        for i in range(ncols):
            ax_subunits = fig.add_subplot(spec[j, i])
            axs.append(ax_subunits)
            
    for i in range(num_subunits):
        w = ws[:, i].reshape(dims)
        axs[i].imshow(w, cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    else:
        for j in range(1, num_left+1):
            axs[i+j].axis('off')
            
    if hasattr(model, 'h_opt') and not hasattr(model, 'fnl_fitted'):
        ax_h_opt = fig.add_subplot(spec[nrows, -1])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
    elif not hasattr(model, 'h_opt') and hasattr(model, 'nl_params_opt'): 
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        nl = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
        xrng = model.nl_xrange
        
        ax_nl.plot(xrng, nl)
        ax_nl.set_title('Fitted nonlinearity')
        ax_nl.spines['top'].set_visible(False)
        ax_nl.spines['right'].set_visible(False)    
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif hasattr(model, 'h_opt') and hasattr(model, 'nl_params_opt'):
        ax_h_opt = fig.add_subplot(spec[nrows, -2])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)    
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)                
        nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
        xrng = model.nl_xrange
        
        ax_nl.plot(xrng, nl0)
        ax_nl.plot(xrng, nl_opt)
        ax_nl.set_title('Fitted nonlinearity')
        ax_nl.spines['top'].set_visible(False)
        ax_nl.spines['right'].set_visible(False)    
        
        ax_pred = fig.add_subplot(spec[nrows, :-2])
    else:
        ax_pred = fig.add_subplot(spec[nrows, :])
    
    y_pred = model.predict(X_test, y_test)
    
    if len_time is not None: 
        n = get_n_samples(len_time / 60, dt)
    else:
        n = y_test.shape[0]

    t_pred = np.arange(n)

    pred_score = model.score(X_test, y_test)

    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')    
        
    ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'{pred_score:.3f}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left" , frameon=False)
    ax_pred.set_title('Prediction performance')
    
    ax_pred.set_xlabel('Time (s)', fontsize=12)
    ax_pred.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax_pred.tick_params(axis='y', colors='black')

    fig.tight_layout()

def plot_subunits3d(model, X_test, y_test, dt=None, shift=None, model_name=None, response_type='spike', len_time=1):
    
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    model_name = str(type(model)).split('.')[-1][:-2] if model_name is None else model_name 

    dims = model.dims
    dt = model.dt if dt is None else dt
    shift = 0 if shift is None else -shift
    t_tRF = np.linspace(-(dims[0]-shift)*dt, shift*dt, dims[0]+1)[1:]
    t_hRF = np.linspace(-(dims[0]+1)*dt, -1*dt, dims[0]+1)[1:]

    ws = uvec(model.w_opt)
    
    num_subunits = ws.shape[1]
    
    sRFs = []
    tRFs = []
    for i in range(num_subunits):
        sRF, tRF = get_spatial_and_temporal_filters(ws[:, i], dims)
        sRFs.append(sRF)
        tRFs.append(tRF)
    
    sRFs = np.stack(sRFs)
    
    vmax = np.max([np.abs(sRFs.max()), np.abs(sRFs.min())])
    
    fig = plt.figure(figsize=(8, 4))
    
    ncols = num_subunits if num_subunits > 5 else 5    
    nrows = 2
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows+1, figure=fig)  
    axs = []
    
    for i in range(num_subunits):
        ax_sRF = fig.add_subplot(spec[0, i])       
        ax_sRF.imshow(sRFs[i], cmap=plt.cm.bwr, vmax=vmax, vmin=-vmax)
        ax_sRF.set_xticks([])
        ax_sRF.set_yticks([])
        ax_sRF.set_title(f'S{i}')
    
        ax_tRF = fig.add_subplot(spec[1, i])       
        ax_tRF.plot(t_tRF, tRFs[i], color='black')
        ax_tRF.spines['top'].set_visible(False)
        ax_tRF.spines['right'].set_visible(False)        
        
    if hasattr(model, 'h_opt') and not hasattr(model, 'fnl_fitted'):
        ax_h_opt = fig.add_subplot(spec[nrows, -1])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif not hasattr(model, 'h_opt') and hasattr(model, 'nl_params_opt'): 
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        nl = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
        xrng = model.nl_xrange
        
        ax_nl.plot(xrng, nl)
        ax_nl.set_title('Fitted nonlinearity')
        ax_nl.spines['top'].set_visible(False)
        ax_nl.spines['right'].set_visible(False)    
        
        ax_pred = fig.add_subplot(spec[nrows, :-1])
        
    elif hasattr(model, 'h_opt') and hasattr(model, 'nl_params_opt'):
        ax_h_opt = fig.add_subplot(spec[nrows, -2])
        ax_h_opt.plot(t_hRF, model.h_opt, color='black')
        ax_h_opt.set_title('History Filter')
        ax_h_opt.spines['top'].set_visible(False)
        ax_h_opt.spines['right'].set_visible(False)    
        
        ax_nl = fig.add_subplot(spec[nrows, -1])
        nl0 = model.fnl_fitted(model.nl_params, model.nl_xrange)                
        nl_opt = model.fnl_fitted(model.nl_params_opt, model.nl_xrange)
        xrng = model.nl_xrange
        
        ax_nl.plot(xrng, nl0)
        ax_nl.plot(xrng, nl_opt)
        ax_nl.set_title('Fitted nonlinearity')
        ax_nl.spines['top'].set_visible(False)
        ax_nl.spines['right'].set_visible(False)    
        
        ax_pred = fig.add_subplot(spec[nrows, :-2])
        
    else:
        ax_pred = fig.add_subplot(spec[nrows, :])
    
    y_pred = model.predict(X_test, y_test)
    
    if len_time is not None: 
        n = get_n_samples(len_time / 60, dt)
    else:
        n = y_test.shape[0]

    t_pred = np.arange(n)

    pred_score = model.score(X_test, y_test)

    if response_type == 'spike':
        markerline, stemlines, baseline = ax_pred.stem(t_pred * dt, y_test[t_pred], linefmt='black',
                            markerfmt='none', use_line_collection=True, label=f'{response_type}')
        markerline.set_markerfacecolor('none')
        plt.setp(baseline,'color', 'none')
    else:
        ax_pred.plot(t_pred * dt, y_test[t_pred], color='black', label=f'{response_type}')    
        
    ax_pred.plot(t_pred * dt, y_pred[t_pred], color='C3', linewidth=3, label=f'{pred_score:.3f}')
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)
    ax_pred.legend(loc="upper left" , frameon=False)
    ax_pred.set_title('Prediction performance')    
        
    ax_pred.set_xlabel('Time (s)', fontsize=12)
    ax_pred.set_ylabel(f'{response_type}', fontsize=12, color='black')
    ax_pred.tick_params(axis='y', colors='black')
        
    fig.tight_layout()