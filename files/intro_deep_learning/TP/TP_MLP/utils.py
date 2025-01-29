import numpy as np

def plot_contours(ax, model, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    ax: matplotlib axes object
    W: weight matrix
    b: bias vector
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    _,_,_,S = model.forward(np.c_[xx.ravel(), yy.ravel()])
    pred = np.argmax(S, axis=1)
    Z = pred.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    
    return out

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    return xx, yy
