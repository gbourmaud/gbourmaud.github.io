import torch as t

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
    _,_,_,S = model.forward(t.hstack((t.atleast_2d(xx.ravel()).T, t.atleast_2d(yy.ravel()).T)))
    pred = t.argmax(S, axis=1)
    Z = pred.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    
    return out

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = t.meshgrid(t.arange(x_min, x_max, h),t.arange(y_min, y_max, h))
    return xx, yy
