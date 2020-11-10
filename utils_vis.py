import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import datetime

def plot_figure(X, title1, title2, figtitle, nb=208,time=60, dt=2, color='rainbow_r'):
    # make these smaller to increase the resolution
    dx, dy = dt, 0.2

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(0, nb*dy, dy), slice(14, 19.01, dx)]

    z = 120*X.transpose()

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    #z = z[:-1, :-1]
    levels = MaxNLocator(nbins=120).tick_values(0, 120)
    levels1 = MaxNLocator(nbins=12).tick_values(0, 120)


    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap(color)#('jet_r')
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    #fig, (ax0, ax1) = plt.subplots(nrows=2,figsize=(12, 12))
    fig, ax0 = plt.subplots(nrows=1,figsize=(15, 7))

    im = ax0.pcolormesh(x, y, z, cmap=cmap)#, norm=norm)
    cbar0 = fig.colorbar(im, ax=ax0)
    ax0.set_title(title1, fontsize=25)
    ax0.set_xlabel('time (PM)', fontsize=20)
    ax0.set_ylabel('distance (km)', fontsize=20)
    cbar0.set_label('km/h', fontsize=20)
    # contours are *point* based plots, so convert our bound into point
    # centers
    #cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                      #y[:-1, :-1] + dy/2., z, levels=levels1,
                      #cmap=cmap)
    #cbar1=fig.colorbar(cf, ax=ax1)
    #ax1.set_title(title2, fontsize=20)
    #ax1.set_xlabel('time (min)', fontsize=15)
    #ax1.set_ylabel('distance (km)', fontsize=15)
    #cbar1.set_label('km/h', fontsize=15)
    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    ax0.tick_params(axis='both', which='major', labelsize=15)
    #ax1.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    
    plt.savefig('Figs/'+figtitle, dpi=600)
    plt.show()
    

def plotter(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(211)
    plt.title('training and validation loss')
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()

    plt.tight_layout()

    plt.show()