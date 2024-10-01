import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as IMG

def get_line_points(w, h, l):
    a=l[0]
    b=l[1]
    c=l[2]
    
    if(a != 0):
        top_x = -c/a #intersection between l and y=0
        bottom_x = (-b*(h-1)-c)/a #intersection between l and y=h-1
    else:
        top_x = -1.
        bottom_x = -1.
        
    if(b != 0):
        left_y = -c/b #intersection between l and x=0
        right_y = (-a*(w-1)-c)/b #intersection between l and x=w-1
    else:
        left_y = -1.
        right_y = -1.
        
    pts = ((top_x,0), (bottom_x,h-1), (0,left_y), (w-1,right_y))
    
    x_draw = []
    y_draw = []
    for pt in pts:
        if(pt[0]>=0 and pt[0]<=w-1 and pt[1]>=0 and pt[1]<=h-1):
            x_draw.append(pt[0]) 
            y_draw.append(pt[1])
    
    return x_draw[0:2], y_draw[0:2]



    
def gui_F(im1, im2, F12):


    h1 = im1.size[1]
    w1 = im1.size[0]
    
    h2 = im2.size[1]
    w2 = im2.size[0]


    fig, axs = plt.subplots(ncols=2)
    
    axs[0].imshow(im1)
    axs[1].imshow(im2)
    
    
    line_ax0, = axs[0].plot([],[],'b-')#init obj ligne vide dans image gauche
    line_ax1, = axs[1].plot([],[],'b-')#init obj ligne vide dans image droite
    marker_ax0, = axs[0].plot([],[],'r+',markersize=10)#init obj marker vide dans image gauche
    marker_ax1, = axs[1].plot([],[],'r+',markersize=10)#init obj marker vide dans image droite 

    def on_move(event):    
        nonlocal line_ax0, line_ax1, marker_ax0, marker_ax1
        if event.inaxes:
            # print(f'data coords {event.xdata} {event.ydata},',
            #       f'pixel coords {event.x} {event.y}')
            
            # print(event.inaxes, axs[0], axs[1])
            if(event.inaxes==axs[0]):#curseur dans image de gauche
                line_ax0.set_xdata([])#efface la ligne dans l'image gauche car je vais en tracer une dans la droite
                line_ax0.set_ydata([])
                marker_ax1.set_xdata([])#efface le marquer dans l'image gauche car je vais en tracer un dans la droite
                marker_ax1.set_ydata([])
                #print('gauche')
                ax_clic = axs[0]
                ax_line = axs[1]
                F = F12.T
                h = h2
                w = w2
                line_ax = line_ax1
                marker_ax = marker_ax0
            else:#curseur dans image de droite
                line_ax1.set_xdata([])#efface la ligne dans l'image droite car je vais en tracer une dans la gauche
                line_ax1.set_ydata([])
                marker_ax0.set_xdata([])#efface le marquer dans l'image gauche car je vais en tracer un dans la droite
                marker_ax0.set_ydata([])
                #print('droite')
                ax_clic = axs[1]
                ax_line = axs[0]
                F = F12
                h = h1
                w = w1
                line_ax = line_ax0
                marker_ax = marker_ax1
                
            marker_ax.set_xdata(event.xdata) #trace marker
            marker_ax.set_ydata(event.ydata)
            
            p_hom = np.array([event.xdata, event.ydata, 1])
            l = F.dot(p_hom.T)
            
            x_draw, y_draw = get_line_points(w, h, l)
            # print(f'x_draw {x_draw},',
            #       f'y_draw {y_draw}')

            line_ax.set_xdata(x_draw)#trace ligne
            line_ax.set_ydata(y_draw)
            plt.draw()
        return
    
    plt.connect('motion_notify_event', on_move)
    
    plt.show()