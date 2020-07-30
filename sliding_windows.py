import numpy as np

def window(image,window_size=(64,64),x_step=0.5,y_step=0.05,x_range=(0,1),y_range=(0,1),scale=1.5):
    windows=[]
    h,w=np.array(image).shape[:2]
    
    for y in range(int(y_range[0]*h), int(y_range[1]*h),int(y_step*h)):
        win_width =int(window_size[1] + scale * (y-(y_range[0]*h)))
        win_height = int(window_size[0] + scale * (y-(y_range[0]*h)))
        
        if y+win_height >int (y_range[1]*h) or win_width > w:
            break
        
        x_steps = int(x_step*win_width)
        for x in range(int(x_range[0]*w), int(x_range[1]*w) - win_width + x_steps , x_steps):
            windows.append((x , y , x + win_width , y + win_height))
             
            
        return np.array(windows)
