import cv2

class Extract():
    
    def __init__(self,win_size,pix_per_cell=(8,8),cells_per_block=(2,2),hog_bins=9):
        self.win_size=win_size
        self.pix_per_cell=pix_per_cell
        self.cells_per_block=cells_per_block
        self.hog_bins=hog_bins
        
        cell_size=pix_per_cell
        block_size=(cells_per_block[0]*cell_size[0], cells_per_block[1]*cell_size[1])
        
        
        self.HOFDescriptor=cv2.HOGDescriptor(win_size, block_size, (8,8), cell_size, hog_bins)
        
    def compute(self,image):
        if image.shape[:2] != self.win_size:
            image = cv2.resize(image, self.win_size, interpolation=cv2.INTER_AREA)
        
        fea_vec = self.HOFDescriptor.compute(image)[:,0]
        return fea_vec
    