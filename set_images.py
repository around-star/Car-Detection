import cv2
import os

def set_images():
    #POSTIVE IMAGES
    pic_num=1
    pos_folder="vehicles"
    for i in os.listdir(pos_folder):
        num_pos=1
        for j in os.listdir(os.path.join(pos_folder,i)):
            img=cv2.imread(os.path.join(os.path.join(pos_folder,i),j),1)
            cv2.imwrite("Pos_img/"+str(pic_num)+'.png',img)
            pic_num+=1
            if (num_pos>=500):
                break
            num_pos+=1
    
    #NEGATIVE IMAGES        
    pic_num=1
    neg_folder="non-vehicles"
    for i in os.listdir(neg_folder):
        num_neg=1
        for j in os.listdir(os.path.join(neg_folder,i)):
            img=cv2.imread(os.path.join(os.path.join(neg_folder,i),j),1)
            cv2.imwrite("Neg_img/"+str(pic_num)+'.png',img)
            pic_num+=1
            if (num_neg>=2000):
                break
            num_neg+=1

set_images()