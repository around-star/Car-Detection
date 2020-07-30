import sliding_windows
import feature_extract
import pickle
import cv2
import numpy as np

file1 = open('model.pkl', 'rb')
model = pickle.load(file1)
file1.close()

file2 = open('scaler.pkl', 'rb')
scaler = pickle.load(file2)
file2.close()

def output(image):
    windows = sliding_windows.window(image)
    
    for window in windows:
        extract = feature_extract.Extract((window[3]-window[1], window[2]-window[0]))
        feature = extract.compute(image[window[1]:window[3], window[0]:window[2]])
        feature = scaler.transform(np.expand_dims(feature, axis=0))
        result = model.predict(feature)
        if result == 1:
            image = cv2.rectangle(image, (window[1], window[0]), (window[3], window[2]), (220, 40, 150), 3)    
            cv2.imwrite('result.jpg', image)
            
        print(result)
            
        
    
    
test_img = cv2.imread("test.png")
output(test_img)