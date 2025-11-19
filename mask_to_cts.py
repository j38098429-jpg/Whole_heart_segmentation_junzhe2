
import numpy as np
import cv2
def mask_to_pts(predicted_mask, class_list):
    
    cts_dict = {"0": None, "1": None, "2": None}
    
    for idx, class_label in enumerate(class_list[1:4]):
        contours_pred, image = cv2.findContours(
            (predicted_mask == idx + 1).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        # choose largest component
        shape = np.shape(contours_pred)

        if shape[0] > 1:
            pt_max = 0
            for i in range(shape[0]):
                if pt_max < np.shape(contours_pred[i])[0]:
                    pt_max = np.shape(contours_pred[i])[0]
                    out = np.reshape(contours_pred[i], (pt_max, 2))
        else:
            out = np.reshape(contours_pred, (shape[1], 2))
            
        cts_dict[class_label] = out
        
    return cts_dict