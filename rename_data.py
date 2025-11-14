import os
import cv2
import time

option = input( "option -> " )
if option == '1': # convert to png
    for person in os.listdir( "train_datas" ):
        for path in os.listdir( f"train_datas/{person}" ):
            cv2.imwrite( f"./train_datas/{person}/{path.lower()}".replace( "jpg", "png" ).replace( "jpeg", "png" ), cv2.imread( f"./train_datas/{person}/{path}" ) )
            if path.find( "jpg" ) != -1:
                os.remove( f"./train_datas/{person}/{path}" )

else: # rename
    for path in os.listdir( "raw_data" ):
        cv2.imwrite( f"./treated_data/{time.time()}.png", cv2.imread( f"./raw_data/{path}" ) )