#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import argparse

import math
import pose.posenetPython.posenet as posenet
import yolo3.filter as fil

class PoseDetection():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=int, default=101)
        parser.add_argument('--scale_factor', type=float, default=0.7125)
        self.args = parser.parse_args()
        self.sess=tf.Session()
        self.model_cfg, self.model_outputs = posenet.load_model(self.args.model, self.sess)
        self.output_stride = self.model_cfg['output_stride']
    
    def detect_pose(self, image):
        input_image, display_image, output_scale = posenet.receive_cap(
                image, scale_factor=self.args.scale_factor, output_stride=self.output_stride)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.sess.run(
                self.model_outputs,
                feed_dict={'image:0': input_image}
            )

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=self.output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

            

            # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

        return overlay_image, keypoint_coords,keypoint_scores
poseDetection=PoseDetection()


def main():

    
    if True:#with tf.Session() as sess:
        ruta="C:/Users/nicoa/OneDrive/Documentos/entrenamiento/P17_kar_izq/"
        rutaEspejo="C:/Users/nicoa/OneDrive/Documentos/entrenamiento/P17_kar_der/"
        f= open(ruta+"salida5.txt","w+")
        fEspejo= open(rutaEspejo+"salida5.txt","w+")
        i=500
        j=500
        people=fil.PoeplFilter()
        cap = cv2.VideoCapture(0)
        keypoint_scores=[]
        
        while True:
            sucess, img=cap.read()
            cuts=people.get_people(img)
            if len(cuts)>0:
                flipHorizontal = cv2.flip(cuts[0], 1)
                i=points(ruta,f,cuts[0],"a",i)
                j=points(rutaEspejo,fEspejo,flipHorizontal,"b",j)
                #else :
                #    print(len(slice),slice)
            else:
                break
                #overlay_image=poseDetection.detect_pose(img)
            

            
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def points(ruta,f,imgen,flag,i):
    overlay_image,keypoint_coords,keypoint_scores=poseDetection.detect_pose(imgen)
                #i=0
            
    slice=keypoint_scores[0]
    slice=slice[5:11]
    incluir=True
    for car in slice:
        
        if car<0.1 :
            incluir=False
    if incluir:
        res=''
        for cord in keypoint_coords[0][5:11]:
            v1=str(int(math.ceil(cord[0])))
            v2=str(int(math.ceil(cord[1])))
            res +=v1+'|'+v2+'|'                    
        print(i)
        i+=1
        f.write(str(i)+'|'+res+"%d\r\n")
        cv2.imwrite(ruta+str(i)+".png", overlay_image)
    cv2.imshow('posenet'+flag, overlay_image)
    return i

if __name__ == "__main__":
    main()




