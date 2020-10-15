import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import argparse
import services.gesture.GestureValidatorInterface as gest
import math
import pose.posenetPython.posenet as posenet
from PIL import Image
from numpy import asarray




class GestureValidationImp(gest.GestureValidationInterface):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=int, default=101)
        parser.add_argument('--scale_factor', type=float, default=0.7125)
        self.args = parser.parse_args()
        self.sess=tf.Session()
        self.model_cfg, self.model_outputs = posenet.load_model(self.args.model, self.sess)
        self.output_stride = self.model_cfg['output_stride']
    
    def detect_pose(self, image, commands:list):
        imageT=Image.fromarray(image)
        imageT=imageT.resize(size=(400,600))
        image=asarray(imageT)
        
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

        cv2.imshow('pose',overlay_image)
        cv2.waitKey(1)

        return overlay_image,''