# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

# python3.10 run.py --cytomine_host "http://cytomine.imu.edu.my" --cytomine_public_key "71981a88-4e97-4551-a403-59d0dfccf5fd" --cytomine_private_key "36344b44-7539-4772-b1fb-d9ae7ec72182" --cytomine_id_project "77742" --cytomine_id_software "73644795" --cytomine_id_images "76061444" --cytomine_id_roi_term "1021768" --cytomine_id_cell_term "1021760" --cytomine_segment_th "0.5" --log_level "WARNING" --cytomine_area_th "0"

from __future__ import print_function, unicode_literals, absolute_import, division


import sys
import numpy as np
# from pathlib import Path
import os
import cytomine
from shapely.geometry import shape, box, Polygon, Point, MultiPolygon
from shapely import wkt
from shapely.ops import unary_union
from shapely.affinity import affine_transform
from glob import glob
from sldc.locator import mask_to_objects_2d

from tifffile import imread
from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, User, JobData, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection

from PIL import Image
from skimage import io, color, filters, measure
from scipy.ndimage import zoom

# import matplotlib.pyplot as plt
import time
import cv2
import math
import csv

from argparse import ArgumentParser
import json
import logging
import logging.handlers
import shutil

__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "1.0.1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet
from torchvision import transforms


# Define the UNet with DenseNet Encoder
class UNetWithDenseNetEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetWithDenseNetEncoder, self).__init__()
        # Use DenseNet121 as the encoder
        densenet = DenseNet(growth_rate=32, block_config=(2, 2, 2, 2),
                            num_init_features=64, bn_size=4, drop_rate=0)
        # Extract the features (remove the classifier part)
        self.encoder = densenet.features
        self.enc_channels = [64, 128, 64, 128, 64]  # Adjusted channels for DenseNet

        # Decoder layers
        self.up1 = nn.ConvTranspose2d(self.enc_channels[4], self.enc_channels[3], kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(self.enc_channels[3] + self.enc_channels[3], self.enc_channels[3], kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(self.enc_channels[3], self.enc_channels[2], kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.enc_channels[2] + self.enc_channels[2], self.enc_channels[2], kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose2d(self.enc_channels[2], self.enc_channels[1], kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(self.enc_channels[1] + self.enc_channels[1], self.enc_channels[1], kernel_size=3, padding=1)
        self.up4 = nn.ConvTranspose2d(self.enc_channels[1], self.enc_channels[0], kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(self.enc_channels[0] + self.enc_channels[0], self.enc_channels[0], kernel_size=3, padding=1)

        # Final convolution layer with 1 output channel
        self.final = nn.Conv2d(self.enc_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc0 = self.encoder[0](x)
        enc1 = self.encoder[4](enc0)
        enc2 = self.encoder[5](enc1)
        enc3 = self.encoder[6](enc2)
        enc4 = self.encoder[7](enc3)

        # Decoder path
        x = self.up1(enc4)
        x = torch.cat([x, enc3], dim=1)
        x = F.relu(self.conv1(x))

        x = self.up2(x)
        x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc2], dim=1)
        x = F.relu(self.conv2(x))

        x = self.up3(x)
        x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc1], dim=1)
        x = F.relu(self.conv3(x))

        x = self.up4(x)
        x = F.interpolate(x, size=enc0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc0], dim=1)
        x = F.relu(self.conv4(x))

        # Final convolution
        x = self.final(x)

        # Upsample the final output to match the target mask size (e.g., 256x256)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # Apply sigmoid activation for binary output
        x = torch.sigmoid(x)

        return x



def run(cyto_job, parameters):
    logging.info("----- PCa-Semantic-UNet-DenseNet v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project
    th_seg = parameters.cytomine_segment_th

    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()

    # # Paths where ONNX and OpenVINO IR models will be stored.
    # # ir_path = weights_path.with_suffix(".xml")
    # # ir_path = "/models/pc-cb-2class_dn21adam_best_model_100ep.xml"
    # ir_path = "/models/pc-cb-3class-v2_dn21adam_best_model_100ep.xml"

    # # Instantiate OpenVINO Core
    # core = ov.Core()

    # # Read model to OpenVINO Runtime
    # #model_ir = core.read_model(model=ir_path)

    # # Load model on device
    # compiled_model = core.compile_model(model=ir_path, device_name='CPU')
    # output_layer = compiled_model.output(0)

    # modelpath="./models/bloodsegment_unet_100ep.pth"
    modelpath="/models/best_unet_dn21_pytable_blood_segment_v4_bceloss_100.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = UNetWithDenseNetEncoder().to(device)  # Assuming binary segmentation with 1 output channel
    # model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    
    # ------------------------

    print("Model successfully loaded!")
    job.update(status=Job.RUNNING, progress=20, statusComment=f"Model successfully loaded!")

    #Select images to process
    images = ImageInstanceCollection().fetch_with_filter("project", project.id)       
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = parameters.cytomine_id_images
        list_imgs2 = list_imgs.split(',')
        
    print('Print list images:', list_imgs2)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")

    #Set working path
    working_path = os.path.join("tmp", str(job.id))
   
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)

    ###################################
    try:

        id_project=project.id   
        output_path = os.path.join(working_path, "classification_results.csv")
        f= open(output_path,"w+")

        f.write("AnnotationID;ImageID;ProjectID;JobID;TermID;UserID;Area;Perimeter;Hue;Value;WKT \n")
        
        #Go over images
        for id_image in list_imgs2:

            print('Current image:', id_image)
            imageinfo=ImageInstance(id=id_image,project=parameters.cytomine_id_project)
            imageinfo.fetch()
            calibration_factor=imageinfo.resolution
            wsi_width=imageinfo.width
            wsi_height=imageinfo.height
            

            roi_annotations = AnnotationCollection(
                terms=[parameters.cytomine_id_roi_term],
                project=parameters.cytomine_id_project,
                image=id_image, #conn.parameters.cytomine_id_image
                showWKT = True,
                includeAlgo=True, 
            )
            roi_annotations.fetch()
            print(roi_annotations)

            #Go over ROI in this image
            for roi in roi_annotations:
                ########################################
                try:
                    #Get Cytomine ROI coordinates for remapping to whole-slide
                    #Cytomine cartesian coordinate system, (0,0) is bottom left corner
                    # print("----------------------------ROI------------------------------")
                    print(".", sep=' ', end='', flush=True)
                    roi_geometry = wkt.loads(roi.location)
                    # print("ROI Geometry from Shapely: {}".format(roi_geometry))
                    # print("ROI Bounds")
                    # print(roi_geomsetry.bounds)
                    min_x=roi_geometry.bounds[0]
                    min_y=roi_geometry.bounds[1]
                    max_x=roi_geometry.bounds[2]
                    max_y=roi_geometry.bounds[3]

                    roi_width=max_x - min_x
                    roi_height=max_y - min_y

                    print("ROI width = ", roi_width)
                    print("ROI height = ", roi_height)
                    
                    #Dump ROI image into local PNG file
                    roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                    roi_png_filename=os.path.join(roi_path+'/'+str(roi.id)+'.png')
                    # print("roi_png_filename: %s" %roi_png_filename)
                    is_algo = User().fetch(roi.user).algo
                    roi.dump(dest_pattern=roi_png_filename,mask=True,alpha=not is_algo)
                    #roi.dump(dest_pattern=os.path.join(roi_path,"{id}.png"), mask=True, alpha=True)

                    roi_im=Image.open(roi_png_filename).convert('RGB')
                    original_image_size = roi_im.size[::-1]  # PIL gives size as (width, height), we need (height, width)

                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),  # Resize to the same size used during training
                        transforms.ToTensor()         # Convert the image to a PyTorch tensor
                    ])
                    roi_resize = transform(roi_im)  # Apply transformations
                    image_tensor = roi_resize.unsqueeze(0)  # Add batch dimension (1, C, H, W)

                    # Forward pass
                    # th_seg = 0.5
                    image_tensor = image_tensor.to(device)
                    with torch.no_grad():
                        output = model(image_tensor)  # Model outputs segmentation

                    prediction = (output > th_seg).float()  # Shape: [1, 1, H, W]
                    prediction = prediction.squeeze(1).long()  # Shape: [1, original_H, original_W]
                    prediction = prediction.squeeze(0)

                    # Get predicted segmentation output
                    original_width, original_height = roi_im.size
                    zoom_factors = (original_height / prediction.shape[0], original_width / prediction.shape[1])
                    prediction = prediction.cpu().numpy()
                    seg_preds_resized = zoom(prediction, zoom_factors, order=0)  # Nearest neighbor interpolation (order=0)

                    # Zoom factor for WSI
                    bit_depth = 8 #imageinfo.bitDepth if imageinfo.bitDepth is not None else 8
                    zoom_factor = 1
                    transform_matrix = [zoom_factor, 0, 0, -zoom_factor, min_x, max_y]
                    extension = 10
                    cytomine_annotations = AnnotationCollection()
                    id_terms=parameters.cytomine_id_cell_term
                    fg_objects = mask_to_objects_2d(seg_preds_resized)

                    for i, (fg_poly, _) in enumerate(fg_objects):
                        upscaled = affine_transform(fg_poly, transform_matrix)
                        outer_boundary = Polygon(upscaled.exterior)
                        upscaled = outer_boundary

                        Annotation(
                            location=upscaled.wkt,
                            id_image=id_image,
                            id_terms=[id_terms],
                            id_project=project.id).save()                    

                except:
                # finally:
                    # # print("end")
                    print("An exception occurred. Proceed with next annotations")
                #################################################


            end_prediction_time=time.time()

            f.write("\n")
            f.write("Image ID;Class Prediction;Class 0 (Others);Class 1 (Necrotic);Class 2 (Tumor);Total Prediction;Execution Time;Prediction Time\n")
            # f.write("{};{};{};{};{};{};{};{}\n".format(id_image,im_pred,pred_c0,pred_c1,pred_c2,pred_total,end_time-start_time,end_prediction_time-start_prediction_time))
            
        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "classification_results.csv").save()
        job_data.upload(output_path)
    #################################################
    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")
    #################################################

    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

