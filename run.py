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
from io import BytesIO
from openvino.runtime import Dimension, PartialShape
import openvino as ov

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


def process_prediction(output, i, j, roi_im_size, combined_mask, patch_size, th_seg):
    prediction = (output > th_seg).float().cpu().numpy()
    prediction = prediction.squeeze()
    
    # Ensure zoom_factors matches the dimensions of prediction
    if prediction.ndim == 2:  # 2D array (height x width)
        zoom_factors = (roi_im_size[1] / prediction.shape[0], roi_im_size[0] / prediction.shape[1])
    elif prediction.ndim == 3:  # 3D array (channels x height x width)
        zoom_factors = (1, roi_im_size[1] / prediction.shape[1], roi_im_size[0] / prediction.shape[2])
    
    # Resize prediction to match the original ROI dimensions
    seg_preds_resized = zoom(prediction, zoom_factors, order=1)
    
    # Update the combined mask
    combined_mask[i:i + patch_size, j:j + patch_size] = np.logical_or(
        combined_mask[i:i + patch_size, j:j + patch_size],
        seg_preds_resized
    ).astype(np.uint8)


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

    ir_path = "/models/best_unet_dn21_pytable_blood_segment_v4_bceloss_100.xml"
    core = ov.Core()
    model_ir = core.read_model(model=ir_path)
    # compiled_model = core.compile_model(model=ir_path, device_name='GPU')    
    # Set the model input shape to dynamic
    model_ir.reshape({0: PartialShape([Dimension.dynamic(), 3, 256, 256])})
    compiled_model = core.compile_model(model=model_ir, device_name='CPU')
    output_layer = compiled_model.output(0)
    
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

    ###################################
    try:

        id_project=project.id   
         
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
                # try:
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

                roi_width=int(max_x - min_x)
                roi_height=int(max_y - min_y)

                combined_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)


                print("ROI width = ", roi_width)
                print("ROI height = ", roi_height)
                print(min_x)
                print(min_y)

                patch_size = parameters.patch_size
                overlap = 0.5
                step = int(patch_size * (1 - overlap))  # 50% overlap
                num_patches_x = (roi_width + step - 1) // step
                num_patches_y = (roi_height + step - 1) // step
                print(f'Patch X: {num_patches_x}, Patch Y: {num_patches_y}')
                print(f'Step X: {step}, Step Y: {step}')

                batch_size=64
                batch = []
                coordinates = []
                id_terms=parameters.cytomine_id_cell_term

                for i in range(0, roi_height, step):
                    for j in range(0, roi_width, step):
                        if i + patch_size > roi_height:
                                i = roi_height - patch_size
                        if j + patch_size > roi_width:
                            j = roi_width - patch_size
                        # Prepare each patch as before
                        patch_x = int(min_x) + j
                        patch_y = int(wsi_height - max_y) + i
                        response = cyto_job.get_instance()._get(
                            "{}/{}/window-{}-{}-{}-{}.{}".format("imageinstance", id_image, patch_x, patch_y, patch_size, patch_size, "png"), {}
                        )

                        if response.status_code in [200, 304] and response.headers['Content-Type'] == 'image/png':
                            roi_im = Image.open(BytesIO(response.content))
                            transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor()
                            ])
                            roi_resize = transform(roi_im)
                            batch.append(roi_resize)
                            coordinates.append((i, j))

                        # Process the batch when it reaches the desired size
                        if len(batch) >= batch_size:
                            batch_tensor = torch.stack(batch).to(device)
                            batch_numpy = batch_tensor.cpu().numpy()  # Move to CPU and convert to NumPy

                            with torch.no_grad():
                                results = compiled_model([batch_numpy])  # Pass the input to OpenVINO model
                                outputs = results[output_layer]  # Extract output from the specified output layer

                            # Convert the outputs back to a PyTorch tensor if needed
                            outputs_tensor = torch.from_numpy(outputs)

                            for output, (i, j) in zip(outputs_tensor, coordinates):
                                process_prediction(
                                    output=output,
                                    i=i,
                                    j=j,
                                    roi_im_size=roi_im.size,  # Pass the size of the ROI image
                                    combined_mask=combined_mask,
                                    patch_size=patch_size,
                                    th_seg=th_seg
                                )

                            # Reset batch and coordinates
                            batch = []
                            coordinates = []

                # Process any remaining patches
                if batch:
                    batch_tensor = torch.stack(batch).to(device)
                    batch_numpy = batch_tensor.cpu().numpy()
                    # with torch.no_grad():
                    #     outputs = model(batch_tensor)
                    with torch.no_grad():
                        results = compiled_model([batch_numpy])  # Pass the input to OpenVINO model
                        outputs = results[output_layer]  # Extract output from the specified output layer

                    # Convert the outputs back to a PyTorch tensor if needed
                    outputs_tensor = torch.from_numpy(outputs)

                    for output, (i, j) in zip(outputs_tensor, coordinates):
                        process_prediction(
                                output=output,
                                i=i,
                                j=j,
                                roi_im_size=roi_im.size,  # Pass the size of the ROI image
                                combined_mask=combined_mask,
                                patch_size=patch_size,
                                th_seg=th_seg
                            )

                # Zoom factor for WSI
                bit_depth = 8
                zoom_factor = 1
                transform_matrix = [zoom_factor, 0, 0, -zoom_factor, min_x, max_y]
                extension = 10
                fg_objects = mask_to_objects_2d(combined_mask)    

                job.update(status=Job.RUNNING, progress=30, statusComment='Uploading annotations...')

                annotations = AnnotationCollection()

                for fg_poly, _ in fg_objects:
                    upscaled = affine_transform(fg_poly, transform_matrix)
                    annotations.append(Annotation(
                        location=upscaled.wkt,
                        id_image=id_image,
                        id_terms=[id_terms],
                        id_project=project.id
                    ))
                # Save all annotations at once
                annotations.save()                    
                        # except:
                        #     print("An exception occurred. Proceed with next annotations")                                

                # except:
                # # finally:
                #     # # print("end")
                #     print("An exception occurred. Proceed with next annotations")
                #################################################

            end_prediction_time=time.time()
        
        job.update(status=Job.RUNNING, progress=85, statusComment="Completing...")

    #################################################
    finally:
        logging.debug("Leaving run()")
    #################################################

    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

