import os
import io
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
import cv2
from cv2 import dnn_superres

 ## Detect function
def crop_face(img):

    h, w, c = img.shape    
    gray = cv2.cvtColor(img[:,:int(w/2),:], cv2.COLOR_RGB2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 50, maxRadius= 150)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        x,y,r = detected_circles[0][0]
        crop_img = img[y-r:y+r, x-r:x+r,:]
        return crop_img
    else:
        h, w, c = img.shape
        crop_img = img[:,int(w/8):int(w/2)-int(w/8),:]
        return crop_img

def init(input_fol):

    ## Transform method
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((120,120)),
        transforms.GaussianBlur(9, sigma=(0.1, 2.0)),
    ])            
            
    ##Load trained model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("model/FSRCNN-small_x3.pb")
    sr.setModel("fsrcnn", 3)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained = "model/trained.pt"
    model = torch.jit.load(trained, map_location=device)
    model.eval()

    print("Load model suscess!")


    ##Prepare data
    labels = open('Eklipse/test_data/hero_names.txt','r').read().replace("Dr._M","Drm").split('\n')[:-1]
    labels.sort()
    gt = os.listdir(input_fol)

    print('All Done!')

    return transform, model, sr, labels, gt

def output(results, output_path):
    f = open(output_path, "a")
    for i in results:
        f.write(f"{i}")
    f.close()
    print("Output file suscess!")

def run(input_path, transform, model, sr, labels, gt):
    results = []
    for path in gt:
        o_img = cv2.imread(f"{input_path}/{path}")
        o_img = cv2.cvtColor(o_img, cv2.COLOR_BGR2RGB)

        img = transforms.GaussianBlur(5, sigma=(0.1, 2.0))(torch.tensor(o_img))
        img = sr.upsample(o_img)
        
        crop_img = crop_face(img)
        crop_img = transform(crop_img)
        pred = model(torch.unsqueeze(crop_img,0).to(device))
        pred = nn.Softmax(1)(pred)
        pred = labels[pred.argmax(1)]
        
        result = f"{path}\t{pred}\n"
        results.append(result)
    return results
    
def main(args):
    transform, model, sr, labels, gt = init(args.input_path)
    results = run(args.input_path, transform, model, sr, labels, gt)
    output(results, args.output_path)
    print("All Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input_path",
        type=str,
        default="Eklipse/test_data/test_images",
        help="The path of folder you want to predict.",
    )
    parser.add_argument(
        "--output_path", type=str, default="predict.txt", 
        help="The path of txt file you want to save result.",
    )
    
    args = parser.parse_args()

    main(args)
