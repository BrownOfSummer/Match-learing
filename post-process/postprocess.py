# -*- coding:utf-8 -*-
from __future__ import print_function
import json
import os
import sys
import numpy as np
from nms import NMS
import cv2

def load_meta(meta_path):
    if not os.path.exists(meta_path):
        print('ERROR: can not find meta file in {}'.format(meta_path))
        return []
    meta = list()
    with open(meta_path, 'r') as f:
        meta = json.loads( f.read() )
    return meta

def load_net_out(net_out_path, shape):
    if not os.path.exists(net_out_path):
        print('ERROR: can not find {}'.format(net_out_path))
        return []
    net_out = np.loadtxt(net_out_path, dtype=np.float32)
    net_out = net_out.reshape(tuple(shape))
    return net_out

def expit_c(x):
    # sigmoid
    x = float(x)
    return 1.0 / ( 1 + np.exp(-x) )
def max_c(a, b):
    a, b = float(a), float(b)
    if( a > b ):
        return a
    return b

def box_constructor(meta, net_out_in):
    H, W, _ = meta['out_size'] # 13, 13, 85
    C = meta['classes'] # classes=12
    B = meta['num'] # num=5
    
    threshold = float(meta['thresh'])
    anchors = np.asarray(meta['anchors'], dtype=np.float32)
    arr_max, sum, tempc = 0.0, 0.0, 0.0
    boxes = list()
    
    net_out = net_out_in.reshape([H, W, B, net_out_in.shape[2]/B]) #[13, 13, 5, 17]
    Classes = net_out[:, :, :, 5:] # '0,1,2,...,Cel,SAMSUNG'
    Bbox_pred = net_out[:, :, :, :5] # coordinates, confidence
    probs = np.zeros((H, W, B, C), dtype=np.float32) # (13, 13, 5, 12)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max=0
                sum=0;
                Bbox_pred[row, col, box_loop, 4] = expit_c(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = np.exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2 * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = np.exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2 * box_loop + 1] / H
                #SOFTMAX BLOCK, no more pointer juggling
                for class_loop in range(C):
                    arr_max=max_c(arr_max,Classes[row,col,box_loop,class_loop])
                
                for class_loop in range(C):
                    Classes[row,col,box_loop,class_loop]=np.exp(Classes[row,col,box_loop,class_loop]-arr_max)
                    sum+=Classes[row,col,box_loop,class_loop]
                
                for class_loop in range(C):
                    tempc = Classes[row, col, box_loop, class_loop] * Bbox_pred[row, col, box_loop, 4]/sum                    
                    if(tempc > threshold):
                        probs[row, col, box_loop, class_loop] = tempc
    
    
    #NMS                    
    return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W,5))
    #print(probs.reshape(H*W*B, C))
    #print('\n',Bbox_pred.reshape(H*B*W, 5))

def process_box(meta, b, h, w, threshold):
	max_indx = np.argmax(b.probs)
	max_prob = b.probs[max_indx]
	label = meta['labels'][max_indx]
	if max_prob > threshold:
		left  = int ((b.x - b.w/2.) * w)
		right = int ((b.x + b.w/2.) * w)
		top   = int ((b.y - b.h/2.) * h)
		bot   = int ((b.y + b.h/2.) * h)
		if left  < 0    :  left = 0
		if right > w - 1: right = w - 1
		if top   < 0    :   top = 0
		if bot   > h - 1:   bot = h - 1
		mess = '{}'.format(label)
		return (left, right, top, bot, mess, max_indx, max_prob)
	return None

def findboxes(meta, net_out):
    # meta
    boxes = list()
    boxes=box_constructor(meta,net_out)
    return boxes

def post_process_boxes(meta, boxes, im, save = True):
        """
        draw the boxes on image, save it to json
        """
	# meta
	#meta = self.meta
        threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	
	resultsForJSON = []
	for b in boxes:
		boxResults = process_box(meta, b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
                resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)

	if not save: return imgcv

	#outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	#img_name = os.path.join(outfolder, os.path.basename(im))
        img_name = 'inference_result.jpg'
        if True:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		#return

	cv2.imwrite(img_name, imgcv)
if __name__ == '__main__':
    meta = load_meta('../darkflow/tiny-model/tiny-8250-pb/tiny-yolo-cel.meta')
    net_out = load_net_out('net_out.txt', meta['out_size'])
    image_path = './output1_0457.jpg'
    H, W, _ = meta['out_size'] # 13,13
    C = meta['classes'] # 12
    B = meta['num'] # 5
    h, w = 416, 416
    threshold = 0.5
    boxes = findboxes(meta, net_out)
    if True:
        post_process_boxes(meta, boxes, image_path, )
    else:
        for b in boxes:
    	    boxResults = process_box(meta, b, h, w, threshold)
            left, right, top, bot, mess, max_indx, confidence = boxResults
            print(boxResults)
