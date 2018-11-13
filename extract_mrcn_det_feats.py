import os
import os.path as osp
import numpy as np
import h5py
from scipy.misc import imread, imresize
import torch
from torch.autograd import Variable
# mrcn path
import _init_paths
from mrcn import inference_no_imdb

#%%

# load mrcn model
mrcn = inference_no_imdb.Inference()

# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def image_to_head(head_feats_dir, image_id):
  """Returns
  head: float32 (1, 1024, H, W)
  im_info: float32 [[im_h, im_w, im_scale]]
  """
  feats_h5 = osp.join(head_feats_dir, str(image_id)+'.h5')
  feats = h5py.File(feats_h5, 'r')
  head, im_info = feats['head'], feats['im_info']
  return np.array(head), np.array(im_info)

def det_to_pool5_fc7(mrcn, det, net_conv, im_info):
  """
  Arguments:
    det: object instance
    net_conv: float32 (1, 1024, H, W)
    im_info: float32 [[im_h, im_w, im_scale]]
  Returns:
    pool5: Variable (cuda) (1, 1024)
    fc7  : Variable (cuda) (1, 2048)
  """
  box = np.asarray(det['box'])  # [[xywh]]
  box = xywh_to_xyxy(box)  # [[x1y1x2y2]]
  pool5, fc7 = mrcn.box_to_pool5_fc7(net_conv, im_info, box)  # (1, 2048)
  return pool5, fc7

#%%

import config   
import json
from tqdm import tqdm
from collections import defaultdict
ds = 'refcoco+'   
kwargs = config.dataset[ds] 
dataset = kwargs.get('dataset')
splitBy = kwargs.get('splitBy')
splits =  kwargs.get('splits')
js = json.load(open('/media/manoj/hdd/VQD/MAttNet/detections/{}_{}/res101_coco_minus_refer_notime_dets.json'
                    .format(dataset,splitBy)
                    ))
data = []
for split in splits  + ['train']:
    data_json = osp.join('/home/manoj/vqd/cache/prepro', dataset +"_"+ splitBy , split +'.json')
    with open(data_json,'r') as f:
        d = json.load(f)
        data.extend(d)
      

qid2ent = defaultdict(list)
for ent in js:
    qid2ent[ent['image_id']].append(ent)

print ("Dataset [{}] loaded....".format(dataset))
#%%
n = len(data)
imgid = np.random.randint(0,len(data))

# feats_h5
file_name = '{}_{}_det_feats.h5'.format(dataset,splitBy)
feats_h5 = osp.join('/media/manoj/hdd/VQD/MAttNet/adsf/cache/feats',file_name)

imgid2info = {}
for ent in data:
    imgid2info[ent['image_id']]= ent['image_info']

maxo = max([ len(val) for key,val in qid2ent.items()])

nb_images =  len(qid2ent.keys())
hdf5_file = h5py.File(feats_h5, 'w')

features_shape = (
    nb_images,
    maxo,
    2048
)

boxes_shape = (
    features_shape[0],
    features_shape[1],
    4
)

  
with h5py.File(feats_h5, libver='latest') as fd:
   
    features = fd.create_dataset('features', features_shape,dtype='f')
    boxes = fd.create_dataset('boxes', boxes_shape,dtype='f')
    coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
    widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
    heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')
    num_boxes = fd.create_dataset('num_boxes', shape=(features_shape[0],), dtype='int32')

    for i,cocoid in tqdm(enumerate(imgid2info),total=nb_images):
        coco_ids[i] = int(cocoid)
        widths[i] = int(imgid2info[cocoid]['width'])
        heights[i] = int(imgid2info[cocoid]['height'])
        
        file_name = imgid2info[cocoid]['file_name']
        IMGDIR ='/media/manoj/hdd/VQA/Images/mscoco'
        if 'train' in file_name:
            img_path = osp.join(IMGDIR,"train2014",file_name)
        elif 'val' in file_name:
            img_path = osp.join(IMGDIR,"val2014",file_name)
        
        if not osp.isfile(img_path):
            print ("file doesnot exist")
            
            
        with  torch.no_grad():
            net_conv, im_info   = mrcn.extract_head(img_path = img_path)
        det = {}
        det['box'] = [b['box'] for b in qid2ent[cocoid]]  
        box = np.array(det['box'])        
        #box is save in xywh format        
        featuresarr = np.zeros((features_shape[1],features_shape[2]))
        boxesarr = np.zeros((boxes_shape[1],boxes_shape[2]))        
        L = len(box)
        _, det_fc7 = det_to_pool5_fc7(mrcn, det, net_conv, im_info)
        fc7_set = det_fc7.detach().cpu().numpy()

        num_boxes[i] = int(L)       
        featuresarr[0:L,:] = fc7_set
        boxesarr[0:L,:] = box
        
        features[i, :, :] = featuresarr    
        boxes[i, :, :] = boxesarr
    

hdf5_file.close()
print('{} written.'.format(feats_h5))






#%%

#        net_conv, im_info   = mrcn.extract_head(img_path = img_path)
#        det = {}
#        det['box'] = [b['box'] for b in qid2ent[imgid]]  
#        det_pool5, det_fc7 = det_to_pool5_fc7(mrcn, det, net_conv, im_info)
#        fc7_set = det_fc7.data.cpu().numpy()
#        pool5_set = det_pool5.data.cpu().numpy()
#    
#        fc7_set[i] = det_fc7.data.cpu().numpy()
#        pool5_set[i] = det_pool5.data.cpu().numpy()
#
#
#hdf5_file.close()
#print('{} written.'.format(feats_h5))



