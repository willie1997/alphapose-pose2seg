"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
sys.path.append('./')
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.writer import DataWriter
from modeling.build_model import Pose2Seg
from pycocotools import mask as maskUtils
"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, help='experiment configure file name',
                    default='pretrained_models/ap_ours.yaml')
parser.add_argument('--checkpoint', type=str, help='checkpoint file name', 
                    default='pretrained_models/fast_dcn_res50_256x192.pth')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="inpimgs")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="out")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--format', type=str, default='coco',
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video', action='store_true', default=False)
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = False

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def check_input():
    # for images
    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            im_names = open(inputlist, 'r').readlines()
        elif len(inputpath) and inputpath != '/':
            for root, dirs, files in os.walk(inputpath):
                im_names = files
        elif len(inputimg):
            im_names = [inputimg]
        return 'image', im_names


def print_finish_info():
    print('===========================> Finish Model Running.')
    if args.save_img:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')

def get_kp(all_results):
    output = []
    for im_res in all_results:
        im_name = im_res['imgname']
        for human in im_res['result']:
            keypoints = []
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            kp = np.array(keypoints).reshape(17,3)
            kp[(np.where(kp[:,2]<0.8)), 2] = 0.
            output.append(kp)
    return np.array(output)



if __name__ == "__main__":
    mode, input_source = check_input()
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode).start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print(f'Loading pose model from {args.checkpoint}...')
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_model.to(args.device)
    pose_model.eval()



    # Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize
    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()
    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    try:
        for i in im_names_desc:
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, os.path.basename(im_name))
                    continue
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))
        print_finish_info()
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
        writer.stop()
        det_loader.stop()
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues
            writer.commit()
            writer.clear_queues()
            # det_loader.clear_queues()
    final_result = writer.results()
    output_kp = get_kp(final_result)


    model_path = './pretrained_models/pose2seg_release.pkl'
    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(model_path)
    model.eval() 
    print("Results have been written to json.")
    
    img_path = './inpimgs/392.jpg'
    imgs = cv2.imread(img_path)
    output = model([imgs], [output_kp])
    ids = 0 
    for mask in output[0]:
        
        # maskencode = maskUtils.encode(np.asfortranarray(mask))
        temp = np.zeros(imgs.shape)
        one = np.ones(imgs.shape) * 255
        temp[np.where(mask!=0)] = one[np.where(mask!=0)]
        # cv2.imshow('mask', temp/255.)
        # cv2.waitKey()
        cv2.imwrite('mask_%d.jpg'%(ids), temp)
        ids+=1
        # maskencode['counts'] = maskencode['counts'].decode('ascii')
        # print('ok')
        # ids+=1