import sys
import os
import yaml
import torch
import torchvision
import argparse
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(r'/workspace/cxy/spikingjelly/spikingjelly')
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from activation_based.ann2snn.converter import Converter

from models.yolo import Model
from models.common import DetectMultiBackend
from utils.general import LOGGER, intersect_dicts, check_dataset, check_img_size, colorstr, increment_path, scale_coords, cv2, xyxy2xywh, non_max_suppression
from utils.dataloaders import create_dataloader, LoadImages 
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def detect(model, source, device, project=ROOT / 'runs/detect', name='exp', exist_ok=False, save_txt=False, save_crop=False, save_conf=False,
           augment=False, conf_thres=0.01, iou_thres=0.1, classes=None, agnostic_nms=False, max_det=1000,  line_thickness=3, 
           hide_labels=False, hide_conf=False, visualize=False):
    stride, names, pt = model.stride, model.names, model.pt
    dataset = LoadImages(source, img_size=opt.imgsz, stride=stride, auto=pt)
    save_img = not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    seen, windows = 0, []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        print(pred)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print(pred)
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)

    # Print results
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

def main(opt, dnn=False, half=False):
    weights, cfg, hyp, data, batch_size, single_cls, workers= opt.weights, opt.cfg, opt.hyp, opt.data, opt.batch_size, opt.single_cls, opt.workers
    
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    device = select_device(opt.device, batch_size=opt.batch_size)
    data_dict = check_dataset(data)
    nc = int(data_dict['nc'])  # number of classes
    # ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    # model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    # exclude = ['anchor'] if (cfg or hyp.get('anchors')) else []
    # csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    # model.load_state_dict(csd, strict=False)  # load
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    print(model.stride)
    # gs = max(int(model.stride.max()), 32)
    gs = max(int(model.stride), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    train_path, val_path = data_dict['train'], data_dict['val']
    # Trainloader for object detection
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    val_loader = create_dataloader( val_path,
                                    imgsz,
                                    batch_size // WORLD_SIZE * 2,
                                    gs,
                                    single_cls,
                                    hyp=hyp,
                                    cache=opt.cache,
                                    rect=True,
                                    rank=-1,
                                    workers=workers * 2,
                                    pad=0.5,
                                    prefix=colorstr('val: '))[0]

    
    # source = '/workspace/cxy/spikingjelly/coco128/images/train2017'
    source = '/workspace/cxy/spikingjelly/airplane'
    # detect(model, source, device)
    model_converter = Converter(mode='Max', dataloader=train_loader)

    snn_model = model_converter(model)
    # print('SNN accuracy:')
    # val(snn_model, device, test_data_loader, T=400)
    detect(snn_model, source, device)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)