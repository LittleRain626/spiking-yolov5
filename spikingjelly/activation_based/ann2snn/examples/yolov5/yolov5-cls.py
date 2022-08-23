from operator import mod
import sys
import os
import torch
import torchvision
import torch.nn.functional as F
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

from models.experimental import attempt_load
from utils.torch_utils import reshape_classifier_output
from utils.general import LOGGER, Profile
from utils.augmentations import classify_transforms
from utils.dataloaders import create_classification_dataloader, LoadImages

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=ROOT / 'yolov5s-cls.pt', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=100, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. cuda:0 or cpu')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')

    return parser.parse_known_args()[0] if known else parser.parse_args()
def val(model, device, dataloader, pbar=None, criterion=None, T=None):
    model.eval().to(device)
    training = model is not None
    pred, targets, loss, dt = [], [], 0, (Profile(), Profile(), Profile())
    n = len(dataloader)  # number of batches
    action = 'validating'
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(dataloader, desc, n, not training, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0)
    
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        if T is None:
                y = model(images)
        else:
            for t in range(T):
                if t == 0:
                    y = model(images)
                else:
                    y += model(images)
        pred.append(y.argsort(1, descending=True)[:, :5])
        targets.append(labels)
        if criterion:
            loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
    LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")
    for i, c in model.names.items():
        aci = acc[targets == i]
        top1i, top5i = aci.mean(0).tolist()
        LOGGER.info(f"{c:>24}{aci.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")
    return top1, top5, loss
def predict(model, source):
    # Load model
    model.to(opt.device)
    dataset = LoadImages(source, img_size=opt.imgsz, transforms=classify_transforms(opt.imgsz))
    for seen, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # Image
        im = im.unsqueeze(0).to(opt.device)
        im = im.float()#  if model.fp16 else im.float()
        # Inference
        results = model(im)
        # Post-process
        p = F.softmax(results, dim=1)  # probabilities
        i = p.argsort(1, descending=True)[:, :5].squeeze().tolist()  # top 5 indices
        # if save:
        #    imshow_cls(im, f=save_dir / Path(path).name, verbose=True)
        LOGGER.info(f"{s}{opt.imgsz}x{opt.imgsz} {', '.join(f'{model.names[j]} {p[0, j]:.2f}' for j in i)}")
    # Print results
    return p


def main(opt):
    batch_size, device = opt.batch_size, opt.device
    if Path(opt.model).is_file() or opt.model.endswith('.pt'):
        model = attempt_load(opt.model, device='cpu', fuse=False)
    # dataset_dir = '../../../dataset/cifar10'
    dataset_dir = '/workspace/cxy/pear_flaw_detection/datasets/cifar10'
    nc = 10
    reshape_classifier_output(model, nc)  # update class count
    
    # Dataloader
    data = Path(dataset_dir)
    test_dir = data / 'test' if (data / 'test').exists() else data / 'val'  # data/test or data/val
    dataloader = create_classification_dataloader(path=test_dir,
                                                  imgsz=opt.imgsz,
                                                  batch_size=batch_size,
                                                  augment=False,
                                                  rank=-1,
                                                  workers=opt.workers)

    source = '/workspace/cxy/spikingjelly/airplane'
    # classify
    print('ANN accuracy:')
    # val(model, device, dataloader)
    # predict(model, source)
    print('CNN2SNN converter: ')
    model_converter = Converter(mode='Max', dataloader=dataloader)
    snn_model = model_converter(model)
    print('SNN accuracy:')
    # val(snn_model, device, dataloader, T=400)
    predict(snn_model, source)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)