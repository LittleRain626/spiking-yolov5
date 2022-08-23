import torch
import torchvision
from tqdm import tqdm

import numpy as np
import sys
sys.path.append(r'/workspace/cxy/spikingjelly/spikingjelly')
from activation_based.ann2snn.converter import Converter
from activation_based.ann2snn.utils import download_url
from activation_based.ann2snn.sample_models import cifar10_resnet


def val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
            else:
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
            # print("***********",out[0].cpu().numpy().shape)
            print(out.shape)
            correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
        acc = correct / total
        print('Validating Accuracy: %.3f' % (acc))
    return acc

def main():
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda:1'
    dataset_dir = '~/dataset/cifar10'
    batch_size = 100
    T = 400

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    model = cifar10_resnet.ResNet18()
    # print(model)
    model.load_state_dict(torch.load('SJ-cifar10-resnet18_model-sample.pth', map_location=torch.device('cpu')))

    train_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=transform,
        download=True)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    test_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=transform,
        download=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=100,
        shuffle=True,
        drop_last=False)

    # print('ANN accuracy:')
    # val(model, device, test_data_loader)
    print('Converting...')
    model_converter = Converter(mode='Max', dataloader=test_data_loader)
    snn_model = model_converter(model)
    # print("**********************************")
    # print(snn_model)
    print('SNN accuracy:')
    val(snn_model, device, test_data_loader, T=T)

if __name__ == '__main__':
    print('Downloading SJ-cifar10-resnet18_model-sample.pth')
    download_url("https://ndownloader.figshare.com/files/26676110",'./SJ-cifar10-resnet18_model-sample.pth')
    main()

