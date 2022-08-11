import pathlib
from torch.utils.data import DataLoader, Dataset
import pathlib
from PIL import Image
from random import choice
from torchvision import transforms
import matplotlib.pyplot as plt

def Image_Transformer():
    return {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]),
        'inv': transforms.Compose([
            transforms.Normalize(mean=(-1., -1., -1.), std=(2., 2., 2.)),
            transforms.ToPILImage()
        ])
    }

class Dataset_Maps(Dataset):
    def __init__(self, imgs_dir, size=(600, 300)):
        self.imgs_dir = imgs_dir
        self.imgs_list = [str(l) for l in pathlib.Path(self.imgs_dir).glob('*/*.jpg')]
        self.transformer = Image_Transformer()['train']
    
    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, index):
        img_pil = Image.open(self.imgs_list[index])
        img_tensor = self.transformer(img_pil)
        return (img_tensor[...,:600], img_tensor[...,600:]) # real_img, map_img

if __name__ == '__main__':

    imgs_dir = './content/maps/'
    maps_dataset = Dataset_Maps(imgs_dir)
    print (f'número de imagens: {len(maps_dataset)}')
    real_img, map_img = choice(maps_dataset)

    print (f'real_img.shape: {real_img.shape}, real_img.min(): {real_img.min()}, real_img.max(): {real_img.max()}')
    print (f'map_img.shape: {map_img.shape}, map_img.min(): {map_img.min()}, map_img.max(): {map_img.max()}')

    inv_transformer = Image_Transformer()['inv']

    img1 = inv_transformer(real_img)
    img2 = inv_transformer(map_img)
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()