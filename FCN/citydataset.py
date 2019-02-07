import os
import torch
from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import torchvision.transforms as tvt

cityscapes_classes = np.array([
    [  0,  0,  0],#static
    [111, 74,  0],#dynamic
    [ 81,  0, 81],#ground
    [128, 64,128],#road
    [244, 35,232],#sidewalk
    [250,170,160],#parking
    [230,150,140],#rail track
    [ 70, 70, 70],#building
    [102,102,156],#wall
    [190,153,153],#fence
    [180,165,180],#guard rail
    [150,100,100],#bridge
    [150,120, 90],#tunnel
    [153,153,153],#pole
    [153,153,153],#polegroup
    [250,170, 30],#traffic light
    [220,220,  0],#traffic sign
    [107,142, 35],#vegetation
    [152,251,152],#terrain
    [ 70,130,180],#sky
    [220, 20, 60],#person
    [255,  0,  0],#rider
    [  0,  0,142],#car
    [  0,  0, 70],#truck
    [  0, 60,100],#bus
    [  0,  0, 90],#caravan
    [  0,  0,110],#trailer
    [  0, 80,100],#train
    [  0,  0,230],#motorcycle
    [119, 11, 32],#bicycle
    [ 0, 0,142]#license plate,
])
maps_classes = np.array([
    [255,255,251],
    [203,222,174],
    [171,208,251],
    [231,229,224],
    [243,239,235],
    [255,150,63]
])

facades_classes = np.array([
    [255,154,47],
    [194,0,47],
    [0,56,248],
    [252,766,30],
    [0,247,238],
    [0,129,249],
    [101,255,160],
    [197,2533,90],
    [0,24,215]
])

classes_city = np.array([
    [128,64,123], # violet route
    [0,15,137], # bleu voiture
    [222,218,63], # Jaune Panneau
    [253,162,59], # Orange Feux de signalisation
    [0,0,0], # noir panneau
    [72,72,72], #gris batiment
    [151,251,157], # vert pelouse
    [107,141,53], # vert arbre
    [251,24,226], #rose trottoir
    [62,130,176], #bleu ciel
    [83,0,79], #violet sortie de route
    [188,152,152], #beige muret ?
    [253,164,168], #beige parking
    [149,100,102], #beige panneau route
    [224,0,62], #rouge personne
    [120,5,43], #rouge scooter, velo
    [0,62,98], #Turquoise bus
    [0,19,161], #bleu trotinette
    [254,0,24], #rouge enfant
    [115,75,23], #kaki parasol
    [167,142,165], #gris poteau
    [0,7,70] #bleu camion
])

train_transforms = tvt.Compose([
    tvt.ToPILImage(),
    tvt.Resize((256,256)),
    tvt.ToTensor(),
    tvt.Normalize([0.485,0.456,0.406],[1.,1.,1.])]
)
mask_transforms = tvt.Compose([
    tvt.ToPILImage(),
    tvt.Resize((256,256)),
])
class city(Dataset):
    def __init__(
        self,
        mode,
        classes=classes_city,
        transforms=train_transforms,
        mask_transforms=mask_transforms,
        folder=r"./datasets/cityscapes/",
        use_cuda=True
    ):
        super(city, self).__init__()
        self.path = os.path.join(folder, mode)
        self.ims = sorted(
            list(
                map(
                    lambda file:os.path.join(self.path, file),os.listdir(self.path)
                )
            )
        )
        self.transforms = transforms
        self.mask_transform = mask_transforms
        self.use_cuda = use_cuda
        self.classes = classes
    def __len__(self):
        return len(self.ims)
    def applyCuda(self,x):
        return x.cuda() if self.use_cuda else x
    def __getitem__(self, idx):
        im, mask = self.get_data(idx)
        if self.transforms is not None:
            im = self.transforms(im)
            mask= np.array(self.mask_transform(mask))
        mask = self.make_mask(mask)
        return {'image':im, 'target':mask}
    def get_data(self, idx):
        I_M = imread(self.ims[idx])
        h,w,c= I_M.shape
        I = I_M[:,:int(w/2)]
        M = I_M[:,int(w/2):]
        return I,M
    def make_mask(self, mask):
        def find_cluster(vec, classes=self.classes):
            rscores = np.zeros((256*256, len(classes)))
            for i in range(len(classes)):
                rscores[:, i] = np.linalg.norm(vec - np.repeat(classes[i].reshape(1, 3), 256 * 256, axis=0), axis=1)
            vc = np.argmin(rscores, axis=1)
            return vc

        def find_cluster_torch(vec, classes=self.classes):
            rscores = torch.zeros((256 * 256, len(classes)))
            for i in range(len(classes)):
                rscores[:, i] = torch.norm(
                    torch.cuda.FloatTensor(vec.reshape(-1, 3)) - torch.cuda.FloatTensor(
                        classes[i].reshape(1, 3)).repeat(256 * 256, 1),
                    dim=1
                )

            vc = rscores.argmin(dim=1)
            return vc
        clusters = find_cluster_torch(mask.reshape(-1,3))
        mask = clusters.view(256,256).type(torch.LongTensor)
        return mask

