import random 
import torch
from kornia import morphology as morph
import cv2
from torchvision import transforms

def get_random_kernel():
    k = torch.rand(3,3).round()
    k[1,1] = 1
    return k

class Erosion:
    def __init__(self):
        self.fn = morph.erosion

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]

class Dilation:
    def __init__(self):
        self.fn = morph.dilation

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]

class Opening:
    def __init__(self):
        self.fn = morph.opening

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]

class Closing:
    def __init__(self):
        self.fn = morph.closing

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]
        
class RandomApply(torch.nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    img = cv2.imread(r'E:\MS_AI\Project_1\resources\icdar17_new\0\0_590-IMG_MAX_1086746_310.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    
    o = Opening()(img)
    e = Erosion()(img)
    d = Dilation()(img)
    c = Closing()(img)
    
    fig, axs = plt.subplots(1,4, figsize=(12,4))
    
    axs[0].imshow(e.squeeze())
    axs[0].set_title('Erosion')
    axs[0].axis('off')
    
    axs[1].imshow(d.squeeze())
    axs[1].set_title('Dilation')
    axs[1].axis('off')
    
    axs[2].imshow(o.squeeze())
    axs[2].set_title('Opening')
    axs[2].axis('off')
    
    axs[3].imshow(c.squeeze())
    axs[3].set_title('Closing')
    axs[3].axis('off')




