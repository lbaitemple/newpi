import pkg_resources
from torchvision  import models
from torchvision import transforms
import torch
import cv2
from PIL import Image

class LiNet():
    #create private members of the class
    imgfile = ""
    img =[]
    classfyname = ""
    classfyconfidence = 0
    transform = transforms.Compose([            #[1]
            transforms.Resize(256),                    #[2]
            transforms.CenterCrop(224),                #[3]
            transforms.ToTensor(),                     #[4]
            transforms.Normalize(                      #[5]
            mean=[0.485, 0.456, 0.406],                #[6]
             std=[0.229, 0.224, 0.225]                  #[7]
             )])
    alexnet = models.alexnet(pretrained=True)
    labels =[]

    def __init__(self, fname="dog.jpg"):
        self.imgfile=fname
        self.setfileName(fname)
        labelfile = pkg_resources.resource_filename('talkback', 'data/imagenet_classes.txt')
        with open(labelfile) as f:
            self.labels = [line.strip() for line in f.readlines()]


    def setfileName(self, fname):
        self.imgfile = fname
        self.img = Image.open(self.imgfile)

    def setImage(self, img):
        self.img = img

    def setCV2Image(self, img):
        img_t = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = Image.fromarray(img_t)
        self.img = img_t

    def eval(self):
        img_t = self.transform(self.img)
        batch_t = torch.unsqueeze(img_t, 0)
        self.alexnet.eval()
        out = self.alexnet(batch_t)
        _, index = torch.max(out, 1)

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        self.classfyname = self.labels[index[0]]
        self.classfyconfidence = percentage[index[0]].item()
        # print(labels[index[0]], percentage[index[0]].item())

    def getClassfyName(self):
        return self.classfyname

    def getPrecentage(self):
        return self.classfyconfidence

    def printResult(self):
        print ("result is :{} ({}%)".format(self.classfyname, self.classfyconfidence))
