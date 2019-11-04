import cv2
from talkback.nnet import LiNet
from PIL import Image

cap = cv2.VideoCapture(0)
a = LiNet()
while True:

    ret, img = cap.read()
#    img_t = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img_t = Image.fromarray(img_t)
#    a.setImage(img_t)
    img2 = img
    a.setCV2Image(img2)
    a.eval()
    print(a.printResult())

    imS = cv2.resize(img, (320, 240))
    cv2.imshow(a.getClassfyName(), imS)
 #   cv2.resizeWindow(a.getClassfyName(), 320, 240)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

