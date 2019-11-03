from talkback.nnet import LiNet

a = LiNet("dog.jpg")
a.eval()

a.printResult()
#print(a.getClassfyName())