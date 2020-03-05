import cv2
vidcap = cv2.VideoCapture('/Users/juser/dev/deep-learning/data/bunny.mp4')

print("cap is opened: {}".format(vidcap.isOpened()))
success,image = vidcap.read()
print(success)
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
