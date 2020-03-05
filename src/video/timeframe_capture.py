import cv2


start_time_ms = 1000
stop_time_ms = 2000
vidcap = cv2.VideoCapture('/Users/juser/data/video/test1.mp4')

count = 0
success = True

while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) < start_time_ms:
    success, image = vidcap.read()

print(success)
while success and vidcap.get(cv2.CAP_PROP_POS_MSEC) <= stop_time_ms:
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    cv2.imwrite("/Users/juser/data/video/out/frame%d.jpg" % count, image)
    count += 1
