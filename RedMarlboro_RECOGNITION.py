import cv2
import numpy as np

MATCH_RATE=27
detector=cv2.xfeatures2d.SIFT_create()

ZERO=0
flannParam=dict(algorithm=ZERO,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread("marlboro.png",0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

cam=cv2.VideoCapture(0)
while True:

    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    MatchPoint=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            MatchPoint.append(m)
    if(len(MatchPoint)>MATCH_RATE):
        tp=[]
        qp=[]
        for m in MatchPoint:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        #Here you can specify what you want to heppen when the Marlboro is placed
        print ("success")
        break
        print (len(MatchPoint))

    cv2.imshow('LocateIMG',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



   
