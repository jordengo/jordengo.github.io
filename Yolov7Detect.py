import argparse
import cv2
from detect import detect as FDetect

NewRecordRtnList = [] # 本次迭代被偵測出的物件


# decide range from DetectAreaRangePercentage with screen width
def SettingDetectRange(DetectAreaRangePercentage,ImageWidth, ImageHeight):
	leftX = int( ImageWidth * ( 1 - (DetectAreaRangePercentage * 0.01 ) ) / 2 )
	rightX = int( ImageWidth * ( 1 - ( 1 - (DetectAreaRangePercentage * 0.01 ) ) / 2  )   )
	return [ (leftX,0) , (rightX,ImageHeight) ]


# parameter setting
ImageFile = 'testphoto.jpg'         # image file
DetectAreaRangePercentage = 62      # valid detecting range : Area as a percentage of the total area
DetectAreaObjectPercentage = 3.5    # each Object that should be detected   area as a percentage of the total area    
RecordRtnList = []                  # record each loop detected object coodinate


# global variable defined as above



# fetched image info
img =  cv2.imread(ImageFile)
tHeight, tWidth, tChannel = img.shape # image height、width 
Area =  tHeight * tWidth  # image total area



# start detecting object 
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='./yolov7-e6e.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default=img, help='source')  # file/folder, 0 for webcam    'inference/images' 
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace',default=True, action='store_true', help='don`t trace model')
opt = parser.parse_args()

RtnList = FDetect(opt, False)
print(RtnList)



# decide valid range
DetectRange = SettingDetectRange(DetectAreaRangePercentage, tWidth, tHeight)
EdgeLeftX = DetectRange[0][0]
EdgeRightX = DetectRange[1][0]
plot_color =  (0,255,0) #BGR
cv2.rectangle(img, ( int( DetectRange[0][0]), int( DetectRange[0][1]) ), ( int( DetectRange[1][0]), int( DetectRange[1][1]) ), plot_color, 3, cv2.LINE_AA)



# start label object to warn user by stategy below
IsPlayWarning = False # 是否要發生警告
for Rtn in RtnList:

	print('------------------------------------------')
	print('coodinate=' + str(int(Rtn[0])) + ',' + str(int(Rtn[1])) + ',' + str(int(Rtn[2])) + ',' + str(int(Rtn[3]))  )
	subarea = abs( int(Rtn[2]) - int(Rtn[0]))  * abs( int(Rtn[3]) - int(Rtn[1]) ) #先計算出 area
	print(  'area=' + str(subarea)  )  
	percentage = subarea/Area*100    #object occupy area of percentage 面積所佔比例
	print( 'percentage=' +  str(percentage) + '%'  )

	plot_color = (0,0,255) #BGR
	
	# 如果物件符合比例且在指定的範圍內，即列為被偵測的物件； 被偵測出的物件要特別框出來 與 暫存到 NewRecordRtnList
	if( percentage >= DetectAreaObjectPercentage and  ( Rtn[0] >= EdgeLeftX and Rtn[0] <= EdgeRightX  or   Rtn[2] >= EdgeLeftX and Rtn[2] <= EdgeRightX  )   ):
		
		cv2.rectangle(img, ( int(Rtn[0]), int(Rtn[1]) ), ( int(Rtn[2]), int(Rtn[3]) ), plot_color, 3, cv2.LINE_AA)

		NewRecordRtnList.append(Rtn) 

		# 被偵測出的物件 決定要不要 發出警告聲音
		for Record in RecordRtnList:

			#左上 #左下  #右上 #右下 其中一個點在範圍內 就代表物件有重疊 將此填滿標註
			if(    ( Record[0] <= Rtn[0] and Rtn[0] <= Record[2] and Record[1] <= Rtn[1] and Rtn[1] <= Record[3]  ) or   
				   ( Record[0] <= Rtn[0] and Rtn[0] <= Record[2] and Record[1] <= Rtn[3] and Rtn[3] <= Record[3]  ) or
				   ( Record[0] <= Rtn[2] and Rtn[2] <= Record[2] and Record[1] <= Rtn[1] and Rtn[1] <= Record[3]  ) or
				   ( Record[0] <= Rtn[2] and Rtn[2] <= Record[2] and Record[1] <= Rtn[3] and Rtn[3] <= Record[3]  ) 
			):
				cv2.rectangle(img, ( int(Rtn[0]), int(Rtn[1]) ), ( int(Rtn[2]), int(Rtn[3]) ), (0, 0, 255), -1)  # fill new detected object color

				RecordSubarea = abs( int(Record[2]) - int(Record[0]))  * abs( int(Record[3]) - int(Record[1]) ) #record's area
				if(RecordSubarea < subarea ):
					IsPlayWarning = True  # setting alert voice

				break

RecordRtnList = NewRecordRtnList # 更新紀錄 

while(1):
    cv2.imshow('result',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()


