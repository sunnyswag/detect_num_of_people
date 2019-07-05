import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
# show = np.zeros([5,2],dtype = int)

# count = 1
# # while True:
# #     num = r['rois'].shape[0]
# count = count + 1
# show=np.concatenate((show,[[1,2]]),axis = 0)
# show=np.delete(show,[0,1],axis=0)
# print(show)
# print(show.shape[0])
# count =50
# print(25//25)
# if count % 25 == 0 :
# 	coutn= count//25
# 	print(coutn)
# 	print(1//2)
# 	num = 6
# 	if num >= 5:
# 		show = np.concatenate((show,[[coutn,num+2]]),axis=0)
# 	elif num >= 10:
# 		show = np.concatenate((show,[[coutn,num+4]]),axis=0)
# 	elif num >= 15:
# 		show = np.concatenate((show,[[coutn,num+6]]),axis=0)
# 	elif num >= 20:
# 		show =np.concatenate((show,[[coutn,num+10]]),axis=0)
# 	show=np.concatenate((show,[[coutn,num+3]]),axis = 0)
# print(show)

# n=20
# X=np.arange(n)
# Y=(1-X/float(n))*np.random.uniform(5, 15, n)


# barlist=plt.bar(X,Y)

# plt.xlim(-0.8,n)
# plt.ylim(0,30)

# plt.xlabel("Time")
# plt.ylabel('Num of people')

# plt.xticks(np.linspace(0,n,5))
# plt.yticks([0,5,10,15,20,25,30,35,40])

# for x,y in zip(X,Y):
# 	if y >= 5 :
# 		barlist[x].set_color('orange')
# 	plt.text(x,y+0.5,'%.1f'%y,ha='center',va='bottom')

# plt.show()
data = [
 [ 1,  3],
 [ 2,  4],
 [ 3,  14],
 [ 4, 14],
 [ 5, 13],
 [ 6, 13],
 [ 7, 13],
 [ 8,  26],
 [ 9,  27],
 [10, 28],
 [11, 27],
 [12, 27],
 [13, 28],
 [14, 10],
 [15, 10],
 [16, 11],
 [17, 12],
 [18, 11],
 [19, 33],
 [20, 31],
 [21,  32],
 [22,  35],
 [23, 32],
 [24, 12],
 [25, 11],
 [26, 12],
 [27, 11],
 [28, 9],
 [29, 12],
 [30, 13],
 [31, 13],
 [32,  6],
 [33,  2],
 [34,  4],
 [35,  7],
 [36, 11],
 [37, 20],
 [38, 28],
 [39, 2]]

data1 = [
 [ 1,  2],
 [ 2,  2],
 [ 3,  2],
 [ 4, 11],
 [ 5, 18],
 [ 6, 17],
 [ 7, 11],
 [ 8,  6],
 [ 9,  2],
 [10,  6],
 [11, 10],
 [12, 13],
 [13, 18],
 [14, 21],
 [15, 13],
 [16, 11],
 [17, 18],
 [18, 21],
 [19, 17],
 [20, 17],
 [21,  2],
 [22,  2],
 [23, 10],
 [24, 18],
 [25, 19],
 [26, 18],
 [27, 12],
 [28, 11],
 [29, 20],
 [30, 26],
 [31, 13],
 [32,  6],
 [33,  2],
 [34,  4],
 [35,  7],
 [36, 11],
 [37, 20],
 [38, 30],
 [39, 2]]

data2 = [
 [ 1,  19],
 [ 2,  12],
 [ 3,  19],
 [ 4, 20],
 [ 5, 14],
 [ 6, 3],
 [ 7, 2],
 [ 8,  10],
 [ 9,  10],
 [10,  12],
 [11, 18],
 [12, 28],
 [13, 30],
 [14, 18],
 [15, 7],
 [16, 21],
 [17, 20],
 [18, 26],
 [19, 14],
 [20, 2],
 [21,  2],
 [22,  12],
 [23, 13],
 [24, 14],
 [25, 17],
 [26, 2],
 [27, 10],
 [28, 12],
 [29, 11],
 [30, 3],
 [31, 3],
 [32,  2],
 [33,  3],
 [34,  2],
 [35,  4],
 [36, 7],
 [37, 10],
 [38, 14],
 [39, 2]]

data3 = [
 [ 1,  11],
 [ 2,  12],
 [ 3,  14],
 [ 4, 4],
 [ 5, 2],
 [ 6, 2],
 [ 7, 2],
 [ 8,  3],
 [ 9,  3],
 [10,  6],
 [11, 14],
 [12, 30],
 [13, 18],
 [14, 19],
 [15, 10],
 [16, 11],
 [17, 21],
 [18, 26],
 [19, 18],
 [20, 18],
 [21,  2],
 [22,  2],
 [23, 3],
 [24, 6],
 [25, 7],
 [26, 18],
 [27, 11],
 [28, 2],
 [29, 6],
 [30, 19],
 [31, 21],
 [32, 11],
 [33,  5],
 [34,  2],
 [35,  4],
 [36, 7],
 [37, 14],
 [38, 13],
 [39, 17]]

data4 = [
 [ 1,  14],
 [ 2,  5],
 [ 3,  17],
 [ 4, 17],
 [ 5, 30],
 [ 6, 7],
 [ 7, 3],
 [ 8,  5],
 [ 9,  6],
 [10,  10],
 [11, 18],
 [12, 17],
 [13, 19],
 [14, 10],
 [15, 12],
 [16, 19],
 [17, 19],
 [18, 22],
 [19, 7],
 [20, 3],
 [21,  5],
 [22,  10],
 [23, 13],
 [24, 13],
 [25, 28],
 [26, 12],
 [27, 4],
 [28, 7],
 [29, 10],
 [30, 10],
 [31, 6],
 [32, 2]]

data5 = [
 [ 1,  18],
 [ 2,  14],
 [ 3,  19],
 [ 4, 13],
 [ 5, 12],
 [ 6, 12],
 [ 7, 17],
 [ 8,  12],
 [ 9,  6],
 [10,  7],
 [11, 10],
 [12, 5],
 [13, 11],
 [14, 14],
 [15, 17],
 [16, 14],
 [17, 19],
 [18, 17],
 [19, 14],
 [20, 17],
 [21, 17],
 [22, 18],
 [23, 13],
 [24, 11],
 [25, 11],
 [26, 13],
 [27, 11],
 [28, 11],
 [29, 10],
 [30, 3]]

# def matplot(show):
# 	barlist=plt.bar(show[:,0],show[:,1],align="center")

# 	ax = plt.gca()
# 	ax.spines['right'].set_color('none')
# 	ax.spines['top'].set_color('none')

# 	plt.xlim(0,show.shape[0]+1)
# 	plt.ylim(0,40)

# 	plt.xlabel("Time")
# 	plt.ylabel('Num of people')

# 	plt.xticks(show[:,0])
# 	plt.yticks([0,5,10,15,20,25,30,35,40])


# 	# plt.plot(color="limegreen",linewidth=5,linestyle="-",label="1")
# 	# plt.plot(color="dodgerblue",linewidth=5,linestyle="-",label="2")
# 	# plt.plot(color="orange",linewidth=5,linestyle="-",label="3")
# 	# plt.plot(color="m",linewidth=5,linestyle="-",label="4")
# 	plt.legend(loc='upper left',frameon=False)

# 	for x,y in zip(show[:,0],show[:,1]):
# 		print((x,y))
# 		if y <= 10 :
# 			barlist[x-1].set_color('limegreen')
# 		elif y <=20 :
# 			barlist[x-1].set_color('dodgerblue')
# 		elif y <= 30 :
# 			barlist[x-1].set_color('orange')
# 		else:
# 			barlist[x-1].set_color('m')

# 		plt.text(x,y+0.5,'%.0f'%y,ha='center',va='bottom')

# 	plt.savefig("plot1.png")
# 	plt.show()


zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')

# def matplot(show):
# 	fig,ax = plt.subplots()
# 	limegreen = np.zeros([2,2],dtype = int)
# 	dodgerblue = np.zeros([2,2],dtype = int)
# 	orange = np.zeros([2,2],dtype = int)
# 	m = np.zeros([2,2],dtype = int)
# 	for x,y in zip(show[:,0],show[:,1]):
# 		# print((x,y))
# 		if y < 10 :
# 			limegreen = np.concatenate((limegreen,[[x,y]]),axis = 0)
# 		elif y < 20 :
# 			dodgerblue = np.concatenate((dodgerblue,[[x,y]]),axis = 0)
# 		elif y < 30 :
# 			orange = np.concatenate((orange,[[x,y]]),axis = 0)
# 		else :
# 			m = np.concatenate((m,[[x,y]]),axis = 0)
# 	limegreen = deletFT(limegreen)
# 	dodgerblue = deletFT(dodgerblue)
# 	orange = deletFT(orange)
# 	m = deletFT(m)
# 	limegreen1 = ax.bar(limegreen[:,0],limegreen[:,1],color='limegreen')
# 	dodgerblue1 = ax.bar(dodgerblue[:,0],dodgerblue[:,1],color='dodgerblue')
# 	orange1 = ax.bar(orange[:,0],orange[:,1],color='orange')
# 	m1 = ax.bar(m[:,0],m[:,1],color='m')
# 	ax.set_title('1路车当前时段车厢内人数分布图',fontproperties=zhfont,size="20")
# 	ax.set_xlim(0,show.shape[0]+1)
# 	ax.set_ylim(0,40)
# 	ax.set_ylabel('人数',fontproperties=zhfont,size="15")
# 	ax.set_xlabel('2018-12-03',size="15")
# 	ax.set_xticks([1,5,10,15,20,25,30])
# 	ax.set_xticklabels((r'$09:36$',r'$09:41$',r'$09:46$',r'$09:51$',r'$09:56$',r'$10:01$',r'$10:06$'))
# 	ax.set_yticks([0,5,10,15,20,25,30,35,40])
# 	leg = ax.legend((limegreen1[0],dodgerblue1[0],orange1[0],m1[0]),('少','正常','较多','爆满'))
# 	#leg = ax.legend((limegreen1[0],dodgerblue1[0]),('少','正常'))
# 	for text in leg.texts : text.set_font_properties(zhfont)

# 	def autolabel(rects):
# 		for rect in rects :
# 			height = rect.get_height()
# 			ax.text(rect.get_x()+rect.get_width()/2.0,1.01*height,'%d'%int(height),ha='center',va='bottom')

# 	autolabel(limegreen1)
# 	autolabel(dodgerblue1)
# 	autolabel(orange1)
# 	autolabel(m1)


# 	fig = mpl.pyplot.gcf()
# 	fig.set_size_inches(18.5, 10.5)
# 	fig.savefig('show.png', dpi=100)
# 	# plt.figure(figsize=(100,50))
# 	plt.show()

def matplot(show):
	fig,ax = plt.subplots()
	limegreen = np.zeros([2,2],dtype = int)
	dodgerblue = np.zeros([2,2],dtype = int)
	orange = np.zeros([2,2],dtype = int)
	m = np.zeros([2,2],dtype = int)
	for x,y in zip(show[:,0],show[:,1]):
		# print((x,y))
		if y < 10 :
			limegreen = np.concatenate((limegreen,[[x,y]]),axis = 0)
		elif y < 20 :
			dodgerblue = np.concatenate((dodgerblue,[[x,y]]),axis = 0)
		elif y < 30 :
			orange = np.concatenate((orange,[[x,y]]),axis = 0)
		else :
			m = np.concatenate((m,[[x,y]]),axis = 0)
	limegreen = deletFT(limegreen)
	dodgerblue = deletFT(dodgerblue)
	orange = deletFT(orange)
	m = deletFT(m)
	limegreen1 = ax.bar(limegreen[:,0],limegreen[:,1],color='limegreen')
	dodgerblue1 = ax.bar(dodgerblue[:,0],dodgerblue[:,1],color='dodgerblue')
	orange1 = ax.bar(orange[:,0],orange[:,1],color='orange')
	m1 = ax.bar(m[:,0],m[:,1],color='m')
	ax.set_title('1路车当前时段车厢内人数分布图(2018-12-03)',fontproperties=zhfont,size="20")
	ax.set_xlim(0,show.shape[0]+1)
	ax.set_ylim(0,40)
	ax.set_ylabel('人数',fontproperties=zhfont,size="15")
	#ax.set_xlabel('2018-12-03',size="15")
	ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34])
	ax.set_xticklabels((r'9:15,宋渡大桥',r'晨光园区',r'瑞祥商贸区',r'红谷皮具',r'财富广场',r'客运中心',r'教育局',r'一中西校区',r'釜江广场',
						r'大转盘',r'怡和嘉苑',r'商业广场',r'富州花园',r'梅泽花园',r'韦家巷',r'文庙',r'东门口',
						r'13:44,宋渡大桥',r'晨光园区',r'瑞祥商贸区',r'红谷皮具',r'财富广场',r'客运中心',r'教育局',r'一中西校区',r'釜江广场',
						r'大转盘',r'怡和嘉苑',r'商业广场',r'富州花园',r'梅泽花园',r'韦家巷',r'文庙',r'东门口',
						),fontproperties=zhfont)
						# r'8:40,东门口',r'文庙',r'韦家巷',r'梅泽花园',r'富州公园',r'商业广场',r'怡和嘉苑',r'大转盘',r'釜江广场',
						# r'一中西校区',r'教育局',r'客运中心',r'财富广场',r'红谷皮具',r'瑞祥商贸城',r'晨光园区',r'宋渡大桥',
						# r'12:52,东门口',r'文庙',r'韦家巷',r'梅泽花园',r'富州公园',r'商业广场',r'怡和嘉苑',r'大转盘',r'釜江广场',
						# r'一中西校区',r'教育局',r'客运中心',r'财富广场',r'红谷皮具',r'瑞祥商贸城',r'晨光园区',r'宋渡大桥',
						# r'14:10,东门口',r'文庙',r'韦家巷',r'梅泽花园',r'富州公园',r'商业广场',r'怡和嘉苑',r'大转盘',r'釜江广场',
						# r'一中西校区',r'教育局',r'客运中心',r'财富广场',r'红谷皮具',r'瑞祥商贸城',r'晨光园区',r'宋渡大桥'
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(10) 
		# specify integer or one of preset strings, e.g.
		#tick.label.set_fontsize('x-small') 
		tick.label.set_rotation('vertical')
	leg = ax.legend((limegreen1[0],dodgerblue1[0],orange1[0],m1[0]),('少','正常','较多','爆满'))
	#leg = ax.legend((limegreen1[0],dodgerblue1[0]),('少','正常'))
	for text in leg.texts : text.set_font_properties(zhfont)

	def autolabel(rects):
		for rect in rects :
			height = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2.0,1.01*height,'%d'%int(height),ha='center',va='bottom')

	autolabel(limegreen1)
	autolabel(dodgerblue1)
	autolabel(orange1)
	autolabel(m1)


	fig = mpl.pyplot.gcf()
	fig.set_size_inches(18.5, 10.5)
	fig.savefig('宋渡大桥_9:15.png', dpi=100)
	# plt.figure(figsize=(100,50))
	plt.show()

def deletFT(abc):
	abc = np.delete(abc,[0,1],axis = 0)
	return abc

data6 = [
 [ 1,  4],
 [ 2,  4],
 [ 3,  7],
 [ 4,  6],
 [ 5, 10],
 [ 6, 11],
 [ 7, 15],
 [ 8, 11],
 [ 9, 14],
 [10, 13],
 [11, 10],
 [12,  1],
 [13,  1],
 [14,  1],
 [15,  1],
 [16,  1],
 [17,  0],
 [18, 13],
 [19, 13],
 [20, 14],
 [21, 14],
 [22, 15],
 [23, 27],
 [24, 32],
 [25, 20],
 [26, 18],
 [27, 31],
 [28, 36],
 [29, 35],
 [30, 17],
 [31, 15],
 [32, 11],
 [33, 11],
 [34, 0]]

data11 = [
 [ 1,  12],
 [ 2,  14],
 [ 3,  16],
 [ 4,  21],
 [ 5, 22],
 [ 6, 24],
 [ 7, 28],
 [ 8, 33],
 [ 9, 34],
 [10, 9],
 [11, 5],
 [12,  5],
 [13,  5],
 [14,  4],
 [15,  4],
 [16,  3],
 [17,  0],
 [18, 15],
 [19, 16],
 [20, 19],
 [21, 16],
 [22, 19],
 [23, 19],
 [24, 23],
 [25, 19],
 [26, 12],
 [27, 14],
 [28, 12],
 [29, 13],
 [30, 10],
 [31, 6],
 [32, 4],
 [33, 4],
 [34, 0]]

data22 = [
 [ 1,  9],
 [ 2,  9],
 [ 3,  11],
 [ 4,  13],
 [ 5, 18],
 [ 6, 17],
 [ 7, 24],
 [ 8, 21],
 [ 9, 9],
 [10, 10],
 [11, 7],
 [12,  4],
 [13,  3],
 [14,  2],
 [15,  2],
 [16,  2],
 [17,  0],
 [18, 4],
 [19, 4],
 [20, 7],
 [21, 6],
 [22, 10],
 [23, 11],
 [24, 15],
 [25, 11],
 [26, 14],
 [27, 13],
 [28, 10],
 [29, 1],
 [30, 1],
 [31, 1],
 [32, 1],
 [33, 1],
 [34, 0],
 [35, 11],
 [36, 13],
 [37, 15],
 [38, 21],
 [39, 26],
 [40, 23],
 [41, 28],
 [42, 30],
 [43, 22],
 [44, 26],
 [45, 25],
 [46, 18],
 [47, 17],
 [48, 17],
 [49, 8],
 [50, 6],
 [51, 0]]

data33 = [
 [ 1,  5],
 [ 2,  8],
 [ 3,  8],
 [ 4,  9],
 [ 5, 10],
 [ 6, 15],
 [ 7, 18],
 [ 8, 20],
 [ 9, 16],
 [10, 23],
 [11, 28],
 [12, 30],
 [13, 26],
 [14, 21],
 [15, 22],
 [16, 10],
 [17,  0],
 [18,  8],
 [19,  9],
 [20,  9],
 [21,  9],
 [22,  9],
 [23,  8],
 [24, 11],
 [25, 20],
 [26, 21],
 [27, 20],
 [28, 13],
 [29, 19],
 [30, 17],
 [31, 15],
 [32, 9],
 [33, 9],
 [34, 0]]

data44 = [
 [ 1,  6],
 [ 2,  8],
 [ 3, 12],
 [ 4, 15],
 [ 5, 16],
 [ 6, 18],
 [ 7, 24],
 [ 8, 22],
 [ 9, 29],
 [10, 31],
 [11, 32],
 [12, 30],
 [13, 35],
 [14, 19],
 [15, 15],
 [16, 17],
 [17,  0],
 [18,  13],
 [19,  13],
 [20,  14],
 [21,  14],
 [22,  15],
 [23,  27],
 [24, 32],
 [25, 20],
 [26, 18],
 [27, 31],
 [28, 36],
 [29, 35],
 [30, 17],
 [31, 15],
 [32, 11],
 [33, 11],
 [34, 0]]

matplot(np.array(data44))

# elif video_path:
#         import cv2
#         # Video capture
#         vcapture = cv2.VideoCapture(video_path)
#         width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = vcapture.get(cv2.CAP_PROP_FPS)

#         # Define codec and create video writer
#         file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
#         vwriter = cv2.VideoWriter(file_name,
#                                   cv2.VideoWriter_fourcc(*'MJPG'),
#                                   fps, (width, height))

#         count = 1
#         success = True
#         while success:
#             #print("frame: ", count)
#             # Read next image
#             success, image = vcapture.read()
#             if success:
#             	if count % 7500 == 0:
# 	                # OpenCV returns images as BGR, convert to RGB
# 	                image = image[..., ::-1]
# 	                # Detect objects
# 	                r = model.detect([image], verbose=0)[0]
# 	                # Color splash
# 	                num = r['rois'].shape[0]
# 	                contn = count//7500
# 	                if num <= 5:
# 	                	show = np.concatenate((show,[[contn,num+2]]),axis=0)
# 	                elif num <= 10:
# 	                	show = np.concatenate((show,[[contn,num+4]]),axis=0)
# 	                elif num <= 15:
# 	                	show = np.concatenate((show,[[contn,num+6]]),axis=0)
# 	                elif num <= 20:
# 	                	show =np.concatenate((show,[[contn,num+10]]),axis=0)
# 	                print(show)

# 	                count += 1
#         vwriter.release()
#     matplot(np.delete(show,[0,1],axis=0))