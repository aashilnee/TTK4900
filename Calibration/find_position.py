import calibration
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import triangulation as tri
import time
import matplotlib.patches as mpatches



from stereo_calibration import StereoManager, CameraParameters, StereoParameters

sm = StereoManager()
sm.load_calibration("/Users/aashi/Documents/stereoMaps/stereoMap_padding_4.pickle")



# TODO test images
img_l_path = 'test_images/left_2178.png' #'test_images/1_left.bmp'
img_r_path = 'test_images/right_2178.png' #'test_images/1_right.bmp'

img_left = cv2.imread(img_l_path)
img_right = cv2.imread(img_r_path)

################## CALIBRATION #########################################################
#img_right, img_left = calibration.undistortRectify(img_right, img_left)
########################################################################################

# calibration, bare med pickle
#img_left, img_right = sm.rectify_images(img_left, img_right)


######## POINTS #######################################################
#center_left = [(444, 156), (519, 158), (439, 207), (426, 438), (676, 168), (606, 452)] # (x,y) 1.bmp, test 1
#center_right = [(403, 222), (471, 222), (398, 270), (387, 493), (623, 229), (555, 510)] # (x,y) 1.bmp

#center_left = [(264,101), (208,564), (1021,463), (803,137), (756,524), (709,406)]  # 24.bmp, test 2
#center_right = [(183,185), (70,573), (878,540), (700,192), (507,583), (488,465)]   # 24.bmp

#center_left = [(817, 36), (948, 92), (1078, 165), (771, 137), (1051, 290), (705, 269)]   # 14.bmp, test 3
#center_right = [(775, 88), (906, 139), (1050, 210), (724, 193), (1016, 349), (662, 328)]  # 14.bmp

#center_left = [(326, 351), (588, 558), (335, 574), (549, 341), (542, 562), (331, 509)]   # 12.bmp, test 4
#center_right = [(311, 409), (549, 614), (310, 624), (522, 401), (505, 618), (307, 562)]  # 12.bmp

#center_left = [(661, 102), (509, 234), (624, 322), (814, 392), (350, 402), (1180, 375)]   # 200, fiskebilde 1
#center_right = [(591, 179), (463, 309), (569, 397), (770, 467), (320, 473), (1144, 452)]  # 200

#center_left = [(1101, 183), (1093, 392), (190, 115), (685, 346), (195, 317), (394, 399)]   # 1994, fiskebilde 2
#center_right = [(953, 254), (833, 471), (167, 195), (628, 420), (166, 393), (367, 472)]    # 1994

#center_left = [(715, 109), (750, 146), (809, 160), (233, 380), (863, 314), (1173, 388)]   # 338, fiskebilde 3
#center_right = [(636, 183), (701, 221), (775, 232), (194, 454), (829, 389), (1139, 466)]  # 338

center_left = [(608, 636), (900, 338), (1056, 36), (277, 155), (213, 330), (100, 84)]   # 2178, fiskebilde 4
center_right = [(201, 693), (767, 415), (940, 101), (249, 235), (180, 405), (81, 165)]  # 2178

color =  (0,0,255) # (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

print("center left: ", center_left)
print("center left: ", center_left[0])




######################### PLOT ##################################################
'''
left_x = []
left_y = []

right_x = []
right_y = []

for i, j in zip(center_left, center_right):
    #print("center left: ", center_left[i][0])
    #print(i[0])
    left_x.append(i[0])
    left_y.append(i[1])
    #print(left_y)

    right_x.append(j[0])
    right_y.append(j[1])

    colors = [np.random.randint(0, 10) for i in range(len(center_left))]  # tODO

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(left_x, left_y)
ax2.scatter(right_x, right_y, c=colors) # TODO c=np.random.rand(len(right_x),3)
ax1.axis([0, 1280, 0, 720])
ax2.axis([0, 1280, 0, 720])
plt.show()

c = np.random.rand(len(right_x), 3)
print("c: ", c)
'''
# TODO ##################


'''
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(left_x, left_y, 'ro')
ax2.plot(right_x, right_y, 'o', color)
ax1.axis([0, 1280, 0, 720])
ax2.axis([0, 1280, 0, 720])
plt.show()
'''
################################### DRAW ##################################
'''
for center_l in center_left:
    im2show_left = cv2.circle(img_left, center_l, 3, color, 5)
    cv2.putText(im2show_left, str(center_l), (center_l[0]-40, center_l[1] - 15), cv2.FONT_ITALIC, 0.6, color, 2)

for center_r in center_right:
    im2show_right = cv2.circle(img_right, center_r, 3, color, 5)
    cv2.putText(im2show_right, str(center_r), (center_r[0]-40, center_r[1] - 15), cv2.FONT_ITALIC, 0.6 , color, 2)
'''
left_xx = []
left_yy = []

right_xx = []
right_yy = []
col = []

stored_z = []
stored_x = []
stored_y = []

j = 1
i = 0
for center_l, center_r in zip(center_left, center_right):
    #colorss =[np.random.randint(0, 255) for i in range(len(center_l))] #(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # TODO fiskebilde
    #colors = [np.random.randint(0, 255) for i in range(len(center_left))]  # tODO

    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    color = (0,0,200)

    colors = np.random.rand(len(right_xx)+1, 3)


    im2show_left = cv2.circle(img_left, center_l, 3, color, 5)
    #cv2.putText(im2show_left, str(center_l), (center_l[0] - 40, center_l[1] - 15), cv2.FONT_ITALIC, 0.6, color, 2)
    cv2.putText(im2show_left, str(j), (center_l[0], center_l[1] - 10), cv2.FONT_ITALIC, 0.6, color, 2)


    im2show_right = cv2.circle(img_right, center_r, 3, color, 5)
    #cv2.putText(im2show_right, str(center_r), (center_r[0] - 40, center_r[1] - 15), cv2.FONT_ITALIC, 0.6, color, 2)
    cv2.putText(im2show_right, str(j), (center_r[0], center_r[1] - 10), cv2.FONT_ITALIC, 0.6, color, 2)



    ############################### POSITON ##################################

    left = center_l
    left_x = left[0]
    left_y = left[1]
    right = center_r
    right_x = right[0]
    right_y = right[1]
    sl_key = np.array([left], dtype=np.float32)
    sr_key = np.array([right], dtype=np.float32)
    xyz = sm.stereopixel_to_real(sl_key, sr_key)
    print("---------------------------------------------------------", j)
    print("xyz: ", xyz)

    stored_z.append(int(xyz[0][2]))

    stored_x.append(xyz[0][0])
    stored_y.append(-xyz[0][1])

    j += 1






    #### plot ############## MATPLOT ###############

    color = list(color)

    #color = ''.join(str(color).split(','))

    #color = color.format(*color)


    col.append(color)



    left_xx.append(center_l[0])
    left_yy.append(center_l[1])


    right_xx.append(center_r[0])
    right_yy.append(center_r[1])



    #print("left_ x ", left_xx)

################### BAR CHART, DEPTH ##############################
print("ZZ", stored_z)
print("XX", stored_x)
print("YY", stored_y)


plt.style.use('seaborn-whitegrid')

# Set the limit
plt.ylim(0,2000)


x = ['1', '2', '3', '4', '5', '6']
y_pos = stored_z

x_pos = [i+1 for i, _ in enumerate(x)]

plt.bar(x_pos, y_pos, color='green')


# function to add value labels
# adapted from https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i+1, y[i], y[i], ha = 'center')

# calling the function to add value labels
addlabels(x_pos, y_pos)

plt.xlabel("Selected points in image")
plt.ylabel("Estimated depth (mm)")
plt.title("Estimated depth from stereo images")
plt.show()

################### plot estimated x/y coordinates ##############################

#plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.plot(stored_x, stored_y, 'ro')

plt.axis([-560, 380, -120, 320])  # TODO HER

lab = 1
# zip joins x and y coordinates in pairs
for x,y in zip(stored_x,stored_y):

    label = str(lab)#"{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    lab += 1

plt.title("Estimated x/y coordinates from stereo images")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


'''
np.random.seed(19680801)

fig, (ax1, ax2) = plt.subplots(1, 2)
for test in ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:red', 'tab:pink']:
    #ax1.scatter(left_x, left_y)
    #ax2.scatter(right_x, right_y, c=colors, ) # TODO c=np.random.rand(len(right_x),3)
    ax1.scatter(left_x, left_y, c=test, s=100, label=test,
                alpha=0.3, edgecolors='none')
    ax2.scatter(right_x, right_y, c=test, s=100, label=test,
               alpha=0.3, edgecolors='none')
    ax1.axis([0, 1280, 0, 720])
    ax2.axis([0, 1280, 0, 720])

ax2.legend()
ax2.grid(True)

plt.show()
'''



fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(left_xx, left_yy, c='b', s=70, label='test',
        alpha=0.8, edgecolors='none')
ax2.scatter(right_xx, right_yy, c='b', s=70, label='test',
            alpha=0.8, edgecolors='none')
ax1.axis([0, 1280, 0, 720])
ax2.axis([0, 1280, 0, 720])

#ax2.legend()
ax2.grid(True)
ax1.grid(True)

plt.show()

############################### POSITON ##################################
for i in enumerate(center_left):
    #print(i)
    #print(i[0])

    num = i[0]

    left = center_left[num]
    left_x = left[0]
    left_y = left[1]

    right = center_right[num]
    right_x = right[0]
    right_y = right[1]



    sl_key = np.array([left], dtype=np.float32)
    sr_key = np.array([right], dtype=np.float32)

    xyz = sm.stereopixel_to_real(sl_key, sr_key)

    print("---------------------------------------------------------", left, right)

    print("xyz: ", xyz)



    ###### TIDLIGERE 3D-estimering ###################################################
    '''
    disparity = left_x - right_x

    #print(left, disparity)

    # Stereo camera setup parameters
    baseline = 122  # Distance between cameras [cm]
    # f = 1.8             # Camera lens focal length [mm]
    FOV = 70  # Camera field of view in the horizontal plane [degrees]

    depth = tri.find_depth(disparity, img_right, img_left, baseline, FOV)


    print("---------------------------------------------------------", left, disparity)
    print("depth: ", str(depth))
    '''
    ####################################################################################

'''
    left = center_left[i]
    left_x = left[0]
    right = center_right[0]

    print(left)
'''


#TODO FINNE PUNKT ############################################
'''
img = np.hstack((img_left, img_right))

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)

# displaying the image
cv2.imshow('image', img)

# setting mouse handler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event)

# wait for a key to be pressed to exit
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()

'''
#TODO FINNE PUNKT ############################################



############ visualize images ################################
#img = np.hstack((img_left, img_right)) #TODO visualisere  begge bilder, kommenter ut når bare ett bilde !!!!!!


# Save image
path = 'test_images/test_results'


vis_img =  img_left.copy() # img.copy() TODO begge bilder
cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # kan skalere bilde i imshow
cv2.imshow("img", vis_img)



k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite(os.path.join(path, 'test_2178_points_single.jpg'), img_left)  #  img   TODO begge bilder
    cv2.destroyAllWindows()



