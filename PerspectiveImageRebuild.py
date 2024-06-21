# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:06:33 2023

@author: Ahmed H. Hanfy
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import screeninfo
import glob
import sys
import os
px = 1/plt.rcParams['figure.dpi']

LineName = ["Z Projection line",
            "Y Projection line",
            "Chord line",
            "Inclined Line"]


# %% OpenCV colors
class CVColor:
    # Define class variables for commonly used colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    GREENBLUE = (255, 128, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    DARK_CYAN = (0, 139, 139)
    MAGENTA = (255, 0, 255)
    FUCHSIPINK = (255, 128, 255)
    GRAY = (128, 128, 128)
    ORANGE = (0, 128, 255)
# %% Inputs

# imgPath = '2023_01_23\\*.JPG'
# imgPath = '2023_02_07\\*.JPG'
# imgPath = 'Oil Visualization\\P3-Ref\\*.JPG'
# imgPath = 'D:\\PhD\\TEAMAero\\2023_02_07 - Oil visualization (FullyOpen and HalfOpen)\\Half-open\\With suction\\Test 7\\*.JPG'
# imgPath = r'D:\PhD\TEAMAero\TFAST oil\Oil Test 2\*.jpg'
# imgPath = r'D:\TFAST\TEAMAero experiments\Roughness study\Smooth profile (P1)\2023_05_25\Test 18 Oil (100mm)\*.jpg'
# imgPath = r'..\D4.1\100mm.jpg'


# imgPath = r'..\D4.1\35mm.jpg'
# imgPath = '2022_09_14\\Oil Test 2\\*.JPG'
imgPath = r'D:\TFAST\TEAMAero experiments\Roughness study\Smooth profile (P1)\2023_05_25\Test 18 Oil (100mm)\*.jpg'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\Roughness study\\Rough profile (P4)\\2023_04_24\\Test 16-OH\\*.jpg'
# imgPath = 'C:\\Users\\Hady-PC\\Desktop\\PhD\\TFAST\\TEAMAero Experiments\\2023_04_24\\Test 15-OD\\*.jpg'
# imgPath = 'D:\\TFAST\\TEAMAero experiments\\2023_05_10\\Test 17-OH 35mm\\*.jpg'
imShp = (1500, 1000)

Vz, AngleZ = [(), []]
Vy, AngleY = [(), []]
near_chord_points = []
far_chord_points = []
# 35mm
# Vz, AngleZ = [(622, -597), [-68.44998641717251, 59.57217944927443]]
# Vy, AngleY = [(1250, 10000), [85.100757508169, -89.90]]
# near_chord_points = [(93, 742), (1495, 890)]
# far_chord_points = [(281, 267), (1192, 374)]
# 100mm
# Vz, AngleZ = [(1604, -2429),[-63.675013533377644, -86.01227442754819]]
# Vy, AngleY = [(5656, 67172),[81.73003841301885, 82.92801073407301]]
# near_chord_points=[(82, 647), (1379, 795)]
# far_chord_points=[(287, 231), (1409, 356)]
# imShp = (1350,900)
# imShp = (1600,900)
SPI = 2  # Scale Point Index
ProfileSurfacePath = r'..\ProfileSurface3.csv'
CordLen = 99.549
# vertical distance between leading and trailing (on the same scale as cord)
LeaadingToTrailingH = 18.88657
SpanLength = 100
file_name = 'TransformedImages-test'
cir_radius = 1
# %% Functions


def screenMidLoc(shp):
    screen = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height
    # x_pos = (screen_width - shp[1]) // 2 + screen_width
    x_pos = (screen_width - shp[1]) // 2
    y_pos = (screen_height - shp[0]) // 2

    return x_pos, y_pos


def XCheck(x, Shp, slope, a):
    if   x >= 0 and x <= Shp[1]:                           p2 = (x, Shp[0])
    elif x >= 0 and x >  Shp[1]: y2 = int(Shp[1]*slope+a); p2 = (Shp[1],y2)
    elif x <  0 and x <= Shp[1]: y2 = int(a);              p2 = (0,y2)
    return p2


def InclinedLine(P1, P2=(), imgShape=(), slope=None):
    if len(imgShape) < 1:
        print('Image shape is not provided, program aborting ...')
        sys.exit()

    if len(P2) > 0 and slope is None:
        dx = P1[0]-P2[0];   dy = P1[1]-P2[1]
        if dx != 0: slope = dy/dx
    elif len(P2) == 0 and slope is np.inf: dx = 0;
    else: dx = -1 

    if slope != 0 and slope is not None and slope is not np.inf:
        a = P1[1] - slope*P1[0]
        Xmax = int((imgShape[0]-a)/slope)
        Xmin = int(-a/slope)
        if Xmin >= 0 and Xmin <= imgShape[1]:
            p1 = (Xmin, 0)
            p2 = XCheck(Xmax, imgShape, slope, a)
        elif Xmin >= 0 and Xmin > imgShape[1]:
            y = int(imgShape[1]*slope+a)
            p1 = (imgShape[1], y)
            p2 = XCheck(Xmax, imgShape, slope, a)
        else:
            y1 = int(a)
            p1 = (0, y1)
            p2 = XCheck(Xmax, imgShape, slope, a)
        return p1, p2, slope, a
    elif dx == 0:
        return (P1[0], 0), (P1[0], imgShape[0]), np.Inf, 0
    else:
        return (0, P1[1]), (imgShape[1], P1[1]), 0, P1[1]


def extract_coordinates(event, x, y, flags, parameters):
    # Record starting (x,y) coordinates on left mouse button click and draw
    # line that cross allover the image and store it in a global variable in
    # case of Horizontal or Vertical lines it takes the average between points
    # Drawing steps: 1- push the left mouse on the first point
    # .............. 2- pull the mouse cursor to the second point
    # .............. 3- the software will draw a thick red line (indecating the
    # ................. mouse locations) and green line indecating the
    # ................. generated averaged line
    # .............. 4- to confrim press left click anywhere on the image, or
    # ................. to delete the line press right click anywhere on the
    # ................. image
    # .............. 5- press anykey to proceed
    # Output: array [Point 1, point 2, slope, y-intercept]
    global line_coordinates
    global TempLine; global Reference
    global Temp;   global clone
    global ClickCount

    if   parameters[0] == "Z Projection line": color = CVColor.BLUE # Blue
    elif parameters[0] == "Y Projection line": color = CVColor.GREEN # Green
    elif parameters[0] == "X Projection line": color = CVColor.RED # Red
    else: color = CVColor.YELLOW  # ................................. Yellow

    # Record starting (x,y) coordinates on left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        ClickCount += 1
        if len(TempLine) == 2:
            line_coordinates = TempLine
        elif len(TempLine) == 0: TempLine = [(x, y)]

    # Record ending (x,y) coordintes on left mouse bottom release
    elif event == cv2.EVENT_LBUTTONUP:
        if len(TempLine) < 2:
            TempLine.append((x, y))
            # print('Starting: {}, Ending: {}'.format(TempLine[0],TempLine[1]))

            # Draw temprary line
            cv2.line(Temp, TempLine[0], TempLine[1], CVColor.ORANGE, 2)
            P1, P2, m, a = InclinedLine(TempLine[0], TempLine[1],
                                        parameters[1])
            cv2.line(Temp, P1, P2, color, 1)

            cv2.imshow(parameters[0], Temp)

        elif ClickCount == 2:
            # storing the vertical line
            Temp = clone.copy()
            cv2.imshow(parameters[0], clone)
            P1, P2, m, a = InclinedLine(line_coordinates[0],
                                        line_coordinates[1], parameters[1])
            cv2.line(Temp, P1, P2, color, 1)
            avg = [P1, P2, m, a]

            Reference.append(avg)
            print(f'stored line coordinates: {line_coordinates[0]}, {line_coordinates[1]}')
            clone = Temp.copy()
            cv2.imshow(parameters[0], clone)

    # Delete draw line before storing
    elif event == cv2.EVENT_RBUTTONDOWN:
        TempLine = []
        if ClickCount > 0: ClickCount -= 1
        Temp = clone.copy()
        cv2.imshow(parameters[0], Temp)


def LineDraw(img, lineType, LineNameInd, Intialize=False):
    global line_coordinates
    global TempLine; global Reference
    global Temp;   global clone
    global ClickCount
    clone = img.copy()
    Temp = clone.copy()
    TempLine = []
    ClickCount = 0
    if Intialize:
        Reference = []
        line_coordinates = []

    shp = img.shape
    prams = [LineName[LineNameInd], shp, lineType]

    # win_x, win_y = screenMidLoc(shp)
    # cv2.namedWindow(LineName[LineNameInd], cv2.WINDOW_NORMAL)
    # cv2.moveWindow(LineName[LineNameInd], win_x, win_y)
    cv2.imshow(LineName[LineNameInd], clone)
    cv2.setMouseCallback(LineName[LineNameInd], extract_coordinates, prams)
    # Wait until user press some key
    cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
    return clone


def IncParameters(P1, P2):
    dx = P1[0]-P2[0]
    dy = P1[1]-P2[1]
    if dy != 0 and dx != 0:
        slope = dy/dx
        a = P1[1] - slope*P1[0]
        return slope, a
    elif dx == 0:
        return 0, np.Inf
    else:
        return 0, 0


def IntersectionPoint(M, A, Ref):
    theta1 = np.arctan(M[0])*180/np.pi
    theta2 = np.arctan(M[1])*180/np.pi
    Pint = []; Xint = np.inf; Yint = np.inf
    if theta1 != 0 and theta2 != 0:
        if theta1 - theta2 != 0:
            Xint = (A[1]-A[0])/(M[0]-M[1])
            Yint = M[0]*Xint + A[0]
        else: print('Lines are parallel')
    elif theta1 == 0 and theta2 != 0:
        Xint = Ref[0]
        Yint = M[1]*Xint + A[1]
    elif theta2 == 0 and theta1 != 0:
        Xint = Ref[1]
        Yint = M[0]*Xint + A[0]
    else: print('Lines are parallel')
    Pint = (round(Xint), round(Yint))
    return Pint, [theta1, theta2]


def FindProfilePoints(LeadingChordPint, TrailingChordPint, ChordVanish, Vy,
                      ProfilePoints, CordLen, AOA, Aqu=0):
    global clone
    # ChordVanish [VanishPoint, ChordAngleInDeg]
    # Locating the points on chord
    ChordInPixel = np.sqrt((LeadingChordPint[0]-TrailingChordPint[0])**2+(LeadingChordPint[1]-TrailingChordPint[1])**2)
    PLPChint = np.sqrt((ChordVanish[0][0]-LeadingChordPint[0])**2+(ChordVanish[0][1]-LeadingChordPint[1])**2)
    PTPChint = np.sqrt((ChordVanish[0][0]-TrailingChordPint[0])**2+(ChordVanish[0][1]-TrailingChordPint[1])**2)
    NewProfilePoints = []
    for i in range(len(ProfilePoints)):
        # Find point distribution on the chordline
        Nom = ChordInPixel*PLPChint*ProfilePoints[i][0]
        PxdashPTdash = CordLen - ProfilePoints[i][0]
        if ChordVanish[0][0] > TrailingChordPint[0]:
            LPx = Nom/(CordLen*PLPChint - PxdashPTdash*ChordInPixel)
        elif ChordVanish[0][0] < LeadingChordPint[0]:
            LPx = -Nom/(ProfilePoints[i][0]*ChordInPixel-PTPChint*CordLen)
        Pxx = round(LeadingChordPint[0]+LPx*np.cos(ChordVanish[1]*np.pi/180))
        Pxy = round(LeadingChordPint[1]+LPx*np.sin(ChordVanish[1]*np.pi/180))
        Px = (Pxx, Pxy)
        # Find Point distribution on Y-Vanishing lines
        myPx, ayPx = IncParameters(Vy, Px)
        mxPT, axPT = IncParameters(Vx, TrailingChordPint)

        VxPxint, VxPxAngle = IntersectionPoint([myPx, mxPT], [ayPx, axPT],
                                               [LeadingChordPint[0],
                                                LeadingChordPint[1]])

        PxPxVx = np.sqrt((Px[0]-VxPxint[0])**2+(Px[1]-VxPxint[1])**2)
        PxVy = np.sqrt((Px[0] - Vy[0])**2+(Px[1] - Vy[1])**2)
        PxdashPxVxdash = PxdashPTdash * np.sin(AOA)
        PpdashPxVxdash = ProfilePoints[i][1]+PxdashPxVxdash
        if PxdashPxVxdash*PxVy - PpdashPxVxdash*PxPxVx != 0:
            LPy = PxPxVx*PxVy*ProfilePoints[i][1]/(PxdashPxVxdash*PxVy-PpdashPxVxdash*PxPxVx)
            Ppx = round(LPy * np.cos(VxPxAngle[0] * np.pi / 180) + Pxx)
            if VxPxAngle[0] > 0:
                Ppy = round(Pxy - LPy * np.sin(VxPxAngle[0] * np.pi / 180))
            else:
                Ppy = round(Pxy - LPy * np.sin(-VxPxAngle[0] * np.pi / 180))
            Pp = (Ppx, Ppy)
            cv2.circle(clone, Pp, radius=cir_radius,
                       color=CVColor.BLUE, thickness=2)
            cv2.circle(clone, Pp, radius=cir_radius,
                       color=CVColor.RED, thickness=-1)
            cv2.putText(clone, f'{i+Aqu}', (Pp[0]-10, Pp[1]-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            NewProfilePoints.append(Pp)
        else:
            print('Unsupported point', ProfilePoints[i])
    return NewProfilePoints


def RebuildPrespactive(img, NearProfilePnt, FarProfilePnt, ProfilePoints, SPI,
                       LengthToPixelRatio, ScaledPlot=False):
    Npt = len(NearProfilePnt)
    Fpt = len(FarProfilePnt)
    SpanPixLen = np.sqrt((FarProfilePnt[SPI][0]-NearProfilePnt[SPI-1][0])**2+(FarProfilePnt[SPI][1]-NearProfilePnt[SPI-1][1])**2)
    Hieght = round(SpanPixLen)
    TransformedImage = np.zeros((Hieght, 1, 3), np.uint8)
    if Npt == Fpt:
        Npt = len(NearProfilePnt)
        FullWidth = 0
        for i in range(Npt-1):
            input_pts = np.float32([FarProfilePnt[i],
                                    FarProfilePnt[i+1],
                                    NearProfilePnt[i+1],
                                    NearProfilePnt[i]])

            ChordPixellen = np.sqrt((ProfilePoints[i+1][0]-ProfilePoints[i][0])**2+(ProfilePoints[i+1][1]-ProfilePoints[i][1])**2)
            Width = round(LengthToPixelRatio*ChordPixellen)
            FullWidth += Width
            output_pts = np.float32([[0, 0],
                                     [Width - 1, 0],
                                     [Width - 1, Hieght - 1],
                                     [0, Hieght - 1]])
            # Compute the perspective transform M
            M = cv2.getPerspectiveTransform(input_pts, output_pts)
            out = cv2.warpPerspective(img, M, (Width, Hieght),
                                      flags=cv2.INTER_LINEAR)

            TransformedImage = np.concatenate((TransformedImage, out), axis=1)

        if ScaledPlot:
            fig, ax = plt.subplots(figsize=(int(FullWidth*px),
                                            int(SpanPixLen*px)))
            ax.imshow(TransformedImage)
        return TransformedImage, (FullWidth, round(SpanPixLen))
    else:
        print('Profile points are not balanced, program aborted!')


def ImportSchlierenImages(path):
    img_list = []
    n = 0
    files = sorted(glob.glob(path), reverse=False)
    n1 = len(files)
    if n1 > 0:
        for name in files:
            with open(name):
                img = cv2.imread(name)
                img_list.append(img)
                # img_list.append(img[1200:3300,1350:4500])
            n += 1
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(n/(n1/20)),
                                               int(5*n/(n1/20))))
            sys.stdout.flush()
        print('')
        print('Number of imported images: ', n)
    else:
        print('No files found!')
        sys.exit()
    return img_list, n


# %% Code
if __name__ == '__main__':
    imgs, n = ImportSchlierenImages(imgPath)
    # img = cv2.imread(imgPath)
    OimShp = imgs[0].shape
    imgR = cv2.resize(imgs[0], imShp)
    new_img_shp = imgR.shape
    # imgR = cv2.cvtColor(cv2.Canny(imgR, 50, 100), cv2.COLOR_BGR2RGB)
    ProfilePoints = np.genfromtxt(ProfileSurfacePath, delimiter=",",
                                  dtype=float)

    P0 = (int(imShp[0]/2), int(imShp[1]/2))
    # Profile inclination
    AOA = np.arcsin(LeaadingToTrailingH / CordLen)
    print('profile angle of attack: ', round(180*AOA/np.pi, 2), 'deg')

    # principle point of the image (center of the image)
    cv2.circle(imgR, P0, radius=3, color=CVColor.GREEN, thickness=-1)

    # Draw Projection lines to find Z-vanishing point
    if len(AngleZ) == 0:
        LineDraw(imgR,  'Inc', 0, Intialize=True)  # ...... Line 1, Ref[0]
        LineDraw(clone, 'Inc', 0)  # ...................... Line 2, Ref[1]
    else:
        Reference = []
        Line1slope = np.tan(AngleZ[0]*np.pi/180)
        p1, p2, a, b = InclinedLine(Vz, imgShape=new_img_shp, slope=Line1slope)
        clone = imgR.copy()
        Reference.append([p1, p2, a, b])
        cv2.line(clone, p1, p2, CVColor.BLUE, 1)

        Line2slope = np.tan(AngleZ[1]*np.pi/180)
        p1, p2, a, b = InclinedLine(Vz, imgShape=new_img_shp, slope=Line2slope)
        Reference.append([p1, p2, a, b])
        cv2.line(clone, p1, p2, CVColor.BLUE, 1)

    if len(Reference) > 1:
        a1  =  Reference[0][3]; a2 = Reference[1][3]
        m1  =  Reference[0][2]; m2 = Reference[1][2]

        # Finding the Z-vanishing point (PZ)
        Vz, AngleZ = IntersectionPoint([m1, m2], [a1, a2],
                                       [Reference[0][0][0],
                                        Reference[1][0][0]])
        print("Z-vanishing point: ", Vz, "Z-rays angle", AngleZ)

        for i, s in enumerate(AngleZ):
            if s < 0: AngleZ[i] += 180
        AngleZ.sort()
        Dt = AngleZ[1]-AngleZ[0]
        if len(Vz) > 1:
            cv2.circle(clone, Vz, radius=3, color=CVColor.RED, thickness=-1)
        else:
            print('Lines are Parallel!')

        # Draw Projection lines to find Y-vanishing point
        if len(AngleY) == 0:
            LineDraw(clone, 'Inc', 1)  # ....................... Line 3, Ref[2]
            LineDraw(clone, 'Inc', 1)  # ....................... Line 4, Ref[3]
        else:
            Line1slope = np.tan(AngleY[0] * np.pi / 180)
            p1, p2, a, b = InclinedLine(Vy, imgShape=new_img_shp,
                                        slope=Line1slope)

            Reference.append([p1, p2, a, b])
            cv2.line(clone, p1, p2, CVColor.GREEN, 1)
            Line2slope = np.tan(AngleY[1]*np.pi/180)
            p1, p2, a, b = InclinedLine(Vy, imgShape=new_img_shp,
                                        slope=Line2slope)
            Reference.append([p1, p2, a, b])
            cv2.line(clone, p1, p2, CVColor.GREEN, 1)

        # Draw Projection of profile chord
        if len(near_chord_points) == 0:
            LineDraw(clone, 'Inc', 2)  # .... Line 5, Ref[4]
        else:
            p1, p2, a, b = InclinedLine(near_chord_points[0],
                                        near_chord_points[1],
                                        imgShape=new_img_shp)
            Reference.append([p1, p2, a, b])
            cv2.line(clone, p1, p2, CVColor.YELLOW, 1)

        if len(far_chord_points) == 0:
            LineDraw(clone, 'Inc', 2)  # .... Line 6, Ref[5]
        else:
            p1, p2, a, b = InclinedLine(far_chord_points[0],
                                        far_chord_points[1],
                                        imgShape=new_img_shp)
            Reference.append([p1, p2, a, b])
            cv2.line(clone, p1, p2, CVColor.YELLOW, 1)

        if len(Reference) > 5:
            m3 = Reference[2][2]; a3 = Reference[2][3]  # ...... Line 3, Ref[2]
            m4 = Reference[3][2]; a4 = Reference[3][3]  # ...... Line 4, Ref[3]
            m5 = Reference[4][2]; a5 = Reference[4][3]  # ...... Line 5, Ref[4]
            m6 = Reference[5][2]; a6 = Reference[5][3]  # ...... Line 6, Ref[5]

            # Finding Y-vanishing point
            Vy, AngleY = IntersectionPoint([m3, m4], [a3, a4],
                                           [Reference[2][0][0],
                                            Reference[3][0][0]])

            # Finding Chord vanishing point
            PChint, IntAngleCh = IntersectionPoint([m5, m6], [a5, a6],
                                                   [Reference[4][0][0],
                                                    Reference[5][0][0]])

            IntersectionVeri = 0
            if len(Vy) > 1 and len(PChint) > 1:
                cv2.circle(clone, Vy, radius=3,
                           color=CVColor.YELLOW, thickness=-1)
                print("Y-vanishing point: ", Vy, "Y-rays angle", AngleY)
                # Line connecting two vanishing points (Z,Y)
                cv2.line(clone, Vz, Vy, CVColor.GRAY, 1)

                # Calculation of the focal length
                dz = np.asarray(Vz)-np.asarray(P0)
                dy = np.asarray(Vy)-np.asarray(P0)
                f = np.sqrt(-np.dot(dz, dy))
                print('Focal length is:', round(f, 2), 'pixel')

                # World coordinates of vanishing points
                Vzdash = np.asarray(Vz + (f,))
                Vydash = np.asarray(Vy + (f,))
                P0dash = np.asarray(P0 + (0,))

                # Finding the third vanishing point
                dzdash = Vzdash - P0dash
                dydash = Vydash - P0dash
                Vxdash = np.cross(dzdash, dydash)
                Vx = (int((Vxdash[0] * f / Vxdash[2]) + P0[0]),
                      int((Vxdash[1] * f / Vxdash[2]) + P0[0]))
                print("X-vanishing point: ", Vx)

                cv2.circle(clone, PChint, radius=3,
                           color=CVColor.YELLOW, thickness=-1)
                print("chord lines intersection point: ", PChint)

                # Finding intersection points between Y projection lines
                # and near profile chord
                # left z-line (1)
                P4int, IntAngle4 = IntersectionPoint([m5, m1], [a5, a1],
                                                     [Reference[4][0][0],
                                                      Reference[0][0][0]])

                # cv2.putText(clone, 'P4', P4int,cv2.FONT_HERSHEY_SIMPLEX, 1,0)
                # Right z-line (2)
                P5int, IntAngle5 = IntersectionPoint([m5, m2], [a5, a2],
                                                     [Reference[4][0][0],
                                                      Reference[1][0][0]])
                # cv2.putText(clone,'P5',P5int, cv2.FONT_HERSHEY_SIMPLEX, 1, 0)

                # Finding intersection points between Y projection lines
                # and far profile chord # left z-line
                P6int, IntAngle6 = IntersectionPoint([m6, m1], [a6, a1],
                                                     [Reference[5][0][0],
                                                      Reference[0][0][0]])
                # cv2.putText(clone, 'P6', P6int, cv2.FONT_HERSHEY_SIMPLEX,1,0)
                # right z-line
                P7int, IntAngle7 = IntersectionPoint([m6, m2], [a6, a2],
                                                     [Reference[5][0][0],
                                                      Reference[1][0][0]])
                # cv2.putText(clone, 'P7', P7int, cv2.FONT_HERSHEY_SIMPLEX,1,0)

                IntersectionVeri += 1
                if len(P4int) > 1:
                    cv2.circle(clone, P4int, radius=3,
                               color=CVColor.YELLOW, thickness=-1)
                    my1, ay1 = IncParameters(Vy, P4int)
                    cv2.line(clone, Vy, P4int, CVColor.GREEN, 1)
                    print("Near Chord intersection with Y-projection line 1: ",
                          P4int)
                    IntersectionVeri += 1
                if len(P5int) > 1:
                    cv2.circle(clone, P5int, radius=3,
                               color=CVColor.YELLOW, thickness=-1)
                    my2, ay2 = IncParameters(Vy, P5int)
                    cv2.line(clone, Vy, P5int, CVColor.GREEN, 1)
                    cv2.line(clone, Vx, P5int, CVColor.RED, 1)

                    print("Near Chord intersection with Y-projection line 2: ",
                          P5int)

                    IntersectionVeri += 1
                if len(P6int) > 1:
                    cv2.circle(clone, P6int, radius=3,
                               color=CVColor.YELLOW, thickness=-1)
                    my3, ay3 = IncParameters(Vy, P6int)
                    cv2.line(clone, Vy, P6int, CVColor.GREEN, 1)

                    print("Far Chord intersection with Y-projection line 1: ",
                          P6int)

                    IntersectionVeri += 1
                if len(P7int) > 1:
                    cv2.circle(clone, P7int, radius=3,
                               color=CVColor.YELLOW, thickness=-1)
                    my4, ay4 = IncParameters(Vy, P7int)
                    cv2.line(clone, Vy, P7int, CVColor.GREEN, 1)
                    cv2.line(clone, Vx, P7int, CVColor.RED, 1)

                    print("Far Chord intersection with Y-projection line 2: ",
                          P7int)

                    IntersectionVeri += 1

                if IntersectionVeri > 4:
                    # Point between upper x line and far y line of near profile
                    # Near chord
                    mx1, ax1 = IncParameters(Vx, P4int)
                    P5xint, IntAnglex5 = IntersectionPoint([mx1, my2],
                                                           [ax1, ay2],
                                                           [P4int[0],
                                                            P4int[1]])
                    cv2.circle(clone, P5xint, radius=3,
                               color=CVColor.YELLOW, thickness=-1)
                    # cv2.putText(clone, 'P5xint',
                    #             P5xint, cv2.FONT_HERSHEY_SIMPLEX, 1, 0)
                    cv2.line(clone, Vx, P5xint, CVColor.RED, 1)
                    cv2.line(clone, P5int, P5xint, CVColor.GREEN, 1)
                    if Vx[0] > P0[0]:
                        cv2.line(clone, P5xint, P4int, CVColor.RED, 1)
                    mx2, ax2 = IncParameters(Vx, P5int)
                    P4xint, IntAnglex4 = IntersectionPoint([mx2, my1],
                                                           [ax2, ay1],
                                                           [P5int[0],
                                                            P5int[1]])

                    cv2.circle(clone, P4xint, radius=3,
                               color=CVColor.YELLOW, thickness=-1)
                    # cv2.putText(clone, 'P4xint', P4xint,
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, 0)

                    NearProfile = FindProfilePoints(P4int, P5int,
                                                    [PChint, IntAngleCh[0]],
                                                    Vy, ProfilePoints,
                                                    CordLen, AOA)

                    FarProfile = FindProfilePoints(P6int, P7int,
                                                   [PChint, IntAngleCh[1]],
                                                   Vy, ProfilePoints, CordLen,
                                                   AOA, Aqu=len(NearProfile))
                    Npt = len(NearProfile)
                    HiResNearProfile = []
                    HiResFarProfile = []
                    ScaleCoef = OimShp[1]/imShp[0]
                    for i in range(Npt):
                        Xnear = round(ScaleCoef*NearProfile[i][0])
                        Ynear = round((ScaleCoef*NearProfile[i][1]))
                        HiResNearProfile.append((Xnear, Ynear))
                        cv2.circle(imgs[0], (Xnear, Ynear), radius=3,
                                   color=CVColor.YELLOW, thickness=-1)

                        Xfar = round(ScaleCoef*FarProfile[i][0])
                        Yfar = round(ScaleCoef*FarProfile[i][1])
                        HiResFarProfile.append((Xfar, Yfar))
                        cv2.circle(imgs[0], (Xfar, Yfar), radius=3,
                                   color=CVColor.YELLOW, thickness=-1)

                    # win_x, win_y = screenMidLoc(imShp)
                    # cv2.namedWindow('Model Domain', cv2.WINDOW_NORMAL)
                    # cv2.moveWindow('Model Domain', win_x, win_y)
                    segmants = np.zeros_like(clone, np.uint8)
                    for i in range(5):
                        pts = np.array([FarProfile[i],
                                        FarProfile[i+1],
                                        NearProfile[i+1],
                                        NearProfile[i]])
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(segmants, [pts], True, CVColor.RED,
                                      thickness=2)
                        cv2.fillPoly(segmants, [pts], CVColor.DARK_CYAN)
                    mask = segmants.astype(bool)
                    clone[mask] = cv2.addWeighted(clone, 0.3, segmants,
                                                  1 - 0.3, 0)[mask]

                    cv2.imshow('Model Domain', clone)
                    cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);

                    # win_x, win_y = screenMidLoc(OimShp)
                    # cv2.namedWindow('Full Domain', cv2.WINDOW_NORMAL)
                    # cv2.moveWindow('Full Domain', win_x, win_y)
                    cv2.imshow('Full Domain', imgs[0])
                    cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1);

                    # Calculating original distances beteen points
                    minLen = OimShp[1]
                    for i in range(Npt):
                        SpanPixLen = np.sqrt((HiResFarProfile[SPI][0]-HiResNearProfile[SPI-1][0])**2+(HiResFarProfile[SPI][1]-HiResNearProfile[SPI-1][1])**2)
                        if SpanPixLen < minLen: minLen = SpanPixLen
                    LengthToPixelRatio = SpanPixLen / SpanLength

                    # Output directory generator
                    Folders = imgPath.split('\\')
                    FileDirectory = ''
                    for i in range(len(Folders)-1):
                        FileDirectory += f'{Folders[i]}\\'
                    NewFileDirectory = os.path.join(FileDirectory, file_name)
                    if not os.path.exists(NewFileDirectory):
                        os.mkdir(NewFileDirectory)
                    print('Model Domain:' , u"stored \u2713" if cv2.imwrite(fr'{NewFileDirectory}\RefD.png', clone)   else "Failed !")
                    ImgSerial = 0
                    for i in imgs:
                        TransformedImage, ImgDim = RebuildPrespactive(i,
                                                                      HiResNearProfile,
                                                                      HiResFarProfile,
                                                                      ProfilePoints,
                                                                      SPI,
                                                                      LengthToPixelRatio, 
                                                                      ScaledPlot=True)

                        filename = fr'{NewFileDirectory}\TransformedImage-[{ImgSerial:04d}].jpg'
                        if TransformedImage.shape[0] > 0:
                            cv2.imwrite(filename, TransformedImage)
                            ImgSerial += 1
                    print('Output image dimenations: ',ImgDim,'px')
                else: print('Impossible lines!')
        else: print('Insufficient data!')
    else: print('Insufficient data!') 

# Draft code:
    # Divide an arc into equally angled lines =================================
    # dt = Dt/Ndiv
    # theta = IntAngle[1]
    # print(IntAngle[0],theta, Dt, dt)
    # for i in range(Ndiv-1):
    #     theta -= dt
    #     Mseg = np.tan(theta*np.pi/180)
    #     Aseg = Pint[1]-Mseg*Pint[0]
    #     x2 = (imShp[0]-Aseg)/Mseg
    #     PLim = (int(x2), int(imShp[0]))
    #     cv2.line(clone, Pint, PLim, (0,0,255), 1)
    # print(Pint)


# Finding Focal length using orthotriangles ===================================
# The perpendicular line from principle point on vanishing points line (PPzy)
    # mv1, av1 = IncParameters(Vz,Vy)
    # if mv1 != 0 and mv1 != np.inf:
    #     Pmv1 = -1/mv1
    #     Pav1 = P0[1]-Pmv1*P0[0]
    # elif mv1 == np.inf:
    #     Pmv1 = 0; Pav1 = P0[1]
    # else:
    #     Pmv1 = np.inf; Pav1 = 0

    # Pzy, Pv1Angle = IntersectionPoint([mv1,Pmv1],[av1,Pav1],[P0[0],P0[1]])
    # cv2.line(clone, P0, Pzy, (192,192,192), 1)

    # # PPzy line length
    # PPzy = np.sqrt((P0[0]-Pzy[0])**2+(P0[1]-Pzy[1])**2)

    # # Finding the distance between PPzy and the vanishing points Y and Z
    # VyPzy = np.sqrt((Pzy[0]-Vy[0])**2+(Pzy[1]-Vy[1])**2)
    # VzPzy = np.sqrt((Pzy[0]-Vz[0])**2+(Pzy[1]-Vz[1])**2)

    # Projection center distance from (PPzy)
    # OPzy = np.sqrt(VyPzy*VzPzy)

    # f = np.sqrt(OPzy**2-PPzy**2)

# Finding line with with the knowlage of angle
    # if IntersectionVeri > 4:
    #     AngY1 = [thetaY1,thetaY2];  AngC = [thetaY2,IntAngle3[0]]
    #     for i, s in enumerate(AngY1):
    #         if s < 0: AngY1[i] += 180
    #     AngY1.sort(); DAngY = AngY1[1]-AngY1[0]
    #     print("orthogonal Y-lines angle:", DAngY)
    #     for i, s in enumerate(AngC):
    #         if s < 0: AngC[i] += 180
    #     AngC.sort(); DAngC = AngC[1]-AngC[0]
    #     print("Cord to orthogonal Y-line 1:", DAngC)

    #     # Finding X projection lines
    #     XOrthLen = np.sqrt(CordLen**2 + LeaadingToTrailingH**2 - 2 * CordLen * LeaadingToTrailingH * np.cos(DAngC * np.pi/180))
    #     NewCAngle = np.arccos((CordLen**2 + XOrthLen**2 - LeaadingToTrailingH**2) / (2 * CordLen * XOrthLen))
    #     mx = np.tan((IntAngle3[0]-NewCAngle)*np.pi/180)
    #     ax = P4int[1]-mx*P4int[0]
    #     P6int, IntAngle4 = IntersectionPoint([m6,m1],[a6,a1],
    #                                  [Reference[5][0][0],Reference[0][0][0]])
    #     print("X-vanishing point: ",P6int)
    #     print(NewCAngle*180/np.pi)

# Apply point transformation from original lengths to the prespactive domain

    # cv2.line(clone, Vx, P4xint, (0,0,225), 1)
    # cv2.line(clone, P5int, P5xint, (0,225,0), 1)
    # if Vx[0] < P0[0]: cv2.line(clone, P4xint, P5int, (0,0,255), 1)

    # Far chord
    # mx3,ax3 = IncParameters(Vx,P6int)
    # P6xint, IntAnglex6 = IntersectionPoint([mx3,my4],[ax3,ay4],
    #                                        [P6int[0],P6int[1]])
    # cv2.circle(clone, P6xint, radius=3, color=(0, 255, 255), thickness=-1)
    # cv2.line(clone, Vx, P6xint, (0,0,225), 1)
    # cv2.line(clone, P7int, P6xint, (0,225,0), 1)
    # if Vx[0] > P0[0]: cv2.line(clone, P6xint, P6int, (0,0,255), 1)

    # Locating the points on chord
    # P4P5 = np.sqrt((P4int[0]-P5int[0])**2+(P4int[1]-P5int[1])**2)
    # P4PChint = np.sqrt((PChint[0]-P4int[0])**2+(PChint[1]-P4int[1])**2)
    # P5PChint = np.sqrt((PChint[0]-P5int[0])**2+(PChint[1]-P5int[1])**2)

    # for i in range(len(ProfilePoints)):
    #     # Find point distribution on the chordline
    #     Nom = P4P5*P4PChint*ProfilePoints[i][0]
    #     PxdashP5dash = CordLen - ProfilePoints[i][0]
    #     if PChint[0] > P5int[0]:
    #         LPx = Nom/(CordLen*P4PChint - PxdashP5dash*P4P5)
    #     elif PChint[0] < P4int[0]:
    #         LPx = -Nom/(ProfilePoints[i][0]*P4P5-P5PChint*CordLen)
    #     Pxx = round(LPx * np.cos(IntAngleCh[0] * np.pi / 180) + P4int[0])
    #     Pxy = round(P4int[1] + LPx * np.sin(IntAngleCh[0] * np.pi / 180))
    #     Px = (Pxx,Pxy)

    #     # Find Point distribution on Y-Vanishing lines
    #     myPx, ayPx = IncParameters(Vy,Px)
    #     VxPxint, VxPxAngle = IntersectionPoint([myPx,mx2],[ayPx,ax2],
    #                                            [P4int[0],P4int[1]])
    #     PxPxVx = np.sqrt((Px[0]-VxPxint[0])**2+(Px[1]-VxPxint[1])**2)
    #     PxVy = np.sqrt((Px[0]-Vy[0])**2+(Px[1]-Vy[1])**2)
    #     PxdashPxVxdash = PxdashP5dash * np.sin(AOA)
    #     PpdashPxVxdash = ProfilePoints[i][1]+PxdashPxVxdash
    #     if PxdashPxVxdash*PxVy - PpdashPxVxdash*PxPxVx != 0:
    #         LPy = PxPxVx*PxVy*ProfilePoints[i][1]/(PxdashPxVxdash*PxVy-PpdashPxVxdash*PxPxVx)
    #         Ppx = round(LPy * np.cos(VxPxAngle[0] * np.pi / 180) + Pxx)
    #         if VxPxAngle[0] > 0:
    #             Ppy = round(Pxy - LPy * np.sin(VxPxAngle[0] * np.pi / 180))
    #         else:
    #             Ppy = round(Pxy - LPy * np.sin(-VxPxAngle[0] * np.pi / 180))

    #         Pp = (Ppx,Ppy)
    #         print(ProfilePoints[i][1],Pp,LPy,VxPxAngle[0])
    #         cv2.circle(clone, Pp, radius=2, color=(255, 0, 255),thickness=-1)
