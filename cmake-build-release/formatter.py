import math
from math import sqrt

train_images_unread = open("train-images.idx3-ubyte", "rb")

train_images = train_images_unread.read()

train_images = train_images[16:len(train_images)]

nuls = [0]*784

def interpolate(a, b ,w):
    return a + w*(b-a)

def geta(x,y):
    if x == 0:
        if y < 0:
            return math.pi/2*3
        return math.pi/2
    ang = math.atan(y/x)
    if x < 0:
        ang += math.pi
    return ang

def center(sav):
    xs = 0
    ys = 0
    c = 0
    for x in range(28):
        for y in range(28):
            xs += x * sav[x+ 28*y]
            ys += y * sav[x+ 28*y]
            c += sav[x+ 28*y]
    xof = -(13.5-xs/c)
    yof = -(13.5-ys/c)
    nsav = []
    for y in range(28):
        for x in range(28):
            vals = []
            for yofi in [0, 1]:
                for xofi in [0,1]:
                    val = 0
                    if int(x+xof)+xofi >= 0 and int(x+xof)+xofi <= 27 and int(y+yof)+yofi >= 0 and int(y+yof)+yofi <= 27:
                        val = sav[int(x+xof) + xofi+ 28*(int(y+yof)+yofi)]
                    vals.append(val)
            y1 = interpolate(vals[0],vals[1], (x+xof)%1)
            y2 = interpolate(vals[2],vals[3], (x + xof) % 1)
            nsav.append(interpolate(y1, y2, (y+yof)%1))
    return nsav

scales = 10

def scale(sav):
    sf= 10
    for x in range(28):
        for y in range(28):
            if sav[x+28*y] >0.5:
                ang = geta(x-13.5, y-13.5)
                if ang >= math.pi/4*7 or ang < math.pi/4:
                    y1 = 13.5 + scales / (x - 13.5) * (y - 13.5)
                    dt = sqrt(scales * scales + (y1 - 13.5) * (y1 - 13.5))
                    if sf > dt / sqrt((x - 13.5) * (x - 13.5) + (y - 13.5) * (y - 13.5)):
                        sf = dt / sqrt((x - 13.5) * (x - 13.5) + (y - 13.5) * (y - 13.5))
                elif ang < math.pi/4*3:
                    x1 = 13.5 + scales / (y - 13.5) * (x - 13.5)
                    dt = sqrt(scales * scales + (x1 - 13.5) * (x1 - 13.5))
                    if sf > dt / sqrt((x - 13.5) * (x - 13.5) + (y - 13.5) * (y - 13.5)):
                        sf = dt / sqrt((x - 13.5) * (x - 13.5) + (y - 13.5) * (y - 13.5))
                elif ang < math.pi/3*5:
                    y1 = 13.5 + scales / (-1*(x - 13.5)) * (y - 13.5)
                    dt = sqrt(scales * scales + (y1 - 13.5) * (y1 - 13.5))
                    if sf > dt / sqrt((x - 13.5) * (x - 13.5) + (y - 13.5) * (y - 13.5)):
                        sf = dt / sqrt((x - 13.5) * (x - 13.5) + (y - 13.5) * (y - 13.5))
                elif ang < math.pi/4*7:
                    x1 = 13.5 + scales / (-1*(y - 13.5)) * (x - 13.5)
                    dt = sqrt(scales * scales + (x1 - 13.5) * (x1 - 13.5))
                    if sf > dt / sqrt((x - 13.5) * (x - 13.5) + (y - 13.5) * (y - 13.5)):
                        sf = dt / sqrt((x - 13.5) * (x - 13.5) + (y - 13.5) * (y - 13.5))
    nw = list(nuls)
    for x in range(28):
        for y in range(28):
            nlx = 13.5 + (x-13.5) * sf
            nly = 13.5 + (y-13.5) * sf
            ntx = 13.5 + ((x+1)-13.5) * sf
            nty = 13.5 + ((y+1) - 13.5) * sf
            fx =  [1-(nlx%1)]
            curoer = int(nlx)+1
            while curoer < int(ntx):
                fx.append(1)
                curoer += 1
            fx.append(ntx%1)
            fy = [1-(nly%1)]
            curoer = int(nly) + 1
            while curoer < int(nty):
                fy.append(1)
                curoer += 1
            fy.append(nty%1)
            for xof in range(int(ntx)-int(nlx)+1):
                for yof in range(int(nty)-int(nly)+1):
                    try:
                        nw[(int(nlx)+xof) + 28*(int(nly)+yof)] += fx[xof]*fy[yof]*sav[x+y*28]
                        if nw[(int(nlx) + xof) + 28 * (int(nly) + yof)] > 1:
                            nw[(int(nlx) + xof) + 28 * (int(nly) + yof)] = 1
                    except:
                        pass
    return nw

tick = 0

sav = []

with open("inpc.txt", "w") as f:
    for cola in train_images:
        sav.append(cola/256)
        tick += 1
        if tick % 784 == 0:
            sav = center(sav)
            sav = scale(sav)
            for col in sav:
                if col >= 0.5:
                    print(1, file=f, end = " ")
                else:
                    print(-1, file=f, end=" ")
            sav = []