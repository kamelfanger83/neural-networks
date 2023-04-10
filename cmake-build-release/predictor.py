import random
import math
import pygame
import time
from math import sqrt

params = open("{}.txt".format(input()))


def sig(x):
    return math.tanh(x)

def draw(baaas):
    global xm
    global ym
    for row in range(28):
        for line in range(28):
            color = baaas[row*28+line]*255
            try:
                pygame.draw.rect(screen, [color, color, color], (line*xm/28, row*ym/28, xm/28, ym/28))
            except:
                print(color)

def gethighest(li):
    rec = -9999
    ind = 0
    s = -9999
    sind = 0
    for d in range(len(li)):
        if li[d] > rec:
            s = rec
            sind = ind
            rec = li[d]
            ind = d
    r = "I am {}% sure that this is a {}".format(int((rec+1)*50), ind)
    if s > -0.98:
        r += " BUT it has a {}% chance of beeing a {}".format(int((s+1)*50), sind)
    return r

xm = 560
ym = 560

con = True

showex = False

filer = [-1, -28]#, 1, 28]#, -29,  -27, -56, -2, 2, 27, 29, 56]

las = [-1, -1]

drag = False

sav = []

for zer in range(784):
    sav.append(0)

nuls = list(sav)

pygame.init()
screen = pygame.display.set_mode((xm, ym))

draw(sav)
pygame.display.update()

def get_ind(pos):
    return int(pos[0]/xm*28) + int(pos[1]/ym*28)*28

def readlis():
    global params
    layers = int(params.readline())
    neurons = []
    for reader in range(layers):
        thisn = int(params.readline())
        neurons.append(thisn)
    ws = []
    bs = []
    for wlayer in range(1,layers):
        tl = []
        for wneuron in range(neurons[wlayer]):
            tr = []
            for wlneuron in range(neurons[wlayer-1]):
                thisw = float(params.readline())
                tr.append(thisw)
            tl.append(tr)
        ws.append(tl)
    for blayer in range(1,layers):
        tl = []
        for bneuron in range(neurons[blayer]):
            thisb = float(params.readline())
            tl.append(thisb)
        bs.append(tl)
    print("finished reading ")
    return neurons, ws, bs


class networks:
    def __init__(self):
        self.shape, self.weights, self.biases = readlis()

    def predict(self, inp):
        a = inp
        for layer in range(1, len(self.shape)):
            na = []
            for neuron in range(self.shape[layer]):
                val = 0
                for laneuron in range(self.shape[layer - 1]):
                    val += a[laneuron] * self.weights[layer - 1][neuron][laneuron]
                val += self.biases[layer - 1][neuron]
                na.append(sig(val))
            a = list(na)
        return a

network = networks()

if showex:
    train_imagesu = open("output.txt").read()
    train_images = []
    num = ""
    for el in train_imagesu:
        if el != " ":
            num += el
        else:
            train_images.append(float(num))
            num = ""
    for _ in range(1000):
        image = random.randint(0,50000)
        draw(train_images[image*784:(image + 1)*784])
        pygame.display.update()
        print(network.predict(train_images[image*784:(image + 1)*784]))
        time.sleep(1)

def gr(brush):
    if abs(brush) > 27:
        brush = brush%28
    if brush < 0:
        return -1*brush%28
    else:
        return brush%28

def interpolate(a, b ,w):
    return a + w*(b-a)

def center():
    global sav
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
    sav = list(nsav)

def geta(x,y):
    if x == 0:
        if y < 0:
            return math.pi/2*3
        return math.pi/2
    ang = math.atan(y/x)
    if x < 0:
        ang += math.pi
    return ang

def scale():
    global sav
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
    sav = list(nw)

scales = 10

while con:
    ev = pygame.event.get()
    for event in ev:
        if event.type == pygame.MOUSEBUTTONDOWN:
            drag = True
        if event.type == pygame.MOUSEBUTTONUP:
            drag = False
        if event.type == pygame.QUIT:
            con = False
        if event.type == pygame.KEYDOWN:
            if event.key == 112:
                print(gethighest(network.predict(sav)))
                start = time.perf_counter()
                sav = list(nuls)
                draw(sav)
                pygame.display.update()
            if event.key == 101:
                sav = list(nuls)
                draw(sav)
                pygame.display.update()
            if event.key == 99:
                center()
                draw(sav)
                pygame.display.update()
            if event.key == 115:
                scale()
                draw(sav)
                pygame.display.update()
            if event.key == 111:
                center()
                scale()
                #sav = [int(k + 0.5) for k in sav]
                draw(sav)
                pygame.display.update()
    if drag:
        ind = get_ind(pygame.mouse.get_pos())
        sav[ind] = 1
        for brush in filer:
            t = ind + brush
            ox = ind%28
            oy = int(ind/28)
            tx = t%28
            ty = int(t/28)
            if ind + brush > 0 and ind + brush < 784 and abs(ox-tx) <= 2 and abs(oy-ty) <= 2:
                if sav[ind + brush] == 0:
                    sav[ind + brush ] = 1
                if sav[ind + brush] < 1 and ind != las:
                    sav[ind + brush] += 0.2
                    sav[ind + brush] = min(1, sav[ind + brush])
        draw(sav)
        las = ind
        pygame.display.update()