import random
import math
import pygame
import time
from math import sqrt

params = open("{}.txt".format(input()))

def sig(x):
    return math.tanh(x)

def draw(baaas, ins):
    global xm
    global ym
    for row in range(28):
        for line in range(28):
            color = (baaas[row*28+line]+1)*127
            try:
                pygame.draw.rect(screen, [color, color, color], (line*xm/28, row*ym/28, xm/28, ym/28))
            except:
                print(color)
    pygame.draw.rect(screen, [0]*3, (0, ym, xm, ym+100))
    ini = 0
    for inp in ins:
        pygame.draw.circle(screen, [255]*3, ((ini+.5)/inam*xm, ym+100-(inp+1)*50), min(10, xm/inam/2))
        ini += 1

xm = 560
ym = 560

con = True

showex = False

sav = [0]*784

pygame.init()
screen = pygame.display.set_mode((xm, ym+100))

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
inam = network.shape[0]
ins = [0]*inam

drag = False

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
            continue
    if drag:
        pos = pygame.mouse.get_pos()
        if pos[1] > ym:
            ini = int(pos[0]/xm*inam)
            ins[ini] = (ym+100-pos[1])/50-1
            sav = network.predict(ins)
            draw(sav, ins)
            pygame.display.update()