import os
import matplotlib.pyplot as plt
import glob


LG_SIZE = (4032, 3024)
MOTO_SIZE = (3120, 4160)
SONY_SIZE = (4000, 6000)


for im in glob.glob('data/LG-Nexus-5x/*.jpg'):
    img = plt.imread(im)
    if img.shape[:2] != LG_SIZE:
        os.system('rm' + ' ' + im.replace('(', '\(').replace(')', '\)'))
        print(im)

for im in glob.glob('data/Motorola-X/*.jpg'):
    img = plt.imread(im)
    if img.shape[:2] != MOTO_SIZE:
        os.system('rm' + ' ' + im.replace('(', '\(').replace(')', '\)'))
        print(im)

for im in glob.glob('data/Sony-NEX-7/*.jpg'):
    img = plt.imread(im)
    if img.shape[:2] != SONY_SIZE:
        os.system('rm' + ' ' + im.replace('(', '\(').replace(')', '\)'))
        print(im)
