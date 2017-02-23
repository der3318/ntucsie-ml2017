#!/usr/bin/python
import sys
from PIL import Image

img1 = Image.open( sys.argv[1] )
img2 = Image.open( sys.argv[2] )
pixels1 = img1.load()
pixels2 = img2.load()
for i in range( img1.size[0] ):
    for j in range( img1.size[1] ):
        pixels2[i, j] = (0, 0 ,0 ,0) if pixels1[i, j] == pixels2[i, j] else pixels2[i, j]
img2.save("ans_two.png")

