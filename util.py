import ast
import numpy as np
from bresenham import bresenham
import matplotlib.pyplot as plt
import os

def get_np_image(stroke_str):
	strokes = ast.literal_eval(stroke_str)
	img_arr = np.zeros((256,256),dtype=int)
	for stroke in strokes:
		for i in range(len(stroke[0])-1):
			points = bresenham(stroke[0][i],stroke[1][i],stroke[0][i+1],stroke[1][i+1])
			for x,y in points:
				img_arr[x][y] = 1
	return img_arr

def show_np_image(rec):
	word = rec['word'].values[0]
	path = rec['path'].values[0]
	img = np.load(path)
	plt.imshow(img)
	plt.title(word)
	plt.show()

