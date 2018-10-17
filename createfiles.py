import pandas as pd
from glob import glob
import os
from tqdm import tqdm
import ast
import numpy as np
from bresenham import bresenham
import matplotlib.pyplot as plt
from util import get_np_image

inputdir = os.path.join('.','input','test')
m = 255

img_info_df = pd.DataFrame({'key_id':[],'path':[],'word':[]})

for filepath in tqdm(glob(os.path.join(inputdir,'*.csv'))):
	df = pd.read_csv(filepath)[['key_id','drawing']]
	for record in tqdm(df.values):
		img_arr = get_np_image(record[1])
		key_id = record[0]
		path = str(key_id)+'.npy'
		np.save(os.path.join(inputdir,path),img_arr)
		img_info_df = img_info_df.append({'key_id':key_id,'path':path}, ignore_index=True)

img_info_df.to_csv('img_info_test.csv',index=False)

