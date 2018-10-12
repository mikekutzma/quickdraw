import pandas as pd
from glob import glob
import os
from tqdm import tqdm
import ast
import numpy as np
from bresenham import bresenham
import matplotlib.pyplot as plt
from util import get_numpy_image

inputdir = os.path.join('.','input','train')
m = 255

img_info_df = pd.DataFrame({'key_id':[],'path':[],'word':[]})

for filepath in tqdm(glob(os.path.join(inputdir,'*.csv'))):
	df = pd.read_csv(filepath)[['key_id','drawing','word']]
	for record in tqdm(df.sample(n=1000,random_state=1).values):
		img_arr = get_np_image(record[1])
		key_id = record[0]
		word = record[2].strip().replace(' ','_')
		path = str(key_id)+'_'+word+'.npy'
		np.save(os.path.join(inputdir,path),img_arr)
		img_info_df = img_info_df.append({'key_id':key_id,'path':path,'word':word}, ignore_index=True)

img_info_df.to_csv('img_info_train.csv',index=False)

