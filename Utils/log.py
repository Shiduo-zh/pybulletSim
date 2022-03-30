import numpy as np
from exptools.logging.context import logger_context
from exptools.logging import logger
import os
import imageio
import csv

def log(log_dir,run_ID,name,**kwargs):
    with logger_context(log_dir, run_ID, name):
            # all logging files will be stored under log_dir/run_ID/
            # You can log images or gifs
            images=kwargs['images']
            tag=kwargs['episode']
            logger.log_gif(
                tag,
                images.astype(np.uint8),
                1,
                duration= 0.1
            )
            text=kwargs['reward']
            logger.log_text(text,tag,color='blue')

def log_gif(log_dir,run_id,episode,images,duration=0.002):
    file_path=os.path.join(log_dir,run_id,'gif')
    if(not os.path.exists(file_path)):
        os.makedirs(file_path)
    filename = os.path.join(file_path, "{}-{}.gif".format('episode', episode))
    if isinstance(images, np.ndarray) or (len(images) > 0 and len(images[0].shape)) == 3:
            imageio.mimwrite(filename, images, format= "GIF", duration= duration)
    else:
            imageio.mimwrite(filename, images, format= "GIF", duration= duration)

def log_scalar(log_dir,run_id,infos):
    file_path=os.path.join(log_dir,run_id)
    if(not os.path.exists(file_path)):
        os.makedirs(file_path)
    filename=os.path.join(file_path,'episodeInfos.csv')
    f=open(filename,'a',encoding='utf-8',newline='')
    csv_writer=csv.writer(f)
    if(infos['episode']==0):
        csv_writer.writerow(['episodeID','Reward','steps',
        'entropy','value_loss','action_loss','totalsteps'])
    
    csv_writer.writerow([infos['episode'],infos['reward'],infos['steps']
    ,infos['entropy'],infos['value_loss'],infos['action_loss'],infos['totalsteps']])
    
    