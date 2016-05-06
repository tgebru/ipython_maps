
# coding: utf-8

# In[1]:

#Get stats on housing data and timelapse images
import os
import pickle

borrough='kings'

f=open('/imagenetdb/tgebru/scrape/%s_lat_lng_rot_url.txt'%borrough,'rb')
lines= f.readlines()
i=0
num_lines=len(lines)
num_tot_ims=0
num_real_ims=0
lat_lng_date_dict={}
lat_lng_rot_date_dict={}
lat_lng_rot_date_file_dict={}
#MIN_SIZE=100e3
MIN_SIZE=50e3
num_lines=len(lines)
line_i=0

#Save files of loaded images

f_loaded=open('/afs/cs.stanford.edu/u/tgebru/cvpr2016/%s_loaded_lat_lng_rot_url.txt'%borrough,'w')
f_unloaded=open('/afs/cs.stanford.edu/u/tgebru/cvpr2016/%s_unloaded_lat_lng_rot_url.txt'%borrough,'w')
for l in lines:
    line_i += 1
    if num_lines % line_i ==10000:
        print 'processing line %d out of %d'%(line_i,num_lines)
    parts=l.split('\t')
    fname=parts[-1].strip()#.replace('gsv_time_fixed','gsv_time_fixed_unwarp')
    if os.path.exists(fname):
        num_tot_ims += 1
        #Only add the images to the dict if they are totally loaded
        if os.stat(fname).st_size>(MIN_SIZE):
            num_real_ims += 1
            #Save the URLS of the loaded images for visualization
            f_loaded.write(l)
            lat=parts[0].split('_')[0].strip()
            lng=parts[0].split('_')[1].strip()
            rot=parts[0].split('_')[2].strip()
            date_str=parts[-1].split('_')[-1][:-5]
            
            #lat_lng_date dict
            if '%s_%s'%(lat,lng) not in lat_lng_date_dict:
                lat_lng_date_dict['%s_%s'%(lat,lng)]={}
                lat_lng_date_dict['%s_%s'%(lat,lng)][date_str]=1
            elif date_str not in lat_lng_date_dict['%s_%s'%(lat,lng)]:
                lat_lng_date_dict['%s_%s'%(lat,lng)][date_str] =1
            else:
                lat_lng_date_dict['%s_%s'%(lat,lng)][date_str] += 1
                
            #lat_lng_rot_date dict
            if '%s_%s_%s'%(lat,lng,rot) not in lat_lng_rot_date_dict:
                lat_lng_rot_date_dict['%s_%s_%s'%(lat,lng,rot)]=[date_str]
            else:
                 lat_lng_rot_date_dict['%s_%s_%s'%(lat,lng,rot)].append(date_str)
            lat_lng_rot_date_file_dict['%s_%s_%s_%s'%(lat,lng,rot,date_str)]=fname
        else:
          f_unloaded.write(l)
  
    else:
        f_unloaded.write(l)
f.close()
f_loaded.close()
f_unloaded.close()
with open('lat_lng_rot_date_dict.pickle','wb') as f:
    pickle.dump(lat_lng_rot_date_dict,f)
        
with open('lat_lng_rot_date_file_dict.pickle','wb') as f:
    pickle.dump(lat_lng_rot_date_file_dict,f)
                
with open('lat_lng_date_dict.pickle','wb') as f:
    pickle.dump(lat_lng_date_dict,f)
