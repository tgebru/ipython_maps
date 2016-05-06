#Get #of ims stored for timealpse by year and other stats
#Put in txt format so that ipython can load and plot
#doing this because I can't have mysql right now for some reason


import MySQLdb
conn = MySQLdb.connect(host='imagenet.stanford.edu', port=3306, db='demo', user='tgebru')
c1 = conn.cursor()

#Number of images per time point
query='select im_date,count(*) as num_ims into outfile '/imagenetdb3/mysql_tmp_dir/timelapse_num_ims.txt' from fixed_timelapse_times where  small=0 and corrupt=0 and downloaded=1 and dup=0 group by im_date'


#Number of images per year for each zipcode/district
'select year(im_date),zipcode,count(*) as num_ims into outfile "timelapse_num_ims_year_zip.txt" from fixed_timelapse_times where zipcode>0 and small=0 and corrupt=0 and downloaded=1 and dup=0 group by year(im_date),zipcode order by year(im_date),zipcode;'
