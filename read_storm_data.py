import math


def read_data(file_number, path_folder):
	# return: vector = [delta_d, phi]
	#path_folder = 'G:\\Lab\\storm_data\\cluster_0'
	path_file = path_folder + '\\' + str(file_number) + '.txt'
	f = open(path_file, 'r')
		
	f_read = f.read();
	f_read = f_read.split('\n')
	length_f_read = len(f_read)
	vector = []
		
	for i in range(length_f_read - 4):
		lat_1 = float(f_read[i])
		long_1 = float(f_read[i+1])
		lat_2 = float(f_read[i+2])
		long_2 = float(f_read[i+3])
		
			
		delta_long = long_2 - long_1;
		delta_lat  = lat_2  - lat_1;
		
		# vector = [delta_d, phi]
		phi = 0.0
		delta_d = math.sqrt(delta_lat**2 + delta_long**2);
		if (delta_long == 0 and delta_lat < 0):
			phi = 3*math.pi/2
		elif (delta_long == 0 and delta_lat > 0):
			phi = math.pi/2;
		elif (delta_long == 0 and delta_lat ==0 ):
			phi = 0;
		else:
			tang_phi = delta_lat/delta_long;
			
			if (delta_lat < 0 and delta_long > 0):	
				phi = math.atan(tang_phi) + math.pi		# pi/2 < phi < pi
			elif (delta_lat < 0 and delta_long < 0):
				phi = math.atan(tang_phi) + math.pi	# pi < phi < 3pi/2
			elif (delta_lat > 0 and delta_long < 0):
				phi = math.atan(tang_phi) + 2*math.pi		# 3pi/2 < phi < 2pi
			elif (delta_lat > 0 and delta_long > 0):
				phi = math.atan(tang_phi)			# 0< phi < pi/2
			
		vector.append([delta_d, phi])
		#vector.append([factor_d*delta_d + factor_phi*phi, factor_lat*delta_lat + factor_long*delta_long])
		
	f.close()
	
	return vector;
			
			
			
