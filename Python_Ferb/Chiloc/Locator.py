class CityLocator(Chiloc):
	"""
	"""
	
	def __init__(self, place_city = '北京市'):
	
		Chiloc.__init__(self)
		self.name = '北京市 ' + self.name
		
	
	
	def distance(self, other):
	
		url = 'http://api.map.baidu.com/directionlite/v1/walking'
		ak = 'H3bQs5XVuBaLnoQ3CvIzZUiEYrr5Bym4'
		
		url = url + '?' + 'origin=' + str(getlnglat(self.name)[1]) +',' + str(getlnglat(self.name)[0]) 
		+ '&destination=' + str(getlnglat(other.name)[1]) + ',' + str(getlnglat(other.name)[0]) + '&ak=' + ak
		
		req = urlopen(url)
		# decode as unicode
		res = req.read().decode()
		json = json.loads(res)
		
		try:
			distance = json['result']['routes'][0]['distance']
		except:
			print(f'Error Message: {json["message"]}')
		
		return distance
	
 
	
			
