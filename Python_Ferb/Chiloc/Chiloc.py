class Chiloc():
	"""
	This class aims to catch the longitude and latitude of given place.
	
	"""
	
	def __init___(self, place_name = '北京大学国家发展研究院'):
		
		self.name = place_name
		self.lng = self.getlnglat(self.name)[0]
		self.lat = self.getlnglat(self.name)[1]
		
	def getlnglat(self, place_name):
		
		place_name = self.name
		url = 'http://api.map.baidu.com/geocoder/v2/'
		output = 'json'
		ak = 'H3bQs5XVuBaLnoQ3CvIzZUiEYrr5Bym4'
		# re-code mandrian
		add = quote(place_name) 
		url = url + '?' + 'address=' + add + '&output=' + output + '&ak=' + ak
		req = urlopen(url)
		# decode as unicode 
		res = req.read().decode()
		json = json.loads(res)
		
		self.lng = json['result']['location']['lng']
		self.lat = json['result']['location']['lat']
		
		return self.lng, self.lat
	
	
	def __repr__(self):
	
		print(f'Place: {self.name}, lng: {self.lng}, lat: {self.lat}')
	
	