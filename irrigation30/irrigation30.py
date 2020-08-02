import time
from math import sin, cos, sqrt, atan2, radians
import pandas as pd
import ee
from shapely.geometry import box
import folium
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def authenticate():
    # Trigger the authentication flow.
    ee.Authenticate()


class Irrigation30():

    # Set the max number of samples used in the clustering
    MAX_SAMPLE = 100000
    # Technically, RESOLUTION can be a parameter in __init___
    #     But we did not fully test resolutions different from 30 m.
    RESOLUTION = 30
    # Reference: https://hess.copernicus.org/articles/19/4441/2015/hessd-12-1329-2015.pdf
    # "If NDVI at peak is less than 0.40, the peak is not counted as cultivation."
    #     The article uses 10-day composite NDVI while we use montly NDVI.
    #     To account for averaging effect, our threshold is slightly lower than 0.4.
    CROP_NDVI_THRESHOLD = 0.3
    # Estimated based on http://www.fao.org/3/s2022e/s2022e07.htm#TopOfPage
    WATER_NEED_THRESHOLD = 100
    # Rename ndvi bands to the following
    NDVI_LABELS = ['ndvi' + str(i).zfill(2) for i in range(1, 13)]
    # Give descriptive name for the month
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # List of colors used to plot each cluster
    CLUSTER_COLORS = ['red', 'blue', 'orange', 'yellow', 'darkgreen',
                      'lightgreen', 'lightblue', 'purple', 'pink', 'lightgray']

    def __init__(self, center_lat=43.771114, center_lon=-116.736866, edge_len=0.005, year=2018, num_clusters=2):
        '''
        Parameters:
            center_lat: latitude for the location coordinate
            center_lon: longitude for the location coordinate
            edge_len: edge length in degrees for the rectangle given the location coordinates
            year: year the satellite data should pull images for
            num_clusters: should be in range 2-10'''

        # Initialize the library.
        ee.Initialize()

        # Error handle parameter issues
        if (type(center_lat) == float and (center_lat >= -90 and center_lat <= 90)):
            self.center_lat = center_lat
        else:
            raise ValueError('Please enter float value for latitude between -90 and 90')
            exit()

        if (type(center_lon) == float and (center_lon >= -180 and center_lon <= 180)):
            self.center_lon = center_lon
        else:
            raise ValueError('Please enter float value for longitude between -180 and 180')
            exit()

        if (type(edge_len) == float and (edge_len <= 0.5 and edge_len >= 0.005)):
            self.edge_len = edge_len
        else:
            raise ValueError('Please enter float value for edge length between 0.5 and 0.005')
            exit()

        # (range is 2017 to year prior)
        if ((type(year) == int) and (year >= 2017 and year <= int(time.strftime("%Y")) - 1)):
            self.year = year
        else:
            raise ValueError(
                'Please enter an integer value for year > 2017 and less than the current year')
            exit()

        # n_clusters (2-10)
        if ((type(num_clusters) == int) and (num_clusters >= 2 and num_clusters <= 10)):
            self.num_clusters = num_clusters
        else:
            raise ValueError(
                'Please enter an integer value for the number of clusters between 2 and 10')
            exit()

        # initialize remaining variables
        self.label = []
        self.comment = dict()
        self.avg_ndvi = np.zeros((2, 12))
        self.temperature_max = []
        self.temperature_min = []
        self.temperature_avg = []
        self.precipitation = []
        self.image = ee.Image()
        self.nClusters = 0
        self.simple_label = []
        self.simple_image = ee.Image()
        self.base_asset_directory = None

        # Create the bounding box using GEE API
        self.aoi_ee = self.__create_bounding_box_ee()
        # Estimate the area of interest
        self.dist_lon = self.__calc_distance(
            self.center_lon - self.edge_len / 2, self.center_lat, self.center_lon + self.edge_len / 2, self.center_lat)
        self.dist_lat = self.__calc_distance(
            self.center_lon, self.center_lat - self.edge_len / 2, self.center_lon, self.center_lat + self.edge_len / 2)
        print('The selected area is approximately {:.2f} km by {:.2f} km'.format(
            self.dist_lon, self.dist_lat))

        # Estimate the amount of pixels used in the clustering algorithm
        est_total_pixels = round(self.dist_lat * self.dist_lon *
                                 (1000**2) / ((Irrigation30.RESOLUTION)**2))
        self.nSample = min(Irrigation30.MAX_SAMPLE, est_total_pixels)
        # print('The estimated percentage of pixels used in the model is
        # {:.0%}.'.format(self.nSample/est_total_pixels))

        self.model_projection = "EPSG:3857"

    def __create_bounding_box_ee(self):
        '''Creates a rectangle for pulling image information using center coordinates and edge_len'''
        return ee.Geometry.Rectangle([self.center_lon - self.edge_len / 2, self.center_lat - self.edge_len / 2, self.center_lon + self.edge_len / 2, self.center_lat + self.edge_len / 2])

    def __create_bounding_box_shapely(self):
        '''Returns a box for coordinates to plug in as an image add-on layer'''
        return box(self.center_lon - self.edge_len / 2, self.center_lat - self.edge_len / 2, self.center_lon + self.edge_len / 2, self.center_lat + self.edge_len / 2)

    @staticmethod
    def __calc_distance(lon1, lat1, lon2, lat2):
        '''Calculates the distance between 2 coordinates'''
        # Reference: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
        # approximate radius of earth in km
        R = 6373.0
        lon1 = radians(lon1)
        lat1 = radians(lat1)
        lon2 = radians(lon2)
        lat2 = radians(lat2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    def __pull_Sentinel2_data(self):
        '''Output monthly Sentinel image dataset for a specified area with NDVI readings for the year
        merged with GFSAD30 and GFSAD1000 information'''
        band_blue = 'B2'  # 10m
        band_green = 'B3'  # 10m
        band_red = "B4"  # 10m
        band_nir = 'B8'  # 10m

        start_date = str(self.year) + '-1-01'
        end_date = str(self.year) + '-12-31'
        Q1_end = str(self.year) + '-4-01'
        Q2_end = str(self.year) + '-7-01'
        Q3_end = str(self.year) + '-10-01'

        self.Sentinel_RGB = (ee.ImageCollection('COPERNICUS/S2')
                             .filterDate(start_date, end_date)
                             .filterBounds(self.aoi_ee)
                             .select(band_red, band_green, band_blue)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))

        self.Sentinel_RGB_Q1 = self.Sentinel_RGB.filterDate(
            start_date, Q1_end).median().clip(self.aoi_ee)
        self.Sentinel_RGB_Q2 = self.Sentinel_RGB.filterDate(
            Q1_end, Q2_end).median().clip(self.aoi_ee)
        self.Sentinel_RGB_Q3 = self.Sentinel_RGB.filterDate(
            Q2_end, Q3_end).median().clip(self.aoi_ee)
        self.Sentinel_RGB_Q4 = self.Sentinel_RGB.filterDate(
            Q3_end, end_date).median().clip(self.aoi_ee)

        # Create image collection that contains the area of interest
        Sentinel_IC = (ee.ImageCollection('COPERNICUS/S2')
                       .filterDate(start_date, end_date)
                       .filterBounds(self.aoi_ee)
                       .select(band_nir, band_red))

        # Get GFSAD30 image and clip to the area of interest
        GFSAD30_IC = ee.ImageCollection("users/ajsohn/GFSAD30").filterBounds(self.aoi_ee)
        GFSAD30_img = GFSAD30_IC.max().clip(self.aoi_ee)

        def __calc_NDVI(img):
            '''A function to compute Normalized Difference Vegetation Index'''
            ndvi = ee.Image(img.normalizedDifference([band_nir, band_red])).rename(
                ["ndvi"]).copyProperties(img, img.propertyNames())
            composite = img.addBands(ndvi)
            return composite

        def __get_by_month_data(img):
            '''Returns an image after merging the ndvi readings and GFSAD30 data per month'''
            months = ee.List.sequence(1, 12)
            byMonth = ee.ImageCollection.fromImages(
                months.map(lambda m: img.filter(ee.Filter.calendarRange(m, m, 'month')).median().set('month', m)
                           ).flatten())

            # Take all the satellite bands that have been split into months
            # as different images in collection (byMonth), and merge into different bands
            def __mergeBands(image, previous):
                '''Returns an image after merging the image with previous image'''
                return ee.Image(previous).addBands(image).copyProperties(image, image.propertyNames())

            merged = byMonth.iterate(__mergeBands, ee.Image())
            return ee.Image(merged).select(['ndvi'] + ['ndvi_' + str(i) for i in range(1, 12)],
                                           Irrigation30.NDVI_LABELS)

        # Apply the calculation of NDVI
        Sentinel_IC = Sentinel_IC.map(__calc_NDVI).select('ndvi')

        # ---------- GET MONTHLY DATA ---------
        # Get Sentinel-2 monthly data
        # 0 = water, 1 = non-cropland, 2 = cropland, 3 = 'no data'
        byMonth_img = __get_by_month_data(Sentinel_IC) \
            .addBands(GFSAD30_img.rename(['gfsad30'])) \
            .addBands(ee.Image("USGS/GFSAD1000_V1").rename(['gfsad1000'])) \
            .clip(self.aoi_ee)

        # Mask the cropland
        cropland = byMonth_img.select('gfsad30').eq(2)
        byMonth_img_masked = byMonth_img.mask(cropland)

        return byMonth_img_masked

    def __pull_TerraClimate_data(self, band, multiplier=1):
        '''Output monthly TerraClimate image dataset for a specified area for the year'''
        start_date = str(self.year) + '-1-01'
        end_date = str(self.year) + '-12-31'

        # Create image collection that contains the area of interest
        TerraClimate_IC = (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                           .filterDate(start_date, end_date)
                           .filterBounds(self.aoi_ee)
                           .select(band))

        def __get_by_month_data(img):
            '''Returns an image after merging the band readings per month'''
            months = ee.List.sequence(1, 12)
            byMonth = ee.ImageCollection.fromImages(
                months.map(lambda m: img.filter(ee.Filter.calendarRange(m, m, 'month')).median().set('month', m)
                           ).flatten())

            # Take all the satellite bands that have been split into months
            # as different images in collection (byMonth), and merge into different bands
            def __mergeBands(image, previous):
                '''Returns an image after merging the image with previous image'''
                return ee.Image(previous).addBands(image).copyProperties(image, image.propertyNames())

            merged = byMonth.iterate(__mergeBands, ee.Image())
            return ee.Image(merged).select([band] + [band + '_' + str(i) for i in range(1, 12)],
                                           [band + str(i).zfill(2) for i in range(1, 13)])

        # Get TerraClimate monthly data
        byMonth_img = __get_by_month_data(TerraClimate_IC).clip(self.aoi_ee)

        # Calculate the average value by month
        climate_dict = byMonth_img.reduceRegion(reducer=ee.Reducer.mean(
        ), geometry=self.aoi_ee, maxPixels=1e13, scale=Irrigation30.RESOLUTION).getInfo()
        climate_df = pd.DataFrame([climate_dict], columns=[band + str(i).zfill(2)
                                                           for i in range(1, 13)], index=[0])
        climate_arr = climate_df.to_numpy() * multiplier

        return climate_arr

    def __identify_peak(self, y_raw):
        '''Returns peak values and the month for peaking'''
        # Peaks cannot be identified if it's the first or last number in a series
        # To resolve this issue, we copy the series three times
        y = np.concatenate((y_raw, y_raw, y_raw))
        x = np.linspace(0, 35, num=36, endpoint=True)
        peak_index_raw, peak_value_raw = find_peaks(y, height=Irrigation30.CROP_NDVI_THRESHOLD)
        # Sometimes there are multiple peaks in a single crop season
        index_diff = np.diff(peak_index_raw)
        peak_grp = [0]
        counter = 0
        for i in index_diff:
            if i == 2:
                peak_grp.append(counter)
            else:
                counter += 1
                peak_grp.append(counter)
        peak_grp_series = pd.Series(peak_grp, name='peak_grp')
        peak_index_series = pd.Series(peak_index_raw, name='peak_index')
        peak_value_series = pd.Series(peak_value_raw['peak_heights'], name='peak_value')
        peak_grp_df = pd.concat([peak_grp_series, peak_index_series, peak_value_series], axis=1)
        peak_grp_agg_df = peak_grp_df.groupby('peak_grp').agg(
            {'peak_index': np.mean, 'peak_value': np.max})
        peak_index = peak_grp_agg_df['peak_index'].to_numpy()
        peak_value = peak_grp_agg_df['peak_value'].to_numpy()

        peak_lst = [(int(i - 12), Irrigation30.MONTHS[int(i - 12)], j)
                    for i, j in zip(peak_index, peak_value) if i >= 12 and i < 24]
        final_peak_index = [i[0] for i in peak_lst]
        final_peak_month = [i[1] for i in peak_lst]
        final_peak_value = [i[2] for i in peak_lst]
        return final_peak_index, final_peak_month, final_peak_value

    def __identify_label(self, cluster_result):
        '''Plugs in labels for the irrigated and rainfed areas'''
        def __identify_surrounding_month(value, diff):
            '''For the peaked month returns surrounding month data'''
            new_value = value + diff
            if new_value < 0:
                new_value += 12
            elif new_value >= 12:
                new_value -= 12
            return int(new_value)

        def __calc_effective_precipitation(P):
            '''Calculates and prints irrigation labels based on effective precipitation and temperatures'''
            # Reference:
            # Pe = 0.8 P - 25 if P > 75 mm/month
            # Pe = 0.6 P - 10 if P < 75 mm/month
            if P >= 75:
                Pe = 0.8 * P - 25
            else:
                Pe = max(0.6 * P - 10, 0)
            return Pe

        self.label = []
        for i in range(self.nClusters):
            final_peak_index, final_peak_month, final_peak_value = self.__identify_peak(
                self.avg_ndvi[i])
            if len(final_peak_index) == 0:
                self.label.append('Cluster {}: Rainfed'.format(i))
                self.comment[i] = 'rainfed'
            else:
                temp_label = []
                temp_comment = '{}-crop cycle annually | '.format(len(final_peak_index))
                if len(self.precipitation) == 0:
                    self.precipitation = self.__pull_TerraClimate_data('pr')[0]
                if len(self.temperature_max) == 0:
                    self.temperature_max = self.__pull_TerraClimate_data('tmmx', multiplier=0.1)[0]
                    self.temperature_min = self.__pull_TerraClimate_data('tmmn', multiplier=0.1)[0]
                self.temperature_avg = np.mean([self.temperature_max, self.temperature_min], axis=0)
                for p in range(len(final_peak_index)):
                    p_index = final_peak_index[p]
                    # Calcuate the precipiration the month before the peak and at the peak
                    # Depending on whether it's Fresh harvested crop or Dry harvested crop, the water need after the mid-season is different
                    # Reference: http://www.fao.org/3/s2022e/s2022e02.htm#TopOfPage
                    p_lst = [__identify_surrounding_month(p_index, -1), p_index]
                    pr_mean = self.precipitation[p_lst].mean()
                    # Lower temperature reduces water need
                    # Reference: http://www.fao.org/3/s2022e/s2022e02.htm#TopOfPage
                    if self.temperature_avg[p_lst].mean() < 15:
                        temperature_adj = 0.7
                    else:
                        temperature_adj = 1
                    if pr_mean >= Irrigation30.WATER_NEED_THRESHOLD * temperature_adj:
                        temp_label.append('Rainfed')
                        temp_comment = temp_comment + \
                            'rainfed around {}; '.format(final_peak_month[p])
                    else:
                        temp_label.append('Irrigated')
                        temp_comment = temp_comment + \
                            'irrigated around {}; '.format(final_peak_month[p])
                self.label.append('Cluster {}: '.format(i) + '+'.join(temp_label))
                self.comment[i] = temp_comment
        self.simple_label = ['Irrigated' if 'Irrigated' in i else 'Rainfed' for i in self.label]
        self.image = self.image.addBands(
            ee.Image(cluster_result.select('cluster')).rename('prediction'))

    def plot_precipitation(self):
        '''Plots precepitation from TerraClimate'''
        if len(self.precipitation) == 0:
            self.precipitation = self.__pull_TerraClimate_data('pr')[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(Irrigation30.MONTHS, self.precipitation, label='Precipitation')
        plt.title("Precipitation")
        plt.ylabel("Precipitation (mm)")
        plt.xlabel("Month (" + str(self.year) + ")")
        plt.legend()

    def plot_temperature_max_min(self):
        '''Plots max and min temperature from TerraClimate'''
        self.temperature_max = self.__pull_TerraClimate_data('tmmx', multiplier=0.1)[0]
        self.temperature_min = self.__pull_TerraClimate_data('tmmn', multiplier=0.1)[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(Irrigation30.MONTHS, self.temperature_max, label='Max Temperature')
        plt.plot(Irrigation30.MONTHS, self.temperature_min, label='Min Temperature')
        plt.title("Temperature (°C)")
        plt.xlabel("Month (" + str(self.year) + ")")
        plt.ylabel("Temperature (°C)")
        plt.legend()

    def fit_predict(self):
        '''Builds model using startified datapoints from sampled ndvi dataset for training'''

        try:
            self.image = self.__pull_Sentinel2_data()
        except:
            raise RuntimeError('GEE will run into issues due to missing images')

        training_FC = self.image.cast({'gfsad30': "int8"}, ['gfsad30', 'gfsad1000'] + Irrigation30.NDVI_LABELS)\
            .stratifiedSample(region=self.aoi_ee, classBand='gfsad30', numPoints=self.nSample,
                              classValues=[0, 1, 3],
                              classPoints=[0, 0, 0],
                              scale=Irrigation30.RESOLUTION)\
            .select(Irrigation30.NDVI_LABELS)

        # Instantiate the clusterer and train it.
        clusterer = ee.Clusterer.wekaKMeans(self.num_clusters).train(
            training_FC, inputProperties=Irrigation30.NDVI_LABELS)

        # Cluster the input using the trained clusterer.
        cluster_result = self.image.cluster(clusterer)

        print('Model building...')
        cluster_output = dict()
        for i in range(0, self.num_clusters):
            cluster_output[i] = self.image.select(Irrigation30.NDVI_LABELS).mask(cluster_result.select('cluster').eq(
                i)).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.aoi_ee, maxPixels=1e13, scale=30).getInfo()
            if cluster_output[i]['ndvi01'] == None:
                self.nClusters = i
                del cluster_output[i]
                break
            elif i == self.num_clusters - 1:
                self.nClusters = self.num_clusters

        # Reference: https://stackoverflow.com/questions/45194934/eval-fails-in-list-comprehension
        globs = globals()
        locs = locals()
        cluster_df = pd.DataFrame([eval('cluster_output[{}]'.format(i), globs, locs) for i in range(
            0, self.nClusters)], columns=Irrigation30.NDVI_LABELS, index=['Cluster_' + str(i) for i in range(0, self.nClusters)])

        self.avg_ndvi = cluster_df.to_numpy()

        self.__identify_label(cluster_result)

        gee_label_irr = ee.List([0] + [1 * (self.simple_label[i] == "Irrigated")
                                       for i in range(len(self.simple_label))] + [0 for i in range(10 - self.nClusters)])

        # including -1 to be my 'not cropland below'
        # Hard-coding 10 as top max_clusters
        cluster_nums_py = [str(i) for i in range(-1, 10)]

        cluster_nbrs = ee.List(cluster_nums_py)
        gee_label_dict = ee.Dictionary.fromLists(cluster_nbrs, gee_label_irr)

        temp_image = self.image.expression(
            "(b('gfsad30') == 2) ? (b('prediction')) : -1 ").rename('class').cast({'class': 'int'})

        self.simple_image = temp_image \
            .where(temp_image.eq(-1), ee.Number(0)) \
            .where(temp_image.eq(0), ee.Number(gee_label_dict.get('0'))) \
            .where(temp_image.eq(1), ee.Number(gee_label_dict.get('1'))) \
            .where(temp_image.eq(2), ee.Number(gee_label_dict.get('2'))) \
            .where(temp_image.eq(3), ee.Number(gee_label_dict.get('3'))) \
            .where(temp_image.eq(4), ee.Number(gee_label_dict.get('4'))) \
            .where(temp_image.eq(5), ee.Number(gee_label_dict.get('5'))) \
            .where(temp_image.eq(6), ee.Number(gee_label_dict.get('6'))) \
            .where(temp_image.eq(7), ee.Number(gee_label_dict.get('7'))) \
            .where(temp_image.eq(8), ee.Number(gee_label_dict.get('8'))) \
            .where(temp_image.eq(9), ee.Number(gee_label_dict.get('9')))

        print('Model complete!')

    def plot_map(self):
        '''Plot folium map using GEE api - the map includes are of interest box and associated ndvi readings'''

        def add_ee_layer(self, ee_object, vis_params, show, name):
            '''Checks if image object classifies as ImageCollection, FeatureCollection, Geometry or single Image
            and adds to folium map accordingly'''
            try:
                if isinstance(ee_object, ee.image.Image):
                    map_id_dict = ee.Image(ee_object).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles=map_id_dict['tile_fetcher'].url_format,
                        attr='Google Earth Engine',
                        name=name,
                        overlay=True,
                        control=True,
                        show=show
                    ).add_to(self)
                elif isinstance(ee_object, ee.imagecollection.ImageCollection):
                    ee_object_new = ee_object.median()
                    map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles=map_id_dict['tile_fetcher'].url_format,
                        attr='Google Earth Engine',
                        name=name,
                        overlay=True,
                        control=True,
                        show=show
                    ).add_to(self)
                elif isinstance(ee_object, ee.geometry.Geometry):
                    folium.GeoJson(
                        data=ee_object.getInfo(),
                        name=name,
                        overlay=True,
                        control=True
                    ).add_to(self)
                elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
                    ee_object_new = ee.Image().paint(ee_object, 0, 2)
                    map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles=map_id_dict['tile_fetcher'].url_format,
                        attr='Google Earth Engine',
                        name=name,
                        overlay=True,
                        control=True,
                        show=show
                    ).add_to(self)

            except:
                print("Could not display {}".format(name))

        # Add EE drawing method to folium.
        folium.Map.add_ee_layer = add_ee_layer

        myMap = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=11)
        aoi_shapely = self.__create_bounding_box_shapely()
        folium.GeoJson(aoi_shapely, name="Area of Interest").add_to(myMap)

        # Add Prediction / Cluster Label layer
        start = time.time()
        visParams = {'min': 0, 'max': self.nClusters - 1,
                     'palette': self.CLUSTER_COLORS[:self.nClusters]}
        myMap.add_ee_layer(self.image.select('prediction'), visParams, show=True, name='Prediction')
        end = time.time()
        print("ADDED PREDICTION LAYER \t\t--> " +
              str(round((end - start) / 60, 2)) + " min")

        # Add Sentinel-2 RGB quarterly layers
        start = time.time()
        visParams = {'max': 4000}
        myMap.add_ee_layer(self.Sentinel_RGB_Q1, visParams, show=False, name="Sentinel2-Q1")
        myMap.add_ee_layer(self.Sentinel_RGB_Q2, visParams, show=False, name="Sentinel2-Q2")
        myMap.add_ee_layer(self.Sentinel_RGB_Q3, visParams, show=False, name="Sentinel2-Q3")
        myMap.add_ee_layer(self.Sentinel_RGB_Q4, visParams, show=False, name="Sentinel2-Q4")
        end = time.time()
        print("ADDED S2 RGB LAYERS \t\t--> " + str(round((end - start) / 60, 2)) + " min")

        # Add GFSAD1000 Layer
        start = time.time()
        visParams = {'min': 0, 'max': 5, 'palette': [
            'black', 'green', 'a9e1a9', 'yellow', 'ffdb00', '#ffa500']}
        #     0: Non-croplands (black)
        #     1: Croplands: irrigation major (green)
        #     2: Croplands: irrigation minor (lighter green)
        #     3: Croplands: rainfed (yellow)
        #     4: Croplands: rainfed, minor fragments (yellow orange)
        #     5: Croplands: rainfed, rainfed, very minor fragments (orange)
        myMap.add_ee_layer(self.image.select('gfsad1000'), visParams, show=False, name='GFSAD1000')
        end = time.time()
        print("ADDED GFSAD1000 LAYER \t\t--> " + str(round((end - start) / 60, 2)) + " min")

        # Add NDVI Monthly layers
        start = time.time()
        visParams = {'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}
        for i in range(1, 13):
            temp_band = 'ndvi' + str(i).zfill(2)
            month_label = Irrigation30.MONTHS[i - 1]
            myMap.add_ee_layer(self.image.select(temp_band), visParams,
                               show=False, name='NDVI ' + month_label)
        end = time.time()
        print("ADDED MONTHLY NDVI LAYERS \t--> " + str(round((end - start) / 60, 2)) + " min")

        myMap.add_child(folium.LayerControl())
        folium.Marker([self.center_lat, self.center_lon], tooltip='center').add_to(myMap)

        print('\n============ Prediction Layer Legend ============')
        # print the comments for each cluster
        for i in range(self.nClusters):
            print('Cluster {} ({}): {}'.format(i, Irrigation30.CLUSTER_COLORS[i], self.comment[i]))
        print('============ GFSAD1000 Layer Legend ============')
        print('Croplands: irrigation major (green)')
        print('Croplands: irrigation minor (lighter green)')
        print('Croplands: rainfed (yellow)')
        print('Croplands: rainfed, minor fragments (yellow orange)')
        print('Croplands: rainfed, rainfed, very minor fragments (orange)')
        print('================================================')
        return myMap

    def plot_avg_ndvi(self):
        '''Plotting for ndvi readings'''
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(0, self.nClusters):
            plt.plot(Irrigation30.MONTHS,
                     self.avg_ndvi[i], label=self.label[i], color=Irrigation30.CLUSTER_COLORS[i])
        plt.ylabel("Avg. NDVI")
        plt.xlabel("Month (" + str(self.year) + ")")
        plt.title("NDVI Temporal Signature")
        plt.legend()

    def set_asset_directory(self, base_asset_directory):
        if (type(base_asset_directory) == str):
            self.base_asset_directory = base_asset_directory
            print("BASE ASSET DIRECTORY:", self.base_asset_directory)
        else:
            raise ValueError(
                "Please enter a string for base_asset_directory like '/users/<GEE_USERNAME>/'"
            )
            exit()

    def write_image_asset(self, image_asset_id, write_simple_version=False):
        '''Writes predicted image out as an image to Google Earth Engine as an asset'''
        # image_asset_id = self.base_asset_directory + '/' + image_asset_id
        if self.base_asset_directory == None:
            raise ValueError(
                "Please set a base_asset_directory like 'users/<GEE_USERNAME>/' with set_base_asset_directory(dir). Your asset will be stored in this asset folder in Google Earth Engine."
            )
            exit()

        image_asset_id = self.base_asset_directory + image_asset_id

        print("BASE ASSET DIRECTORY:", self.base_asset_directory)
        print("IMAGE ASSET PATH:", image_asset_id)

        if write_simple_version == False:
            task = ee.batch.Export.image.toAsset(
                crs=self.model_projection,
                region=self.aoi_ee,
                image=self.image,
                scale=self.RESOLUTION,
                assetId=image_asset_id,
                maxPixels=1e13
            )
            task.start()
        else:
            task = ee.batch.Export.image.toAsset(
                crs=self.model_projection,
                region=self.aoi_ee,
                image=self.simple_image,
                scale=self.RESOLUTION,
                assetId=image_asset_id,
                maxPixels=1e13
            )
            task.start()

    def write_image_google_drive(self, filename, write_simple_version=False):
        '''Writes predicted image out as an image to Google Drive as a TIF file'''
        if write_simple_version == False:
            task = ee.batch.Export.image.toDrive(
                crs=self.model_projection,
                region=self.aoi_ee,
                image=self.image.select('prediction'),
                scale=self.RESOLUTION,
                description=filename,
                maxPixels=1e13
            )
        else:
            task = ee.batch.Export.image.toDrive(
                crs=self.model_projection,
                region=self.aoi_ee,
                image=self.simple_image,
                scale=self.RESOLUTION,
                description=filename,
                maxPixels=1e13
            )
        print("Writing To Google Drive filename = ", filename + ".tif")
        task.start()
