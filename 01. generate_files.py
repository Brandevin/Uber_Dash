#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box
import plotly.express as px
import geopandas

token = open("mapbox_token.txt").read()

# %%
data=pd.read_csv('uber.csv',index_col=0)[['pickup_datetime','pickup_longitude','dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count']]
# %%
long_range_start=-74.1
long_range_end=-73.7
lat_range_start=40.6
lat_range_end=40.9

step=0.005

longs=['pickup_longitude','dropoff_longitude']
lats=['pickup_latitude','dropoff_latitude']

for long in longs:
    data=data[(data[long]>long_range_start)&(data[long]<long_range_end)]
for lat in lats:
    data=data[(data[lat]>lat_range_start)&(data[lat]<lat_range_end)]

# %%
min_lat=min(data['pickup_latitude'].min(),data['dropoff_latitude'].min())
max_lat=max(data['pickup_latitude'].max(),data['dropoff_latitude'].max())
min_lon=min(data['pickup_longitude'].min(),data['dropoff_longitude'].min())
max_lon=max(data['pickup_longitude'].max(),data['dropoff_longitude'].max())


# %%

divisions_lat=(max_lat-min_lat)/step
lat_list=min_lat+np.arange(0,divisions_lat+1)*step
divisions_lon=(max_lon-min_lon)/step
lon_list=min_lon+np.arange(0,divisions_lon+1)*step
# %%
index=1
quad_properties=[]
for i in np.arange(len(lat_list)-1):
    for j in np.arange(len(lon_list)-1):
        quad_properties+=[[index,lat_list[i],lat_list[i+1],lon_list[j],lon_list[j+1]]]
        indexes_pickup=(data['pickup_longitude']>lon_list[j])&(data['pickup_longitude']<lon_list[j+1])&(data['pickup_latitude']>lat_list[i])&(data['pickup_latitude']<lat_list[i+1])
        indexes_dropoff=(data['dropoff_longitude']>lon_list[j])&(data['dropoff_longitude']<lon_list[j+1])&(data['dropoff_latitude']>lat_list[i])&(data['dropoff_latitude']<lat_list[i+1])
        data.loc[indexes_dropoff,'quad_drop']=index
        data.loc[indexes_pickup,'quad_pickup']=index


        index+=1
#%%
count_by_quad_pick=data.groupby('quad_pickup').count()
count_by_quad_drop=data.groupby('quad_drop').count()
indexes_selected=set.union(set(list(count_by_quad_pick[count_by_quad_pick['pickup_datetime']>=1].index)),set(list(count_by_quad_drop[count_by_quad_drop['pickup_datetime']>=1].index)))


# %%
quad_properties_df=pd.DataFrame(quad_properties,columns=['quad_id','lat','lat2','lon','lon2'])
quad_properties_df=quad_properties_df[quad_properties_df['quad_id'].apply(lambda x: x in indexes_selected)]

quad_properties_df['geometry']=quad_properties_df.apply(lambda x:box(x['lon'],x['lat'],x['lon2'],x['lat2']),axis=1)
#%%
quad_properties_df.to_csv('quad_properties.csv')
data.to_csv('uber_rides_processed.csv')

#%%
quad_properties_df2=pd.read_csv('quad_properties.csv',index_col=0)
quad_properties_df2['geometry']=geopandas.GeoSeries.from_wkt(quad_properties_df2['geometry'])
# %%
gdf = geopandas.GeoDataFrame(quad_properties_df2, crs="EPSG:4326",geometry=quad_properties_df2.geometry)

#%%

# %%
px.choropleth_mapbox(gdf,geojson=gdf.geometry, 
                            locations=gdf.quad_id, mapbox_style="carto-positron",center={'lon':-74,'lat':40.75},zoom=10,opacity=0.5)
# %%

# %%
