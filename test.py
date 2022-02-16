#%%

import pandas as pd
import geopandas as gp
from plotly import express as px
import numpy as np
import matplotlib
import matplotlib.pylab as pylab
import plotly.graph_objects as go
from plotly.subplots import make_subplots

quad_properties_df=pd.read_csv('quad_properties.csv',index_col=0)[['quad_id','geometry']]
# %%

cmap = pylab.cm.viridis

cmaplist = [matplotlib.colors.rgb2hex(cmap(i/10)) for i in range(10)]
token = open("./env/mapbox_token.txt").read()
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
#%%
# %%


data=pd.read_csv('uber_rides_processed.csv',index_col=0)
#%%

start_hour=1
end_hour=20
minimum_number_to_show=10
var='quad_pickup'

# %%
data_grouped=data[(data['hour']>start_hour)&(data['hour']<end_hour)].groupby(var,as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year'],axis=1).rename(columns={'pickup_longitude':'count',var:'quad_id'})
data_grouped=data_grouped[data_grouped['count']>minimum_number_to_show]

# %%
dff=data_grouped.merge(quad_properties_df,on='quad_id')
dff['quad_id']=dff['quad_id'].astype(int)
dff['geometry']=gp.GeoSeries.from_wkt(dff['geometry'])

gdf = gp.GeoDataFrame(dff, crs="EPSG:4326",geometry=dff.geometry)


percentiles=list(map(lambda x:round(x,0),list(gdf['count'].quantile([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]))))
percentiles[0]=percentiles[0]-100
percentiles[-1]=percentiles[-1]+100

labels=[f'<{percentiles[1]:.0f}']
for i in np.arange(len(percentiles)-1):
    if i<=1:continue
    labels+=[f'{percentiles[i-1]:.0f}-{percentiles[i]:.0f}']
labels+=[f'>{percentiles[-2]:.0f}']

gdf['count_bins']=pd.cut(gdf['count'],bins=percentiles,labels=labels)

hover_columns=['count']
#drop_duplicates

hover_columns=list(dict.fromkeys(hover_columns))
fig=px.choropleth_mapbox(gdf, geojson=gdf.geometry, 
                            locations=gdf.index, color='count_bins',category_orders ={'count_bins':labels},height=600, mapbox_style="carto-positron", hover_data=hover_columns,color_discrete_sequence=cmaplist
    ,opacity=0.5,zoom=10,center={'lon':-74,'lat':40.75})
fig.update_layout(mapbox_style="light",margin={"r":0,"t":0,"l":0,"b":0},
    mapbox_accesstoken=token,
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    font=dict(size=8),legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
# %%

# %%
quad_selected=2503
#%%
data_pickup_quad=data[data['quad_pickup']==quad_selected]
data_dropoff_quad=data[data['quad_drop']==quad_selected]

# %%
#histogram by hour
count_by_hour=pd.DataFrame(map(list,zip(*[data_dropoff_quad.groupby('hour').size(),data_pickup_quad.groupby('hour').size()])),columns=['dropoff','pickup']).reset_index().rename(columns={'index': 'hour'})

data_melt=pd.melt(count_by_hour,value_vars=['dropoff','pickup'],id_vars='hour').rename(columns={'value':'Total trips','variable':'type'})

px.bar(data_melt,x='hour',y='Total trips',facet_row='type',color='type',title='Trips by hour on quad')
# %%
#by year

count_by_year=data_dropoff_quad.groupby('year').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'dropoff'}).merge(data_pickup_quad.groupby('year').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'pickup'}),on='year')

data_melt=pd.melt(count_by_year,value_vars=['dropoff','pickup'],id_vars='year').rename(columns={'value':'Total trips','variable':'type'})

px.line(data_melt,x='year',y='Total trips',color='type',title='Trips over time on quad')

# %%

#by number of passengers
count_by_passenger_drop=data_dropoff_quad.groupby('passenger_count').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'dropoff'})

count_by_passenger_pick=data_pickup_quad.groupby('passenger_count').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'pickup'})

fig = make_subplots(rows=2, cols=1, specs=[[{'type':'domain'}],[ {'type':'domain'}]])
fig.add_trace(go.Pie(labels=count_by_passenger_drop.passenger_count, values=count_by_passenger_drop.dropoff, name="Pick Up"),
              1, 1)
fig.add_trace(go.Pie(labels=count_by_passenger_pick.passenger_count, values=count_by_passenger_pick.pickup, name="Drop Off"),
              2, 1)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.5, hoverinfo="label+value+percent+name")

fig.update_layout(
    title_text="Rides by number of passengers",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Pick Up', x=0.5, y=0.18, font_size=12, showarrow=False),
                 dict(text='Drop Off', x=0.5, y=0.82, font_size=12, showarrow=False)])
fig.show()
# %%
