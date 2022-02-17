#%%

import pandas as pd
import geopandas as gp
from plotly import express as px
import numpy as np
import matplotlib
import matplotlib.pylab as pylab
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#%%
quad_properties_df=pd.read_csv('quad_properties.csv',index_col=0)
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
count_by_hour=pd.DataFrame(map(list,zip(*[data_pickup_quad.groupby('hour').size(),data_dropoff_quad.groupby('hour').size()])),columns=['pickup','dropoff']).reset_index().rename(columns={'index': 'hour'})
data_melt=pd.melt(count_by_hour,value_vars=['dropoff','pickup'],id_vars='hour').rename(columns={'value':'Total trips','variable':'type'})

fig=px.bar(data_melt,x='hour',y='Total trips',facet_row='type',color='type',title='Trips by hour on quad',width=600, text_auto=True)
fig.update_layout(
    {'plot_bgcolor':'rgba(0,0,0,0)','paper_bgcolor':'rgba(0,0,0,0)'},
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 1
    ),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        showlegend=False,
        font=dict(size=10)
)
#clear annotations
fig.for_each_annotation(lambda a: a.update(text=''))
axis_id=0
for axis in fig.layout:
    if type(fig.layout[axis]) == go.layout.YAxis:
        if axis_id==0:
            fig.layout[axis].title.text = 'Dropoffs'
            axis_id+=1
        else:
            fig.layout[axis].title.text = 'Pickups'
fig.update_traces(textfont_size=10, textangle=90, cliponaxis=True,textfont=dict(
        color="white"
    ))
fig.update_yaxes(showgrid=False)

# %%
#by year

count_by_year=data_dropoff_quad.groupby('year').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'dropoff'}).merge(data_pickup_quad.groupby('year').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'pickup'}),on='year')
count_by_year=count_by_year[count_by_year['year']!=2015]
data_melt=pd.melt(count_by_year,value_vars=['dropoff','pickup'],id_vars='year').rename(columns={'value':'Total trips','variable':'type'})

fig=px.line(data_melt,x='year',y='Total trips',color='type',title='Trips over time on quad', markers=True)
fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
         xaxis = dict(
        tickmode = 'linear',
        dtick = 1
    ),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig.update_xaxes(showgrid=False)

# %%
color_labels={0:'blue',1:'green',2:'red',3:'cyan',4:'magenta',5:'darkorange',6:'purple'}

#%%

#by number of passengers
count_by_passenger_drop=data_dropoff_quad.groupby('passenger_count').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'dropoff'})

count_by_passenger_pick=data_pickup_quad.groupby('passenger_count').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'pickup'})
#%%
passengers=sorted(list(set(list(count_by_passenger_pick['passenger_count'])+list(count_by_passenger_drop['passenger_count']))))

colors_graph1=list(map(lambda x:color_labels[x],list(count_by_passenger_pick['passenger_count'])))

colors_graph2=list(map(lambda x:color_labels[x],list(count_by_passenger_drop['passenger_count'])))

#%%

fig = make_subplots(rows=2, cols=1, specs=[[{'type':'domain'}],[ {'type':'domain'}]])
fig.add_trace(go.Pie(labels=count_by_passenger_drop.passenger_count, values=count_by_passenger_drop.dropoff, name="Pick Up",sort=False,marker_colors=colors_graph1),
              1, 1)
fig.add_trace(go.Pie(labels=count_by_passenger_pick.passenger_count, values=count_by_passenger_pick.pickup, name="Drop Off",sort=False,marker_colors=colors_graph2),
              2, 1)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.5, hoverinfo="label+value+percent+name")

fig.update_layout(
    title_text="Rides by number of passengers",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Pick Up', x=0.5, y=0.18, font_size=12, showarrow=False),
                 dict(text='Drop Off', x=0.5, y=0.82, font_size=12, showarrow=False)],
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    legend=dict(
    xanchor="left",
    x=-0.3,
    title='Passengers'
))

fig.show()
# %%

#most common origins

def plot_choropleth(dff,var,slct_var_label,hover_columns):
    dff['quad_id']=dff['quad_id'].astype(int)

    dff['geometry']=gp.GeoSeries.from_wkt(dff['geometry'])
    gdf = gp.GeoDataFrame(dff, crs="EPSG:4326",geometry=dff.geometry)

    percentiles=list(map(lambda x:round(x,0),list(gdf[var].quantile([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]))))
    percentiles[0]=percentiles[0]-100
    percentiles[-1]=percentiles[-1]+100

    percentiles = list( dict.fromkeys(percentiles) )
    divisions=len(percentiles)
    cmaplist = [matplotlib.colors.rgb2hex(cmap(i/divisions)) for i in range(divisions)]

    labels=[f'<{percentiles[1]:.0f}']
    for i in np.arange(len(percentiles)-1):
        if i<=1:continue
        labels+=[f'{percentiles[i-1]:.0f}-{percentiles[i]:.0f}']
    labels+=[f'>{percentiles[-2]:.0f}']

    gdf[slct_var_label]=pd.cut(gdf[var],bins=percentiles,labels=labels)
    gdf.set_index('quad_id',inplace=True)


    hover_columns=list(dict.fromkeys(hover_columns))
    fig=px.choropleth_mapbox(gdf, geojson=gdf.geometry, 
                                locations=gdf.index, color=slct_var_label,category_orders ={slct_var_label:labels},height=600, mapbox_style="carto-positron", hover_data=hover_columns,color_discrete_sequence=cmaplist
        ,opacity=0.5,zoom=10,center={'lon':-74,'lat':40.75},title='Testando')
    fig.update_layout(mapbox_style="light",margin={"r":0,"t":0,"l":0,"b":0},
        mapbox_accesstoken=token,
        plot_bgcolor=colors['background'],
        paper_bgcolor='white',
        font_color=colors['text'],
        font=dict(size=8),legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    return fig


def plot_choropleth_from_selected_quad(df,var):
    data_grouped=df.groupby(var).count()[['pickup_longitude']].reset_index().rename(columns={var:'quad_id','pickup_longitude':'count'})

    dff=data_grouped.merge(quad_properties_df,on='quad_id')

    hover_columns=['count']

    fig =plot_choropleth(dff,'count','count_bins',hover_columns)
    
    return fig
plot_choropleth_from_selected_quad(data_dropoff_quad,'quad_pickup')
#%%
plot_choropleth_from_selected_quad(data_pickup_quad,'quad_drop')


# %%
quad_data=quad_properties_df[quad_properties_df['quad_id']==quad_selected]
lat_center=quad_data.iloc[0][['lat','lat2']].mean()
lon_center=quad_data.iloc[0][['lon','lon2']].mean()


fig = go.Figure(data=go.Scattermapbox(
        lon = data_pickup_quad['pickup_longitude'],
        lat = data_pickup_quad['pickup_latitude'],
        text = data_pickup_quad['pickup_datetime_local'],
        mode = 'markers',
        name='pickups',
        marker = go.scattermapbox.Marker(
            size = 3,
            opacity = 0.8,
            color = 'red'
        )))
fig.add_trace(go.Scattermapbox(
        lon = data_dropoff_quad['dropoff_longitude'],
        lat = data_dropoff_quad['dropoff_latitude'],
        text = data_dropoff_quad['pickup_datetime_local'],
        mode = 'markers',
        name='dropoffs',
        marker = go.scattermapbox.Marker(
            size = 3,
            opacity = 0.8,
            color = 'blue'
        )))
fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=lat_center,
            lon=lon_center
        ),
        pitch=0,
        zoom=15
    ),height=600,margin={"r":0,"t":0,"l":0,"b":0},
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        font=dict(
            family="Arial",
            size=10,
            color="black"
        ),
        bgcolor="White",
        bordercolor="Black",
        borderwidth=2
    )
)
fig.show()
# %%

# %%
