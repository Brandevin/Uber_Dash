#%%

import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output,State  # pip install dash (version 2.0.0 or higher)
import dash_leaflet as dl
import geopandas as gp
import json
from dash_extensions.javascript import arrow_function, assign
import dash_leaflet.express as dlx
import numpy as np
import time
import matplotlib.pylab as pylab
import matplotlib

cmap = pylab.cm.viridis
cmaplist = [matplotlib.colors.rgb2hex(cmap(i/10)) for i in range(10)]

token = open("./env/mapbox_token.txt").read()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

data=pd.read_csv('uber_rides_processed.csv',index_col=0)

data_grouped_total=data.groupby('quad_drop',as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year','quad_pickup'],axis=1).rename(columns={'pickup_longitude':'total_dropoffs','quad_drop':'quad_id'}).merge(data.groupby('quad_pickup',as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year','quad_drop'],axis=1).rename(columns={'pickup_longitude':'total_pickups','quad_pickup':'quad_id'}),on='quad_id',how='outer').fillna(0)


quad_properties_df=pd.read_csv('quad_properties.csv',index_col=0)[['quad_id','geometry']]


def blank_figure():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None,paper_bgcolor=colors['background'])
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)

    return fig

app = Dash(__name__)



# App layout

label_var=html.Label('Select Variable:', style={
        'textAlign': 'center',
        'color': colors['text']
    })
variable=dcc.Dropdown(id="slct_var",
                 options=[
                     {"label": "Number of pick ups", "value": 'pickups_selected_hours'},
                     {"label": "Number of drop offs", "value": 'dropoffs_selected_hours'},
                    {"label": "% of pick ups on selected hour range", "value": '%_pickups_selected_hours'},
                    {"label": "% of drop offs on selected hour range", "value": '%_dropoffs_selected_hours'
                    },
],
                 multi=False,
                 value='pickups_selected_hours',
                 style={"z-index": "1000",'height':'30px'}
                 )

label_range=html.Label('Select hour range:', style={
        'textAlign': 'center',
        'color': colors['text']
    })
range_hour=dcc.RangeSlider(0, 23, 1, count=1,
        marks={i: f'{i:02.0f}h' for i in range(0, 24)}, value=[0, 23],id='range_hours')

select_filter_devices= dcc.Input(id="filter_rides", type="number", placeholder="Minimum rides",value=50,style={'height':'30px'})
label_filter_devices=html.Label('Minimum number of rides to show quad:', style={
        'textAlign': 'center',
        'color': colors['text']
    })

button=    html.Button('UPDATE', id='button1',style={'height':'30px','margin':'5px'})
label_button=html.Div(html.Label('Update map', style={
        'textAlign': 'center',
        'color': colors['text']
    }))

main_map=html.Div([html.Div(style={'backgroundColor': colors['background'], 'display': 'flex', 'flex-direction': 'row'},
    children=[html.Div([label_var,variable],style={'padding':10,'flex-basis':'30%','text-align':'center'}),
    html.Div([label_range,range_hour],style={'padding':10,'flex-basis':'35%'}),

    html.Div([label_filter_devices,select_filter_devices],style={'padding':10,'flex-basis':'20%','text-align':'center'}),

    html.Div([label_button,button],style={'padding':10,'flex-basis':'15%'})]),

    html.Br(), #break
    html.H2("", style={'text-align': 'center',
            'color': colors['text']},id='main_map_title'),
    html.H6("(Click on a quad for more information)", style={'text-align': 'center',
            'color': colors['text']}),

    dcc.Graph(id='choropleth', figure = blank_figure(),style={'margin':'30px'})
    ],style={'padding':10,'flex-basis':'60%','text-align':'center'})

app.layout=html.Div([main_map],style={'backgroundColor': colors['background']})

#functions

@app.callback(
    [Output(component_id='choropleth', component_property='figure'),Output(component_id='main_map_title', component_property='children')
     ],
    [Input('button1','n_clicks'),State(component_id='range_hours', component_property='value'),State(component_id='slct_var', component_property='value'),State(component_id='filter_rides', component_property='value'),State(component_id='slct_var', component_property='options')]
)
def update_graph(n_clicks,hour_range,var,minimum_number_to_show,var_options):
    print(var,var_options)
    slct_var_label=[x['label'] for x in var_options if x['value'] == var][0]

    start_hour,end_hour=hour_range
    data_subset=data[(data['hour']>=start_hour)&(data['hour']<=end_hour)]

    data_grouped_filter=data_subset.groupby('quad_drop',as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year','quad_pickup'],axis=1).rename(columns={'pickup_longitude':'dropoffs_selected_hours','quad_drop':'quad_id'}).merge(data_subset.groupby('quad_pickup',as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year','quad_drop'],axis=1).rename(columns={'pickup_longitude':'pickups_selected_hours','quad_pickup':'quad_id'}),on='quad_id',how='outer').fillna(0)

    data_grouped=data_grouped_filter.merge(data_grouped_total,on='quad_id')

    data_grouped['%_pickups_selected_hours']=data_grouped.apply(lambda x:0 if x['total_pickups']==0 else 100*round(x['pickups_selected_hours']/x['total_pickups'],2),axis=1)
    data_grouped['%_dropoffs_selected_hours']=data_grouped.apply(lambda x:0 if x['total_dropoffs']==0 else 100*round(x['dropoffs_selected_hours']/x['total_dropoffs'],2),axis=1)

    if 'drop' in var:
        data_grouped=data_grouped[data_grouped['total_dropoffs']>minimum_number_to_show]
    else:
        data_grouped=data_grouped[data_grouped['total_pickups']>minimum_number_to_show]

    dff=data_grouped.merge(quad_properties_df,on='quad_id')
    dff['quad_id']=dff['quad_id'].astype(int)
    dff['geometry']=gp.GeoSeries.from_wkt(dff['geometry'])

    gdf = gp.GeoDataFrame(dff, crs="EPSG:4326",geometry=dff.geometry)

    print('test2')
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
    print('test')
    gdf[slct_var_label]=pd.cut(gdf[var],bins=percentiles,labels=labels)
    gdf.set_index('quad_id',inplace=True)
    hover_columns=['dropoffs_selected_hours', 'pickups_selected_hours',
       'total_dropoffs', 'total_pickups', '%_pickups_selected_hours',
       '%_dropoffs_selected_hours']

    print('here')    
    hover_columns=list(dict.fromkeys(hover_columns))
    fig=px.choropleth_mapbox(gdf, geojson=gdf.geometry, 
                                locations=gdf.index, color=slct_var_label,category_orders ={slct_var_label:labels},height=600, mapbox_style="carto-positron", hover_data=hover_columns,color_discrete_sequence=cmaplist
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
    print(var)
    if var=='pickups_selected_hours':  
        text=f"Number of pick ups per quadrant between {start_hour:02.0f}h - {end_hour:02.0f}h"
    elif var=='dropoffs_selected_hours':  
        text=f"Number of drop offs per quadrant between {start_hour:02.0f}h - {end_hour:02.0f}h"
    elif var=='%_dropoffs_selected_hours':  
        text=f"% of drop offs per quadrant between {start_hour:02.0f}h - {end_hour:02.0f}h (out of all drop offs in quadrant)"
    elif var=='%_pickups_selected_hours':  
        text=f"% of pick ups per quadrant between {start_hour:02.0f}h - {end_hour:02.0f}h (out of all pick ups in quadrant)"

    return [fig,text]
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)

# %%
