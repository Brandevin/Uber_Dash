#%%

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output,State 
import geopandas as gp
import numpy as np
import matplotlib.pylab as pylab
import matplotlib
from plotly.subplots import make_subplots

cmap = pylab.cm.viridis
cmaplist = [matplotlib.colors.rgb2hex(cmap(i/10)) for i in range(10)]

#colors for pie chart
color_labels={0:'blue',1:'green',2:'red',3:'cyan',4:'magenta',5:'darkorange',6:'purple'}


token = open("./env/mapbox_token.txt").read()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

data=pd.read_csv('uber_rides_processed.csv',index_col=0)

data_grouped_total=data.groupby('quad_drop',as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year','quad_pickup'],axis=1).rename(columns={'pickup_longitude':'total_dropoffs','quad_drop':'quad_id'}).merge(data.groupby('quad_pickup',as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year','quad_drop'],axis=1).rename(columns={'pickup_longitude':'total_pickups','quad_pickup':'quad_id'}),on='quad_id',how='outer').fillna(0)


quad_properties_df=pd.read_csv('quad_properties.csv',index_col=0)


def blank_figure():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None,paper_bgcolor=colors['background'])
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)

    return fig

app = Dash(__name__)
app.title = 'New York Uber rides Dash'


# App layout

title_quad=html.H2("", style={'text-align': 'center',
            'color': colors['text']},id='slct_quad_title')

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
                    {'label':'Ratio Pickups/Dropoffs','value':'ratio_pickup_dropoff'}
],
                 multi=False,
                 value='pickups_selected_hours',
                 style={"z-index": "1000",'height':'30px'}
                 )

main_title=html.H1('New York Uber Rides Dashboard', style={
        'textAlign': 'center',
        'color': colors['text']
    })

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

main_map=html.Div([html.Div(style={'backgroundColor': colors['background'], 'display': 'flex', 'flex-direction': 'row','flex-basis':'70%'},
    children=[
    html.Div([label_var,variable],style={'padding':10,'flex-basis':'70%','text-align':'center'}),
    html.Div([label_filter_devices,select_filter_devices],style={'padding':10,'flex-basis':'30%','text-align':'center'}),

]),
    html.Div(style={'backgroundColor': colors['background'], 'display': 'flex', 'flex-direction': 'row'},
    children=[
    html.Div([label_range,range_hour],style={'padding':10,'flex-basis':'80%'}),
    html.Div([label_button,button],style={'padding':10,'flex-basis':'20%'})]),

    html.Br(), #break
    html.H2("", style={'text-align': 'center',
            'color': colors['text']},id='main_map_title'),
    html.H4("(Click on a quad for more information)", style={'text-align': 'center',
            'color': colors['text']}),

    dcc.Graph(id='choropleth', figure = blank_figure(),style={'margin':'30px'})
    ],style={'padding':10,'flex-basis':'55%','text-align':'center'})

histogram=dcc.Graph(id='histograms', figure = blank_figure(),style={'padding':0,'flex-basis':'33%','text-align':'center'})
line_plot=dcc.Graph(id='line_plots', figure = blank_figure(),style={'padding':0,'flex-basis':'33%','text-align':'center'})
pie_plot=dcc.Graph(id='pie_plots', figure = blank_figure(),style={'padding':0,'flex-basis':'33%','text-align':'center'})

map_destinations=dcc.Graph(id='map_destinations', figure = blank_figure(),style={'padding':10,'text-align':'center','flex-basis':'33%'})
map_origins=dcc.Graph(id='map_origins', figure = blank_figure(),style={'padding':10,'text-align':'center','flex-basis':'33%'})
map_details=dcc.Graph(id='map_details', figure = blank_figure(),style={'padding':10,'text-align':'center','flex-basis':'33%'})

label_destination=html.H2('Most common destinations of people from quad', style={
        'textAlign': 'center',
        'color': colors['text']
    })
label_origin=html.H2('Most common origins of people that go to quad', style={
        'textAlign': 'center',
        'color': colors['text']
    })
label_details=html.H2('Most common pickup and dropoff locations', style={
        'textAlign': 'center',
        'color': colors['text']
    })


maps=html.Div([html.Div([label_destination,map_destinations],style={'padding':10,'text-align':'center','flex-basis':'33%'}),html.Div([label_origin,map_origins],style={'padding':10,'text-align':'center','flex-basis':'33%'}),html.Div([label_details,map_details],style={'padding':10,'text-align':'center','flex-basis':'33%'})],style={'display':'flex','text-align':'center','flex-direction':'row'})

h_division=html.Hr(style={"width": "95%", "color": colors['text'],'border-color':colors['text'],'border':'4px solid','border-radius':'2px'})

v_division=html.Div(style={'border-left':'4px', "color": colors['text'],'border-color':colors['text'],'border':'4px solid','border-radius':'2px'})

title_maps=html.H2("", style={'text-align': 'center',
            'color': colors['text']},id='slct_quad_title_map')



subplots=html.Div([title_quad,histogram,html.Div(style={'backgroundColor': colors['background'], 'display': 'flex', 'flex-direction': 'row'},
    children=[line_plot,pie_plot])
    ],style={'padding':0,'text-align':'center','flex-basis':'30%'})

first_section=html.Div([main_map,v_division,subplots],style={'backgroundColor': colors['background'], 'display': 'flex', 'flex-direction': 'row'})
app.layout=html.Div([main_title,first_section,h_division,title_maps,maps],style={'backgroundColor': colors['background']})

def plot_choropleth(dff,var,slct_var_label,hover_columns):
    dff['quad_id']=dff['quad_id'].astype(int)

    dff['geometry']=gp.GeoSeries.from_wkt(dff['geometry'])
    gdf = gp.GeoDataFrame(dff, crs="EPSG:4326",geometry=dff.geometry)

    percentiles=list(map(lambda x:round(x,2),list(gdf[var].quantile([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]))))
    percentiles[0]=percentiles[0]-100
    percentiles[-1]=percentiles[-1]+100

    percentiles = list( dict.fromkeys(percentiles) )

    divisions=len(percentiles)
    cmaplist = [matplotlib.colors.rgb2hex(cmap(i/divisions)) for i in range(divisions)]
    if var=='ratio_pickup_dropoff':
        labels=[f'<={percentiles[1]:.2f}']
        for i in np.arange(len(percentiles)-1):
            if i<=1:continue
            labels+=[f'<{percentiles[i-1]:.2f}<={percentiles[i]:.2f}']
        labels+=[f'>{percentiles[-2]:.2f}']
    elif slct_var_label =='Total Trips':
        labels=[f'<={percentiles[1]:.1f}']
        for i in np.arange(len(percentiles)-1):
            if i<=1:continue
            labels+=[f'<{percentiles[i-1]:.1f}<={percentiles[i]:.1f}']
        labels+=[f'>{percentiles[-2]:.1f}']
    else:
        labels=[f'<={percentiles[1]:.0f}']
        for i in np.arange(len(percentiles)-1):
            if i<=1:continue
            labels+=[f'<{percentiles[i-1]:.0f}<={percentiles[i]:.0f}']
        labels+=[f'>{percentiles[-2]:.0f}']
    print(slct_var_label,var,percentiles,labels)
    gdf[slct_var_label]=pd.cut(gdf[var],bins=percentiles,labels=labels)
    gdf.set_index('quad_id',inplace=True)


    hover_columns=list(dict.fromkeys(hover_columns))
    fig=px.choropleth_mapbox(gdf, geojson=gdf.geometry, 
                                locations=gdf.index, color=slct_var_label,category_orders ={slct_var_label:labels},height=600, mapbox_style="carto-positron", hover_data=hover_columns,color_discrete_sequence=cmaplist
        ,opacity=0.5,zoom=10,center={'lon':-74,'lat':40.75},title='teste')
    fig.update_layout(mapbox_style="light",margin={"r":0,"t":0,"l":0,"b":0},
        mapbox_accesstoken=token,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        font=dict(size=8),
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
    ))
    return fig

def plot_choropleth_from_selected_quad(df,var):
    data_grouped=df.groupby(var).count()[['pickup_longitude']].reset_index().rename(columns={var:'quad_id','pickup_longitude':'count'})

    dff=data_grouped.merge(quad_properties_df,on='quad_id')

    hover_columns=['count']

    fig =plot_choropleth(dff,'count','Total Trips',hover_columns)
    
    return fig

#function to update main graph
@app.callback(
    [Output(component_id='choropleth', component_property='figure'),Output(component_id='main_map_title', component_property='children')
     ],
    [Input('button1','n_clicks'),State(component_id='range_hours', component_property='value'),State(component_id='slct_var', component_property='value'),State(component_id='filter_rides', component_property='value'),State(component_id='slct_var', component_property='options')]
)
def update_graph(n_clicks,hour_range,var,minimum_number_to_show,var_options):

    slct_var_label=[x['label'] for x in var_options if x['value'] == var][0]

    start_hour,end_hour=hour_range
    data_subset=data[(data['hour']>=start_hour)&(data['hour']<=end_hour)]

    data_grouped_filter=data_subset.groupby('quad_drop',as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year','quad_pickup'],axis=1).rename(columns={'pickup_longitude':'dropoffs_selected_hours','quad_drop':'quad_id'}).merge(data_subset.groupby('quad_pickup',as_index=False).count().drop(['dropoff_longitude','pickup_latitude','dropoff_latitude','passenger_count','pickup_datetime_local','hour','year','quad_drop'],axis=1).rename(columns={'pickup_longitude':'pickups_selected_hours','quad_pickup':'quad_id'}),on='quad_id',how='outer').fillna(0)

    data_grouped=data_grouped_filter.merge(data_grouped_total,on='quad_id')

    data_grouped['%_pickups_selected_hours']=data_grouped.apply(lambda x:0 if x['total_pickups']==0 else 100*round(x['pickups_selected_hours']/x['total_pickups'],2),axis=1)
    data_grouped['%_dropoffs_selected_hours']=data_grouped.apply(lambda x:0 if x['total_dropoffs']==0 else 100*round(x['dropoffs_selected_hours']/x['total_dropoffs'],2),axis=1)

    data_grouped['ratio_pickup_dropoff']=data_grouped.apply(lambda x:10 if x['dropoffs_selected_hours']==0 else round(x['pickups_selected_hours']/x['dropoffs_selected_hours'],2),axis=1)


    if 'drop' in var:
        data_grouped=data_grouped[data_grouped['total_dropoffs']>minimum_number_to_show]
    elif 'pick' in var:
        data_grouped=data_grouped[data_grouped['total_pickups']>minimum_number_to_show]
    else:
        data_grouped=data_grouped[(data_grouped['total_pickups']>minimum_number_to_show)&(data_grouped['total_dropoffs']>minimum_number_to_show)]

    dff=data_grouped.merge(quad_properties_df,on='quad_id')

    hover_columns=['dropoffs_selected_hours', 'pickups_selected_hours',
       'total_dropoffs', 'total_pickups', '%_pickups_selected_hours',
       '%_dropoffs_selected_hours','ratio_pickup_dropoff']

    fig=plot_choropleth(dff,var,slct_var_label,hover_columns)

    if var=='pickups_selected_hours':  
        text=f"Number of pick ups per quadrant between {start_hour:02.0f}h - {end_hour:02.0f}h"
    elif var=='dropoffs_selected_hours':  
        text=f"Number of drop offs per quadrant between {start_hour:02.0f}h - {end_hour:02.0f}h"
    elif var=='%_dropoffs_selected_hours':  
        text=f"% of drop offs per quadrant between {start_hour:02.0f}h - {end_hour:02.0f}h (out of all drop offs in quadrant)"
    elif var=='%_pickups_selected_hours':  
        text=f"% of pick ups per quadrant between {start_hour:02.0f}h - {end_hour:02.0f}h (out of all pick ups in quadrant)"
    elif var=='ratio_pickup_dropoff':
        text=f"Ratio of pickups to dropoffs between {start_hour:02.0f}h - {end_hour:02.0f}h"

    return [fig,text]

#function to update overall plots
@app.callback(
    [Output(component_id='histograms', component_property='figure'),Output(component_id='line_plots', component_property='figure'),Output(component_id='pie_plots', component_property='figure'),
    Output('slct_quad_title','children')],
    [Input('choropleth', 'clickData')]
)
def plot_overall_data(clickData):
    if clickData:
        quad_selected=clickData['points'][0]['location']
    else:
        quad_selected=2503


    data_pickup_quad=data[data['quad_pickup']==quad_selected]
    data_dropoff_quad=data[data['quad_drop']==quad_selected]
    ############################################
    # 1. histogram by hour
    count_by_hour=pd.DataFrame(map(list,zip(*[data_dropoff_quad.groupby('hour').size(),data_pickup_quad.groupby('hour').size()])),columns=['dropoff','pickup']).reset_index().rename(columns={'index': 'hour'})

    data_melt=pd.melt(count_by_hour,value_vars=['dropoff','pickup'],id_vars='hour').rename(columns={'value':'Total trips','variable':'type'})

    fig_hist=px.bar(data_melt,x='hour',y='Total trips',facet_row='type',color='type',title='Trips by hour on quad',width=900, text_auto=True)


    fig_hist.update_layout(
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
    fig_hist.for_each_annotation(lambda a: a.update(text=''))
    axis_id=0
    for axis in fig_hist.layout:
        if type(fig_hist.layout[axis]) == go.layout.YAxis:
            if axis_id==0:
                fig_hist.layout[axis].title.text = 'Dropoffs'
                axis_id+=1
            else:
                fig_hist.layout[axis].title.text = 'Pickups'
    fig_hist.update_traces(textangle=90, cliponaxis=True,textfont=dict(
            color="white"
        ))
    fig_hist.update_yaxes(showgrid=False)


    #2. Line graph by year

    count_by_year=data_dropoff_quad.groupby('year').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'dropoff'}).merge(data_pickup_quad.groupby('year').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'pickup'}),on='year')
    count_by_year=count_by_year[count_by_year['year']!=2015]
    data_melt=pd.melt(count_by_year,value_vars=['dropoff','pickup'],id_vars='year').rename(columns={'value':'Total trips','variable':'type'})

    fig_line=px.line(data_melt,x='year',y='Total trips',color='type',title='Trips over time on quad', markers=True,width=400)
    fig_line.update_layout(
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
    ),
        font=dict(size=10))
    fig_line.update_xaxes(showgrid=False)

    #3. Pie chart of number of passengers
    count_by_passenger_drop=data_dropoff_quad.groupby('passenger_count').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'dropoff'})

    count_by_passenger_pick=data_pickup_quad.groupby('passenger_count').count()[['pickup_longitude']].reset_index().rename(columns={'pickup_longitude':'pickup'})

    #get number of passengers that appear and get corresponding colors
    colors_graph1=list(map(lambda x:color_labels[x],list(count_by_passenger_drop['passenger_count'])))

    colors_graph2=list(map(lambda x:color_labels[x],list(count_by_passenger_pick['passenger_count'])))


    fig_pie = make_subplots(rows=2, cols=1, specs=[[{'type':'domain'}],[ {'type':'domain'}]])
    fig_pie.add_trace(go.Pie(labels=count_by_passenger_drop.passenger_count, values=count_by_passenger_drop.dropoff, name="Pick Up",sort=False,marker_colors=colors_graph1),
                1, 1)
    fig_pie.add_trace(go.Pie(labels=count_by_passenger_pick.passenger_count, values=count_by_passenger_pick.pickup, name="Drop Off",sort=False,marker_colors=colors_graph2),
                2, 1)

    # Use `hole` to create a donut-like pie chart
    fig_pie.update_traces(hole=.5, hoverinfo="label+value+percent+name",textposition='inside')

    fig_pie.update_layout(
        title_text="Rides by number of passengers",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Pick Up', x=0.5, y=0.18, font_size=10, showarrow=False),
                    dict(text='Drop Off', x=0.5, y=0.82, font_size=10, showarrow=False)],
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        legend=dict(
        xanchor="left",
        x=-0.5,
        title='Passengers',
        font=dict(size=10),
        title_font_size=10
    ))

    text_title=f'Overview of Uber rides on quad {quad_selected:.0f}'

    return [fig_hist,fig_line,fig_pie,text_title]

#function to update map plots
@app.callback(
    [Output(component_id='map_origins', component_property='figure'),Output(component_id='map_destinations', component_property='figure'),Output(component_id='map_details', component_property='figure'),
    Output('slct_quad_title_map','children')],
    [Input('choropleth', 'clickData')]
)
def plot_overall_data(clickData):
    if clickData:
        quad_selected=clickData['points'][0]['location']
    else:
        quad_selected=2503

    data_pickup_quad=data[data['quad_pickup']==quad_selected]
    data_dropoff_quad=data[data['quad_drop']==quad_selected]

    #from where people come to reach this quad?
    map_origin=plot_choropleth_from_selected_quad(data_dropoff_quad,'quad_pickup')

    #to where people from this quad go?
    map_destination=plot_choropleth_from_selected_quad(data_pickup_quad,'quad_drop')

    quad_data=quad_properties_df[quad_properties_df['quad_id']==quad_selected]

    lat_center=quad_data.iloc[0][['lat','lat2']].mean()
    lon_center=quad_data.iloc[0][['lon','lon2']].mean()


    fig_details = go.Figure(data=go.Scattermapbox(
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
    fig_details.add_trace(go.Scattermapbox(
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
    fig_details.update_layout(
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
    text=f'Origins and destinations of rides associated with quad {quad_selected:.0f}'


    return [map_origin,map_destination,fig_details,text]
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)

# %%
