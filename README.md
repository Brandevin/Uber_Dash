# Uber_Dash
Dashboard created using plotly and dash to visualize origin and destination of uber rides in new york

# Files
01. generate_files.py: responsible for taking original uber dataset (uber.csv) and assigning quadrants to pickup and dropoff location. Generates uber_rides_processed.csv and quad_properties.csv
02. uber_dashboard.py and test.py: auxiliary codes to plot plotly graphs beforehand
03. app.py: Main app file, final version of the app.
04. Procfile: Specifies information necessary to deploy on heroku.
05. requirements: Specifies python libraries to heroku


# Python Version

Python 3.7

# Libraries needed

dash==2.1.0

geopandas==0.9.0

matplotlib==3.5.0

numpy==1.20.3

pandas==1.3.4

plotly==5.6.0

python_dateutil==2.8.2

Shapely==1.7.1

gunicorn

