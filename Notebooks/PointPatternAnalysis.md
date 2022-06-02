---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

**CURSO**: Análisis Geoespacial, Departamento de Geociencias y Medio Ambiente, Universidad Nacional de Colombia - sede Medellín <br/>
**Profesor**: Edier Aristizábal (evaristizabalg@unal.edu.co) <br />
**Classroom code**: [32cjlau] <br />
**Credits**: The content of this notebook is taken from several sources, such as [Introduction to web mapping by Michael Dorman](https://web-mapping.surge.sh/index.html), [Geographic Data Science with PySAL and the PyData Stack](https://geographicdata.science/book/intro.html) by  Sergio Rey - [Center for Geospatial Sciences, University of California, Riverside](http://spatial.ucr.edu/peopleRey.html), [Dani Arribas-Bel](http://darribas.org/) -  University of Liverpool & Levi J. Wolf; Geographic Information Analysis by David Sullivanb & David Unwin (2010); Spatial Analysis Methods and Practice by George Grekousis (2020); Spatial Analysis using big data by Yoshiki Yamagata and Hajime Seya (2020), and Applied spatial statistics and econometrics by Katarzyna Kopczewska (2021). Every effort has been made to trace copyright holders of the materials used in this book. The author apologies for any unintentional omissions and would be pleased to add an acknowledgment in future editions. 


# Point Pattern Analysis



## Introduction

Points are spatial entities that can be understood in two fundamentally different ways:

* On the one hand, points can be seen as fixed objects in space, which is to say their location is taken as given (*exogenous*). In this interpretation, the location of an observed point is considered as secondary to the value observed at the point. Think of this like measuring the number of cars traversing a given road intersection; the location is fixed, and the data of interest comes from the measurement taken at that location. The analysis of this kind of point data is very similar to that of other types of spatial data such as polygons and lines. 

* On the other hand, an observation occuring at a point can also be thought of as a site of measurement from an underlying geographically-continuous process. In this case, the measurement could theoretically take place anywhere, but was only carried out or conducted in certain locations. Think of this as measuring the length of birds' wings: the location at which birds are measured reflects the underlying geographical process of bird movement and foraging, and the length of the birds' wings may reflect an underlying ecological process that varies by bird. This kind of approach means that both the location and the measurement matter. This is the perspective we will adopt in the rest of the chapter.

When points are seen as events that could take place in several locations but only happen in a few of them, a collection of such events is called a *point pattern*. In this case, the location of points is one of the key aspects of interest for analysis. A good example of a point pattern is geo-tagged photographs: they could technically happen in many locations but we usually find photos tend to concentrate only in a handful of them. Point patterns can be *marked*, if more attributes are provided with the location, or *unmarked*, if only the coordinates of where the event occured are provided. Continuing the photo example, an unmarked pattern would result if only the location where are taken is used for analysis, while we would be speaking of a marked point pattern if other attributes, such as the time, camera model, or a "image quality score" was provided with the location.

Point pattern analysis is thus concerned with the visualization, description, statistical characerization, and modeling of point patterns, trying to understand the generating process that gives rise and explains the observed data. Common questions in this domain include:

- *What does the pattern look like?* 
- *What is the nature of the distribution of points?* 
- *Is there any structure in the way locations are arranged over space? That is, are events clustered? or are they dispersed?*
- *Why do events occur in those places and not in others?* 

These are the most common questions in the analysis of point patterns. 

At this point, it is useful to remind ourselves of an important distinction, that between process and pattern. The former relates to the underlying mechanism that is at work to generate the outcome we end up observing. Because of its abstract nature, we do not get to see it. However, in many contexts, the key focus of any analysis is to learn about *what* determines a given phenomenon and *how* those factors combine to generate it. In this context, "process" is associated with the *how*. "Pattern," on the other hand, relates to the result of that process. In some cases, it is the only trace of the process we can observe and thus the only input we have to work with in order to reconstruct it. Although directly observable and, arguably, easier to tackle, pattern is only a reflection of process. The real challenge is not to characterize the former but to use it to work out the latter.

In this chapter, we provide an introduction to point patters through geo-tagged Flickr photos from Tokyo. We will treat the phenomena represented in the data as events: photos could be taken of any place in Tokyo, but only certain locations are captured. Keep in mind this understanding of Tokyo photos is not immutable: one could conceive cases where it makes sense to take those locations as given and look at the properties of each of them ignoring their "event" aspect. However, in this context, we will focus on those questions that relate to location and the collective shape of locations. The use of these tools will allow us to transform a long list of unintelligible XY coordinates into tangible phenomena with a characteristic spatial structure, and to answer questions about the center, dispersion, and clustering of attractions in Tokyo for Flickr users.



```python
%matplotlib inline

import numpy as np
import pandas as pd
import geopandas as gpd
import pysal
import seaborn as sbn
import contextily as ctx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
```

## The Tokyo photographs dataset

The rise of new forms of data such as geotagged photos uploaded to online services is creating new ways for researchers to study and understand cities. Where do people take pictures? When are those pictures taken? Why certain places attract many more photographers than others? All these questions and more become more than just rethorical ones when we consider, for example,  online photo hosting services as volunteered geographic information (VGI, {cite}`Goodchild2007citizens`). In this chapter we will explore metadata from a sample of georeferenced images uploaded to [Flickr](https://www.flickr.com/) and extracted thanks to the [100m Flickr dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67). In doing so, we will introduce a few approaches that help us better understand the distribution and characteristics of a point pattern. 

To get started, let's load the flickr data first:

```python
# Read remote file
db = pd.read_csv("https://geographicdata.science/book/_downloads/7fb86b605af15b3c9cbd9bfcbead23e9/tokyo_clean.csv")
```

The table contains the following information about the sample of 10,000 photographs: the ID of the user who took the photo; the location expressed as latitude and longitude columns; a transformed version of those coordinates expressed in Pseudo Mercator; the timestamp when the photo was taken; and the url where the picture they refer to is stored online:

```python
db.info()
```
```python
db.head(2)
```

```python
type(db)
```

<!-- #region tags=[] -->
# Read the tokyo map
<!-- #endregion -->

```python
areas = gpd.read_file("https://darribas.org/gds_course/content/data/tokyo_admin_boundaries.geojson")
```

```python
type(areas)
```

```python
areas.plot();
```

```python
areas.head(2)
```

```python
areas.crs
```

The final bit we need to get out of the way is attaching the administrative area code where a photo is located to each area. This can be done with a GIS operation called “spatial join”.

```python
tokyo_gdf = gpd.GeoDataFrame({"geometry": gpd.points_from_xy(db["longitude"], db["latitude"])}, crs="EPSG:4326").join(db)
crosswalk = gpd.sjoin(tokyo_gdf, areas, how="inner")
db["admin_area"] = crosswalk["GID_2"]
```

```python
crosswalk.head(2)
```

```python
db.head(2)
```

```python
db.info()
```

Note that the data is provided as a `.csv` file, so the spatial information is encoded as separate columns, one for each coordinate. This is contrast to how we have consumed spatial data in previous chapters, where spatial information was stored in a single column and encoded in geometry objects.



## Visualization



### Dots on a map

The first step to get a sense of what the spatial dimension of this dataset looks like is to plot it. At its most basic level, we can generate a scatter plot with `seaborn`:


```python
# Generate scatter plot
sbn.jointplot(x='longitude', y='latitude', data=db, s=0.5);
```


This is a good start: we can see dots tend to be concentrated in the center of the covered area in a very (apparently) not random. Furthermore, within the broad pattern, we can also see there seems to be more localised clusters. However, the plot above has two key drawbacks: one, it lacks geographical context; and two, there are areas where the density of points is so large that it is hard to tell anything beyond a blue blurb. 

Start with the context. The easiest way to provide additional context is by ovelaying a tile map from the internet. Let us quickly call `contextily` for that, and integrate it with `jointplot`:

```python
# Generate scatter plot
joint_axes = sbn.jointplot(x='longitude', y='latitude', data=db, s=0.5)
# Add dark basemap
ctx.add_basemap(
    joint_axes.ax_joint,
    crs="EPSG:4326",
    source=ctx.providers.Stamen.TonerLite
);
```


Note how we can pull out the axis where the points are plotted and add the basemap there, specifying the CRS as WGS84, since we are plotting longitude and latitude. Compared to the previous plot, adding a basemap to our initial plot makes the pattern of Flickr data clearer.


### Points meet polygons

The approach presented above works until a certain number of points to plot; tweaking dot transparency and size only gets us so far and, at some point, we need to shift the focus. Having learned about visualizing lattice (polygon) data, an option is to "turn" points into polygons and apply techniques like choropleth mapping to visualize their spatial distribution. To do that, we will overlay a polygon layer on top of the point pattern, *join* the points to the polygons by assigning to each point the polygon where they fall into, and create a choropleth of the counts by polygon. 

This approach is intuitive but of course raises the following question: *what polygons do we use to aggregate the points?* Ideally, we want a boundary delineation that matches as closely as possible the point generating process and partitions the space into areas with a similar internal intensity of points. However, that is usually not the case, no less because one of the main reasons we typically want to visualize the point pattern is to learn about such generating process, so we would typically not know a priori whether a set of polygons match it. If we cannot count on the ideal set of polygons to begin with, we can adopt two more realistic approaches: using a set of pre-existing irregular areas or create a artificial set of regular polygons. Let's explore both.


#### Irregular lattices

To exemplify this approach, we will use the administrative areas we have loaded above. Let's add them to the figure above to get better context 

```python
# Plot photographs with smaller, more translucent dots
ax = db.plot.scatter("longitude", 
                        "latitude",
                        s=0.25,
                        c="xkcd:bright yellow",
                        alpha=0.5,
                        figsize=(9, 9)
                       )
# Add administrative boundaries
areas.plot(ax=ax,
           facecolor="none",
           edgecolor="xkcd:pale lavender"
          )
# remove axis
ax.set_axis_off()
# Add dark basemap
ctx.add_basemap(ax, 
               crs="EPSG:4326",
               source=ctx.providers.Stamen.TonerLite
              )
```

Now we need to know how many photographs each are contains. Our photograph table already contains the area ID, so all we need to do here is counting by area and attaching the count to the `areas` table. We rely here on the `groupby` operator which takes all the photos in the table and "groups" them "by" their administrative ID. Once grouped, we apply the method `size`, which counts how many elements each group has and returns a column indexed on the LSOA code with all the counts as its values. We end by assigning the counts to a newly created column in the `areas` table.

```python
# Create counts
photos_by_area = db.groupby("admin_area").size()
# Assign counts into a table of its own 
# and joins it to the areas table
areas = areas.join(pd.DataFrame({"photo_count": photos_by_area}),on="GID_2")
```

```python
pepito=pd.DataFrame({"photo_count": photos_by_area})
```

```python
areas.head(2)
```

The lines above have created a new column in our `areas` table that contains the number of photos that have been taken within each of the polygons in the table.

At this point, we are ready to map the counts. Technically speaking, this is a choropleth just as we have seen many times before:

```python
# Set up figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot the equal interval choropleth and add a legend
areas.plot(column='photo_count', 
           scheme='natural_breaks', 
           ax=ax,
           legend=True,
           cmap='OrRd', edgecolor='k',
           k=4,
           legend_kwds={"loc": 4}
          )
# Remove the axes
#ax.set_axis_off()
# Set the title
ax.set_title("Quantile map of photo counts by administrative boundary")
# Add dark basemap
ctx.add_basemap(ax, 
               crs="EPSG:4326",
               source=ctx.providers.Stamen.TonerLite
              )
# Draw map
plt.show()
```

The map above clearly shows a concentration of photos in the centre of Tokyo. However, it is important to remember that the map is showing *raw* counts of tweets. In the case to photos, as with many other phenomena, it is crucial to keep in mind the "container geography". In this case, different administrative areas have different sizes. Everything else equal, a larger polygon may contain more photos, simply because it covers a larger space. To obtain a more accurate picture of the _intensity_ of photos by area, what we would like to see is a map of the *density* of photos, not of raw counts. To do this, we can divide the count per polygon by the area of the polygon.

Let's first calculate the area in Sq. metres of each administrative delineation:

```python
areas["area_sqm"] = areas.to_crs(epsg=2459).area * 1e-6
```

And we can add the photo density as well:

```python
areas["photo_density"] = areas["photo_count"] / areas["area_sqm"]
```

```python
areas.sample(5)
```

With the density at hand, creating the new choropleth is similar as above (check the code in the expandable):

```python
# Set up figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot the equal interval choropleth and add a legend
areas.plot(column='photo_density', 
           scheme='quantiles', 
           ax=ax,
           k=3,
           legend=True,
           legend_kwds={"loc": 4}
          )
# Remove the axes
ax.set_axis_off()
# Set the title
ax.set_title("Quantile map of photo density by administrative boundary")
# Add dark basemap
ctx.add_basemap(ax, 
               crs="EPSG:4326",
               source=ctx.providers.Stamen.TonerLite
              )
# Draw map
plt.show()
```

The pattern in the raw counts is similar to that of density, but we can see how some peripheral, large areas are "downgraded" when correcting for their size, while some smaller polygons in the centre display a higher value.


### Regular lattices: Hex-binning

Sometimes we either do not have any polygon layer to use or the ones we have are not particularly well suited to aggregate points into them. In these cases, a sensible alternative is to create an artificial topology of polygons that we can use to aggregate points. There are several ways to do this but the most common one is to create a grid of hexagons. This provides a regular topology (every polygon is of the same size and shape) that, unlike circles, cleanly exhausts all the space without overlaps and has more edges than squares, which alleviates edge problems.

Python has a simplified way to create this hexagon layer *and* aggregate points into it in one shot thanks to the method `hexbin`, which is available in every axis object (e.g. `ax`). Let us first see how you could create a map of the hexagon layer alone:

One solution to get around cluttering relates to what we referred to earlier as moving from "tables to surfaces". We can now recast this approach as a *spatial* or *2-dimensional histogram*. Here, we generate a regular grid (either squared or hexagonal), count how many dots fall within each grid cell, and present it as we would any other choropleth. This is attractive because it is simple, intuitive and, if fine enough, the regular grid removes some of the area distortions choropleths may induce. For this illustration, let us use use hexagonal binning (sometimes called hexbin) because it has slightly nicer properties than squared grids, such as less shape distortion and more regular connectivity between cells. Creating a hexbin 2-d histogram is straightforward in Python using the `hexbin` function:


```python
# Set up figure and axis
f, ax = plt.subplots(1, figsize=(12, 9))
# Generate and add hexbin with 50 hexagons in each dimension, no borderlines, half transparency, and the reverse viridis colormap
hb = ax.hexbin(
    db['x'], 
    db['y'],
    gridsize=50, 
    linewidths=0,
    alpha=0.5, 
    cmap='viridis_r'
)
# Add basemap
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)
# Add colorbar
plt.colorbar(hb)
# Remove axes
ax.set_axis_off()
```


Voila, this allows a lot more detail! It is now clear that the majority of photographs relate to much more localised areas and that the previous map was obscuring this. See how all it takes is to set up the figure and call `hexbin` directly using the set of coordinate columns (`db["longitude"]` and `db["latitude"]`). Additional arguments we include is the number of hexagons by axis (`gridsize`, 50 for a 50 by 50 layer), and the transparency we want (80%). Additionally, we include a colorbar to get a sense of how counts are mapped to colors. Note that we need to pass the name of the object that includes the `hexbin` (`hb` in our case), but keep in mind this is optional, you do not need to always create one.

### Kernel Density Estimation (KDE)

Using a hexagonal binning can be a quick solution when we do not have a good polygon layer to overlay the points directly and some of its properties, such as the equal size of each polygon, can help alleviate some of the problems with a "bad" irregular topology (one that does not fit the underlying point generating process). However, it does not get around the issue of the modifiable areal unit problem (M.A.U.P.). At the end of the day, we are still imposing arbitrary boundary lines and aggregating based on them, so the possibility of mismatch with the underlying distribution of the point pattern is very real.

One way to work around this problem is to avoid aggregating into another geography altogether. Instead, we can aim at estimating the *continuous* observed probability distribution. The most commonly used method to do this is the so called *kernel density estimate* (KDE). The idea behind KDEs is to count the number of points in a *continious* way. Instead of using discrete counting, where you include a point in the count if it is inside a certain boundary and ignore it otherwise, KDEs use functions (kernels) that include points but give different weights to each one depending of how far of the location where we are counting the point is.

The actual algorithm to estimate a kernel density is not trivial but its application in Python is rather simplified by the use of Seaborn. KDE's however are fairly computationally intensive. When you have a large point pattern like we do in the `tokyo` example (10,000 points), its computation can take a bit long. To get around this issue, we create a random subset, which retains the overall structure of the pattern, but with much fewer points. Let's take a subset of 1,000 random points from our original table: 

```python
# Set up figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Generate and add KDE with a shading of 50 gradients 
# coloured contours, 75% of transparency,
# and the reverse viridis colormap
sbn.kdeplot(
    db['x'], 
    db['y'],
    n_levels=30, 
    shade=True,
    alpha=0.55, 
    cmap='viridis_r'
)
# Add basemap
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)
# Remove axes
ax.set_axis_off()
```


The result is a smoother output that captures the same structure of the hexbin but "eases" the transitions between different areas. This provides a better generalisation of the theoretical probability that a picture *might* occur at any given point. This is useful in some cases, but is mainly of use to escape the restrictions imposed by a regular grid of hexagons or squares. 

Seaborn greatly streamlines the process and boils it down to a single line. The method `sns.kdeplot` (which we can also use to create a KDE of a single variable) takes the X and Y coordinate of the points as the only compulsory attributes. In addition, we specify the number of levels we want the color gradient to have (`n_levels`), whether we want to color the space in between each level (`share`, yes), and the colormap of choice.


### Leaflet
The Leaflet.heat library is a geoprocessing plugin for the Leaflet library. The term “plugin”, in this context, means that Leaflet.heat is an extension of the Leaflet library and thus cannot be used without Leaflet.  Leaflet.heat does just one thing—drawing a heatmap to display the density of point data. The Leaflet.heat plugin is one of many created to extend the core functionality of Leaflet98.

Drawing individual points can be visually overwhelming, slow, and hard to comprehend when there are too many of them. Heatmaps are a convenient alternative way of conveying the distribution of large amounts of point data. A heatmap is basically a way to summarize and display point density, instead of showing each and every individual point. Technically speaking, a heatmap uses two-dimensional Kernel Density Estimation (KDE) to calculate a smooth surface representing point density.

The following example draws a heatmap of all rare plant observations from the plants table on CARTO. Overall, there are 23,827 points in the plants. Drawing markers, or even the simpler circle markers, for such a large amount of points is usually not a good idea. First, it may cause the browser to become unresponsive due to computational intensity. Second, it is usually difficult to pick the right styling (size, opacity, etc.) for each point so that dense locations are clearly distinguished on the various map zoom levels the user may choose. This is where the density-based heatmaps become very useful. Particularly, the Leaflet.heat library automatically re-calculates the heatmap for each zoom level, allowing the user to conveniently explore point density on both large and small scales.


```
<!DOCTYPE html>
<html>
<head>
    <title>Point density map</title>
    <meta name="viewport" content="width=device-width,  initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.css"/>
    <script src="https://code.jquery.com/jquery-3.6.0.js"
  integrity="sha256-H+K7U5CnXl1h5ywQfKtSj8PCmoN9aaq30gDh27Xc0jk="
  crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.js"></script>
    <style>
        body {
            padding: 0;
            margin: 0;
        }
        html, body, #map {
            height: 100%;
            width: 100%;
        }
    </style>
</head>
<body>
    <div id="map"></div>

    <script>

        var map = L.map("map").setView([32, 35], 8);
        L.tileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", 
            {attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}
        ).addTo(map);
        
        // Define variables
        var url = "https://michaeldorman.carto.com/api/v2/sql?format=GeoJSON&q=";
        var sqlQuery = "SELECT the_geom FROM plants";
        
        // load GeoJSON from an external file
        $.getJSON(url + sqlQuery, function(data) {
            var locations = [];
            $.each(data.features, function(key, value) {
                var coords = value.geometry.coordinates;
                var location = [coords[1], coords[0], 0.5];
                locations.push(location);
            });
            L.heatLayer(locations, {radius: 20, minOpacity: 0.5}).addTo(map); 
        });
                
            </script>
   
   <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"></script>

</body>
</html>
```


## Centrography

Centrography is the analysis of centrality in a point pattern. By "centrality," we mean the general location and dispersion of the pattern. If the hexbin above can be seen as a "spatial histogram", centrography is the point pattern equivalent of measures of central tendency such as the mean. These measures are useful because they allow us to summarise spatial distributions in smaller sets of information (e.g. a single point). Many different indices are used in centrography to provide an indication of "where" a point pattern is, how tightly the point pattern clusters around its center, or how irregular its shape is. 



### Tendency

A common measure of central tendency for a point pattern is its *center of mass*. For marked point patterns, the center of mass identifies a central point close to observations that have higher values in their marked attribute. For unmarked point patterns, the center of mass is equivalent to the *mean center*, or average of the coordinate values. In addition, the *median center* is analogous to the *median* elsewhere, and represents a point where half of the data is above or below the point & half is to its left or right. We can analyze the mean center with our flickr point pattern using the `pointpats` package in Python. 


```python
from pointpats import centrography
```


```python
mean_center = centrography.mean_center(db[['x', 'y']])
med_center = centrography.euclidean_median(db[['x', 'y']])
```

It is easiest to visualize this by plotting the point pattern and its mean center alongside one another:


```python
# Generate scatter plot
joint_axes = sbn.jointplot(
    x='x', y='y', data=db, s=0.75, height=9
)
# Add mean point and marginal lines
joint_axes.ax_joint.scatter(
    *mean_center, color='red', marker='x', s=50, label='Mean Center'
)
joint_axes.ax_marg_x.axvline(mean_center[0], color='red')
joint_axes.ax_marg_y.axhline(mean_center[1], color='red')
# Add median point and marginal lines
joint_axes.ax_joint.scatter(
    *med_center, color='limegreen', marker='o', s=50, label='Median Center'
)
joint_axes.ax_marg_x.axvline(med_center[0], color='limegreen')
joint_axes.ax_marg_y.axhline(med_center[1], color='limegreen')
# Legend
joint_axes.ax_joint.legend()
# Add basemap
ctx.add_basemap(
    joint_axes.ax_joint, 
    source=ctx.providers.CartoDB.Positron
)
# Clean axes
joint_axes.ax_joint.set_axis_off()
# Display
plt.show()
```


The discrepancy between the two centers is caused by the skew; there are many "clusters" of pictures far out in West and South Tokyo, whereas North and East Tokyo is densely packed, but drops off very quickly. Thus, the far out clusters of pictures pulls the mean center to the west and south, relative to the median center. 



### Dispersion

A measure of dispersion that is common in centrography is the *standard distance*. This measure provides the average distance away from the center of the point cloud (such as measured by the center of mass). This is also simple to compute using `pointpats`, using the `std_distance` function:


```python
centrography.std_distance(db[['x','y']])
```
This means that, on average, pictures are taken around 8800 metres away from the mean center. 

Another helpful visualization is the *standard deviational ellipse*, or *standard ellipse*. This is an ellipse drawn from the data that reflects both its center and dispersion. To visualize this, we first compute the axes and rotation using the `ellipse` function in `pointpats`:


```python
major, minor, rotation = centrography.ellipse(db[['x','y']])
```


```python
major
```

```python
minor
```

```python
from matplotlib.patches import Ellipse

# Set up figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot photograph points
ax.scatter(db['x'], db['y'], s=0.75)
ax.scatter(*mean_center, color='red', marker='x', label='Mean Center')
ax.scatter(*med_center, color='limegreen', marker='o', label='Median Center')

# Construct the standard ellipse using matplotlib
ellipse = Ellipse(xy=mean_center, # center the ellipse on our mean center
                  width=major*2, # centrography.ellipse only gives half the axis
                  height=minor*2, 
                  angle = numpy.rad2deg(rotation), # Angles for this are in degrees, not radians
                  facecolor='none', 
                  edgecolor='red', linestyle='--',
                  label='Std. Ellipse')
ax.add_patch(ellipse)

ax.legend()
# Display
# Add basemap
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)
plt.show()
```


### Extent

The last collection of centrography measures we will discuss characterizes the extent of a point cloud. Four shapes are useful, and reflect varying levels of how "tightly" they bind the pattern. 

Below, we'll walk through how to construct each example and visualize them at the end. To make things more clear, we'll use the flickr photos for the most prolific user in the dataset (ID: `95795770`) to show how different these results can be.


```python
db.head(2)
```

```python
user = db.query('user_id == "95795770@N00"')
coordinates = user[['x','y']].values
```

First, we'll compute the **convex hull**, which is the tighest *convex* shape that encloses the user's photos. By *convex*, we mean that the shape never "doubles back" on itself; it has no divets, valleys, crenelations, or holes. All of its interior angles are smaller than 180 degrees.  This is computed using the `centrography.hull` method.


```python
convex_hull_vertices = centrography.hull(coordinates)
```

Second, we'll compute the **alpha shape**, which can be understood as a "tighter" version of the convex hull. One way to think of a convex hull is that it's the space left over when rolling a **really** large ball or circle all the way around the shape. The ball is so large relative to the shape, its radius is actually infinite, and the lines forming the convex hull are actually just straight lines! 

In contrast, you can think of an alpha shape as the space made from rolling a *small* balls around the shape. Since the ball is smaller, it rolls into the dips & valleys created between points. As that ball gets bigger, the alpha shape becomes the convex hull. But, for small balls, the shape can get very tight indeed. In fact, if alpha gets too small, it "slips" through the points, resulting in *more than one hull!* As such, the `pysal` package has an `alpha_shape_auto` function to find the smallest *single* alpha shape, so that you don't have to guess at how big the ball needs to be. 


```python
import libpysal
alpha_shape, alpha, circs = libpysal.cg.alpha_shape_auto(coordinates, return_circles=True)
```

```python
alpha_shape
```

To illustrate, the figure below has the tightest single alpha shape shown in green and the original source points shown in black. The "bounding" circles shown in the figure all have a radius of $8652$ meters. The circles are plotted where our "bounding" disk touches two or three of the points in the point cloud. You can see that the circles "cut into" the convex hull, shown in blue dashed lines, up until they touch two (or three) points. Any tighter, and the circle would disconnect one of the points on the boundary of the alpha shape. 


```python
from descartes import PolygonPatch #to plot the alpha shape easily
f,ax = plt.subplots(1,1, figsize=(9,9))

# Plot a green alpha shape
ax.add_patch(
    PolygonPatch(
        alpha_shape, 
        edgecolor='green', 
        facecolor='green', 
        alpha=.2, 
        label = 'Tighest single alpha shape'
    )
)

# Include the points for our prolific user in black
ax.scatter(
    *coordinates.T, color='k', marker='.', label='Source Points'
)

# plot the circles forming the boundary of the alpha shape
for i, circle in enumerate(circs):
    # only label the first circle of its kind
    if i == 0:
        label = 'Bounding Circles'
    else:
        label = None
    ax.add_patch(
        plt.Circle(
            circle, 
            radius=alpha, 
            facecolor='none', 
            edgecolor='r', 
            label=label
        )
    )

# add a blue convex hull
ax.add_patch(
    plt.Polygon(
        convex_hull_vertices, 
        closed=True, 
        edgecolor='blue', 
        facecolor='none', 
        linestyle=':', 
        linewidth=2,
        label='Convex Hull'
    )
)

# Add basemap
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)

plt.legend();
```
We will cover three more bounding shapes, all of them rectangles or circles. First, two kinds of **minimum bounding rectangles**. They both are constructed as the tightest *rectangle* that can be drawn around the data that contains all of the points. One kind of minimum bounding rectangle can be drawn just by considering vertical and horizontal lines. However, diagonal lines can often be drawn to construct a rectangle with a smaller area. This means that the **minimum rotated rectangle** provides a tighter rectangular bound on the point pattern, but the rectangle is askew or rotated. 

For the minimum rotated rectangle, we will use the `minimum_rotated_rectangle` function from the `pointpats.centrography` module. 


```python
min_rot_rect = centrography.minimum_rotated_rectangle(coordinates)
```

And, for the minimum bounding rectangle without rotation, we will use the `minimum_bounding_rectangle` function from the `pointpats` package.


```python
min_rect_vertices = centrography.minimum_bounding_rectangle(coordinates)
```

Finally, the **minimum bounding circle** is the smallest circle that can be drawn to enclose the entire dataset. Often, this circle is bigger than the minimum bounding rectangle. It is implemented in the `minimum_bounding_circle` function in `pointpats`. 


```python
(center_x, center_y), radius = centrography.minimum_bounding_circle(coordinates)
```

Now, to visualize these, we'll convert the raw vertices into matplotlib patches: 


```python
from matplotlib.patches import Polygon, Circle, Rectangle
from descartes import PolygonPatch

# Make a purple alpha shape
alpha_shape_patch = PolygonPatch(
    alpha_shape, 
    edgecolor='purple', 
    facecolor='none', 
    linewidth=2,
    label='Alpha Shape'
)

# a blue convex hull
convex_hull_patch = Polygon(
    convex_hull_vertices, 
    closed=True, 
    edgecolor='blue', facecolor='none', 
    linestyle=':', linewidth=2,
    label='Convex Hull'
)

# a green minimum rotated rectangle
min_rot_rect_patch = Polygon(
    min_rot_rect, 
    closed=True, 
    edgecolor='green', 
    facecolor='none', 
    linestyle='--', 
    label='Min Rotated Rectangle', 
    linewidth=2
)

# compute the width and height of the 
min_rect_width = min_rect_vertices[2] - min_rect_vertices[0]
min_rect_height = min_rect_vertices[2] - min_rect_vertices[0]

# a goldenrod minimum bounding rectangle
min_rect_patch = Rectangle(
    min_rect_vertices[0:2], 
    width = min_rect_width,
    height = min_rect_height,
    edgecolor='goldenrod', facecolor='none', 
    linestyle='dashed', linewidth=2, 
    label='Min Bounding Rectangle', 
)

# and a red minimum bounding circle
circ_patch = Circle(
    (center_x, center_y), 
    radius=radius,
    edgecolor='red', 
    facecolor='none', 
    linewidth=2,
    label='Min Bounding Circle'
)
```

Finally, we'll plot the patches together with the photograph locations below:


```python
f,ax = plt.subplots(1, figsize=(10,10))

ax.add_patch(alpha_shape_patch)
ax.add_patch(convex_hull_patch)
ax.add_patch(min_rot_rect_patch)
ax.add_patch(min_rect_patch)
ax.add_patch(circ_patch)

ax.scatter(db.x, db.y, s=.75, color='grey')
ax.scatter(user.x, user.y, s=100, color='r', marker='x')
ax.legend(ncol=1, loc='center left')

# Add basemap
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)
plt.show()
```


Each gives a different impression of the area enclosing the user's range of photographs. In this, you can see that the the alpha shape is much tighter than the rest of the shapes. The minimum bounding rectangle & circle are the "loosest" shapes, in that they contain the most area outside of the user's typical area. But, they're also the simplest shapes to draw and understand. 




### [Turf.js](http://turfjs.org/)
Turf.js is a JavaScript library for client-side geoprocessing. It includes a comprehensive set of functions, covering a wide range of geoprocessing tasks. The Turf.js library also has excellent [documentation](http://turfjs.org/docs/), including a small visual example for each of its function


```
<!DOCTYPE html>
<html>
<head>
    <title>Agregar contenido</title>
    <meta name="viewport" content="width=device-width,  initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.css"/>
    <script src="https://code.jquery.com/jquery-3.6.0.js"
  integrity="sha256-H+K7U5CnXl1h5ywQfKtSj8PCmoN9aaq30gDh27Xc0jk="
  crossorigin="anonymous"></script>
  <script src="https://npmcdn.com/@turf/turf/turf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.js"></script>
    <style>
        body {
            padding: 0;
            margin: 0;
        }
        html, body, #map {
            height: 100%;
            width: 100%;
        }
    </style>
</head>
<body>
    <div id="map"></div>

<script>

// Bounds
var bounds = [-74.26801, 0.49708, -75.90094, 7.36403];

// Initialize map
var map = L.map("map").fitBounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]]);
L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", 
    {attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}
).addTo(map);

// Define layers
var pnt_layer = L.layerGroup().addTo(map);
var tin_layer = L.layerGroup().addTo(map);

// Calculate random points
var points = turf.randomPoint(20, {bbox: bounds});
L.geoJSON(points, {
    onEachFeature: function(feature, layer) {
        layer.on("drag", drawTIN);
    },
    pointToLayer: function(geoJsonPoint, latlng) {
        return L.marker(latlng, {draggable: true});
    }
}).addTo(pnt_layer);

// Function to draw TIN
function drawTIN() {
    tin_layer.clearLayers();
    points = pnt_layer.toGeoJSON();
    tin = turf.tin(points);
    tin = L.geoJSON(tin);
    tin.addTo(tin_layer);
}
drawTIN();

    </script>

</body>
</html>
```


## Randomness & clustering

Beyond questions of centrality and extent, spatial statistics on point patterns are often concerned with how *even* a distribution of points is. By this, we mean whether points tend to all cluster near one another or disperse evenly throughout the problem area. Questions like this refer to the *intensity* or *dispersion* of the point pattern overall. In the jargon of the last two chapters, this focus resembles the goals we examined when we introduced [global spatial autocorrelation](06_spatial_autocorrelation): what is the overall degree of *clustering* we observe in the pattern? Spatial statistics has devoted plenty of effort to understand this kind of clustering. This section will cover methods useful for identifying clustering in point patterns.

The first set of techniques, **quadrat** statistics, receive their name after their approach to split the data up into small areas (quadrants). Once created, these "buckets" are used to examinee the uniformity of counts across them. The second set of techniques all derive from Ripley (1988), and involve measurements of the distance between points in a point pattern. 


```python
from pointpats import distance_statistics, QStatistic, random, PointPattern
```

For the purposes of illustration, it also helps to provide a pattern derived from a known *completely spatially random* process. That is, the location and number of points is totally random; there is neither clustering nor dispersion. In point pattern analysis, this is known as a *poisson point process*. 

To simulate these processes from a given point set, you can use the `pointpats.random` module. 


```python
random_pattern = random.poisson(coordinates, size=len(coordinates))
```


```python
f,ax = plt.subplots(1, figsize=(9, 9))
plt.scatter(*coordinates.T, color='k', marker='.', label='Observed photographs')
plt.scatter(*random_pattern.T, color='r', marker='x', label='Random')
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)
ax.legend(ncol=1, loc='center left')
plt.show()
```


As you can see, the simulation (by default) works with the bounding box of the input point pattern. To simulate from more restricted areas formed by the point pattern, pass those hulls to the simulator! For example, to generate a random pattern within the alpha shapes: 


```python
random_pattern_ashape = random.poisson(alpha_shape, size=len(coordinates))
```


```python
f,ax = plt.subplots(1, figsize=(9, 9))
plt.scatter(*coordinates.T, color='k', marker='.', label='Observed')
plt.scatter(*random_pattern_ashape.T, color='r', marker='x', label='Random')
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)
ax.legend(ncol=1, loc='center left')
plt.show()
```


### Quadrat statistics

Quadrat statistics examine the spatial distribution of points in an area in terms of the count of observations that fall within a given cell. By examining whether observations are spread *evenly* over cells, the quadrat approach aims to estimate whether points are spread out, or if they are clustered into a few cells. Strictly speaking, quadrat statistics examine the *evenness* of the distribution over cells using a $\chi^2$ statistical test common in the analysis of contingency tables. 

In the `pointpats` package, you can visualize the results using the following `QStatistic.plot()` method. This shows the grid used to count the events, as well as the underlying pattern:


```python
qstat = QStatistic(coordinates, nx = 2, ny = 2)
qstat.plot()
```


In this case, for the default of a three by three grid spanning the point pattern, we see that the central square has over 350 observations, but the surrounding cells have many fewer flickr photographs. This means that the chi-squared test (which compares how likely this distribution is if the cell counts are uniform) will be statistically significant, with a very small p-value:


```python
qstat.chi2_pvalue
```
In contrast, our totally random point process will have nearly the same points in every cell:


```python
qstat_null = QStatistic(random_pattern)
qstat_null.plot()
```


This means its p-value will be large, and likely not significant:


```python
qstat_null.chi2_pvalue
```
Be careful, however: the fact that quadrat counts are measured in a *regular tiling* that is overlaid on top of the potentially irregular extent of our pattern can mislead us. In particular, irregular *but random* patterns can be mistakenly found "significant" by this approach. Consider our random set generated within the alpha shape polygon, with the quadrat grid overlaid on top:

```python
qstat_null_ashape = QStatistic(random_pattern_ashape)
qstat_null_ashape.plot()
```


The quadrat test finds this to be *statistically nonrandom*, while our simulating process ensured that *within the given study area*, the pattern is a complete spatially-random process. 

```python
qstat_null_ashape.chi2_pvalue
```
Thus, quadrat counts can have issues with irregular study areas, and care should be taken to ensure that clustering is not mistakenly identified. One way to interpret the quadrat statistic that reconciles cases like the one above is to think of it as a test that considers both the uniformity of points *and* the shape of their extent to examine whether the resulting pattern is uniform across a regular grid. In some cases, this is a useful tool; in others this needs to be used with caution.


### Ripley's alphabet of functions

The second group of spatial statistics we consider focuses on the distributions of two quantities in a point pattern: nearest neighbor distances and what we will term "gaps" in the pattern. They derive from seminal work by {cite}`Ripley1991` on how to characterize clustering or co-location in point patterns. Each of these characterize an aspect of the point pattern as we increase the distance range from each point to calculate them. 

The first function, Ripley's $G$, focuses on the distribution of nearest neighbor distances. That is, the $G$ function summarises the distances between each point in the pattern and their nearest neighbor. In the plot below, this nearest neighbor logic is visualized with the red dots being a detailed view of the point pattern and the black arrows indicating the nearest neighbor to each point. Note that sometimes two points are *mutual* nearest neighbors (and so have arrows going in both directions) but some are not. 


```python tags=["hide-input"]
# this code should be hidden in the book, and only the plot visible!
f,ax = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)
ax[0].scatter(*random_pattern.T, color='red')
ax[1].scatter(*random_pattern.T, color='red',
              zorder=100, marker='.', label='Points')
nn_ixs, nn_ds = PointPattern(random_pattern).knn(1)
first = True
for coord, nn_ix, nn_d in zip(random_pattern, nn_ixs, nn_ds):
    dx, dy = random_pattern[nn_ix].squeeze() - coord
    arrow = ax[1].arrow(*coord, dx,dy, 
                length_includes_head=True, 
                overhang=0, head_length=300*3,
                head_width=300*3, width=50*3,
                linewidth=0, facecolor='k',
                head_starts_at_zero=False)
    if first:
        plt.plot((1e100, 1e101), (0,1), color='k', 
                 marker='<', markersize=10,
                 label='Nearest Neighbor to Point')
    first = False

ax[0].axis([1.554e7, 1.556e7, 4240000, 4260000])
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].set_xticks([])
ax[0].set_yticks([])
f.tight_layout()
ax[1].legend(bbox_to_anchor = (.5,-.06), fontsize=16)
plt.show()
```


Ripley's $G$ keeps track of the proportion of points for which the nearest neighbor is within a given distance threshold, and plots that cumulative percentage against the increasing distance radiuses. The distribution of these cumulative percentage has a distinctive shape under completely spatially random processes. The intuition behind Ripley's G goes as follows: we can learn about how similar our pattern is to a spatially random one by computing the cumulative distribution of nearest neighbor distances over increasing distance thresholds, and comparing it to that of a set of simulated patterns that follow a known spatially-random process. Usually, a spatial Poisson point process is used as such reference distribution. 

To do this in the `pointpats` package, we can use the `g_test` function, which computes both the `G` function for the empirical data *and* these hypothetical replications under a completely spatially random process.


```python
g_test = distance_statistics.g_test(
    coordinates, support=40, keep_simulations=True
)
```

Thinking about these distributions of distances, a "clustered" pattern must have more points near one another than a pattern that is "dispersed"; and a completely random pattern should have something in between. Therefore, if the $G$ function increases *rapidly* with distance, we probably have a clustered pattern. If it increases *slowly* with distance, we have a dispersed pattern. Something in the middle will be difficult to distinguish from pure chance.

We can visualize this below. On the left, we plot the $G(d)$ function, with distance-to-point ($d$) on the horizontal axis and the fraction of nearest neighbor distances smaller than $d$ on the right axis. In red, the empirical cumulative distribution of nearest neighbor distances is shown. In blue, simulations (like the `random` pattern shown in the previous section) are shown. The bright blue line represents the average of all simulations, and the darker blue/black band around it represents the middle 95% of simulations. 

In this plot, we see that the red empirical function rises much faster than simulated completely spatially random patterns. This means that the observed pattern of this user's flickr photographs are *closer* to their nearest neighbors than would be expected from a completely spatially random pattern. The pattern is *clustered.*


```python tags=["hide-input"]
f,ax = plt.subplots(1,2,figsize=(9,3), 
                    gridspec_kw=dict(width_ratios=(6,3)))
# plot all the simulations with very fine lines
ax[0].plot(g_test.support, g_test.simulations.T, color='k', alpha=.01)
# and show the average of simulations
ax[0].plot(g_test.support, np.median(g_test.simulations, axis=0), color='cyan', 
         label='median simulation')


# and the observed pattern's G function
ax[0].plot(g_test.support, g_test.statistic, label = 'observed', color='red')

# clean up labels and axes
ax[0].set_xlabel('distance')
ax[0].set_ylabel('% of nearest neighbor\ndistances shorter')
ax[0].legend()
ax[0].set_xlim(0,2000)
ax[0].set_title(r"Ripley's $G(d)$ function")

# plot the pattern itself on the next frame
ax[1].scatter(*coordinates.T)

# and clean up labels and axes there, too
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_title('Pattern')
f.tight_layout()
plt.show()
```


The second function we introduce is Ripley's $F$. Where the $G$ function works by analyzing the distance *between* points in the pattern, the *F* function works by analyzing the distance *to* points in the pattern from locations in empty space. That is why the $F$ function is called the "the empty space function", since it characterizes the typical distance from arbitrary points in empty space to the point pattern. More explicitly, the $F$ accumulates, for a growing distance range, the percentage of points that can be found within that range from a random point pattern generated within the extent of the observed pattern. If the pattern has large gaps or empty areas, the $F$ function will increase slowly. But, if the pattern is highly dispersed, then the $F$ function will increase rapidly. The shape of this cumulative distribution is then compared to those constructed by calculating the same cumulative distribution between the random pattern and an additional, random one generated in each simulation step.

We can use similar tooling to investigate the $F$ function, since it is so mathematically similar to the $G$ function. This is implemented identically using the `f_test` function in `pointpats`. Since the $F$ function estimated for the observed pattern increases *much* more slowly than the $F$ functions for the simulated patterns, we can be confident that there are many gaps in our pattern; i.e. the pattern is *clustered*. 


```python
f_test = distance_statistics.f_test(
    coordinates, support=40, keep_simulations=True
)
```


```python tags=["hide-input"]
f,ax = plt.subplots(
    1,2,figsize=(9,3), gridspec_kw=dict(width_ratios=(6,3))
)

# plot all the simulations with very fine lines
ax[0].plot(f_test.support, f_test.simulations.T, color='k', alpha=.01)
# and show the average of simulations
ax[0].plot(f_test.support, numpy.median(f_test.simulations, axis=0), color='cyan', 
         label='median simulation')


# and the observed pattern's F function
ax[0].plot(f_test.support, f_test.statistic, label = 'observed', color='red')

# clean up labels and axes
ax[0].set_xlabel('distance')
ax[0].set_ylabel('% of nearest point in pattern\ndistances shorter')
ax[0].legend()
ax[0].set_xlim(0,2000)
ax[0].set_title(r"Ripley's $F(d)$ function")

# plot the pattern itself on the next frame
ax[1].scatter(*coordinates.T)

# and clean up labels and axes there, too
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_title('Pattern')
f.tight_layout()
plt.show()
```


Ripley's "alphabet" extends to several other letter-named functions that can be used for conducting point pattern analysis in this vein. Good "next steps" in your point pattern analysis journey include the book by {cite}`Baddeley2015`; and the `pointpats` documentation for guidance on how to run these in Python.


## Identifying clusters

The previous two sections on exploratory spatial analysis of point patterns provide methods to characterize whether point patterns are dispersed or clustered in space. Another way to see the content in those sections is they help us explore the degree of overall *clustering*. However, knowing that a point pattern is clustered does not necessarily give us information about where that (set of) cluster(s) resides. To do this, we need to switch to a method able to identify areas of high density of points within our pattern. In other words, in this section we focus on the existence and location of *clusters*. This distinction between cluster*ing* and cluster*s* of points is analogue to that discussed in the context of spatial autocorrelation. The notion is the same, the differences in the techniques we examine in each part of the book relate to the unique nature of points we referred to in the beginning of the notebook. Remember that, while the methods we explored in the earlier notebooks take the location of the spatial objects (points, lines, polygons) as given and focus on understanding the configurations of values within those locations; the methods discussed in this chapter understand points as events that happen in particular locations but that could happen in a much broader set of places. Factoring in this underlying relevance of the location of an object itself is what makes the techniques in this notebook distinct.

## DBSCAN clustering

DBSCAN (Density-Based Spatial Clustering of Applications) is a density-based clustering method, which means that points that are closely packed together are assigned into the same cluster and given the same ID. The DBSCAN algorithm has two parameters, which the user needs to specify:

* ε —The maximal distance between points to be considered within the same cluster
* minPts —The minimal number of points required to form a cluster

In short, all groups of at least minPts points, where each point is within ε or less from at least one other point in the group, are considered to be separate clusters and assigned with unique IDs. All other points are considered “noise” and are not assigned with an ID.

DBSCAN is a widely used algorithm that originated in the area of knowledge discovery and machine learning and that has since spread into many areas, including the analysis of spatial points. In part, its popularity resides in its intellectual simplicity and computational tractability. In some ways, we can think of DBSCAN as a point pattern counterpart of the local statistics. They do however differ in fundamental ways. Unlike the local statistics we have seen earlier, DBSCAN is not based on an inferential framework, but it is instead a deterministic algorithm. This implies that, unlike the measures seen before, we will not be able to estimate a measure of the degree to which the clusters found are compatible with cases of spatial randomness.  

From the point of view of DBSCAN, a cluster is a concentration of at least `m` points, each of them within a distance of `r` of at least another point in the cluster. Following this definition, the algorithm classifies each point in our pattern into three categories:

* *Noise*, for those points outside a cluster.
* *Cores*, for those points inside a cluster whith at least `m` points in the cluster within distance `r`.
* *Borders* for points inside a cluster with less than `m` other points in the cluster within distance `r`.

The flexibility (but also some of the limitations) of the algorithm resides in that both `m` and `r` need to be prespecified by the user before running DBSCAN. This is a critical point, as their value can influence significantly the final result. Before exploring this in greater depth, let us get a first run at computing `DBSCAN` in Python:


```python
# Define DBSCAN
clusterer = DBSCAN()
# Fit to our data
clusterer.fit(db[["x", "y"]])
```

Following the standard interface in scikit-learn, we first define the algorithm we want to run (creating the `clusterer` object) and then we *fit* it to our data. Once fit, `clusterer` contains the required information to access all the results of the algorithm. The `core_sample_indices_` attribute contains the indices (order, starting from zero) of each point which is classified as a *core*. We can have a peek into it to see what it looks like:


```python
# Print the first 5 elements of `cs`
clusterer.core_sample_indices_[:5]
```
The printout above tells us that the second (remember, Python starts counting at zero!) point in the dataset is a core, as it is the 23rd, 31st, 36th, and 43rd. This attribute has a variable length, depending on how many cores the algorithm finds.

The second attribute of interest is `labels_`:

```python
clusterer.labels_[:5]
```
The labels object always has the same length as the number of points used to run DBSCAN. Each value represents the index of the cluster a point belongs to. If the point is classified as *noise*, it receives a -1. Above, we can see that the second point belongs to cluster 1, while the others in the list are effectively not part of any cluster. To make thinks easier later on, let us turn the labels into a `Series` object that we can index in the same way as our collection of points:


```python
lbls = pandas.Series(clusterer.labels_, index=db.index)
```

Now we already have the clusters, we can proceed to visualize them. There are many ways in which this can be done. We will start just by coloring points in a cluster in red and noise in grey:


```python
# Setup figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Subset points that are not part of any cluster (noise)
noise = db.loc[lbls==-1, ['x', 'y']]
# Plot noise in grey
ax.scatter(noise['x'], noise['y'], c='grey', s=5, linewidth=0)
# Plot all points that are not noise in red
# NOTE how this is done through some fancy indexing, where
#      we take the index of all points (tw) and substract from
#      it the index of those that are noise
ax.scatter(db.loc[db.index.difference(noise.index), 'x'], \
           db.loc[db.index.difference(noise.index), 'y'], \
          c='red', linewidth=0)
# Add basemap
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)
# Remove axes
ax.set_axis_off()
# Display the figure
plt.show()
```


This is a first good pass. The algorithm is able to identify a few clusters with high density of photos. However, as we mentioned {ref}`when discussing DBSCAN <concepts_H:dbscan>`, this is all contingent on the parameters we arbitrarily set. Depending on the maximum radious (`eps`) we set, we will pick one type of cluster or another: a higher (lower) radious will translate in less (more) local clusters. Equally, the minimum number of points required for a cluster (`min_samples`) will affect the implicit size of the cluster. Both parameters need to be set before running the algorithm, so our decision will affect the final outcome quite significantly.

For an illustration of this, let's run through a case with very different parameter values. For example, let's pick a larger radious (e.g. 500m) and a smaller number of points (e.g. 10):

```python
# Set up algorithm
algo = DBSCAN(eps=500, min_samples=10)
# Fit to Tokyo projected points
algo.fit(db[["x", "y"]])
# Store labels
lbls = pd.Series(algo.labels_, index=db.index)
```

```python
# Setup figure and axis
f, ax = plt.subplots(1, figsize=(6, 6))
# Assign labels to tokyo table dynamically and
# subset points that are not part of any cluster (noise)
noise = db.assign(lbls=lbls)\
             .query("lbls == -1")
# Plot noise in grey
ax.scatter(noise["x"], 
           noise["y"], 
           c='grey', 
           s=5, 
           linewidth=0
          )
# Plot all points that are not noise in red
# NOTE how this is done through some fancy indexing, where
#      we take the index of all points (tw) and substract from
#      it the index of those that are noise
ax.scatter(db.loc[db.index.difference(noise.index), 
                     "x"
                    ], 
           db.loc[db.index.difference(noise.index), 
                     "y"
                    ],
           c="red", 
           linewidth=0
          )
# Display the figure
plt.show()
```

The output is now very different, isn't it? This exemplifies how different parameters can give rise to substantially different outcomes, even if the same data and algorithm are applied.


If we change those parameters, can pick up more general patterns. For example, let us say a cluster needs to, at least, have roughly 1% of all the points in the dataset:

```python
# Obtain the number of points 1% of the total represents
minp = numpy.round(db.shape[0] * 0.01)
minp
```
At the same time, let us expand the maximum radious to say, 500 metres. Then we can re-run the algorithm and plot the output, all in the same cell this time:


```python
# Rerun DBSCAN
clusterer = DBSCAN(eps=500, min_samples=minp)
clusterer.fit(db[['x', 'y']])
# Turn labels into a Series
lbls = pandas.Series(clusterer.labels_, index=db.index)
# Setup figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Subset points that are not part of any cluster (noise)
noise = db.loc[lbls==-1, ['x', 'y']]
# Plot noise in grey
ax.scatter(noise['x'], noise['y'], c='grey', s=5, linewidth=0)
# Plot all points that are not noise in red
# NOTE how this is done through some fancy indexing, where
#      we take the index of all points (db) and substract from
#      it the index of those that are noise
ax.scatter(
    db.loc[db.index.difference(noise.index), 'x'],
    db.loc[db.index.difference(noise.index), 'y'],
    c='red', 
    linewidth=0
)
# Add basemap
ctx.add_basemap(
    ax, 
    source=ctx.providers.CartoDB.Positron
)
# Remove axes
ax.set_axis_off()
# Display the figure
plt.show()
```


As we have seen, the choice of parameters plays a crucial role in the number, shape and type of clusters founds in a dataset. To allow an easier exploration of these effects, in this section we will turn the computation and visualization of `DBSCAN` outputs into a single function. This in turn will allow us to build an interactive tool later on.

Below is a function that accomplishes just that:

```python
def clusters(db, eps, min_samples):
    '''
    Compute and visualize DBSCAN clusters    
    ...
    
    Arguments
    ---------
    db          : (Geo)DataFrame
                  Table with at least columns `X` and `Y` for point coordinates
    eps         : float
                  Maximum radious to search for points within a cluster
    min_samples : int
                  Minimum number of points in a cluster
    '''
    algo = DBSCAN(eps=eps, min_samples=min_samples)
    algo.fit(db[['x', 'y']])
    lbls = pd.Series(algo.labels_, index=db.index)

    f, ax = plt.subplots(1, figsize=(6, 6))
    noise = db.loc[lbls==-1, ['x', 'y']]
    ax.scatter(noise['x'], noise['y'], c='grey', s=5, linewidth=0)
    ax.scatter(db.loc[db.index.difference(noise.index), 'x'], \
               db.loc[db.index.difference(noise.index), 'y'], \
              c='red', linewidth=0)
    return plt.show()
```

The function takes the following three arguments:

1. `db`: a `(Geo)DataFrame` containing the points on which we will try to find the clusters.
1. `eps`: a number (maybe with decimals, hence the `float` label in the documentation of the function) specifying the maximum distance to look for neighbors that will be part of a cluster.
1. `min_samples`: a count of the minimum number of points required to form a cluster.

Let us see how the function can be used. For example, let us replicate the plot above, with a minimum of 10 points and a maximum radious of 500 metres:

```python
clusters(db, 500, 10)
```

Voila! With just one line of code, we can create a map of `DBSCAN` clusters. How cool is that? 

However, this could be even more interesting if we didn't have to write each time the parameters we want to explore. To change that, we can create a quick interactive tool that will allow us to modify both parameters with sliders. To do this, we will use the library [`ipywidgets`](https://ipywidgets.readthedocs.io). Let us first do it and then we will analyse it bit by bit:

```python
from ipywidgets import interact, fixed
interact(
    clusters,                 # Method to make interactive
    db=fixed(db),          # Data to pass on db (does not change)
    eps=(50, 500, 50),        # Range start/end/step of eps
    min_samples=(50, 300, 50) # Range start/end/step of min_samples
);
```

Phew! That is cool, isn't it? Once passed the first excitement, let us have a look at how we built it, and how you can modify it further on. A few points on this:

* First, `interact` is a method that allows us to pass an arbitrary function (like `clusters`) and turn it into an interactive widget where we modify the values of its parameters through sliders, drop-down menus, etc.
* What we need to pass to `interact` is the name of the function we would like to make interactive (`clusters` in this case), and all the parameters it will take.
* Since in this case we do not wish to modify the dataset that is used, we pass `tokyo` as the `db` argument in `clusters` and fixate it by passing it first to the `fixed` method.
* Then both the radious `eps` and the minimum cluster size `min_samples` are passed. In this case, we do want to allow interactivity, so we do not use `fixed`. Instead, we pass a tuple that specifies the range and the step of the values we will allow to be used. 
* In the case of `eps`, we use `(50, 500, 50)`, which means we want `r` to go from 50 to 500, in jumps of 50 units at a time. Since these are specified in metres, we are saying we want the range to go from 50 to 500 metres in increments of 50 metres.
* In the case of `min_samples`, we take a similar approach and say we want the minimum number of points to go from 50 to 300, in steps of 50 points at a time.

The above results in a little interactive tool that allows us to play easily and quickly with different values for the parameters and to explore how they affect the final outcome.


### Turf.js
The turf.clustersDbscan function is an implementation of the DBSCAN algorithm in Turf.js. The function accepts an input point layer and returns the same layer with a new property named cluster. The cluster property contains the cluster ID assigned using DBSCAN. Noise points are not assigned with an ID and thus are not given a cluster property. The first parameter of the turf.clustersDbscan function is maxDistance (ε), which determines what is the maximal distance between two points to be considered within the same cluster, in kilometers. In our example (see below), we will use a maximal distance of 10 kilometers. The minPoints (minPts) parameter has a default value of 3, which we will not override.


```
<!DOCTYPE html>
<html>
<head>
    <title>Agregar contenido</title>
    <meta name="viewport" content="width=device-width,  initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.css"/>
    <script src="https://code.jquery.com/jquery-3.6.0.js"
  integrity="sha256-H+K7U5CnXl1h5ywQfKtSj8PCmoN9aaq30gDh27Xc0jk="
  crossorigin="anonymous"></script>
  <script src="https://npmcdn.com/@turf/turf/turf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.js"></script>
    <style>
        body {
            padding: 0;
            margin: 0;
        }
        html, body, #map {
            height: 100%;
            width: 100%;
        }
    </style>
</head>
<body>
    <div id="map"></div>

    <script>

        // Initialize map
        var map = L.map("map").setView([31.5, 35], 9);
        L.tileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", 
            {attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}
        ).addTo(map);
        
        // Set base URL
        var url = "https://michaeldorman.carto.com/api/v2/sql?format=GeoJSON&q=";
        
        // Set SQL Query
        var sqlQuery = "SELECT name_lat, the_geom " + 
            "FROM plants WHERE " + 
            "name_lat='Iris atrofusca' OR " + 
            "name_lat='Iris atropurpurea' OR " +
            "name_lat='Iris mariae' OR " +
            "name_lat='Iris petrana'";
        
        // Color function
        function getColor(species) {
            if(species == "Iris mariae") return "yellow";
            if(species == "Iris petrana") return "brown";
            if(species == "Iris atrofusca") return "black"; 
            if(species == "Iris atropurpurea") return "orange"; 
        }
        
        // Style function
        function setStyle(feature) {
            return {
                fillColor: getColor(feature.properties.name_lat),
                weight: 1,
                opacity: 1,
                color: "black",
                fillOpacity: 0.5
            };
        }
        
        // load GeoJSON from CARTO database
        $.getJSON(url + sqlQuery, function(data) {
            turf.clusterEach(data, "name_lat", function(cluster, clusterValue, currentIndex) {
                L.geoJSON(cluster, {
                    onEachFeature: function(feature, layer) {
                        layer.bindPopup("<i>" + clusterValue + "</i>");
                    },
                    pointToLayer: function(geoJsonPoint, latlng) {
                        return L.circleMarker(latlng);
                    },
                    style: setStyle
                }).addTo(map);
                clustered = turf.clustersDbscan(cluster,20);
                turf.clusterEach(clustered, "cluster", function(cluster2, clusterValue2, currentIndex2) {
                    var ch = turf.convex(cluster2);
                    ch.properties.name_lat = clusterValue;
                    L.geoJSON(ch, {style: setStyle}).addTo(map);
                });
            });
        });
            </script>

</body>
</html>
```


## Clustering as feature engineering

One unique way to use spatial or feature information within your data as a feature in your existing models is to use clustering. This can provide an indication of whether an observation exists in a given “place” geographically, or if an observation is a specific “kind” of observation.

Sometimes, this data reduction of many correlated variates into a derived feature can be useful in training models. This is more useful when the spatial location of a given observation indicates something useful about what kind of observation is taken at that location.

While it would be best to use an explicitly-spatial model to examine this structure, we can cheat a little bit and use cluster labels themselves as features. For example, to cluster the listings based on their location, we can use hierarchical DBSCAN, an improved variant of the DBSCAN algorithm. we use the Airbnb nightly rental prices in San Diego.

```python
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import numpy as np
```

```python
airbnbs = gpd.read_file("https://geographicdata.science/book/_downloads/dcd429d1761a2d0efdbc4532e141ba14/regression_db.geojson")
```

```python
f,ax = plt.subplots(1,figsize=(12, 12))
airbnbs.plot(ax=ax, marker='.')
ctx.add_basemap(
    ax, 
    crs=airbnbs.crs.to_string(), 
    source=ctx.providers.Stamen.Toner
)
```

```python
airbnbs_albers = airbnbs.to_crs(epsg=3311)
coordinates = np.column_stack((airbnbs_albers.geometry.x, airbnbs_albers.geometry.y))
```

Here, we’ll look for relatively large clusters of Airbnbs, those with about 100 listings or more.

```python
labels = HDBSCAN(min_cluster_size=25).fit(coordinates).labels_
```

The spatial distribution of these clusters gives us a sense of the geographical distribution of the observations. To help us visualize the clusters, we can construct the convex hull of the observations in each dectected cluster:

```python
hulls = airbnbs_albers[['geometry']].dissolve(by=labels).convex_hull
```

Since humans tend to make locational decisions hierarchically (in that they pick San Diego, then they pick a particular part of San Diego (such as the Gaslamp Quarter), then they pick a house in that part), this clustering process might give us a reasonable insight into the enclaves of competition between Airbnbs:

```python
f, ax = plt.subplots(1, figsize=(9, 9))
airbnbs_albers.plot(column=labels,
                    categorical=True,
                    alpha=0.5,
                    legend=False,
                    ax=ax, marker='.'
                    )
hulls[hulls.index >=0].boundary.plot(color='k', ax=ax,)
ctx.add_basemap(ax, 
                       crs=airbnbs_albers.crs.to_string(), 
                       source=ctx.providers.Stamen.Toner
                      )
```

Regardless, this cluster label certainly communicates some information about the price of a listing, since the distributions of prices are substantially different across the detected clusters:

```python
f = plt.figure(figsize=(8,3))
ax = airbnbs_albers.boxplot("price", by=labels, 
                            flierprops=dict(marker=None), 
                            ax=plt.gca())
ax.set_xlabel("competition cluster")
ax.set_ylabel("price ($)")
plt.gcf().suptitle(None)
ax.set_title("Price distribution by detected cluster")
ax.set_ylim(0,1250)
plt.show()
```

Feature engineering is a powerful way to enrich your data analysis capabilities. It’s often within reach of your existing data analysis methods: at a minimum, it only requires that new variables are constructed from your existing data. At a maximum, feature engineering gives you the ultimate linkage key, a flexible and powerful tool with which you can unlock the value held in many other datasets that you may have access to.

The main operations and methods that are involved in feature engineering, such as determining what the average value is in the area near each observation or identifying whether observations exist in a “cluster,” are fundamentally simple operations. Together, though, they build large, rich, and useful datasets that can be used directly in your existing methods of analysis.
