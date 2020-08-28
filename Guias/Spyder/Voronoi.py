import numpy as np
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

from shapely.ops import cascaded_union

from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
from geovoronoi import voronoi_regions_from_coords, points_to_coords


gdf = gpd.read_file("data/preschools.shp")
gdf.head()

gdf.shape

boundary = gpd.read_file("data/uppsala.shp")
boundary


fig, ax = plt.subplots(figsize=(12, 10))
boundary.plot(ax=ax, color="gray")
gdf.plot(ax=ax, markersize=3.5, color="black")
ax.axis("off")
plt.axis('equal')
plt.show()


fig, ax = plt.subplots(figsize=(16, 18))
gdf.to_crs(epsg=3857).plot(ax=ax)
ctx.add_basemap(ax)
plt.title('Preschools in Uppsala', fontsize=40, fontname="Palatino Linotype", color="grey")
ax.axis("off")
#plt.axis('equal')
plt.show()

boundary.crs

boundary = boundary.to_crs(epsg=3395)
gdf_proj = gdf.to_crs(boundary.crs)


boundary_shape = cascaded_union(boundary.geometry)
boundary_shape

coords = points_to_coords(gdf_proj.geometry)

# Calculate Voronoi Regions
poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, boundary_shape)

fig, ax = subplot_for_map()

plot_voronoi_polys_with_points_in_area(ax, boundary_shape, poly_shapes, pts, poly_to_pt_assignments)

ax.set_title('Voronoi regions of Schools in Uppsala')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(14,12))

plot_voronoi_polys_with_points_in_area(ax, boundary_shape, poly_shapes, pts)

ax.set_title('Voronoi regions of Schools in Uppsala')

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(14,12))

plot_voronoi_polys_with_points_in_area(ax, boundary_shape, poly_shapes, pts, poly_to_pt_assignments,
                                       voronoi_and_points_cmap='tab20c',
                                      points_markersize=20)

ax.set_title('Upssala Preschools - Voronoi Regions')
ax.axis("off")
plt.tight_layout()
plt.show()
