#Imports from PlotMesh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask
import intake
import xarray as xr
import holoviews as hv
import datashader as ds
import datashader.transfer_functions as tf
import datashader.utils as du
import geoviews as gv
from holoviews import opts
import holoviews.operation.datashader as dshade
from holoviews.operation.datashader import datashade, shade, rasterize
import cmocean
hv.extension('bokeh')

#Imports from BoundingBox
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import Gridliner, LongitudeFormatter, LatitudeFormatter
import contextily as cx

#Imports for Tropycal
import tropycal.tracks as tracks

#Imports for ML
from sklearn.neighbors import BallTree

#Other imports
from scipy.interpolate import griddata
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import full_NN_pipeline

#Constants
R = full_NN_pipeline.EARTH_RADIUS_KM

#Box Edges
westLon, eastLon, northLat, southLat = -77.262338, -75.367403, 36.553085, 34.552033
centroid_lon, centroid_lat = (westLon + eastLon)/2.0, (northLat + southLat)/2.0

#Storm Times
Storm_times = {
    "Irene": [pd.Timestamp("2011-08-25 00:00:00"), pd.Timestamp("2011-08-28 00:00:00")],
}

#CORA timeseries
catalog = intake.open_catalog("s3://noaa-nos-cora-pds/CORA_V1.1_intake.yml",storage_options={'anon':True})
ds_native = catalog["CORA-V1.1-fort.63-timeseries"].to_dask()
ds_native = ds_native.rename({"x": "lon", "y": "lat", "node": "node"})
ds_500m = catalog["CORA-V1.1-Grid-timeseries"].to_dask()
ds_500m = ds_500m.rename({"nodes": "node"})

#Storm Tracks

hurdat = tracks.TrackDataset(
    basin="north_atlantic",
    source="hurdat"
)

Storm_times = full_NN_pipeline.filter_storms_fast(
    hurdat,
    300,
    centroid_lat=centroid_lat,
    centroid_lon=centroid_lon,
    earth_radius_km=R,
)
print("Grabbed Storms")
ds_native_spatial_subset = full_NN_pipeline.get_spatial_subset(
    ds_native,
    west_lon=westLon,
    east_lon=eastLon,
    south_lat=southLat,
    north_lat=northLat,
)
print("Grabbed spatial subset")
ds_500m_spatial_subset = full_NN_pipeline.get_spatial_subset(
    ds_500m,
    west_lon=westLon,
    east_lon=eastLon,
    south_lat=southLat,
    north_lat=northLat,
)
print("Gridded to 500m")
std_all, node_coords, zeta_subset_6hr = full_NN_pipeline.get_zeta_stddevs_test(
    ds_500m_spatial_subset,
    Storm_times,
    earth_radius_km=R,
)
print("Got zeta std deviations")
ds_storm_all = full_NN_pipeline.get_storm_features(hurdat, node_coords, Storm_times)
print("Got storm features")
ds_depths_subset = full_NN_pipeline.get_bathymetry(ds_native_spatial_subset, ds_500m_spatial_subset, node_coords)
print("Got bathymetry")

combined_ds = full_NN_pipeline.combine_features(std_all, ds_storm_all, ds_depths_subset, zeta_subset_6hr, node_coords)
print("Combined features")
combined_ds_newfeats = full_NN_pipeline.add_features(combined_ds, earth_radius_km=R)
storms_to_drop = [
    storm for storm in full_NN_pipeline.MANUALLY_EXCLUDED_STORMS
    if storm in combined_ds_newfeats["storm"].values
]
if storms_to_drop:
    combined_ds_newfeats = combined_ds_newfeats.drop_sel(storm=storms_to_drop)
    print(f"Dropped manually excluded storms: {storms_to_drop}")
print("Added new features")

train_storms, val_storms, test_storms = full_NN_pipeline.stratified_storm_train_val_test_split(
    combined_ds_newfeats
)
X_train, X_val, X_test, y_train, y_val, y_test, feature_vars = full_NN_pipeline.apply_train_val_test_split_for_tcn(
    combined_ds_newfeats,
    train_storms,
    val_storms,
    test_storms,
    target_var="std_dev",
    target_mode="final",
)
print(f"Storm split sizes -> train: {len(train_storms)}, val: {len(val_storms)}, test: {len(test_storms)}")
print(f"TCN feature variables ({len(feature_vars)}): {feature_vars}")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
