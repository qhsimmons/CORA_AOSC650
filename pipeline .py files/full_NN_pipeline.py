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

EARTH_RADIUS_KM = 6371.0
MANUALLY_EXCLUDED_STORMS = {"AL061996", "AL081999", "AL052019"}
TCN_SEQUENCE_LENGTH = 13

#Data Collection & Feature Construction

def filter_storms(hurdat, threshold_km, centroid_lat, centroid_lon, earth_radius_km=EARTH_RADIUS_KM):
    #Read in storms, find closest approach, keep only storms under some threshold
    #Output Storm_times dictionary
    
    Storm_times_dict = {}
    for storm_id in hurdat.data.keys():
        storm = hurdat.get_storm(storm_id)

        if storm.year < 2019 or storm.year > 2022:
            continue
            print("Skipped a storm")
            
        df = storm.to_dataframe()

        # Skip empty storms
        if df.empty:
            continue

        # Compute distance to OBX for all timesteps
        dists = haversine(
            df["lat"].values,
            df["lon"].values,
            centroid_lat,
            centroid_lon,
            earth_radius_km=earth_radius_km,
        )

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist < threshold_km:
            closest_time = df["time"].iloc[min_idx]

            start_time = closest_time - timedelta(hours=72)
            end_time = closest_time + timedelta(hours=6)

            Storm_times_dict[storm_id] = (start_time, end_time)
                
        #find closest
        #if closest < 300km, tot_storms += 1, valid_storms_dict[storm] = [closest_approach_time_minus72hours, closest_approach_time_plus6hrs]

    return Storm_times_dict

def filter_storms_fast(
    hurdat,
    threshold_km,
    centroid_lat,
    centroid_lon,
    year_start=2021,
    year_end=2022,
    earth_radius_km=EARTH_RADIUS_KM,
    required_steps=14,
    step_hours=6,
):
    """
    Filter HURDAT2 storms by proximity and year range.
    Avoids calling `get_storm()` for every storm.
    """
    Storm_times_dict = {}
    
    for storm_id, storm_data in hurdat.data.items():
        if storm_id in MANUALLY_EXCLUDED_STORMS:
            continue

        year = storm_data["year"]  # storm year stored as metadata

        # Skip storms outside year range
        if year < year_start or year > year_end:
            continue

        # Grab full track arrays
        lats = np.array(storm_data["lat"])
        lons = np.array(storm_data["lon"])
        times = np.array(storm_data["time"], dtype="datetime64[ns]")

        # Compute all distances to your target point
        dists = haversine(
            lats,
            lons,
            centroid_lat,
            centroid_lon,
            earth_radius_km=earth_radius_km,
        )
        if len(dists) == 0:
            continue

        # Find closest approach
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist < threshold_km:
            closest_time = times[min_idx]
            start_time = pd.Timestamp(closest_time - np.timedelta64(72, "h"))
            end_time = pd.Timestamp(closest_time + np.timedelta64(6, "h"))
            expected_times = pd.date_range(
                start=start_time,
                periods=required_steps,
                freq=f"{step_hours}h",
            )
            available_times = pd.to_datetime(times)

            if np.isin(expected_times.values, available_times.values).all():
                Storm_times_dict[storm_id] = (start_time, end_time)

    return Storm_times_dict

def get_bathymetry(ds_unstructured, ds_structured, node_coords):

    #Native Mesh
    points = np.column_stack((ds_unstructured["lon"].values, ds_unstructured["lat"].values))
    bathy_vals = ds_unstructured["depth"].values

    #structured grid
    target_points = np.column_stack((ds_structured["lon"].values, ds_structured["lat"].values))

    # Interpolate
    bathy_interp_vals = griddata(
        points,
        bathy_vals,
        target_points,
        method="linear"
    )
    
    bathy_interp = xr.DataArray(
        bathy_interp_vals,
        dims=(["node"]),
        coords={"node": ds_structured["node"]}
    )

    return bathy_interp

def get_spatial_subset(ds, west_lon, east_lon, south_lat, north_lat):
    x_subset = ((ds.lon >= west_lon) & (ds.lon <= east_lon)).compute()
    y_subset = ((ds.lat >= south_lat) & (ds.lat <= north_lat)).compute()

    boolmask = (x_subset & y_subset).compute()
    nodeIndex = np.where(boolmask)[0]

    ds_subset = ds.isel(node=nodeIndex)

    return ds_subset

def _build_zeta_subset(ds, Storm_times):
    zeta_list = []

    for storm, (start_t, end_t) in Storm_times.items():
        time_subset = ds["zeta"].sel(time=slice(start_t, end_t))
        time_subset = time_subset.sel(time=time_subset["time"].dt.hour.isin([0, 6, 12, 18]))
        time_subset = time_subset.expand_dims(storm=[storm]).assign_coords(
            time=np.arange(time_subset.sizes["time"])
        )
        zeta_list.append(time_subset)

    if not zeta_list:
        raise ValueError("No storm windows produced any zeta samples.")

    return xr.concat(zeta_list, dim="storm", coords="minimal", compat="override")

def _build_neighbor_index(ds, radius_km, earth_radius_km):
    lat = np.deg2rad(ds["lat"].values)
    lon = np.deg2rad(ds["lon"].values)
    coords = np.vstack([lat, lon]).T
    tree = BallTree(coords, metric='haversine')
    r = radius_km / earth_radius_km
    neighbors = tree.query_radius(coords, r=r)

    max_neighbors = max(len(nbrs) for nbrs in neighbors)
    neighbor_idx = np.full((len(neighbors), max_neighbors), -1, dtype=np.int32)
    neighbor_mask = np.zeros((len(neighbors), max_neighbors), dtype=bool)

    for i, nbrs in enumerate(neighbors):
        count = len(nbrs)
        neighbor_idx[i, :count] = nbrs
        neighbor_mask[i, :count] = True

    return neighbor_idx, neighbor_mask

def _compute_storm_local_std(zeta_values, neighbor_idx, neighbor_mask, min_valid_neighbors=5):
    n_nodes, n_time = zeta_values.shape
    std_feature = np.full((n_nodes, n_time), np.nan, dtype=np.float32)

    safe_neighbor_idx = np.where(neighbor_idx < 0, 0, neighbor_idx)

    for t in range(n_time):
        values_t = zeta_values[:, t]
        wet_mask = np.isfinite(values_t)
        if not wet_mask.any():
            continue

        gathered = values_t[safe_neighbor_idx]
        valid_neighbors = neighbor_mask & wet_mask[safe_neighbor_idx]
        gathered = np.where(valid_neighbors, gathered, np.nan)

        valid_counts = valid_neighbors.sum(axis=1)
        enough_neighbors = valid_counts >= min_valid_neighbors
        if not enough_neighbors.any():
            continue

        gathered_sum = np.nansum(gathered, axis=1)
        gathered_mean = np.divide(
            gathered_sum,
            valid_counts,
            out=np.full(n_nodes, np.nan, dtype=np.float32),
            where=enough_neighbors,
        )

        centered = np.where(valid_neighbors, gathered - gathered_mean[:, None], np.nan)
        gathered_var = np.divide(
            np.nansum(centered ** 2, axis=1),
            valid_counts,
            out=np.full(n_nodes, np.nan, dtype=np.float32),
            where=enough_neighbors,
        )
        std_t = np.sqrt(gathered_var, dtype=np.float32)
        std_t[(~enough_neighbors) | (~wet_mask)] = np.nan
        std_feature[:, t] = std_t

    return std_feature

def _compute_zeta_stddevs(ds, Storm_times, radius_km=0.8, earth_radius_km=EARTH_RADIUS_KM):
    zeta_subset_6hr = _build_zeta_subset(ds, Storm_times)
    neighbor_idx, neighbor_mask = _build_neighbor_index(ds, radius_km, earth_radius_km)

    std_list = []
    for storm_id in zeta_subset_6hr["storm"].values:
        zeta_storm = zeta_subset_6hr.sel(storm=storm_id).transpose("node", "time").compute()
        std_feature = _compute_storm_local_std(
            np.asarray(zeta_storm.values, dtype=np.float32),
            neighbor_idx,
            neighbor_mask,
        )
        std_list.append(
            xr.DataArray(
                std_feature,
                dims=["node", "time"],
                coords={
                    "node": ds["node"].values,
                    "time": zeta_storm["time"].values,
                },
                name="std_dev",
            ).expand_dims(storm=[storm_id])
        )
        print(f"looped through storm {storm_id}")

    std_all = xr.concat(std_list, dim="storm").transpose("storm", "time", "node")
    return std_all, ds["node"], zeta_subset_6hr

def get_zeta_stddevs(ds, Storm_times, radius_km=0.8, earth_radius_km=EARTH_RADIUS_KM):
    return _compute_zeta_stddevs(
        ds,
        Storm_times,
        radius_km=radius_km,
        earth_radius_km=earth_radius_km,
    )

def get_zeta_stddevs_test(ds, Storm_times, radius_km=0.8, earth_radius_km=EARTH_RADIUS_KM):
    """
    Compute local standard deviation of water elevation (zeta) over neighboring nodes
    for each storm in Storm_times. Fully vectorized version using NumPy broadcasting.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'zeta', 'lat', 'lon' variables.
    Storm_times : dict
        Dictionary of {storm_id: (start_time, end_time)}
    radius_km : float
        Neighbor search radius in km.

    Returns
    -------
    std_all : xarray.DataArray
        Std deviation for each storm, time, and node (storm, time, node)
    node_coords : xarray.DataArray
        Node indices
    zeta_subset_6hr : xarray.DataArray
        Subset of zeta at 6-hour intervals for the storms
    """

    return _compute_zeta_stddevs(
        ds,
        Storm_times,
        radius_km=radius_km,
        earth_radius_km=earth_radius_km,
    )
    

def get_storm_features(hurdat, node_coords, Storm_times):
    storm_list = []

    for storm, (start, end) in Storm_times.items():
        storm_ = hurdat.get_storm((storm))
        ds_storm_ = storm_.to_xarray()
        ds_storm_ = ds_storm_.rename({'lat': 'hur_lat', 'lon': 'hur_lon'})
        ds_storm_6hr_ = ds_storm_.sel(time=ds_storm_['time'].dt.hour.isin([0, 6, 12, 18])).sel(time=slice(start, end))
    
        ds_storm_6hr_ = ds_storm_6hr_.expand_dims(storm=[storm])
        ds_storm_6hr_ = ds_storm_6hr_.assign_coords(time=np.arange(ds_storm_6hr_.sizes["time"]))
    
        storm_list.append(ds_storm_6hr_)
    
    ds_storm_all = xr.concat(storm_list, dim="storm")
    ds_storm_all = ds_storm_all.expand_dims(node=node_coords.values)
    ds_storm_all = ds_storm_all.transpose("storm", "time", "node")

    return ds_storm_all

def combine_features(std_all, ds_storm_all, ds_depths_subset, zeta_subset_6hr, node_coords):
    combined_ds = xr.merge([std_all, ds_storm_all])
    combined_ds["bathy"] = ("node", ds_depths_subset.values)
    combined_ds["lat"] = ("node", zeta_subset_6hr["lat"].values)
    combined_ds["lon"] = ("node", zeta_subset_6hr["lon"].values)
    drop_vars = [var for var in ["extra_obs", "special"] if var in combined_ds]
    if drop_vars:
        combined_ds = combined_ds.drop_vars(drop_vars)

    return combined_ds

def azimuth(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    az = np.arctan2(np.sin(lon2-lon1)*np.cos(lat2), np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
    
    return az

def haversine(lat1, lon1, lat2, lon2, earth_radius_km=EARTH_RADIUS_KM):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = earth_radius_km*c
    
    return d

#Feature Addition

def add_features(combined_ds, earth_radius_km=EARTH_RADIUS_KM, sequence_length=TCN_SEQUENCE_LENGTH):
    combined_ds = combined_ds.copy()

    dist = haversine(
        combined_ds["hur_lat"],
        combined_ds["hur_lon"],
        combined_ds["lat"],
        combined_ds["lon"],
        earth_radius_km=earth_radius_km,
    )
    az = azimuth(combined_ds["hur_lat"], combined_ds["hur_lon"], combined_ds["lat"], combined_ds["lon"])
    
    combined_ds["dist_to_hur"] = dist
    combined_ds["azimuth"] = az
    combined_ds["dx"] = np.sin(combined_ds["azimuth"])
    combined_ds["dy"] = np.cos(combined_ds["azimuth"])
    combined_ds["dx_dist_weighted"] = combined_ds["dist_to_hur"] * combined_ds["dx"]
    combined_ds["dy_dist_weighted"] = combined_ds["dist_to_hur"] * combined_ds["dy"]

    combined_ds = hur_speed(combined_ds, earth_radius_km=earth_radius_km)
    combined_ds = intensification(combined_ds)
    combined_ds = combined_ds.isel(time=slice(1, sequence_length + 1))
    
    return combined_ds

#May want to add an extra 6 hours at the beginning of each storm so I can drop the NaN and have a true 72 hr reading
def hur_speed(ds, earth_radius_km=EARTH_RADIUS_KM):
    lat1 = ds["hur_lat"]
    lon1 = ds["hur_lon"]

    lat0 = lat1.shift(time=1)
    lon0 = lon1.shift(time=1)

    dist = haversine(lat0, lon0, lat1, lon1, earth_radius_km=earth_radius_km)
    
    x_dist = haversine(lat0, lon0, lat0, lon1, earth_radius_km=earth_radius_km)
    y_dist = haversine(lat0, lon0, lat1, lon0, earth_radius_km=earth_radius_km)

    x_sign = xr.where((lon1 - lon0) >= 0, 1.0, -1.0)
    y_sign = xr.where((lat1 - lat0) >= 0, 1.0, -1.0)

    speed = dist / 6.0
    x_speed = (x_dist * x_sign) / 6.0
    y_speed = (y_dist * y_sign) / 6.0

    ds["hur_speed"] = speed
    ds["hur_x_speed"] = x_speed
    ds["hur_y_speed"] = y_speed

    # Project storm motion onto the storm-to-node unit vector and its perpendicular.
    storm_dx = x_speed / (speed + 1e-6)
    storm_dy = y_speed / (speed + 1e-6)

    ds["alignment"] = storm_dx * ds["dx"] + storm_dy * ds["dy"]
    ds["cross_track"] = storm_dx * ds["dy"] - storm_dy * ds["dx"]

    return ds

def intensification(ds):
    vmax1 = ds["vmax"]
    mslp1 = ds["mslp"]

    vmax0 = vmax1.shift(time=1)
    mslp0 = mslp1.shift(time=1)

    d_vmax_dt = (vmax1 - vmax0) / 6.0
    d_mslp_dt = (mslp1 - mslp0) / 6.0

    ds["d_vmax_dt"] = d_vmax_dt
    ds["d_mslp_dt"] = d_mslp_dt

    return ds

#Pre-Processing

def remove_fluff_features(da):
    #Placeholder right now, need to finish code
    
    feats = []
    da = da.drop(feats)

def get_standard_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data)

    return scaler

#Training

def get_storm_df(ds):
    storm_list = []

    vmax_da = ds["vmax"]
    if "node" in vmax_da.dims:
        vmax_da = vmax_da.max(dim="node", skipna=True)

    for storm_id in ds["storm"].values:
        vmax_series = vmax_da.sel(storm=storm_id).values
        valid_idx = np.where(~np.isnan(vmax_series))[0]

        if len(valid_idx) == 0:
            continue  # skip storms with no valid vmax

        last_idx = valid_idx[-1]
        storm_list.append({
            "storm": storm_id,
            "vmax_final_time": float(vmax_series[last_idx]),
        })

    storm_df = pd.DataFrame(storm_list)
    return storm_df

def _get_stratify_bins(storm_df, n_bins, test_size):
    if len(storm_df) < 3:
        return None

    n_test = max(1, int(np.ceil(len(storm_df) * test_size)))
    max_bins = min(n_bins, len(storm_df) // 2, n_test)
    if max_bins < 2:
        return None

    try:
        stratify_bins = pd.qcut(
            storm_df["vmax_final_time"],
            q=max_bins,
            duplicates="drop"
        )
    except ValueError:
        return None

    bin_counts = stratify_bins.value_counts()
    if len(bin_counts) < 2 or (bin_counts < 2).any():
        return None

    return stratify_bins

def stratified_storm_split(ds, n_bins=5, test_size=0.2, random_state=42):
    storm_df = get_storm_df(ds)
    if storm_df.empty:
        raise ValueError("No storms with valid vmax values were found for splitting.")

    stratify_bins = _get_stratify_bins(storm_df, n_bins=n_bins, test_size=test_size)
    split_kwargs = {
        "test_size": test_size,
        "random_state": random_state,
    }
    if stratify_bins is not None:
        split_kwargs["stratify"] = stratify_bins

    train_storms, test_storms = train_test_split(
        storm_df["storm"],
        **split_kwargs,
    )

    return list(train_storms), list(test_storms)

def stratified_storm_train_val_test_split(
    ds,
    n_bins=5,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    random_state=42,
):
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size must sum to 1.0.")

    storm_df = get_storm_df(ds)
    if storm_df.empty:
        raise ValueError("No storms with valid vmax values were found for splitting.")

    train_val_size = train_size + val_size
    train_val_storms, test_storms = stratified_storm_split(
        ds,
        n_bins=n_bins,
        test_size=test_size,
        random_state=random_state,
    )

    train_val_df = storm_df[storm_df["storm"].isin(train_val_storms)].reset_index(drop=True)
    if train_val_df.empty:
        raise ValueError("Train/validation split produced no storms.")

    val_fraction_within_train_val = val_size / train_val_size
    stratify_bins = _get_stratify_bins(
        train_val_df,
        n_bins=n_bins,
        test_size=val_fraction_within_train_val,
    )

    split_kwargs = {
        "test_size": val_fraction_within_train_val,
        "random_state": random_state,
    }
    if stratify_bins is not None:
        split_kwargs["stratify"] = stratify_bins

    train_storms, val_storms = train_test_split(
        train_val_df["storm"],
        **split_kwargs,
    )

    return list(train_storms), list(val_storms), list(test_storms)

def _broadcast_feature_to_time(feature_da, time_coord):
    if "time" in feature_da.dims:
        return feature_da.sel(time=time_coord)

    broadcast = feature_da.expand_dims(time=time_coord)
    desired_dims = ["time"] + [dim for dim in broadcast.dims if dim != "time"]
    return broadcast.transpose(*desired_dims)

def build_tcn_dataset(
    ds,
    storm_ids,
    target_var="std_dev",
    excluded_feature_vars=None,
    require_complete_sequences=True,
    target_mode="final",
    sequence_length=TCN_SEQUENCE_LENGTH,
):
    subset_ds = ds.sel(storm=list(storm_ids))
    subset_ds = subset_ds.isel(time=slice(0, sequence_length))

    drop_vars = ["type", "wmo_basin"]
    subset_ds = subset_ds.drop_vars([v for v in drop_vars if v in subset_ds])

    excluded = {target_var, "zeta_std_dev"}
    if excluded_feature_vars is not None:
        excluded.update(excluded_feature_vars)

    feature_vars = []
    for var_name, data_var in subset_ds.data_vars.items():
        if var_name in excluded:
            continue
        if not np.issubdtype(data_var.dtype, np.number):
            continue
        feature_vars.append(var_name)

    if not feature_vars:
        raise ValueError("No numeric predictor variables remain for TCN input.")

    X_parts = []
    y_parts = []

    for storm_id in storm_ids:
        storm_ds = subset_ds.sel(storm=storm_id)
        target_da = storm_ds[target_var].transpose("time", "node").compute()
        time_coord = target_da["time"]
        target_values = np.asarray(target_da.values, dtype=np.float32)

        feature_arrays = []
        for feature_name in feature_vars:
            feature_da = _broadcast_feature_to_time(storm_ds[feature_name], time_coord)
            feature_da = feature_da.transpose("time", "node").compute()
            feature_arrays.append(np.asarray(feature_da.values, dtype=np.float32))

        X_storm = np.stack(feature_arrays, axis=-1)  # (time, node, feature)
        X_storm = np.transpose(X_storm, (1, 0, 2))   # (node, time, feature)
        y_storm_full = np.transpose(target_values, (1, 0))[:, :, None]  # (node, time, 1)

        finite_target = np.isfinite(y_storm_full[:, :, 0])
        finite_features = np.isfinite(X_storm).all(axis=2)
        finite_final_target = finite_target[:, -1]

        if require_complete_sequences:
            if target_mode == "final":
                valid_sequences = finite_final_target & finite_features.all(axis=1)
            elif target_mode == "sequence":
                valid_sequences = finite_target.all(axis=1) & finite_features.all(axis=1)
            else:
                raise ValueError("target_mode must be 'final' or 'sequence'.")
        else:
            if target_mode == "final":
                valid_sequences = finite_final_target & finite_features.any(axis=1)
            elif target_mode == "sequence":
                valid_sequences = finite_target.any(axis=1) & finite_features.any(axis=1)
            else:
                raise ValueError("target_mode must be 'final' or 'sequence'.")
            X_storm = np.where(np.isfinite(X_storm), X_storm, 0.0)
            y_storm_full = np.where(np.isfinite(y_storm_full), y_storm_full, np.nan)

        if not valid_sequences.any():
            continue

        X_parts.append(X_storm[valid_sequences])
        if target_mode == "final":
            y_parts.append(y_storm_full[valid_sequences, -1, :])
        elif target_mode == "sequence":
            y_parts.append(y_storm_full[valid_sequences])
        else:
            raise ValueError("target_mode must be 'final' or 'sequence'.")

    if not X_parts:
        raise ValueError("No valid TCN sequences were found after filtering NaNs.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y, feature_vars

def apply_split_for_tcn(
    ds,
    train_storms,
    test_storms,
    target_var="std_dev",
    excluded_feature_vars=None,
    require_complete_sequences=True,
    target_mode="final",
    sequence_length=TCN_SEQUENCE_LENGTH,
):
    X_train, y_train, feature_vars = build_tcn_dataset(
        ds,
        train_storms,
        target_var=target_var,
        excluded_feature_vars=excluded_feature_vars,
        require_complete_sequences=require_complete_sequences,
        target_mode=target_mode,
        sequence_length=sequence_length,
    )
    X_test, y_test, _ = build_tcn_dataset(
        ds,
        test_storms,
        target_var=target_var,
        excluded_feature_vars=excluded_feature_vars,
        require_complete_sequences=require_complete_sequences,
        target_mode=target_mode,
        sequence_length=sequence_length,
    )

    return X_train, X_test, y_train, y_test, feature_vars

def apply_train_val_test_split_for_tcn(
    ds,
    train_storms,
    val_storms,
    test_storms,
    target_var="std_dev",
    excluded_feature_vars=None,
    require_complete_sequences=True,
    target_mode="final",
    sequence_length=TCN_SEQUENCE_LENGTH,
):
    X_train, y_train, feature_vars = build_tcn_dataset(
        ds,
        train_storms,
        target_var=target_var,
        excluded_feature_vars=excluded_feature_vars,
        require_complete_sequences=require_complete_sequences,
        target_mode=target_mode,
        sequence_length=sequence_length,
    )
    X_val, y_val, _ = build_tcn_dataset(
        ds,
        val_storms,
        target_var=target_var,
        excluded_feature_vars=excluded_feature_vars,
        require_complete_sequences=require_complete_sequences,
        target_mode=target_mode,
        sequence_length=sequence_length,
    )
    X_test, y_test, _ = build_tcn_dataset(
        ds,
        test_storms,
        target_var=target_var,
        excluded_feature_vars=excluded_feature_vars,
        require_complete_sequences=require_complete_sequences,
        target_mode=target_mode,
        sequence_length=sequence_length,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_vars

def apply_split_on_storm(ds, train_storms, test_storms, target_var="std_dev", excluded_feature_vars=None):
    # Subset in xarray
    train_ds = ds.sel(storm=list(train_storms))
    test_ds  = ds.sel(storm=list(test_storms))

    # Drop unnecessary variables
    drop_vars = ["type", "wmo_basin"]
    train_ds = train_ds.drop_vars([v for v in drop_vars if v in train_ds])
    test_ds  = test_ds.drop_vars([v for v in drop_vars if v in test_ds])

    excluded = {target_var, "zeta_std_dev"}
    if excluded_feature_vars is not None:
        excluded.update(excluded_feature_vars)

    feature_vars = [v for v in train_ds.data_vars if v not in excluded]
    if not feature_vars:
        raise ValueError("No predictor variables remain after excluding target-like columns.")

    def _flatten_subset(subset_ds, storm_ids):
        X_parts = []
        y_parts = []

        for storm_id in storm_ids:
            storm_ds = subset_ds.sel(storm=storm_id)
            target_da = storm_ds[target_var].transpose("time", "node").dropna(dim="time", how="all")

            if target_da.sizes.get("time", 0) == 0:
                continue

            storm_ds = storm_ds.sel(time=target_da["time"])
            feature_da = storm_ds[feature_vars].to_array("feature").transpose("time", "node", "feature")

            target_values = target_da.values
            valid_mask = np.isfinite(target_values)
            if not valid_mask.any():
                continue

            feature_values = feature_da.values
            X_storm = feature_values[valid_mask]
            y_storm = target_values[valid_mask]

            finite_feature_rows = np.isfinite(X_storm).all(axis=1)
            if finite_feature_rows.any():
                X_parts.append(X_storm[finite_feature_rows])
                y_parts.append(y_storm[finite_feature_rows])

        if not X_parts:
            raise ValueError("No valid samples were found after masking target and feature NaNs.")

        return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)

    X_train, y_train = _flatten_subset(train_ds, train_storms)
    X_test, y_test = _flatten_subset(test_ds, test_storms)

    return X_train, X_test, y_train, y_test
