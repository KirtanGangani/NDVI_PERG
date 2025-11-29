import os
from datetime import datetime
import ee
import json
import geemap
import numpy as np
import geemap.foliumap as gee_folium
import leafmap.foliumap as leaf_folium
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.ops import transform
from functools import reduce
import plotly.express as px
import branca.colormap as cm
import folium
import pyproj
from io import StringIO, BytesIO
import requests
import kml2geojson
import base64


def force_stop():
    show_visitor_counter('counter.txt')
    show_credits()
    st.stop()

GCP_PROJECT_ID = 'galvanic-ripsaw-477107-g1' 
EE_INITIALIZED = False

def one_time_setup():
    global EE_INITIALIZED
    
    if EE_INITIALIZED:
        return

    print("--- Starting Earth Engine initialization (On-Demand) ---")

    try:
        ee.Initialize(project=GCP_PROJECT_ID)
        EE_INITIALIZED = True
        print("Earth Engine initialized successfully.")
    except Exception as e:
        print(f"EE Initialization FAILED: {e}")
        raise

# def one_time_setup():
#     credentials_path = os.path.expanduser("~/.config/earthengine/credentials")
#     if os.path.exists(credentials_path):
#         pass  # Earth Engine credentials already exist
#     elif "EE" in os.environ:  # write the credentials to the file
#         ee_credentials = os.environ.get("EE")
#         os.makedirs(os.path.dirname(credentials_path), exist_ok=True)
#         with open(credentials_path, "w") as f:
#             f.write(ee_credentials)
#     else:
#         raise ValueError(
#             f"Earth Engine credentials not found at {credentials_path} or in the environment variable 'EE'"
#         )
#     ee.Initialize()

def show_credits():
    # Add credits
    pdf_path = "User Manual - Kamlan App (PERG v1.0)n.pdf"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode()

    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
    <p style="text-align: center;"><a href="data:application/pdf;base64,{b64_pdf}" download="User_Manual.pdf">User Manual</a></p>
    </div>
    <div style="display: flex; justify-content: center; align-items: center;">
    <p style="text-align: center;">Developed by <a href="https://sustainability-lab.github.io/">Sustainability Lab</a>, <a href="https://www.iitgn.ac.in/">IIT Gandhinagar</a></p>
    </div>
    <div style="display: flex; justify-content: center; align-items: center;">
    <p style="text-align: center;"> Supported by <a href="https://forests.gujarat.gov.in/">Gujarat Forest Department</a>
    </p>
    </div>
    """,
    unsafe_allow_html=True,
    )


def get_gdf_from_file_url(file_url):
    if isinstance(file_url, str):
        if file_url.startswith("https://drive.google.com/file/d/"):
            ID = file_url.replace("https://drive.google.com/file/d/", "").split("/")[0]
            file_url = f"https://drive.google.com/uc?id={ID}"
        elif file_url.startswith("https://drive.google.com/open?id="):
            ID = file_url.replace("https://drive.google.com/open?id=", "")
            file_url = f"https://drive.google.com/uc?id={ID}"

        response = requests.get(file_url)
        bytes_data = BytesIO(response.content)
        string_data = response.text
    else:
        bytes_data = BytesIO(file_url.getvalue())
        string_data = file_url.getvalue().decode("utf-8")

    if string_data.startswith("<?xml"):
        geojson = kml2geojson.convert(bytes_data)
        features = geojson[0]["features"]
        epsg = 4326
        input_gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{epsg}")
    else:
        input_gdf = gpd.read_file(bytes_data)

    return input_gdf


# Function of find best suited statewise EPSG code
def find_best_epsg(geometry):
    if geometry.geom_type == "Polygon":
        centroid = geometry.centroid
    else:
        st.error("Geometry is not Polygon !!!")
        st.stop()
    common_epsg_codes = [
        7756,  # Andhra Pradesh
        7757,  # Arunachal Pradesh
        7758,  # Assam
        7759,  # Bihar
        7760,  # Delhi
        7761,  # Gujarat
        7762,  # Haryana
        7763,  # HimachalPradesh
        7764,  # JammuAndKashmir
        7765,  # Jharkhand
        7766,  # MadhyaPradesh
        7767,  # Maharastra
        7768,  # Manipur
        7769,  # Meghalaya
        7770,  # Nagaland
        7772,  # Orissa
        7773,  # Punjab
        7774,  # Rajasthan
        7775,  # UttarPradesh
        7776,  # Uttaranchal
        7777,  # A&N
        7778,  # Chattisgarh
        7779,  # Goa
        7780,  # Karnataka
        7781,  # Kerala
        7782,  # Lakshadweep
        7783,  # Mizoram
        7784,  # Sikkim
        7785,  # TamilNadu
        7786,  # Tripura
        7787,  # WestBengal
        7771,  # NE India
        7755,  # India
    ]

    for epsg in common_epsg_codes:
        crs = pyproj.CRS.from_epsg(epsg)
        area_of_use = crs.area_of_use.bounds  # Get the bounding box of the area of use

        # check if centroid of polygon lies in teh bounds of the crs
        if (area_of_use[0] <= centroid.x <= area_of_use[2]) and (area_of_use[1] <= centroid.y <= area_of_use[3]):
            return epsg  # Return the best suitable EPSG code


def daterange_str_to_dates(daterange_str):
    start_date, end_date = daterange_str.split("-")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return start_date, end_date


def daterange_dates_to_str(start_date, end_date):
    return f"{start_date.strftime('%Y/%m/%d')}-{end_date.strftime('%Y/%m/%d')}"


def daterange_str_to_year(daterange_str):
    start_date, _ = daterange_str.split("-")
    year = pd.to_datetime(start_date).year
    return str(year)


def shape_3d_to_2d(shape):
    if shape.has_z:
        return transform(lambda x, y, z: (x, y), shape)
    else:
        return shape


def preprocess_gdf(gdf):
    gdf["geometry"] = gdf["geometry"].apply(shape_3d_to_2d)
    gdf["geometry"] = gdf.buffer(0)  # Fixes some invalid geometries
    return gdf


def to_best_crs(gdf):
    best_epsg_code = find_best_epsg(gdf["geometry"].iloc[0])
    gdf = gdf.to_crs(epsg=best_epsg_code)
    return gdf


def is_valid_polygon(geometry_gdf):
    geometry = geometry_gdf.geometry.item()
    return (geometry.type == "Polygon") and (not geometry.is_empty)


def add_geometry_to_maps(map_list, geometry_gdf, buffer_geometry_gdf, opacity=0.0):
    for m in map_list:
        m.add_gdf(
            buffer_geometry_gdf,
            layer_name="Geometry Buffer",
            style_function=lambda x: {"color": "red", "fillOpacity": opacity, "fillColor": "red"},
        )
        m.add_gdf(
            geometry_gdf,
            layer_name="Geometry",
            style_function=lambda x: {"color": "blue", "fillOpacity": opacity, "fillColor": "blue"},
        )

def get_dem_slope_maps(ee_geometry, wayback_url, wayback_title):
    # Create the map for DEM
    dem_map = gee_folium.Map(controls={"scale": "bottomleft"})
    dem_map.add_tile_layer(wayback_url, name=wayback_title, attribution="Esri")

    dem_layer = ee.Image("USGS/SRTMGL1_003")
    # Set the target resolution to 10 meters
    target_resolution = 10
    dem_layer = dem_layer.resample("bilinear").reproject(crs="EPSG:4326", scale=target_resolution).clip(ee_geometry)

    # Generate contour lines using elevation thresholds
    terrain = ee.Algorithms.Terrain(dem_layer)
    contour_interval = 1
    contours = (
        terrain.select("elevation").subtract(terrain.select("elevation").mod(contour_interval)).rename("contours")
    )

    # Calculate the minimum and maximum values
    stats = contours.reduceRegion(reducer=ee.Reducer.minMax(), scale=10, maxPixels=1e13)
    max_value = stats.get("contours_max").getInfo()
    min_value = stats.get("contours_min").getInfo()
    vis_params = {"min": min_value, "max": max_value, "palette": ["blue", "green", "yellow", "red"]}
    dem_map.addLayer(contours, vis_params, "Contours")
    # Create a colormap
    cmap = cm.LinearColormap(colors=vis_params["palette"], vmin=vis_params["min"], vmax=vis_params["max"])
    tick_size = int((max_value - min_value) / 4)
    dem_map.add_legend(
        title="Elevation (m)",
        legend_dict={
            "{}-{} m".format(min_value, min_value + tick_size): "#0000FF",
            "{}-{} m".format(min_value + tick_size, min_value + 2 * tick_size): "#00FF00",
            "{}-{} m".format(min_value + 2 * tick_size, min_value + 3 * tick_size): "#FFFF00",
            "{}-{} m".format(min_value + 3 * tick_size, max_value): "#FF0000",
        },
        position="bottomright",
        draggable=False,
    )

    # Create the map for Slope
    slope_map = gee_folium.Map(controls={"scale": "bottomleft"})
    slope_map.add_tile_layer(wayback_url, name=wayback_title, attribution="Esri")

    # Calculate slope from the DEM
    slope_layer = (
        ee.Terrain.slope(
            ee.Image("USGS/SRTMGL1_003").resample("bilinear").reproject(crs="EPSG:4326", scale=target_resolution)
        )
        .clip(ee_geometry)
        .rename("slope")
    )
    # Calculate the minimum and maximum values
    stats = slope_layer.reduceRegion(reducer=ee.Reducer.minMax(), scale=10, maxPixels=1e13)
    max_value = int(stats.get("slope_max").getInfo())
    min_value = int(stats.get("slope_min").getInfo())
    vis_params = {"min": min_value, "max": max_value, "palette": ["blue", "green", "yellow", "red"]}
    slope_map.addLayer(slope_layer, vis_params, "Slope Layer")
    # Create a colormap
    colormap = cm.LinearColormap(colors=vis_params["palette"], vmin=vis_params["min"], vmax=vis_params["max"])
    tick_size = int((max_value - min_value) / 4)
    slope_map.add_legend(
        title="Slope (degrees)",
        legend_dict={
            "{}-{} deg".format(min_value, min_value + tick_size): "#0000FF",
            "{}-{} deg".format(min_value + tick_size, min_value + 2 * tick_size): "#00FF00",
            "{}-{} deg".format(min_value + 2 * tick_size, min_value + 3 * tick_size): "#FFFF00",
            "{}-{} deg".format(min_value + 3 * tick_size, max_value): "#FF0000",
        },
        position="bottomright",
        draggable=False,
    )
    return dem_map, slope_map

def add_indices(image, nir_band, red_band, blue_band, green_band, swir_band, swir2_band, evi_vars):
    """Calculates and adds multiple vegetation indices to an Earth Engine image."""
    nir = image.select(nir_band).divide(10000)
    red = image.select(red_band).divide(10000)
    blue = image.select(blue_band).divide(10000)
    green = image.select(green_band).divide(10000)
    swir = image.select(swir_band).divide(10000)
    swir2 = image.select(swir2_band).divide(10000)

    # Previously existing indices
    ndvi = image.normalizedDifference([nir_band, red_band]).rename('NDVI')
    evi = image.expression(
        'G * ((NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L))', {
            'NIR': nir, 'RED': red, 'BLUE': blue,
            'G': evi_vars['G'], 'C1': evi_vars['C1'], 'C2': evi_vars['C2'], 'L': evi_vars['L']
        }).rename('EVI')
    evi2 = image.expression(
        'G * (NIR - RED) / (NIR + L + C * RED)', {
            'NIR': nir, 'RED': red,
            'G': evi_vars['G'], 'L': evi_vars['L'], 'C': evi_vars['C']
        }).rename('EVI2')
    try:
        table = ee.FeatureCollection('projects/in793-aq-nb-24330048/assets/cleanedVDI').select(
            ["B2", "B4", "B8", "cVDI"], ["Blue", "Red", "NIR", 'cVDI'])
        classifier = ee.Classifier.smileRandomForest(50).train(
            features=table, classProperty='cVDI', inputProperties=['Blue', 'Red', 'NIR'])
        rf = image.classify(classifier).multiply(ee.Number(0.2)).add(ee.Number(0.1)).rename('RandomForest')
    except Exception as e:
        print(f"Random Forest calculation failed: {e}")
        rf = ee.Image.constant(0).rename('RandomForest')
    
    gujvdi = image.expression(
        '(-3.98 * (BLUE/NIR) + 12.54 * (GREEN/NIR) - 5.49 * (RED/NIR) - 0.19) / ' +
        '(-21.87 * (BLUE/NIR) + 12.4 * (GREEN/NIR) + 19.98 * (RED/NIR) + 1) * 2.29', {
            'NIR': nir, 'RED': red, 'BLUE': blue, 'GREEN': green
        }).clamp(0, 1).rename('GujVDI')
    gujevi = image.expression(
        '0.5 * (NIR - RED) / (NIR + 6 * RED - 8.25 * BLUE - 0.01)', {
            'NIR': nir, 'RED': red, 'BLUE': blue
        }).clamp(0, 1).rename('GujEVI')
    
    mndwi = image.normalizedDifference([green_band, swir_band]).rename('MNDWI')
    savi = image.expression('(1 + L) * (NIR - RED) / (NIR + RED + L)', {
        'NIR': nir, 'RED': red, 'L': 0.5
    }).rename('SAVI')
    mvi = image.expression('(NIR - (GREEN + SWIR)) / (NIR + (GREEN + SWIR))', {
        'NIR': nir, 'GREEN': green, 'SWIR': swir
    }).rename('MVI')
    nbr = image.normalizedDifference([nir_band, swir2_band]).rename('NBR')
    gci = image.expression('(NIR - GREEN) / GREEN', {
        'NIR': nir, 'GREEN': green
    }).rename('GCI')

    return image.addBands([ndvi, evi, evi2, rf, gujvdi, gujevi, mndwi, savi, mvi, nbr, gci])
    
def get_histogram(veg_index, image, geometry, bins):
    # Get image values as a list
    values = image.reduceRegion(reducer=ee.Reducer.toList(), geometry=geometry, scale=10, maxPixels=1e13).get(veg_index)

    # Convert values to a NumPy array
    values_array = np.array(values.getInfo())

    # Compute the histogram on bins
    hist, bin_edges = np.histogram(values_array, bins=bins)

    return hist, bin_edges


def process_date(
    daterange,
    satellite,
    veg_indices,
    satellites,
    buffer_ee_geometry,
    ee_feature_collection,
    buffer_ee_feature_collection,
    result_df,
):
    start_date, end_date = daterange
    daterange_str = daterange_dates_to_str(start_date, end_date)
    prefix = f"Processing {satellite} - {daterange_str}"
    try:
        attrs = satellites[satellite]
        collection = attrs["collection"]
        collection = collection.filterBounds(buffer_ee_geometry)
        collection = collection.filterDate(start_date, end_date)

        bucket = {}
        for veg_index in veg_indices:
            print(veg_index)
            mosaic_veg_index = collection.qualityMosaic(veg_index)
            fc = geemap.zonal_stats(
                mosaic_veg_index, ee_feature_collection, scale=attrs["scale"], return_fc=True
            ).getInfo()
            mean_veg_index = fc["features"][0]["properties"][veg_index]
            bucket[veg_index] = mean_veg_index
            fc = geemap.zonal_stats(
                mosaic_veg_index, buffer_ee_feature_collection, scale=attrs["scale"], return_fc=True
            ).getInfo()
            buffer_mean_veg_index = fc["features"][0]["properties"][veg_index]
            bucket[f"{veg_index}_buffer"] = buffer_mean_veg_index
            bucket[f"{veg_index}_ratio"] = mean_veg_index / buffer_mean_veg_index
            bucket[f"mosaic_{veg_index}"] = mosaic_veg_index

        # Get median mosaic
        bucket["mosaic_visual_max_ndvi"] = collection.qualityMosaic("NDVI")
        bucket["mosaic_visual_median"] = collection.median()
        bucket["image_visual_least_cloud"] = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()

        if satellite == "COPERNICUS/S2_SR_HARMONIZED":
            cloud_mask_probability = fc["features"][0]["properties"]["MSK_CLDPRB"] / 100
        else:
            cloud_mask_probability = None
        bucket["Cloud (0 to 1)"] = cloud_mask_probability
        result_df.loc[daterange_str, list(bucket.keys())] = list(bucket.values())
        count = collection.size().getInfo()
        suffix = f" - Processed {count} images"
        write_info(f"{prefix}{suffix}")
    except Exception as e:
        print(e)
        suffix = f" - Imagery not available"
        write_info(f"{prefix}{suffix}")


def write_info(info, center_align=False):
    if center_align:
        st.write(f"<div style='text-align: center; color:#006400;'>{info}</div>", unsafe_allow_html=True)
    else:
        st.write(f"<span style='color:#006400;'>{info}</span>", unsafe_allow_html=True)

def read_counter(counter_file):
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("0")
    with open(counter_file, "r") as f:
        content = f.read().strip()
        return int(content) if content.isdigit() else 0

def write_counter(counter_file, count):
    with open(counter_file, "w") as f:
        f.write(str(count))

def show_visitor_counter(counter_file="counter.txt"):
    if "visitor_counted" not in st.session_state:
        count = read_counter(counter_file) + 1
        write_counter(counter_file, count)
        st.session_state.visitor_counted = True
    else:
        count = read_counter(counter_file)

    count_str = str(count).zfill(6)

    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; align-items:center; gap:10px;">
            <h3 style="margin:0;">Visitor Counter</h3>
            <div style="display:flex; gap:6px;">
                {''.join([
                    f'<div style="border:2px solid black; border-radius:6px; text-align:center; font-size:20px; padding:6px; width:35px; background:white; color:black; font-weight:bold;">{digit}</div>'
                    for digit in count_str
                ])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )