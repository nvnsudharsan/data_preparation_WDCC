import xarray as xr
import os
import re
from datetime import datetime
import multiprocessing
from threading import Lock
import argparse
import numpy as np

netcdf_lock = Lock()

def drop_history_dim(data_array):
    if "history" in data_array.dims and data_array.sizes["history"] == 1:
        return data_array.squeeze("history", drop=True)
    return data_array

def process_single_netcdf(input_file, input_dir, output_base_dir, metadata_dict):
    input_path = os.path.join(input_dir, input_file)
    match = re.search(r"graphcast_(\d{4})_(\d{2})_(\d{2})", input_file)
    if not match:
        return

    file_year, month, day = match.groups()
    date_str = f"{file_year}-{month}-{day}"
    print(f"Processing file: {input_file} ({date_str})")

    try:
        with netcdf_lock:
            ds = xr.open_dataset(input_path, lock=False)
    except Exception as e:
        print(f"❌ Skipping {input_file} due to read error: {e}")
        with open("skipped_files.txt", "a") as skip_log:
            skip_log.write(f"{input_file} - {str(e)}\n")
        return

    if "lat" in ds and "lon" in ds:
        ds["lat"].attrs.update(metadata_dict["lat"])
        ds["lon"].attrs.update(metadata_dict["lon"])

    if "time" in ds:
        ds["time"].attrs.pop("units", None)
        ds["time"].attrs.pop("calendar", None)
        ds["time"].attrs.update({
            "long_name": "Time",
            "standard_name": "time",
            "axis": "T"
        })

    pressure_vars = ["q", "t", "z", "u", "v", "w"]
    pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    surface_var_names = ["u10m", "v10m", "t2m", "msl", "tp06"]

    surface_ds = {}
    grouped_pressure_data = {var: [] for var in pressure_vars}

    for var_name in ds.data_vars:
        var_data = drop_history_dim(ds[var_name])

        if var_name in metadata_dict:
            var_data.attrs.update(metadata_dict[var_name])

        is_pressure_level = False
        for base_var in pressure_vars:
            for level in pressure_levels:
                if var_name == f"{base_var}{level}":
                    grouped_pressure_data[base_var].append((level, var_data))
                    is_pressure_level = True
                    break
            if is_pressure_level:
                break

        if not is_pressure_level and var_name in surface_var_names:
            surface_ds[var_name] = var_data

    global_attrs = {
        "institution": "The University of Texas at Austin, Austin, Texas, USA",
        "institute_id": "UT-Austin",
        "experiment_id": "ERA5-Based Graphcast Hindcast",
        "source": "Google Deepmind Graphcast-operational output using ERA5 as initial condition",
        "model_id": "Graphcast-ERA5",
        "forcing": "ERA5 Reanalysis Data",
        "parent_experiment_id": "GraphCast Hindcast",
        "contact": (
            "Naveen Sudharsan (naveens@utexas.edu); "
            "Manmeet Singh (manmeet.singh@utexas.edu); "
            "Zong-Liang Yang (liang@jsg.utexas.edu); "
            "Dev Niyogi (dev.niyogi@jsg.utexas.edu)"
        ),
        "references": "DeepMind GraphCast Model Documentation, ERA5 Reanalysis Data",
        "product": "AI model 15-day forecast",
        "experiment": "ERA5-based Graphcast Forecasts",
        "frequency": "6-hourly",
        "creation_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "history": f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} Data processed for NetCDF format",
        "project_id": "UT Austin Graphcast Hindcast",
        "title": f"Graphcast-operational Forecast Data for {date_str}",
        "modeling_realm": "atmosphere",
        "realization": 1,
        "Conventions": "CF-1.8"
    }

    encoding = {"time": {"units": "days since 0001-01-01", "calendar": "proleptic_gregorian"}}

    if surface_ds:
        surface_output_file = os.path.join(output_base_dir, f"surface_variables_{date_str}.nc")
        surface_dataset = xr.Dataset(surface_ds)
        surface_dataset.attrs.update(global_attrs)
        with netcdf_lock:
            surface_dataset.to_netcdf(surface_output_file, encoding=encoding)
        print(f"Saved surface variables to {surface_output_file}")

    for var, level_data in grouped_pressure_data.items():
        if level_data:
            plevels, data_arrays = zip(*sorted(level_data))
            combined_var = xr.concat(data_arrays, dim="plevel")
            combined_var = combined_var.assign_coords(plevel=np.array(plevels, dtype=np.int32))

            combined_var.coords["plevel"].attrs.update({
                "long_name": "Pressure Level",
                "standard_name": "air_pressure",
                "units": "hPa"
            })

            if var == "t":
                combined_var.attrs.update({"long_name": "Temperature", "standard_name": "air_temperature", "units": "K"})
            elif var == "u":
                combined_var.attrs.update({"long_name": "Zonal Wind", "standard_name": "eastward_wind", "units": "m s-1"})
            elif var == "v":
                combined_var.attrs.update({"long_name": "Meridional Wind", "standard_name": "northward_wind", "units": "m s-1"})
            elif var == "w":
                combined_var.attrs.update({"long_name": "Vertical Wind", "standard_name": "upward_air_velocity", "units": "m s-1"})
            elif var == "z":
                combined_var.attrs.update({"long_name": "Geopotential Height", "standard_name": "geopotential_height", "units": "m"})
            elif var == "q":
                combined_var.attrs.update({"long_name": "Specific Humidity", "standard_name": "specific_humidity", "units": "kg kg-1"})

            pressure_dataset = combined_var.to_dataset(name=var)
            pressure_dataset.attrs.update(global_attrs)

            pressure_output_file = os.path.join(output_base_dir, f"{var}_pressure_levels_{date_str}.nc")
            with netcdf_lock:
                pressure_dataset.to_netcdf(pressure_output_file, encoding=encoding)
            print(f"Saved {var} pressure-level variables to {pressure_output_file}")

    ds.close()

def process_netcdf_for_year(year, input_dir, metadata_dict):
    output_base_dir = os.path.join(os.getcwd(), str(year))
    os.makedirs(output_base_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if re.match(f"graphcast_{year}_\d{{2}}_\d{{2}}\\.nc", f)])
    if not files:
        print(f"No NetCDF files found for the year {year} in {input_dir}")
        return

    with multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
        pool.starmap(process_single_netcdf, [(f, input_dir, output_base_dir, metadata_dict) for f in files])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Graphcast NetCDF files for a given year.")
    parser.add_argument("--year", required=True, help="Year to process (e.g., 2024)")
    parser.add_argument("--input_dir", required=False, default="/scratch/09295/naveens/hindcast", help="Directory containing input NetCDF files")

    args = parser.parse_args()
    year_to_process = args.year
    input_directory = args.input_dir

    metadata_dict = {
        "lat": {"long_name": "Latitude", "standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
        "lon": {"long_name": "Longitude", "standard_name": "longitude", "units": "degrees_east", "axis": "X"},
        "msl": {"long_name": "Mean Sea Level Pressure", "standard_name": "air_pressure_at_mean_sea_level", "units": "Pa"},
        "t2m": {"long_name": "2m Temperature", "standard_name": "air_temperature", "units": "K"},
        "v10m": {"long_name": "10m Meridional Wind", "standard_name": "northward_wind", "units": "m s-1"},
        "u10m": {"long_name": "10m Zonal Wind", "standard_name": "eastward_wind", "units": "m s-1"},
        "tp06": {"long_name": "6-Hour Total Precipitation", "standard_name": "precipitation_amount", "units": "m"}
    }

    process_netcdf_for_year(year_to_process, input_directory, metadata_dict)
