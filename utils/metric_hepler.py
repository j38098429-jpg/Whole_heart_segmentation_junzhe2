import os
import numpy as np 
import pandas as pd

from torchvision.transforms import functional as F
from PIL import Image
import medpy.metric.binary as mmb

from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import plotly.subplots as sp

import math

from utils.Echo.camus_clinical_metric import *

from typing import Tuple

def check_variable(data_list):
    c_data_list = []
    for data in data_list:
        if isinstance(data, np.ndarray):  # Check if 'pred' is on GPU
            pred_cpu = data  # Move to CPU
        else:
            pred_cpu = data.cpu().numpy()
        c_data_list.append(pred_cpu)
    return c_data_list

def bool_array_to_numpy(data):
    data = data.astype(int)
    return data

def compute_metrics(label , pred, num_class, size, voxel_spacing , frame_list , to_origianl_size = True, only_ED_ES = None):
    #### input : fr dx dy
    #### label : fr dx dy
    
    target_label_list = list(range(1, num_class-1)) 

    [label, pred] = check_variable([label, pred])

    num_px_mask = {str(key): [] for key in target_label_list}
    num_px_pred = {str(key): [] for key in target_label_list}
    
    c_x = {str(key): [] for key in target_label_list}
    c_y = {str(key): [] for key in target_label_list}
    
    asd = {str(key): [] for key in target_label_list}
    dsc = {str(key): [] for key in target_label_list}
    hd = {str(key): [] for key in target_label_list}
    
    info = {"gt_ed": [] , "gt_es": [] ,"pred_ed":[], "pred_es":[]}
    
    if len(np.shape(pred)) > 4:
        #3D
        for fr in range(np.shape(pred)[4]):
            if only_ED_ES == False:
                if to_origianl_size == True:
                    original_size_pred = np.asarray(F.resize(Image.fromarray( (pred[0,0,:,:,fr]).astype(np.uint8), mode="L"), size, interpolation=F.InterpolationMode.NEAREST))
                
                for lb in (target_label_list):  # Assuming there are three labels: 0, 1, 2

                    # Calculate the mask arrays and their sums for each lb
                    if to_origianl_size == True:
                        label_ = label[fr] == lb
                        pred_ = original_size_pred == lb
                    else:
                        label_ = label[fr] == lb
                        pred_ = pred[fr] == lb
                    
                    label_ = bool_array_to_numpy(label_)
                    pred_ = bool_array_to_numpy(pred_)
                                
                    label_pixels = np.where( label_== 1.0)
                    pred_pixels = np.where( pred_== 1.0)
                    
                    center_x_label = int(np.mean(label_pixels[0]))
                    center_y_label = int(np.mean(label_pixels[1]))
                    
                    center_x_pred = int(np.mean(pred_pixels[0]))
                    center_y_pred = int(np.mean(pred_pixels[1]))
                    
                    c_x[str(lb)].append((center_x_label, center_x_pred))
                    c_y[str(lb)].append((center_y_label, center_y_pred))
                    
                    num_px_mask[str(lb)].append(label_.sum())
                    num_px_pred[str(lb)].append(pred_.sum())
                    
                    dsc[str(lb)].append(mmb.dc(pred_, label_))
                    hd[str(lb)].append(mmb.hd(pred_, label_, voxelspacing = voxel_spacing))
                    asd[str(lb)].append(mmb.asd(pred_, label_, voxelspacing = voxel_spacing))
            else:
                if str(fr + 1) in range(np.shape(pred)[4]):
                    if to_origianl_size == True:
                        original_size_pred = np.asarray(F.resize(Image.fromarray( (pred[0,0,:,:,fr]).astype(np.uint8), mode="L"), size, interpolation=F.InterpolationMode.NEAREST))
                    
                    for lb in (target_label_list):  # Assuming there are three labels: 0, 1, 2

                        # Calculate the mask arrays and their sums for each lb
                        if to_origianl_size == True:
                            label_ = label[fr] == lb
                            pred_ = original_size_pred == lb
                        else:
                            label_ = label[fr] == lb
                            pred_ = pred[fr] == lb
                        
                        label_ = bool_array_to_numpy(label_)
                        pred_ = bool_array_to_numpy(pred_)
                                    
                        label_pixels = np.where( label_== 1.0)
                        pred_pixels = np.where( pred_== 1.0)
                        
                        center_x_label = int(np.mean(label_pixels[0]))
                        center_y_label = int(np.mean(label_pixels[1]))
                        
                        center_x_pred = int(np.mean(pred_pixels[0]))
                        center_y_pred = int(np.mean(pred_pixels[1]))
                        
                        c_x[str(lb)].append((center_x_label, center_x_pred))
                        c_y[str(lb)].append((center_y_label, center_y_pred))
                        
                        num_px_mask[str(lb)].append(label_.sum())
                        num_px_pred[str(lb)].append(pred_.sum())
                        
                        dsc[str(lb)].append(mmb.dc(pred_, label_))
                        hd[str(lb)].append(mmb.hd(pred_, label_, voxelspacing = voxel_spacing))
                        asd[str(lb)].append(mmb.asd(pred_, label_, voxelspacing = voxel_spacing))
            
            if str(fr+1) in frame_list:
                label = bool_array_to_numpy(label)
                original_size_pred = bool_array_to_numpy(original_size_pred)
                
                if str(fr+1) == frame_list[0]:
                    info["gt_ed"] = label[fr] == 1.0
                    info["pred_ed"] = original_size_pred == 1.0
                else:
                    info["gt_es"] = label[fr] == 1.0
                    info["pred_es"] = original_size_pred == 1.0
    else:
        # Loop one patient
        for fr in range(np.shape(pred)[0]):
            if frame_list == None:    
                if to_origianl_size == True:
                    original_size_pred = np.asarray(F.resize(Image.fromarray( (pred[fr,0,...]).astype(np.uint8), mode="L"), size, interpolation=F.InterpolationMode.NEAREST))
                
                for lb in (target_label_list):  # Assuming there are three labels: 0, 1, 2

                    # Calculate the mask arrays and their sums for each lb
                    if to_origianl_size == True:
                        label_ = label[fr] == lb
                        pred_ = original_size_pred == lb
                    else:
                        label_ = label[fr] == lb
                        pred_ = pred[fr] == lb
                    
                    label_ = bool_array_to_numpy(label_)
                    pred_ = bool_array_to_numpy(pred_)
                                
                    label_pixels = np.where( label_== 1.0)
                    pred_pixels = np.where( pred_== 1.0)
                    
                    center_x_label = int(np.mean(label_pixels[0]))
                    center_y_label = int(np.mean(label_pixels[1]))
                    
                    center_x_pred = int(np.mean(pred_pixels[0]))
                    center_y_pred = int(np.mean(pred_pixels[1]))
                    
                    c_x[str(lb)].append((center_x_label, center_x_pred))
                    c_y[str(lb)].append((center_y_label, center_y_pred))
                    
                    num_px_mask[str(lb)].append(label_.sum())
                    num_px_pred[str(lb)].append(pred_.sum())
                    
                    dsc[str(lb)].append(mmb.dc(pred_, label_))
                    hd[str(lb)].append(mmb.hd(pred_, label_, voxelspacing = voxel_spacing))
                    asd[str(lb)].append(mmb.asd(pred_, label_, voxelspacing = voxel_spacing))
            else:
                frame_list = [int(float(num)) for num in frame_list]
                
                if to_origianl_size == True:
                    original_size_pred = np.asarray(F.resize(Image.fromarray( (pred[fr,0,...]).astype(np.uint8), mode="L"), size, interpolation=F.InterpolationMode.NEAREST))
                    
                    if np.shape(label)[1] != size[0]:
                        original_size_label = np.asarray(F.resize(Image.fromarray( (label[fr,...]).astype(np.uint8), mode="L"), size, interpolation=F.InterpolationMode.NEAREST))

                for lb in (target_label_list):  # Assuming there are three labels: 0, 1, 2

                    # Calculate the mask arrays and their sums for each lb
                    if to_origianl_size == True:
                        pred_ = original_size_pred == lb
                        if 'original_size_label' in locals():
                            label_ = original_size_label == lb
                        else:
                            label_ = label[fr] == lb
                    else:
                        pred_ = pred[fr] == lb
                        if 'original_size_label' in locals():
                            label_ = original_size_label == lb
                        else:
                            label_ = label[fr] == lb
                        
                    num_px_mask[str(lb)].append(label_.sum())
                    num_px_pred[str(lb)].append(pred_.sum())
                    
                    if int(fr+1) in frame_list:
                        label_ = bool_array_to_numpy(label_)
                        pred_ = bool_array_to_numpy(pred_)
                                    
                        label_pixels = np.where( label_== 1.0)
                        pred_pixels = np.where( pred_== 1.0)
                        
                        center_x_label = int(np.mean(label_pixels[0]))
                        center_y_label = int(np.mean(label_pixels[1]))
                        
                        center_x_label = 0
                        center_y_label = 0
                        
                        center_x_pred = int(np.mean(pred_pixels[0]))
                        center_y_pred = int(np.mean(pred_pixels[1]))
                        
                        c_x[str(lb)].append((center_x_label, center_x_pred))
                        c_y[str(lb)].append((center_y_label, center_y_pred))
                        
                        dsc[str(lb)].append(mmb.dc(pred_, label_))
                        hd[str(lb)].append(mmb.hd(pred_, label_, voxelspacing = voxel_spacing))
                        asd[str(lb)].append(mmb.asd(pred_, label_, voxelspacing = voxel_spacing))
                    
            if int(fr+1) in frame_list:
                label = bool_array_to_numpy(label)
                original_size_pred = bool_array_to_numpy(original_size_pred)
                    
                if int(fr+1) == frame_list[0]:
                    if 'original_size_label' in locals():
                        info["gt_ed"] = original_size_label == 1.0
                    else:
                        info["gt_ed"] = label[fr] == 1.0
                    info["pred_ed"] = original_size_pred == 1.0
                else:
                    if 'original_size_label' in locals():
                        info["gt_es"] = original_size_label == 1.0
                    else:
                        info["gt_es"] = label[fr] == 1.0
                    info["pred_es"] = original_size_pred == 1.0
                    
        ef_pred, ef_gt, ef_error = cal_EF(info["pred_ed"],info["pred_es"], info["gt_ed"], info["gt_es"], voxel_spacing)
        
        if ef_pred is None or ef_pred == 0 or ef_pred > 200:
            ef_pred = None
            ef_gt = None
            ef_error = None
            
            # return dsc, hd, asd, c_x, c_y, num_px_mask, num_px_pred, ef_pred, ef_gt, ef_error, arr_endo, arr_epi, thres_endo,thres_epi
        
            return dsc, hd, asd, c_x, c_y, num_px_mask, num_px_pred, ef_pred, ef_gt, ef_error, None , None, None, None
        
        else:           
            arr_endo, thres_endo = check_temporal_consistency_errors(1.60e-2, np.array(num_px_pred["1"]))
            # arr_epi, thres_epi = check_temporal_consistency_errors(3.21e-2, np.array(num_px_pred["2"]))
            
            thres_endo  = sum(thres_endo)
            # thres_epi = sum(thres_epi)
            
            arr_endo = np.mean(arr_endo)
            # arr_epi= np.mean(arr_epi)
        
            return dsc, hd, asd, c_x, c_y, num_px_mask, num_px_pred, ef_pred, ef_gt, ef_error, arr_endo, None, thres_endo, None


def save_report_mia(p_id_list, data, dest, title_text,  label = ['ENDO']): #label = ['ENDO' EPI]

    df = pd.DataFrame()
    df_new = pd.DataFrame()
    
    avg_ = []
    std_ = []
    for label_idx, label_name in enumerate(label):
        extracted_list = [d[str(label_idx+1)] for d in data if str(label_idx+1) in d]

        # Convert the lists to a dictionary
        data_dict = dict(zip(p_id_list, extracted_list))

        # Find the maximum length
        max_len = max(len(lst) for lst in extracted_list)

        # Make all lists the same length
        for key in data_dict:
            length_diff = max_len - len(data_dict[key])
            data_dict[key] += [None] * length_diff

        # Create a DataFrame from the dictionary
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        
        bins = 30
        # Normalize the x-axis
        x_norm = np.linspace(0, 1, bins)  # Generates [0, 0.05, 0.1, ... 1.0]

        # Create a new plot
        fig = go.Figure()

        # Add traces (lines), one for each patient
        for i in range(df.shape[0]):
            non_nan_values = [x for x in df.iloc[i] if not math.isnan(x)]
            length = len(non_nan_values)

            new_indices = np.linspace(0, length-1, bins)
            interpolated_list = np.interp(new_indices, np.arange(length), df.iloc[i][:length])
            fig.add_trace(go.Scatter(x=x_norm, y= interpolated_list.T, mode='lines+markers', name='Endo'))
            
            df_new = df_new.append(pd.DataFrame(interpolated_list).T)

        df_new.reset_index(drop=True, inplace=True)
        average_value = df.mean().mean()
        avg_.append(average_value)
        standard_deviation = df.std().mean()
        std_.append(standard_deviation)
        # Convert list to a DataFrame
        new_row = pd.DataFrame([x_norm], columns=df_new.columns)

        
        # Create subplot with 1 row and 2 columns
        fig = sp.make_subplots(rows=1, cols=2,  column_widths=[0.85, 0.15], shared_xaxes=False, shared_yaxes=True)

        new_row_repeated = np.repeat(new_row.values, len(df_new), axis=0)

        # Flatten both new_row_repeated and df_new
        new_row_flat = new_row_repeated.flatten()
        df_new_flat = df_new.values.flatten()

        # Create a density map for your scatter plot
        xy = np.vstack([new_row_flat, df_new_flat])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = new_row_flat[idx], df_new_flat[idx], z[idx]
        
        if label_idx == 0:
            grad = [[0, '#87CEFA'], [1, 'blue']]
        else:
            grad = [[0, '#FFA500'], [1, '#FF4500']]
            
        # Add scatter plots to the first subplot
        fig.add_trace(go.Scatter(x=x, 
                                y=y, 
                                mode='markers', 
                                marker=dict(
                                    size=15,  
                                    color=z, # set color to an array/list of desired values
                                    colorscale=grad, # choose a colorscale
                                    opacity=0.5
                                    )
                                ), 
                    row=1, col=1)

        # Create the histogram data
        counts, bin_edges = np.histogram(df_new.values.flatten(), bins=100, density=True)
        
        # Calculate the bin centers
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        
        # Create a scatter trace for the line
        
        smoothed_counts = savgol_filter(counts, window_length=21, polyorder=3)

        fig.add_trace(go.Scatter(x=smoothed_counts, y=bin_centers, mode='lines', line=dict(color=grad[0][1], width=2)), row=1, col=2)

        # Update layout to remove y-axis title for histogram and set the margin to display properly
        fig.update_layout(showlegend=False, bargap=0.01)
        fig.update_yaxes(range=[0.5, 1.0], row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        fig.update_xaxes(range=[0, 20], row=1, col=2)
        
        # Update layout to add axis titles
        fig.update_layout(title_text=title_text, title_x=0.5,  # Set the title of the entire subplot
                        xaxis_title="Normalized Frame",  # Set the X axis title for the first subplot
                        yaxis_title=title_text,  # Set the Y axis title for the first subplot
                        yaxis=dict(
                        dtick=0.05  # Set the y-axis unit to 0.05 for subplot 1
                    ),
                        xaxis2=dict(
                        dtick=5  # Set the y-axis unit to 0.05 for subplot 1
                    ),
                        showlegend=False, bargap=0.01)


        # Show the plot
        fig.show()
        if not os.path.exists(dest):
            os.makedirs(dest)
        fig.write_image(os.path.join(dest, title_text + "_" + label_name + "_figure.png"))

    return avg_, std_


# lv_area: 1.60e-2
# lv_base_width: 9.69e-2
# lv_length: 2.62e-2
# lv_orientation: 2.29e-2
# myo_area: 3.21e-2
# epi_center_x: 4.56e-3
# epi_center_y: 1.10e-2

# Statistics about the min/max values of the attributes computed on the images
# lv_area: [19632, 112404]
# lv_base_width: [53.75872022286245, 172.8843544106869]
# lv_length: [301.42494919963076, 612.8623418027902]
# lv_orientation: [-7.035926925125878, 15.260000077617264]
# myo_area: [23228, 64134]
# epi_center_x: [183.36958708083102, 479.3726391015824]
# epi_center_y: [285.85869497540625, 609.2086387810838]


def minmax_scaling(data: np.ndarray, bounds: Tuple[float, float] = None) -> np.ndarray:
    """Standardizes data w.r.t. predefined min/max bounds, if provided, or its own min/max otherwise.
    Args:
        data: Data to scale.
        bounds: Prior min and max bounds to use to scale the data.
    Returns:
        Data scaled w.r.t. the predefined or computed bounds.
    """
    # If no prior min/max bounds for the data are provided, compute them from its values
    if bounds is None:
        min, max = data.min(), data.max()
    else:
        min, max = bounds
    return (data - min) / (max - min)

def check_temporal_consistency_errors(threshold: float, array, *args, **kwargs) -> np.ndarray:
    """Identifies instants where the temporal consistency metric exceed a certain threshold value.

    Args:
        threshold: The maximum value above which the absolute value of the temporal consistency metric flags the instant
            as temporally inconsistent.
        *args: Positional arguments to pass to the temporal consistency metric computation.
        **kwargs: Keyword arguments to pass to the temporal consistency metric computation.

    Returns:
        temporal_errors: (n_samples,) Whether each instant is temporally inconsistent (`True`) or not (`False`).
    """
    return np.abs(compute_temporal_consistency_metric(array, *args, **kwargs)), np.abs(compute_temporal_consistency_metric(array, *args, **kwargs)) > threshold


def compute_temporal_consistency_metric(attribute: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Computes the error between attribute values and the interpolation between their previous/next neighbors.

    Args:
        attribute: (n_samples, [1]), The 1D signal to analyze for temporal inconsistencies between instants.
        *args: Positional arguments to pass to the scaling function.
        **kwargs: Keyword arguments to pass to the scaling function.

    Returns:
        metric: (n_samples,) Error between attribute values and the interpolation between their previous/next neighbors.
    """
    attribute = minmax_scaling(attribute, *args, **kwargs)

    # Compute the temporal consistency metric
    prev_neigh = attribute[:-2]  # Previous neighbors of non-edge instants
    next_neigh = attribute[2:]  # Next neighbors of non-edge instants
    neigh_inter_diff = attribute[1:-1] - ((prev_neigh + next_neigh) / 2)
    # Pad edges with 0; since edges are missing a neighbor for the interpolation,
    # they are considered temporally consistent by default
    pad_width = [(1, 1)] + [(0, 0)] * (attribute.ndim - 1)
    neigh_inter_diff = np.pad(neigh_inter_diff, pad_width, constant_values=0)

    return neigh_inter_diff