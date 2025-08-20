""" 
Postprocess the csv files from both the RCBenchmark thrust stand and the Crazyflie. 
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import sys
from scipy.optimize import curve_fit

def parse_cf_csv(cf_csv):
    """ 
    Parse the crazyflie csv and return a pandas dict and do any data manipulation necessary.
    """ 

    cf_df = pd.read_csv(cf_csv, index_col=False, on_bad_lines='skip')
    cf_df['timestep'] -= cf_df['timestep'][0]
    cf_df['timestep'] /= 1000  # go from ms to s

    return cf_df

def parse_rc_csv(rc_csv):
    """ 
    Parse the rcbenchmark csv and return a pandas dict and do any data manipulation necessary. 
    """ 

    rc_df = pd.read_csv(rc_csv, index_col=False, on_bad_lines='skip')
    rc_df['Time (s)'] -= rc_df['Time (s)'][0]

    # Remove all rows where the thrust is NaN
    rc_df = rc_df.dropna(subset=['Thrust (kgf)'])

    return rc_df

def sync_csvfiles(cf_csv, rc_csv):
    """ 
    Synchronize the timestamps of the Crazyflie csv file and the RCBenchmark csv file. 
    
    Inputs:
        cf_csv: the crazyflie csv file
        rc_csv: the rcbenchmark csv file
    Outputs:
    """ 

    cf_df = parse_cf_csv(cf_csv)
    rc_df = parse_rc_csv(rc_csv)

    avg_pwm_cmd = np.mean(cf_df[['m1', 'm2', 'm3', 'm4']], axis=1)
    motorspeed = rc_df['Motor Optical Speed (rad/s)']

    # Make a plot for selecting sync points
    fig_sync, ax_sync = plt.subplots(num="Select time sync")
    ax_sync.plot(cf_df['timestep'], avg_pwm_cmd/np.max(avg_pwm_cmd), 'r.', label="Avg Motor PWM")
    ax_sync.plot(rc_df['Time (s)'], motorspeed/np.max(motorspeed), 'b.', label="Optical Motor Speed")
    ax_sync.set_xlabel("Time (s)")
    ax_sync.set_ylabel("Normalized value")
    ax_sync.legend()
    ax_sync.set_title("Click on two points: first CF, then RC")

    plt.show(block=False)

    # Let the user select 2 points
    print("Please click on the Crazyflie point and then the RC Benchmark point to sync.")
    pts = plt.ginput(2, timeout=-1)
    plt.close(fig_sync)

    # Extract x-values (times)
    cf_time_selected = pts[0][0]
    rc_time_selected = pts[1][0]

    print(f"Selected sync times -> CF: {cf_time_selected:.3f}s, RC: {rc_time_selected:.3f}s")

    # Compute time shift and apply to CF timestamps
    time_offset = rc_time_selected - cf_time_selected
    cf_df['timestep'] += time_offset

    # Resample entire CF dataframe to RC timestamps
    cf_df_resampled = pd.DataFrame({'Time (s)': rc_df['Time (s)']})
    for col in cf_df.columns:
        if col != 'timestep':  # Skip old time column
            cf_df_resampled[col] = np.interp(rc_df['Time (s)'], cf_df['timestep'], cf_df[col])

    # # Replot synced curves
    # fig_synced, ax_synced = plt.subplots(num="Synced Curves")
    # ax_synced.plot(cf_df['timestep'], avg_pwm_cmd/np.max(avg_pwm_cmd), 'r.', label="Avg Motor PWM (CF)")
    # ax_synced.plot(rc_df['Time (s)'], motorspeed/np.max(motorspeed), 'b.', label="Optical Motor Speed (RC)")
    # ax_synced.set_xlabel("Time (s)")
    # ax_synced.set_ylabel("Normalized value")
    # ax_synced.legend()
    # ax_synced.set_title("Time-synced curves")
    # plt.show()

    return cf_df_resampled, rc_df

def combine_all_synced(cf_csvs, rc_csvs):
    """
    Sync all CSV pairs and combine all outputs into single CF and RC dataframes.

    Inputs:
        cf_csvs: a list of csv files for the Crazyflie data. 
        rc_csvs: a list of csv files for the RCBenchmark thrust stand. 

        **** IMPORTANT: we assume both lists are ordered... s.t. cf_csvs[i] and rc_csvs[i] are the same dataset!!
    """
    cf_list, rc_list = [], []

    for i, (cf_csv, rc_csv) in enumerate(zip(cf_csvs, rc_csvs)):
        cf_synced, rc_synced = sync_csvfiles(cf_csv, rc_csv)

        # Add experiment ID
        cf_synced['ID'] = round(rc_synced['Voltage (V)'].mean(), 1)
        rc_synced['ID'] = round(rc_synced['Voltage (V)'].mean(), 1)

        cf_list.append(cf_synced)
        rc_list.append(rc_synced)

    # Combine all synced dataframes into one per source
    cf_combined = pd.concat(cf_list, ignore_index=True)
    rc_combined = pd.concat(rc_list, ignore_index=True)

    # Drop time columns if desired
    cf_combined = cf_combined.drop(columns=['Time (s)'], errors='ignore')
    rc_combined = rc_combined.drop(columns=['Time (s)'], errors='ignore')

    return cf_combined, rc_combined

def battery_compensated_motorspeed_model(pwm, voltage, a, b):

    return a*(voltage + b)*(pwm - 10_000)**(2/3)

def fit_battery_compensated_motorspeed_model(cf_df, rc_df):
    """ 
    Fit a model that maps pwm and input voltage to RPM. 

    Inputs:
        cf_df, rc_df: the dataframes for data between the crazyflie and rcbenchmark. 

    """
    pwm = cf_df['m1'].values
    voltage = rc_df['Voltage (V)'].values
    rpm = rc_df['Motor Optical Speed (RPM)'].values

    popt, _ = curve_fit(lambda xdata, a, b: rpm_map(xdata[0], xdata[1], a, b),
                (pwm, voltage), rpm, p0=(1e-6, 0.1), maxfev=10_000)

    print(f"motor_speed = {popt[0]}*(voltage + {popt[1]})*(pwm - 10_000)^(2/3)")
    return popt, pwm, voltage, rpm

def produce_mappings(cf_df, rc_df, num_motors=4):
    """ 
    Compute the mappings between PWM, RPM, and thrust. 
    Fit y = A x^2 to thrust vs motor speed and plot per experiment ID.

    Inputs:
        cf_df: combined Crazyflie dataframe with ID
        rc_df: combined thrust stand dataframe with ID
    """

    # Unique experiment IDs
    ids = rc_df['ID'].unique()

    # Define colors and markers for plotting
    colors = plt.cm.tab10.colors  # up to 10 distinct colors
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']

    # Prepare data for fitting
    motorspeed = rc_df['Motor Optical Speed (rad/s)'].values
    thrust = np.abs(rc_df['Thrust (kgf)'] * 9.81 / num_motors)  # convert kgf to N and separated by each motor
    pwm = cf_df['m1'].values 

    # Fit quadratic model y = A x^2 (no intercept)
    k_eta = np.sum(thrust * motorspeed**2) / np.sum(motorspeed**4)
    print(f"Fitted quadratic coefficient k_eta = {k_eta:.3e}")

    # Plotting
    fig_tvm, ax_tvm = plt.subplots()
    for i, exp_id in enumerate(ids):
        exp_data = rc_df[rc_df['ID'] == exp_id]
        exp_thrust = np.abs(exp_data['Thrust (kgf)'] * 9.81 / num_motors)
        ax_tvm.scatter(exp_data['Motor Optical Speed (rad/s)'], exp_thrust, 
            color=colors[i % len(colors)],
            s=5,
            alpha=0.8,
            marker=markers[i % len(markers)],
            label=f'Experiment {exp_id}'
        )

    # Add text in top-right corner with the equation
    eq_text = f"$T = {k_eta:.3e} \\cdot \\eta^2$"
    ax_tvm.text(
        0.95, 0.1, eq_text,
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax_tvm.transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )

    # Overlay quadratic fit curve
    motorspeed_fit = np.linspace(0, motorspeed.max()*1.05, 200)
    thrust_fit = k_eta * motorspeed_fit**2
    ax_tvm.plot(motorspeed_fit, thrust_fit, 'k--', label='Quadratic Fit')

    ax_tvm.set_xlabel("Motor Speed (rad/s)")
    ax_tvm.set_ylabel("Thrust (N)")
    ax_tvm.legend()
    ax_tvm.set_title("Single Motor Thrust vs Motor Speed with Quadratic Fit")

    #

    return k_eta

if __name__ == "__main__":
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    rc_dir = os.path.join(data_dir, 'thrust_stand_data')
    cf_dir = os.path.join(data_dir, 'cf_data')

    cf_csvs = [os.path.join(cf_dir, f) for f in os.listdir(cf_dir) if f.endswith('.csv')]
    rc_csvs = [os.path.join(rc_dir, f) for f in os.listdir(rc_dir) if f.endswith('.csv')]

    # Get combine dataframes of all the csv files for the Crazyflie and RCBenchmark. 
    cf_combined, rc_combined = combine_all_synced(cf_csvs, rc_csvs)

    produce_mappings(cf_combined, rc_combined)

    plt.show()