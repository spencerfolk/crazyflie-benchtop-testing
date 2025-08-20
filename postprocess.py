""" 
Postprocess the csv files from both the RCBenchmark thrust stand and the Crazyflie. 
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import sys
from scipy.optimize import curve_fit
import yaml
from mpl_toolkits.mplot3d import Axes3D

def r2_score(y_true, y_pred):
    """
    Compute coefficient of determination R^2.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

def load_sync_offsets(sync_path):
    if os.path.exists(sync_path):
        with open(sync_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_sync_offsets(offsets, sync_path):
    with open(sync_path, "w") as f:
        yaml.safe_dump(offsets, f)

def parse_cf_csv(cf_csv):
    """ 
    Parse the crazyflie csv and return a pandas dict and do any data manipulation necessary.
    """ 

    cf_df = pd.read_csv(cf_csv, index_col=False, on_bad_lines='skip')
    cf_df['timestep'] -= cf_df['timestep'][0]
    cf_df['timestep'] /= 1000  # go from ms to s

    return cf_df

def parse_rc_csv(rc_csv, rpm_col='Motor Optical Speed (rad/s)', threshold=1000, motorspeed_min=250.0):
    """ 
    Parse the rcbenchmark csv and return a pandas dataframe with only steady-state points.
    
    Inputs:
        rc_csv: path to the CSV file.
        rpm_col: name of the column for motor speed in rad/s.
        threshold: allowable rate of change in rad/s per second to keep (default=50).
        rpm_min: minimum motor_speed value to keep (default=250).
    """ 
    # Load data
    rc_df = pd.read_csv(rc_csv, index_col=False, on_bad_lines='skip')
    rc_df['Time (s)'] -= rc_df['Time (s)'][0]

    # Remove all rows where thrust is NaN
    rc_df = rc_df.dropna(subset=['Thrust (kgf)'])

    # Compute discrete derivative of motor speed
    rc_df['dRPM'] = rc_df[rpm_col].diff().abs() / rc_df['Time (s)'].diff()

    # Apply filters: steady-state + motorspeed_min
    steady_df = rc_df[(rc_df['dRPM'] < threshold) & (rc_df[rpm_col] > motorspeed_min)].copy()

    # Drop helper column if not needed
    steady_df = steady_df.drop(columns=['dRPM'])

    return steady_df

def sync_csvfiles(cf_csv, rc_csv, sync_path):
    """ 
    Synchronize the timestamps of the Crazyflie csv file and the RCBenchmark csv file. 
    
    Inputs:
        cf_csv: the crazyflie csv file
        rc_csv: the rcbenchmark csv file
    Outputs:
    """ 

    cf_df = parse_cf_csv(cf_csv)
    rc_df = parse_rc_csv(rc_csv)

    # Load stored offsets
    offsets = load_sync_offsets(sync_path)
    key = f"{os.path.basename(cf_csv)} | {os.path.basename(rc_csv)}"

    if key in offsets:
        # Use stored offset
        time_offset = offsets[key]
        print(f"Using cached sync offset {time_offset:.3f}s for {key}")
    else:
        # Interactive mode. 

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

        # Save offset for next run
        offsets[key] = float(time_offset)
        save_sync_offsets(offsets, sync_path)

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

def combine_all_synced(cf_csvs, rc_csvs, sync_path='sync_file.yaml'):
    """
    Sync all CSV pairs and combine all outputs into single CF and RC dataframes.

    Inputs:
        cf_csvs: a list of csv files for the Crazyflie data. 
        rc_csvs: a list of csv files for the RCBenchmark thrust stand. 

        **** IMPORTANT: we assume both lists are ordered... s.t. cf_csvs[i] and rc_csvs[i] are the same dataset!!
    """
    cf_list, rc_list = [], []

    for i, (cf_csv, rc_csv) in enumerate(zip(cf_csvs, rc_csvs)):
        cf_synced, rc_synced = sync_csvfiles(cf_csv, rc_csv, sync_path=sync_path)

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

def battery_compensated_motorspeed_model(pwm, voltage, a, b, c):

    return a*(voltage + b)*(pwm - c)**(2/3)

def fit_battery_compensated_motorspeed_model(pwm, voltage, motorspeed):
    """ 
    Fit a model that maps pwm and input voltage to motor_speed. 

    Inputs:
        pwm (np array): the 16bit pwm value sent to the motor.
        voltage (np array): the supply voltage.
        motorspeed (np array): the speed of the motor. 

    """

    popt, _ = curve_fit(lambda xdata, a, b, c: battery_compensated_motorspeed_model(xdata[0], xdata[1], a, b, c),
                (pwm, voltage), motorspeed, p0=(1e-6, 0.1, 8_000), maxfev=10_000)

    print(f"motor_speed = {popt[0]}*(voltage + {popt[1]})*(pwm - {popt[2]})^(2/3)")
    return popt

def plot_speed_vs_pwm(cf_df, rc_df, use_3d=False, motor_col='m1'):
    """ 
    Plot motor speed vs PWM grouped by supply voltage. 

    Args:
        cf_df (pd.DataFrame): Synced Crazyflie dataframe (contains PWM + ID).
        rc_df (pd.DataFrame): Synced RC benchmark dataframe (contains speed + ID).
        use_3d (bool): If True, make a 3D plot of PWM vs Voltage vs Motor Speed.
        motor_col (str): Column in cf_df for PWM values (default 'm1').
    """

    # Join on index (since both are synced to same timestamps)
    df = pd.DataFrame({
        'PWM': cf_df[motor_col],
        'Voltage': cf_df['ID'],  # experiment ID = supply voltage
        'Motor Speed (rad/s)': rc_df['Motor Optical Speed (rad/s)']
    })

    if use_3d:
        fig = plt.figure(num="Rotor Speed vs PWM (3D)", figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        grouped = df.groupby('Voltage')
        for voltage, group in grouped:
            group = group.sort_values('PWM')
            ax.plot(group['PWM'], 
                    [voltage]*len(group), 
                    group['Motor Speed (rad/s)'],
                    '.',
                    label=f"{voltage:.1f} V")

        ax.set_xlabel("Motor PWM (16-bit)")
        ax.set_ylabel("Voltage (V)")
        ax.set_zlabel("Motor Speed (rad/s)")
        ax.set_title("Rotor Speed vs PWM vs Voltage")
        ax.legend(title="Voltage")

    else:
        fig, ax = plt.subplots(1, 1, num="Rotor Speed vs PWM", figsize=(8, 6))

        grouped = df.groupby('Voltage')
        for voltage, group in grouped:
            group = group.sort_values('PWM')
            ax.plot(group['PWM'], group['Motor Speed (rad/s)'],  '.',
                    label=f"{voltage:.1f} V", alpha=0.8)

        ax.set_xlabel("Motor PWM (16-bit)")
        ax.set_ylabel("Motor Speed (rad/s)")
        ax.set_title("Rotor Speed vs PWM vs Voltage")
        ax.legend(title="Voltage")
        ax.grid(True)
        fig.tight_layout()

    return fig, ax

def plot_speed_mapping_surface(ax, coeffs, pwm_range, voltage_range, 
                     color='red', alpha=0.6, resolution=50, 
                     use_3d=False, surface=True,
                     pwm_data=None, motorspeed_data=None, voltage_data=None):
    """
    Plot the motor_speed model on the provided axis.

    Parameters:
        ax : matplotlib axis (2D or 3D depending on use_3d)
        coeffs : [a, b, c] coefficients for the model (used in rpm_map)
        pwm_range : (min_pwm, max_pwm)
        voltage_range : (min_voltage, max_voltage)
        color : color of the surface/curve
        alpha : transparency
        resolution : number of grid points
        use_3d : if True, plot in 3D with Voltage axis
        surface : if True, plot surface in 3D; if False, plot scatter of fitted points
        pwm/motorspeed/voltage_data : actual data to compute R2 based off of. 
    """

    # Create grid
    pwm_vals = np.linspace(pwm_range[0], pwm_range[1], 2*resolution)
    voltage_vals = np.linspace(voltage_range[0], voltage_range[1], resolution)
    PWM, V = np.meshgrid(pwm_vals, voltage_vals)

    # Evaluate model (you must have rpm_map(PWM, V, a, b, c) defined elsewhere)
    motor_speed = battery_compensated_motorspeed_model(PWM, V, *coeffs)

    if use_3d:
        if surface:
            ax.plot_surface(PWM, V, motor_speed, color=color, alpha=alpha, edgecolor='none')
        else:
            ax.scatter(PWM.flatten(), V.flatten(), motor_speed.flatten(), 
                       c=color, alpha=alpha, s=10)
        ax.set_xlabel("PWM")
        ax.set_ylabel("Voltage (V)")
        ax.set_zlabel("Motor Speed (rad/s)")
    else:
        # Project onto 2D (Voltage collapsed/ignored → just PWM vs motor_speed at midpoint V)
        mid_v = np.mean(voltage_range)
        RPM_2D = battery_compensated_motorspeed_model(pwm_vals, mid_v, *coeffs)
        ax.plot(pwm_vals, RPM_2D, color=color, linewidth=2, label="Fitted Model")
        ax.set_xlabel("PWM")
        ax.set_ylabel("Motor Speed (rad/s)")
        ax.legend()

    # If data provided, compute R^2 and annotate
    if pwm_data is not None and voltage_data is not None and motorspeed_data is not None:
        motorspeed_pred = battery_compensated_motorspeed_model(pwm_data, voltage_data, *coeffs)
        r2 = r2_score(motorspeed_data, motorspeed_pred)

        # Add annotation (top right of 3D axis or 2D axis)
        if use_3d:
            ax.text2D(0.95, 0.95, f"$R^2 = {r2:.3f}$", 
                    transform=ax.transAxes, 
                    ha='right', va='top', fontsize=12, color=color)
        else:
            ax.text(0.05, 0.95, f"$R^2 = {r2:.3f}$", 
                    transform=ax.transAxes, 
                    ha='left', va='center', fontsize=12, color=color)

    return ax

def produce_mappings(cf_df, rc_df, num_motors=4):
    """ 
    Compute the mappings between PWM, motor_speed, and thrust. 
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
    voltage = rc_df['Voltage (V)'].values
    pwm = cf_df['m1'].values 

    ########## Motor speed to thrust model 

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
            label=f'V_s = {exp_id} V'
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

    ############ PWM + battery to motor speed mapping  
    use_3d = True
    motorspeed_mapping_coeffs = fit_battery_compensated_motorspeed_model(pwm, voltage, motorspeed)
    fig_speedvspwm, ax_speedvspwm = plot_speed_vs_pwm(cf_df, rc_df, use_3d=use_3d, motor_col='m1')
    plot_speed_mapping_surface(ax_speedvspwm, motorspeed_mapping_coeffs, pwm_range=(min(pwm), max(pwm)), voltage_range=(min(voltage),max(voltage)), use_3d=use_3d,
                                surface=True, resolution=10, alpha=0.35,
                                pwm_data=pwm, motorspeed_data=motorspeed, voltage_data=voltage)

    return k_eta

if __name__ == "__main__":
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    rc_dir = os.path.join(data_dir, 'thrust_stand_data_cfbl')
    cf_dir = os.path.join(data_dir, 'cf_data_cfbl')

    cf_csvs = [os.path.join(cf_dir, f) for f in os.listdir(cf_dir) if f.endswith('.csv')]
    rc_csvs = [os.path.join(rc_dir, f) for f in os.listdir(rc_dir) if f.endswith('.csv')]

    # Get combine dataframes of all the csv files for the Crazyflie and RCBenchmark. 
    cf_combined, rc_combined = combine_all_synced(cf_csvs, rc_csvs, sync_path=os.path.join(data_dir, 'sync_file.yaml'))

    produce_mappings(cf_combined, rc_combined)

    plt.show()