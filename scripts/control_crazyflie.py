from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

import time
import os
import numpy as np

"""
Credit: @PratikKunapuli for a lot of this code!
"""

def test_cf_connection(uri="radio://0/80/2M/E7E7E7E7E7"):
    """ 
    Test the connection to the Crazyflie using the CFLib. 

    Inputs:
        uri (str): the URI of the Crazyflie. 
    """ 

    init_drivers()
    cf = Crazyflie()
    cf.open_link(uri)
    print("Connected to Crazyflie!")
    cf.close_link()

    return 

def send_cf_setpoint(cf, duration, thrust, roll, pitch, yaw):
    """
    Send a commander-level setpoint to the Crazyflie. The controller runs in the loop. 

    Inputs:
        duration (float): How long to repeat this command for
        thrust (int): Thrust value to send to the Crazyflie (16 bits) (clipped to 0xFFFF)
        roll, pitch, yaw (floats): Angles or Rate to send depending on firmware settings
    """
    # Set points must be sent continuously to the Crazyflie, if not it will think that connection is lost
    end_time = time.time() + duration
    while time.time() < end_time:
        cf.commander.send_setpoint(roll, pitch, yaw, int(thrust))
        time.sleep(0.01) # 100 Hz

    return

def send_motor_pwm(cf, pwm_values, duration):
    """
    Send motor pwm commands to each motor. This is run in open loop, the controllers are not running for stabilization!

    Inputs:
        pwm_values (list): list of 4 ints [m1, m2, m3, m4] (0-65535)
        duration (float): seconds to hold this PWM
    """
    start = time.time()
    while time.time() - start < duration:
        cf.param.set_value('motorPowerSet.m1', str(pwm_values[0]))
        cf.param.set_value('motorPowerSet.m2', str(pwm_values[1]))
        cf.param.set_value('motorPowerSet.m3', str(pwm_values[2]))
        cf.param.set_value('motorPowerSet.m4', str(pwm_values[3]))
        time.sleep(0.02)  # 50 Hz update

    return

def cf_stop(cf):
    """ 
    Send a stop and disarm command to all the motors. 

    Inputs:
        cf (Crazyflie obj): the crazyflie
    """

    # Stop all motors
    for i in range(30):
        cf.param.set_value('motorPowerSet.m1', '0')
        cf.param.set_value('motorPowerSet.m2', '0')
        cf.param.set_value('motorPowerSet.m3', '0')
        cf.param.set_value('motorPowerSet.m4', '0')
        cf.platform.send_arming_request(False)
        time.sleep(0.1)

    return 

def ramp_motors(cf, start_pwm, end_pwm, step, hold_time, motor_idxs=None):
    """
    Ramp the motor PWM values and log the responses.
    
    Inputs:
    Outputs:
        log_data (dict): {'timestep': np.array, 'm1': np.array, 'm2': np.array, 'm3': np.array, 'm4': np.array}
    """

    # Storage for logged data
    log_data = {
        'timestep': [],
        'm1': [],
        'm2': [],
        'm3': [],
        'm4': []
    }

    motor_pwms = [0, 0, 0, 0]
    pwm = start_pwm

    # Define logging configuration
    lg = LogConfig(name='MotorLogging', period_in_ms=100)  # log every 100 ms
    lg.add_variable('motor.m1', 'uint16_t')
    lg.add_variable('motor.m2', 'uint16_t')
    lg.add_variable('motor.m3', 'uint16_t')
    lg.add_variable('motor.m4', 'uint16_t')

    def _log_motor_data(timestamp, data, logconf):
        """Callback to save motor data into local dict"""
        
        print("time: ", timestamp, "m1: ", data["motor.m1"], "m2: ", data["motor.m2"], "m3: ", data["motor.m3"], "m4: ", data["motor.m4"])
        log_data['timestep'].append(timestamp)
        log_data['m1'].append(data['motor.m1'])
        log_data['m2'].append(data['motor.m2'])
        log_data['m3'].append(data['motor.m3'])
        log_data['m4'].append(data['motor.m4'])

    # Start logging
    cf.log.add_config(lg)
    lg.data_received_cb.add_callback(_log_motor_data)
    lg.start()

    try:
        # Ramp motors
        while pwm <= end_pwm:
            for idx in motor_idxs:
                motor_pwms[idx] = pwm
            send_motor_pwm(cf, motor_pwms, hold_time)
            pwm += step

    finally:

        lg.stop()
        cf_stop(cf)

    # Convert to numpy arrays
    for k in log_data:
        log_data[k] = np.array(log_data[k])

    return log_data

def set_cf_params(scf):
    """ 
    Set the CF params prior to starting an experiment. This ensures that things like the tumble check are disabled. 

    Inputs:
        scf (SyncCrazyflie obj): the sync Crazyflie. 
    """

    print("Setting CF params...")

    cf = scf.cf
    cf.commander.set_client_xmode(False)

    cf.param.set_value('stabilizer.controller', '0') # PID Controller - 1, Mellinger Controller - 2
    cf.param.set_value('supervisor.tmblChckEn', '0') # Disable tumble check
    cf.param.set_value('flightmode.stabModePitch', '0')  # Rate mode
    cf.param.set_value('flightmode.stabModeRoll', '0')  # Rate mode
    cf.param.set_value('motorPowerSet.m1', '0')     # Turn off motor (just to activate motor)
    cf.param.set_value('motorPowerSet.enable', '1')  # Enable motor power set control (open loop commands)

    return 