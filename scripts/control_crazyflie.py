from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

import time
import os

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

def _log_motor_data(timestamp, data, logconf):
    """ 
    Callback function for logging the motor data when it is transmitted. 

    Inputs:
        timestamp (int): the timestamp
        data (dict): dictionary containing the motor data. 
        logconf: logging configuration dict. 
    """

    print("time: ", timestamp, "m1: ", data["motor.m1"], "m2: ", data["motor.m2"], "m3: ", data["motor.m3"], "m4: ", data["motor.m4"])

    return 

def set_up_logging(scf):
    """ 
    Set up the logging configuration for the Crazyflie. 

    Inputs:
        scf (SyncCrazyflie obj): the sync Crazyflie object. 
    """

    print("Setting up logging...")
    # Set up logging
    _lg_motors = LogConfig(name='Motors', period_in_ms=10)
    _lg_motors.add_variable('motor.m1', 'uint16_t')
    _lg_motors.add_variable('motor.m2', 'uint16_t')
    _lg_motors.add_variable('motor.m3', 'uint16_t')
    _lg_motors.add_variable('motor.m4', 'uint16_t')

    # Add the callback to the log config
    scf.cf.log.add_config(_lg_motors)
    _lg_motors.data_received_cb.add_callback(_log_motor_data)

    # Start logging
    _lg_motors.start()

    return 

def ramp_motors(cf, start_pwm=10000, end_pwm=25000, step=1000, hold_time=1.0, motor_idxs=[0, 1, 2, 3]):
    """
    Slowly ramp all motors from start_pwm to end_pwm, hold each step for hold_time.

    Inputs:
        cf (Crazyflie obj): the crazyflie to control.
        start_pwm (int): the starting pwm value for all motors. 
        end_pwm (int): the end pwm value for all motors. 
        step (int): the step change in pwm. 
        hold_time (float): how long to hold at each stepped pwm value. 
        motor_idxs (list): choose which motors to ramp up. 

    """
    motor_pwms = [0, 0, 0, 0]
    pwm = start_pwm
    while pwm <= end_pwm:
        for idx in motor_idxs:
            motor_pwms[idx] = pwm
        send_motor_pwm(cf, motor_pwms, hold_time)
        pwm += step
    # Return to zero at the end
    cf_stop(cf)

    return

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

if __name__=="__main__":

    # uri = "usb://0"
    uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7B3')

    # test_cf_connection(uri)
    # test_tyto_connection(COM_PORT="COM3", baud_rate=115200)

    print("Initializing drivers...")
    init_drivers()
    print("Drivers initialized!")

    try:
        print("Running...")
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

            set_cf_params(scf)
            set_up_logging(scf)

            print("Ramping motors...")
            ramp_motors(scf.cf, start_pwm=15000, end_pwm=30000, step=2000, hold_time=1.0)

            print("Closing link...")
            scf.cf.close_link()

    except KeyboardInterrupt:
        print("Keyboard interrupt detected, stopping motors and closing...")
        cf_stop(scf.cf)
        scf.cf.close_link()