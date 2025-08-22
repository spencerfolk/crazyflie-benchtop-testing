from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

from scripts.control_crazyflie import set_cf_params, ramp_motors, cf_stop

import pandas as pd
import os
import time

# uri = "usb://0"
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7B1')

current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
output_path = os.path.join(os.path.dirname(__file__), 'data', 'cf_data', 'rampup_data_'+current_datetime+'.csv')

print("Initializing drivers...")
init_drivers()
print("Drivers initialized!")

try:
    print("Running...")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

        set_cf_params(scf)

        print("Ramping motors...")
        log_dict = ramp_motors(scf.cf, start_pwm=10000, end_pwm=50000, step=5000, hold_time=1.0, motor_idxs=[0, 1, 2, 3])

        print("Closing link...")
        scf.cf.close_link()

        print("Saving data to csv file...")
        df = pd.DataFrame(log_dict)
        df.to_csv(output_path, index=False)

except KeyboardInterrupt:
    print("Keyboard interrupt detected, stopping motors and closing...")
    cf_stop(scf.cf)
    scf.cf.close_link()