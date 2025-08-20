from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

from scripts.control_crazyflie import set_cf_params, set_up_logging, collective_ramp_motors, cf_stop

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
        collective_ramp_motors(scf.cf, start_pwm=15000, end_pwm=30000, step=2000, hold_time=5.0)

        print("Closing link...")
        scf.cf.close_link()

except KeyboardInterrupt:
    print("Keyboard interrupt detected, stopping motors and closing...")
    cf_stop(scf.cf)
    scf.cf.close_link()