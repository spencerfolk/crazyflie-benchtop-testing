import logging
import time
import csv
import os
import serial
import re
from threading import Thread, Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

# ==========================
# Crazyflie + Anemometer CSV Logger
# ==========================

# Crazyflie URI
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7B3')

# Anemometer serial port
ANEMOMETER_PORT = "COM7"
ANEMOMETER_BAUD = 115200

# Regex to parse anemometer data
TOKEN_RE = re.compile(r'([A-Z]{1,2})\s*([+-]?\d+(?:\.\d+)?)')

def parse_trisonica_line(line):
    pairs = TOKEN_RE.findall(line)
    if not pairs:
        return None
    out = {}
    for k, v in pairs:
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out

# Output CSV
directory_path = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(directory_path, exist_ok=True)
filename = os.path.join(directory_path, "cf_whisker", f"log_{int(time.time())}.csv")

# Shared data container
latest_anemometer = {"U": None, "V": None}

# Stop signal
stop_event = Event()

# ==========================
# Crazyflie Logger
# ==========================
class CFLogger:
    def __init__(self, link_uri):
        self._cf = Crazyflie(rw_cache='./cache')
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.open_link(link_uri)
        self.is_connected = True

    def _connected(self, link_uri):
        print(f"Connected to Crazyflie: {link_uri}")
        lg = LogConfig(name="Wind", period_in_ms=50)  # 20 Hz
        lg.add_variable("windSensor.flowX", "int16_t")
        lg.add_variable("windSensor.flowY", "int16_t")
        try:
            self._cf.log.add_config(lg)
            lg.data_received_cb.add_callback(self._log_data)
            lg.start()
        except KeyError as e:
            print("Log config error:", e)

    def _log_data(self, timestamp, data, logconf):
        ts = time.time()
        Bx, By = data["windSensor.flowX"], data["windSensor.flowY"]
        U, V = latest_anemometer["U"], latest_anemometer["V"]

        writer.writerow({
            "Time": ts,
            "Bx": Bx,
            "By": By,
            "U": U if U is not None else "",
            "V": V if V is not None else ""
        })
        csv_file.flush()
        print(f"{ts:.3f}, Bx={Bx}, By={By}, U={U}, V={V}")

    def _disconnected(self, link_uri):
        print("Crazyflie disconnected")
        self.is_connected = False

    def close(self):
        self._cf.close_link()

# ==========================
# Anemometer Reader
# ==========================
def anemometer_thread():
    try:
        ser = serial.Serial(ANEMOMETER_PORT, ANEMOMETER_BAUD, timeout=1)
        print(f"Connected to anemometer on {ANEMOMETER_PORT}")
    except Exception as e:
        print("Could not open anemometer:", e)
        return

    while not stop_event.is_set():
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            data = parse_trisonica_line(line)
            if data and "U" in data and "V" in data:
                latest_anemometer["U"] = data["U"]
                latest_anemometer["V"] = data["V"]
        except Exception as e:
            if not stop_event.is_set():
                print("Anemometer error:", e)
            break
    ser.close()
    print("Anemometer thread stopped")

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    cflib.crtp.init_drivers()

    with open(filename, "w", newline="") as csv_file:
        fieldnames = ["Time", "Bx", "By", "U", "V"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Start anemometer thread
        t = Thread(target=anemometer_thread, daemon=True)
        t.start()

        # Start Crazyflie logger
        le = CFLogger(uri)

        try:
            while le.is_connected and not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nCtrl-C pressed, stopping...")
            stop_event.set()
            le.close()
            t.join()
        print("Exit complete.")
