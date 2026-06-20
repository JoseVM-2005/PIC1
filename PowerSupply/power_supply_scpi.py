# ============================================================
# SCPI COMMAND REFERENCE — MicroPython PSU
# ============================================================
#
# *IDN? 
#     Returns instrument identification string.
#     Format: "LIP,MicroPythonPSU,0001,1.0"
#
# *RST
#     Performs a full reset:
#       - Disables output
#       - Reconfigures GPIO
#       - Sets DAC/ADC to safe defaults
#       - Enters standby mode
#
# SYST:DEBUG ON | OFF
#     Enables or disables internal debug printing.
#
# ------------------------------------------------------------
# OUTPUT CONTROL
# ------------------------------------------------------------
# OUTP ON
#     Turns the power supply output ON.
#     Executes the full power‑up routine:
#       - ADC config
#       - Enable DAC
#       - Switch to big feedback loop
#       - Set output to 0 V
#
# OUTP OFF
#     Turns the power supply output OFF.
#     Executes the power‑down routine:
#       - Forces output to 0 V
#       - Switches to small feedback loop
#       - Powers down DAC
#
# OUTP?
#     (Not implemented — returns "UNKNOWN")
#
# ------------------------------------------------------------
# SOURCE SETTINGS
# ------------------------------------------------------------
# SOUR:VOLT <value>
#     Sets the output voltage setpoint (in volts).
#     Automatically selects DAC gain (1× or 2×).
#     Range: 0–5 V (values above are capped)
#
# SOUR:CURR <value>
#     Sets the output current limit setpoint (in mA).
#     Automatically selects DAC gain.
#     Range: 0–250 mA (values above are capped)
#
# ------------------------------------------------------------
# PROTECTION LIMITS
# ------------------------------------------------------------
# SOUR:VOLT:PROT <value>
#     Sets the over‑voltage protection threshold (V).
#     Uses a separate DAC channel (B).
#
# SOUR:CURR:PROT <value>
#     Sets the over‑current protection threshold (mA).
#     Uses DAC channel (B).
#
# ------------------------------------------------------------
# MODE SELECTION
# ------------------------------------------------------------
# FUNC VOLT
#     Sets measurement mode to voltage.
#     Internally sets GPIO bits for FVMI mode.
#
# FUNC CURR
#     Sets measurement mode to current.
#     Internally sets GPIO bits for FIMV mode.
#
# ------------------------------------------------------------
# MEASUREMENTS
# ------------------------------------------------------------
# MEAS:VOLT?
#     Returns the measured output voltage (V).
#     Applies calibration table.
#
# MEAS:CURR?
#     Returns the measured output current (mA).
#     Applies calibration table.
#
####### READ ME - SCPI COMMANDS #######

# READ?
#     Returns both voltage and current in CSV format:
#         "<voltage>,<current>"
#     Example: "1.234567,12.345678"
#
# ------------------------------------------------------------
# STATUS / DIAGNOSTICS
# ------------------------------------------------------------
# STAT?
#     Prints a detailed status dump to stdout:
#       - GPIO configuration
#       - ADC raw bytes
#       - DAC raw bytes
#       - 12‑bit decoded DAC values
#
# ------------------------------------------------------------
# ERROR HANDLING
# ------------------------------------------------------------
# ERR:UNKNOWN COMMAND
#     Returned when the header does not match any SCPI command.
#
# ERR:<message>
#     Returned when an exception occurs during command execution.
#
# ============================================================

import sys
import select
from utime import sleep, ticks_ms, ticks_diff
from machine import Pin, I2C, Timer

# ============================
# Initial pi Setup
# ============================

led = Pin(25, Pin.OUT)
LED_state = False
tim = Timer()
DEBUG = False
# check if all's ready with the pi
def tick(timer):
    global LED_state
    LED_state = not LED_state
    led.value(LED_state)

tim.init(freq=2, mode=Timer.PERIODIC, callback=tick)

# SDA is connected to P6 and SCL to P7
i2c = I2C(0, scl=Pin(5), sda=Pin(4), freq=400000)
#dprint(i2c)

#Make sure all's good with the i2c connections
#print("I2C Address     : ")
#for i in i2c.scan():
#    dprint(bin(i), "(", i, ")", "\n")

# Addresses (last two bits of all addresses are determined by pins on the board):
addr_gpio  = 0b0111000
addr_dac   = 0b0001100
addr_eprom = 0b1010000
addr_adc   = 0b1101000


# ===========================
# Calibration Tables
# ===========================

adc_v_table = [
   #(val_meas , DMM)
    (0.012375, -0.0197506),
    (0.013250, -0.0118514),
    (0.015600, 0.003073),
    (0.103247, 0.102980),
    (0.201971, 0.203079),
    (0.401784, 0.403363),
    (0.502110, 0.503442),
    (0.602436, 0.603519),
    (0.702451, 0.703574),
    (0.802853, 0.803664),
    (1.003024, 1.003629),
    (1.103305, 1.103949),
    (1.303660, 1.303995),
    (1.403523, 1.404048),
    (1.503798, 1.503969),
    (1.603437, 1.603983),
    (1.703493, 1.704031),
    (1.800199, 1.804349),
    (1.893444, 1.904385),
    (2.087285, 2.104297),
    (2.187501, 2.204305),
    (2.287196, 2.304125),
    (2.487866, 2.504428),
    (2.688606, 2.704353),
    (2.889661, 2.904365),
    (3.090068, 3.104271),
    (3.290831, 3.304544),
    (3.391213, 3.404676),
    (3.590168, 3.604680),
    (3.788183, 3.804532),
    (3.887398, 3.904541),
    (4.086991, 4.104801),
    (4.187038, 4.204648),
    (4.287086, 4.304618),
    (4.387343, 4.404597),
    (4.588125, 4.604991),
    (4.788140, 4.805093),
    (4.988295, 5.005044),
    (5.089686, 5.105001),
    (5.190759, 5.204987),
    (5.393118, 5.405078),
    (5.494403, 5.505032),
    (5.693229, 5.704964),
    (5.792396, 5.804918),
    (5.991667, 6.004715),
    (5.991563, 6.004722),
    (5.792500, 5.804932),
    (5.891875, 5.904788),
    (5.991563, 6.004735),
    (6.091600, 6.105000),
    (6.161616, 6.174900),
    (7.0,7.0),

]

 #(val_meas , DMM)
adc_i_table = [(2.34375, -2.3531446000000003), (3.96875, 3.6682392), (7.875, 4.9464884), (12.0, 8.9791292), (15.90625, 12.774044), (19.78125, 16.560782), (23.937502, 20.595282), (27.843752000000002, 24.403056), (31.96875, 28.438806), (35.875, 32.236343999999995), (39.78125, 36.047838), (43.906252, 40.081402000000004), (47.812504, 43.875099999999996), (51.71875200000001, 47.668330000000005), (55.84375, 51.709902), (59.75, 55.501639999999995), (63.90625200000001, 59.552328), (67.800004, 63.353506), (71.693758, 67.164222), (75.84375, 71.20351), (79.75, 74.99645), (83.88125199999999, 79.029786), (87.78125, 82.82132800000001), (91.687512, 86.636476), (95.843768, 90.67121), (99.718768, 94.467028), (103.62501, 98.25827000000001), (107.78125, 102.30942), (111.681252, 106.11248), (115.8125, 110.13050000000001), (119.71875, 113.93029999999999), (123.625, 117.73964000000001), (127.81251, 121.91923999999999), (131.71875, 125.70804000000001), (135.84375, 129.74928), (139.75, 133.56045999999998), (143.65625, 137.35548), (147.8125, 141.39607999999998), (151.71875, 145.18076000000002), (155.59375, 149.0019), (159.75002, 153.03648), (163.65626, 156.83972), (167.78126, 160.87364), (171.68752, 164.68264), (175.59376, 168.48193999999998), (179.75002, 172.51664), (183.637516, 176.3109), (187.787508, 180.3562), (191.68752, 184.15348), (195.59376, 187.96638000000002), (199.75, 192.00846), (203.656252, 195.80212), (207.53126000000003, 199.59592), (211.6875, 203.63518), (215.59374399999996, 207.43366), (219.73751280000002, 211.47298), (223.63750399999998, 215.27422), (227.531252, 219.05922), (231.68752, 223.10512), (235.59376000000003, 226.90025999999997), (239.75002, 230.94088000000002), (243.637516, 234.72152), (247.543764, 238.52812), (251.687532, 242.56892000000002), (255.59378, 246.37399999999997), (259.493784, 250.15805999999998), (263.62504, 254.20207999999997), (267.53128, 258.0014), (271.6875, 262.02712), (275.59376, 265.82626), (279.481268, 269.6238), (283.625, 273.67222000000004), (287.53124, 277.46096), (291.68752, 281.49771999999996), (295.581268, 285.29584), (299.475016, 289.0956), (303.62504, 293.12966), (307.53128, 296.91558)]


#   (DMM   , Val)
dac_v_table_g1 = [(-0.51668706, 0), (-0.42471655999999997, 0.01), (-0.10101484000000001, 0.02), (0.031354352, 0.03), (0.041083626, 0.04), (0.050773387999999996, 0.05), (0.061754788000000005, 0.060000000000000005), (0.071493474, 0.07), (0.08124954799999999, 0.08), (0.091010442, 0.09), (0.10075788000000001, 0.09999999999999999), (0.11170268000000001, 0.10999999999999999), (0.21156722, 0.21), (0.31021874, 0.31), (0.41007563999999996, 0.41000000000000003), (0.50994184, 0.51), (0.60986576, 0.61), (0.7097313000000001, 0.71), (0.80960632, 0.8099999999999999), (0.9094984, 0.9099999999999999), (1.0094281999999999, 1.01), (1.1093157999999999, 1.11), (1.2091336, 1.2100000000000002), (1.3090014, 1.3100000000000003), (1.4088764, 1.4100000000000004), (1.5075699999999999, 1.5100000000000005), (1.6074486000000001, 1.6100000000000005), (1.7073262, 1.7100000000000006), (1.8071397999999999, 1.8100000000000007), (1.9070258, 1.9100000000000008), (2.0068962, 2.0100000000000007), (2.1067516, 2.1100000000000008), (2.2066578, 2.210000000000001), (2.3065336000000003, 2.310000000000001), (2.4064008, 2.410000000000001), (2.5062882, 2.510000000000001), (2.6061348, 2.610000000000001), (2.706007, 2.7100000000000013), (2.8046888, 2.8100000000000014), (2.9045858, 2.9100000000000015), (3.0044394, 3.0100000000000016), (3.1043038, 3.1100000000000017), (3.2041595999999997, 3.2100000000000017), (3.3040602, 3.310000000000002), (3.4038890000000004, 3.410000000000002), (3.5037846000000004, 3.510000000000002), (3.6036349999999997, 3.610000000000002), (3.7035188, 3.710000000000002), (3.803362, 3.8100000000000023), (3.9032585999999996, 3.9100000000000024), (4.0019194, 4.0100000000000025), (4.1017952, 4.110000000000002), (4.2016838, 4.210000000000002), (4.3016, 4.310000000000001), (4.4015064, 4.410000000000001), (4.5014098, 4.510000000000001), (4.6012802, 4.61), (4.7011448, 4.71), (4.801029, 4.81), (4.9009362, 4.909999999999999), (4.9898254, 5.009999999999999),]





#   (DMM   , Val)
dac_v_table_g2 = [(-0.51744134, 0), (-0.41286652, 0.01), (-0.07626179599999999, 0.02), (0.031941566, 0.03), (0.041708089999999996, 0.04), (0.051485392, 0.05), (0.061205256, 0.060000000000000005), (0.070942622, 0.07), (0.08068204799999999, 0.08), (0.090415178, 0.09), (0.10013916, 0.09999999999999999), (0.1123456, 0.10999999999999999), (0.21220424, 0.21), (0.30964706, 0.31), (0.40950586, 0.41000000000000003), (0.50935612, 0.51), (0.6092532800000001, 0.61), (0.70913278, 0.71), (0.8090001000000001, 0.8099999999999999), (0.9088768, 0.9099999999999999), (1.0087796, 1.01), (1.1086066, 1.11), (1.208494, 1.2100000000000002), (1.3083816000000001, 1.3100000000000003), (1.4082776, 1.4100000000000004), (1.5081602, 1.5100000000000005), (1.6080507999999998, 1.6100000000000005), (1.707913, 1.7100000000000006), (1.8077956, 1.8100000000000007), (1.9076978, 1.9100000000000008), (2.0075914, 2.0100000000000007), (2.1074626, 2.1100000000000008), (2.2073460000000003, 2.210000000000001), (2.307176, 2.310000000000001), (2.4070755999999998, 2.410000000000001), (2.5069326, 2.510000000000001), (2.606841, 2.610000000000001), (2.7067064, 2.7100000000000013), (2.8042664, 2.8100000000000014), (2.9040694, 2.9100000000000015), (3.003958, 3.0100000000000016), (3.1038464, 3.1100000000000017), (3.203731, 3.2100000000000017), (3.3035808, 3.310000000000002), (3.4034614000000003, 3.410000000000002), (3.5033608000000003, 3.510000000000002), (3.6032318000000005, 3.610000000000002), (3.7031162, 3.710000000000002), (3.8030118, 3.8100000000000023), (3.9028628, 3.9100000000000024), (4.0027479999999995, 4.0100000000000025), (4.1025968, 4.110000000000002), (4.202464, 4.210000000000002), (4.3023404, 4.310000000000001), (4.4022714, 4.410000000000001), (4.5021552, 4.510000000000001), (4.60197, 4.61), (4.701886, 4.71), (4.8017444, 4.81), (4.901642000000001, 4.909999999999999), (5.0014874, 5.009999999999999), (5.101432, 5.1099999999999985), (5.2012431999999995, 5.209999999999998), (5.2987034, 5.309999999999998), (5.3985792, 5.4099999999999975), (5.4984712, 5.509999999999997), (5.5983768, 5.609999999999997), (5.6989394, 5.709999999999996), (5.7993749999999995, 5.809999999999996), (5.899311, 5.909999999999996), (5.9992294, 6.009999999999995), (6.0990802, 6.109999999999995), (6.1990102, 6.209999999999995),]



dac_i_table = [
    (-10.0,-10.0),
    (0.0, 0.0),
    (50.0, 50.0),
    (100.0, 100.0),
    (150.0, 150.0),
    (200.0, 200.0),
    (250.0, 250.0),
    (300.0, 300.0),
]

dac_limit_v_table = [
    (0.0, 0.0),
    (1.0, 1.0),
    (2.0, 2.0),
    (3.0, 3.0),
    (4.0, 4.0),
    (5.0, 5.0),
    (6.0, 6.0),
    (7.0, 7.0),
]

dac_limit_i_table =  [
    (0.0, 0.0),
    (50.0, 50.0),
    (100.0, 100.0),
    (150.0, 150.0),
    (200.0, 200.0),
    (250.0, 250.0),
    (300.0, 300.0),
    (350.0, 350.0),
]



def interpolate(x, table):
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i+1]

        if x0 <= x <= x1:
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    return -90  # out of range



# ============================================
# Function definition for setup and operation
# ============================================


###---------GPIO Stuff:----------###

# GPIO Pins:
## P7 -> Clamp read status (clamped or not)
## P6 -> Feedback 0 = smll (standby), 1 = bg (when turning on: first set to ON, then change to big loop, and vice‑versa)
## P5 -> LoadDAC (=0 for now)
## P4 -> DACGain
## P3 -> FIMV
## P2 -> FVMI
## P1 -> ON
## P0 -> Diswrite (=1 for now)

# command bytes (register):
input_port          = 0b00000000
output_port         = 0b00000001
polarity_inversion  = 0b00000010
configuration       = 0b00000011

def gpio_status():
    i2c.writeto(addr_gpio, bytes([input_port]))
    real_value = i2c.readfrom(addr_gpio, 1)[0]

    i2c.writeto(addr_gpio, bytes([output_port]))
    expected_value = i2c.readfrom(addr_gpio, 1)[0]
    
    i2c.writeto(addr_gpio, bytes([configuration]))
    gpio_inout = i2c.readfrom(addr_gpio, 1)[0]

    a = bin(expected_value)[2:]
    a = "0" * (8 - len(a)) + a

    b = bin(real_value)[2:]
    b = "0" * (8 - len(b)) + b
    
    c = bin(gpio_inout)[2:]
    c = "0" * (8 - len(c)) + c

    dprint("GPIO config   state:", c)
    dprint("GPIO expected state:", a)
    dprint("GPIO current  state:", b)


def config_gpio():
    # all are outputs except for Clamp (P7)
    i2c.writeto(addr_gpio, bytes([configuration, 0b10000000]))
    dprint("Undergoing Configuration...")


def force_measure_mode(mode):
    # 1. Read current OUTPUT register (register 1)
    i2c.writeto(addr_gpio, bytes([output_port]))
    current = i2c.readfrom(addr_gpio, 1)[0]

    if mode == "v":
        # FVMI: P2 = 1, P3 = 0
        new = current | (1 << 2)
        new = new & ~(1 << 3)
        dprint("Changing to FVMI")
    elif mode == "i":
        # FIMV: P3 = 1, P2 = 0
        new = current | (1 << 3)
        new = new & ~(1 << 2)
        dprint("Changing to FIMV")
    else:
        dprint("Please enter a valid operating mode")
        return None

    # 3. Write updated byte back to output register
    i2c.writeto(addr_gpio, bytes([output_port, new]))
    gpio_status()

    return mode



###---------DAC Stuff:----------###


# Always operating with both dacA and dacB until told which to use meaning commands end with 1001
dac_power       = 0b01001001
write_n_update_A  = 0b00110001
write_n_update_B  = 0b00111000
write_to_buffer = 0b00011001


def power_dac(state):
    # Check datasheet for exact bits; here we assume:
    # state = True  -> normal operation (both on)
    # state = False -> power-down both channels
    if state:
        mode = 0b00000000  # both DACs normal mode
    else:
        i2c.writeto(addr_dac, bytes([write_n_update_A, 0x00, 0x00]))
        mode = 0b11111111  # both DACs powered down (example)
    i2c.writeto(addr_dac, bytes([dac_power, 0x00, mode]))  # middle byte is not relevant
    #set limit to max
    limit_low = 0b1111<<4
    limit_high = 0xFF
    i2c.writeto(addr_dac, bytes([write_n_update_B, limit_high , limit_low]))


def dac_2xgain(state):
    pin = 4  # GPIO bit controlling DAC gain

    # 1. Read current OUTPUT register (register 1)
    i2c.writeto(addr_gpio, bytes([output_port]))
    current = i2c.readfrom(addr_gpio, 1)[0]

    # 2. Modify only bit 4
    if state:
        new = current | (1 << pin)      # set bit
    else:
        new = current & ~(1 << pin)     # clear bit

    # 3. Write updated byte back to output register
    i2c.writeto(addr_gpio, bytes([output_port, new]))


def set_voltage(v):
    v_ref = 2.5
    gain = 1

    # 1. Select gain
    if v < 0:
        v = 0
        dprint("Limit voltage must be positive")
    elif v > 6.3:
        v = 6.3
        dprint("Limit voltage too high - capped at 6V (MAX)")
    else:
        dprint(f"V = {v}")

    v_g = interpolate(v, dac_v_table_g1)
    
    if v_g >= v_ref: #then v is outside of the domain of gain 1 -> gain 2
        v_g = interpolate(v, dac_v_table_g2)
        gain = 2
        dprint(f"V = {v_g} - DAC Gain at 2")

    # 2. Set gain pin
    dac_2xgain(gain == 2)

    # 3. Compute DAC code (12 bits)
    D = int((v_g / (v_ref * gain)) * 4096/2)
    if D > 4095:
        D = 4095
        dprint("Voltage at capped at dac max")
    dprint(D)
    # 4. Split into bytes
    high = (D >> 4) & 0xFF          # D11..D4
    low  = (D & 0xF) << 4           # D3..D0 + 0000
    # Send the 3 bytes
    i2c.writeto(addr_dac, bytes([write_n_update_A, high, low]))

    

def set_current(i): #where i is the desired current in mA - max of 250mA
    i_ref = 2.5*50  #make sure this is correct
    gain = 1
    
       
    # 1. Select gain
    if i < 0:
        i = 0
        dprint("Input current must be positive")

    elif i > 2*i_ref:
        i = 250
        dprint("Input current too high - capped at 250mA?")
    else:
        dprint(i)

        
    if i <= i_ref:
        gain = 2

    i = interpolate(i, dac_i_table)
    # 2. Set gain pin
    dac_2xgain(gain == 2)

    # 3. Compute DAC code (12 bits)
    D = int((i / (i_ref * gain)) * 4096)
    if D > 4095:
        D = 4095

    # 4. Split into bytes
    high = (D >> 4) & 0xFF          # D11..D4
    low  = (D & 0xF) << 4           # D3..D0 + 0000

    # Send the 3 bytes
    i2c.writeto(addr_dac, bytes([write_n_update_A, high, low]))



def set_limit_v(v):
    v_ref = 2.5
    gain = 1
    mode_multiplier = 1
    gain = 1
    
    # 1. Select gain
    if v < 0:
        v = 0
        dprint("Limit voltage must be positive")
    elif v > 6:
        v = 6
        dprint("Limit voltage too high - capped at 5V (MAX)")
    else:
        dprint(f"V = {v} - DAC gain at 1")

    v = interpolate(v, dac_limit_v_table)

    if v >= v_ref:
        gain = 2
        dprint(f"V = {v} - DAC Gain at 2")

    # 2. Set gain pin
    dac_2xgain(gain == 2)
    

    # 3. Compute DAC code (12 bits)
    D = int((v / (v_ref * gain)) * 4096/2)
    if D > 4095:
        D = 4095
        dprint("Voltage capped at dac max")
    dprint(f"Voltage limit set to {v}V (D = {D})")
    # 4. Split into bytes
    limit_high = (D >> 4) & 0xFF          # D11..D4
    limit_low  = (D & 0xF) << 4           # D3..D0 + 0000
    # Send the 3 bytes
    i2c.writeto(addr_dac, bytes([write_n_update_B, limit_high , limit_low]))
    
    
    
def set_limit_i(i):
    v_ref = 2.5
    gain = 1
    mode_multiplier = 50
    
    # 1. Select gain
    if i < 0:
        i = 0
        dprint("Limit Current must be positive")
    elif i <= v_ref*50:
        gain = 1
        dprint(f"I = {i} - DAC gain at 1")
    elif i > 250:
        i = 250
        dprint("Limit current too high - capped at 250mA (MAX)")
    else:
        gain = 2
        dprint(f"I = {i} - DAC Gain at 2")

    # 2. Set gain pin
    dac_2xgain(gain == 2)
    
    v = interpolate(i, dac_limit_i_table)
    # 3. Compute DAC code (12 bits)
    D = int((i / (v_ref * gain*mode_multiplier)) * 4096/2)
    if D > 4095:
        D = 4095
        dprint("Current capped at dac max")
    dprint(f"Current limit set to {i}mA (D = {D})")
    # 4. Split into bytes
    limit_high = (D >> 4) & 0xFF          # D11..D4
    limit_low  = (D & 0xF) << 4           # D3..D0 + 0000
    # Send the 3 bytes
    i2c.writeto(addr_dac, bytes([write_n_update_B, limit_high , limit_low]))



def read_dac():
    v_ref = 2.5
    mode_multiplier = 1
    mode = None

    # Read gain from OUTPUT register (P4)
    i2c.writeto(addr_gpio, bytes([output_port]))
    gpio_out = i2c.readfrom(addr_gpio, 1)[0]  #0b76543210
    
    if gpio_out & 0x4 :
        mode_multiplier = 1
        mode = "Voltage (V)"
    elif gpio_out & 0x8:
        mode_multiplier = 1000/20  #conversion to mA and 20x multiplier for current
        mode = "Current (mA)"
    else:
        dprint("Error reading the DAC")
        
        
    gain = (gpio_out >> 4) & 0b1

    sleep(0.1)

    # Readback sequence: write command, then read
    i2c.writeto(addr_dac, bytes([write_to_buffer]))
    data = i2c.readfrom(addr_dac, 8)

    # Extract registers
    A_high = data[0]
    A_low  = data[1]
    B_high = data[6]
    B_low  = data[7]

    # Convert to 12-bit values
    D_A = (A_high << 4) | (A_low >> 4)
    D_B = (B_high << 4) | (B_low >> 4)

    dprint("Raw register A: ", D_A)
    dprint("Raw register B: ", D_B)

    val_out_A = v_ref * (D_A / 4096) * (2**gain) * mode_multiplier
    val_out_B = v_ref * (D_B / 4096) * (2**gain) * mode_multiplier

    dprint(f"{mode} in register A: ", val_out_A)
    dprint(f"{mode} in register B: ", val_out_B)

    return [val_out_A, val_out_B]




###------------ADC Stuff--------------###
# configuration bits:
# not-ready (one-shot trigger or read status) | Channel1 | Channel2 | Not-oneshot/Continuous | Rate(resolution)1 | Rate(resolution)2 | Gain1 | Gain2
# ADC Default:          0b10010000 (aka ready, channel 0, continuous, 240SPS/12bits, gain 1x)
# Operational Default:  0b10011000 (same except 16bits)

def adc_config():
    i2c.writeto(0x00, bytes([0b00000110]))  # general call reset - recommended when powered on
    i2c.writeto(addr_adc, bytes([0b10011000]))
    sleep(0.2)


def read_adc_v():
    v_ref =  2.048
    pga = 1  # gain 1 by default
    mode_multiplier = 5
    mode = None
    
    # Read gain from OUTPUT register (P4)
    i2c.writeto(addr_gpio, bytes([output_port]))
    gpio_out = i2c.readfrom(addr_gpio, 1)[0]  #0b76543210

    if gpio_out & 0x4 :
        mode = "Voltage (V)"
    elif gpio_out & 0x8:
        dprint("Power Supply in Current mode")
        mode = "Current (mA)"
    else:
        dprint("Error reading the ADC")
        return -91 
    
    i2c.writeto(addr_adc, bytes([0b10011000]))
    
    sleep(0.2)
    
    raw = i2c.readfrom(addr_adc, 3)

    data_high = raw[0]
    data_low  = raw[1]
    config    = raw[2]

    # Extract 15-bit magnitude
    magnitude = ((data_high & 0x7F) << 8) | data_low

    # Determine sign
    if data_high & 0x80:
        # Negative number in two's complement
        signed = magnitude - 0x8000
    else:
        signed = magnitude

    # Convert to voltage
    val_meas = signed * 62.5e-6 / pga * mode_multiplier


    # Print results
    dprint(f"ADC val_meas:", val_meas)
    dprint("Ready =", ((config & 0x80) == 0))

    # Overflow detection
    if signed == -32768 or signed == 32767:
        dprint("Warning: Value outside ADC measuring limit")
    elif val_meas > 6.8:
        dprint("Warning: according to adc, input resulted in DAC maxed")
        return -92

    if mode in ("Voltage (V)","Current (mA)") :
        return interpolate(val_meas, adc_v_table)
    else:
        dprint("Warning: according to adc, mode is not properly configured")
        return -93
    
        
        

def read_adc_i():
    v_ref =  2.048
    pga = 1  # gain 1 by default
    mode_multiplier = 500
    mode = None
    
    # Read gain from OUTPUT register (P4)
    i2c.writeto(addr_gpio, bytes([output_port]))
    gpio_out = i2c.readfrom(addr_gpio, 1)[0]  #0b76543210

    if gpio_out & 0x4 :
        dprint("Power Supply in Voltage mode")
        mode = "Voltage (V)"

    elif gpio_out & 0x8:
        mode = "Current (mA)"
    else:
        dprint("Error reading the ADC")
        return -94

    i2c.writeto(addr_adc, bytes([0b10111000]))
    
    sleep(0.2)
    
    raw = i2c.readfrom(addr_adc, 3)

    data_high = raw[0]
    data_low  = raw[1]
    config    = raw[2]

    # Extract 15-bit magnitude
    magnitude = ((data_high & 0x7F) << 8) | data_low

    # Determine sign
    if data_high & 0x80:
        # Negative number in two's complement
        signed = magnitude - 0x8000
    else:
        signed = magnitude

    # Convert to Current
    val_meas = signed * 62.5e-6 / pga * mode_multiplier


    # dprint results
    dprint(f"ADC val_meas: {val_meas}")
    dprint("Ready =", ((config & 0x80) == 0))

    # Overflow detection
    if signed == -32768 or signed == 32767:
        dprint("Warning: Value outside ADC measuring limit")
#     elif val_meas > 250:
#         dprint("Warning: according to adc, input resulted in DAC maxed")
#         return -99

    if mode in ("Voltage (V)", "Current (mA)"):
        return interpolate(val_meas, adc_i_table)
    else:
        dprint("Warning: according to adc, mode is not properly configured")
        return -95




###------------Basic Operation-----------###

def wait_for_i2c():
    while True:
        try:
            devs = i2c.scan()
            if len(devs) > 0:
                dprint("I2C ready:", devs)
                return
            else:
                dprint("I2C not ready, retrying...")
        except:
            dprint("I2C error, retrying...")

        sleep(0.5)


def power_routine(state):
    if state is True:
        adc_config()
        sleep(0.2)
        i2c.writeto(addr_gpio, bytes([output_port, 0b00000111]))  # Turn on
        sleep(0.2)
        power_dac(state)
        sleep(0.2)
        i2c.writeto(addr_gpio, bytes([output_port, 0b01000111]))  # Change to big feedback loop
        sleep(0.2)
        gpio_status()
        adc_config()
        sleep(0.2)
        set_voltage(0)
        dprint("Power up routine complete - ready to operate")
        
    elif state is False:
        set_voltage(0)
        #set_current(0)
        sleep(0.2)
        i2c.writeto(addr_gpio, bytes([output_port, 0b00000111]))  # Change to small feedback loop
        sleep(0.2)
        power_dac(state)
        sleep(0.2)
        i2c.writeto(addr_gpio, bytes([output_port, 0b00000101]))  # Turn off
        sleep(0.2)
        gpio_status()
        dprint("Power down routine complete - ready to operate")
    else:
        dprint("Power-Supply in stand-by mode")


def standby_mode():
    status = False

    power_dac(status)

    status_dump()
    config_gpio()
    i2c.writeto(addr_gpio, bytes([output_port, 0b00000101]))  # Status of gpio pin (see list above)
    status_dump()
    dprint("Power-Supply in stand-by mode")

def to_bin(val, bits):
    s = bin(val)[2:]
    return "0" * (bits - len(s)) + s

def status_dump():
    # -------------------------
    # GPIO
    # -------------------------
    i2c.writeto(addr_gpio, bytes([input_port]))
    real_value = i2c.readfrom(addr_gpio, 1)[0]

    i2c.writeto(addr_gpio, bytes([output_port]))
    expected_value = i2c.readfrom(addr_gpio, 1)[0]
    
    i2c.writeto(addr_gpio, bytes([configuration]))
    gpio_inout = i2c.readfrom(addr_gpio, 1)[0]

    a = to_bin(expected_value, 8)
    b = to_bin(real_value, 8)
    c = to_bin(gpio_inout, 8)

    # -------------------------
    # ADC
    # -------------------------
    i2c.writeto(addr_adc, bytes([0b10011000]))
    sleep(0.2)
    raw1 = i2c.readfrom(addr_adc, 3)

    i2c.writeto(addr_adc, bytes([0b10111000]))
    sleep(0.2)
    raw2 = i2c.readfrom(addr_adc, 3)

    adc1 = to_bin(raw1[0],8) + to_bin(raw1[1],8) + to_bin(raw1[2],8)
    adc2 = to_bin(raw2[0],8) + to_bin(raw2[1],8) + to_bin(raw2[2],8)

    # -------------------------
    # DAC
    # -------------------------
    i2c.writeto(addr_dac, bytes([write_to_buffer]))
    data = i2c.readfrom(addr_dac, 8)

    D_A = (data[0] << 4) | (data[1] >> 4)
    D_B = (data[6] << 4) | (data[7] >> 4)

    # -------------------------
    # SINGLE LINE OUTPUT
    # -------------------------
    return (
        "GPIO_CONF=" + c +
        ",GPIO_EXP=" + a +
        ",GPIO_ACT=" + b +
        ",ADC1=" + adc1 +
        ",ADC2=" + adc2 +
        ",DACA=" + str(D_A) +
        ",DACB=" + str(D_B)
    )


def read_both():

    v = read_adc_v()
    i = read_adc_i()

    return v, i


# ============================
# SCPI COMMAND PARSER
# ============================
def dprint(*args):
    if DEBUG:
        print(*args)
        
def scpi_execute(cmd):

    cmd = cmd.strip().upper()
    parts = cmd.split()

    header = parts[0]
    arg = parts[1] if len(parts) > 1 else None

    try:

        # -------------------------
        # Identification
        # -------------------------
        if header == "*IDN?":
            return "LIP,MicroPythonPSU,0001,1.0"

        # -------------------------
        # Reset
        # -------------------------
        if header == "*RST":
            standby_mode()
            return None
        
        # -------------------------
        # Debug prints
        # -------------------------
        if header == "SYST:DEBUG":
            global DEBUG

            if arg == "ON":
                DEBUG = True
                return "DEBUG ON"

            if arg == "OFF":
                DEBUG = False
                return "DEBUG OFF"
            
        # -------------------------
        # Output control
        # -------------------------
        if header == "OUTP":
            if arg == "ON":
                power_routine(True)
                return None

            if arg == "OFF":
                power_routine(False)
                return None

        if header == "OUTP?":
            return "UNKNOWN"  # optional future implementation

        # -------------------------
        # Source Voltage
        # -------------------------
        if header == "SOUR:VOLT":
            set_voltage(float(arg))
            return None

        if header == "SOUR:CURR":
            set_current(float(arg))
            return None

        # -------------------------
        # Protection Limits
        # -------------------------
        if header == "SOUR:VOLT:PROT":
            set_limit_v(float(arg))
            return None

        if header == "SOUR:CURR:PROT":
            set_limit_i(float(arg))
            return None

        # -------------------------
        # Mode selection
        # -------------------------
        if header == "FUNC":

            if arg == "VOLT":
                force_measure_mode("v")
                return None

            if arg == "CURR":
                force_measure_mode("i")
                return None

        # -------------------------
        # Measurements
        # -------------------------
        if header == "MEAS:VOLT?":
            return str(read_adc_v())

        if header == "MEAS:CURR?":
            return str(read_adc_i())
        
        if header == "READ?":
            v, i = read_both()
            return str(v) + "V , " + str(i) + "mA"

        # -------------------------
        # Status
        # -------------------------
        if header == "STAT?":
            return status_dump()

        return "ERR:UNKNOWN COMMAND"

    except Exception as e:
        return "ERR:" + str(e)



# ============================
# MAIN LOOP (SCPI SERVER)
# ============================

try:
    wait_for_i2c()
    standby_mode()
    sleep(0.2)

    print("SCPI Power Supply Ready")
    

    while True:
        now = ticks_ms()

        # 1. Handle SCPI commands (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip()
            if cmd:
                #print(cmd)  # echo
                response = scpi_execute(cmd)
                if response is not None:
                    print(response)


        sleep(0.01)


finally:

    power_routine(False)
    dprint("Emergency shutdown")




