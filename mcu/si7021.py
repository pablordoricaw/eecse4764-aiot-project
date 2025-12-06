# Si7021 Temperature & Humidity Sensor Driver for MicroPython
# Based on: https://gist.github.com/minyk/7c3070bc1c2766633b8ff1d4d51089cf
# For Adafruit Si7021 Breakout Board

from time import sleep_ms

# Default Address
SI7021_I2C_DEFAULT_ADDR = 0x40

# Commands
CMD_MEASURE_RELATIVE_HUMIDITY = 0xF5
CMD_MEASURE_TEMPERATURE = 0xF3
CMD_RESET = 0xFE

class Si7021:
    def __init__(self, i2c, addr=SI7021_I2C_DEFAULT_ADDR):
        self.i2c = i2c
        self.addr = addr
        self.cbuffer = bytearray(2)
        self.cbuffer[1] = 0x00
        
    def write_command(self, command_byte):
        self.cbuffer[0] = command_byte
        self.i2c.writeto(self.addr, self.cbuffer)
        
    def reset(self):
        """Reset the sensor"""
        self.cbuffer[0] = CMD_RESET
        self.i2c.writeto(self.addr, self.cbuffer)
        sleep_ms(50)

    def read_temperature(self):
        """Read temperature in Celsius"""
        self.write_command(CMD_MEASURE_TEMPERATURE)
        sleep_ms(25)
        temp = self.i2c.readfrom(self.addr, 3)
        temp2 = temp[0] << 8
        temp2 = temp2 | temp[1]
        return (175.72 * temp2 / 65536) - 46.85
    
    def read_temperature_fahrenheit(self):
        """Read temperature in Fahrenheit"""
        celsius = self.read_temperature()
        return (celsius * 9/5) + 32

    def read_humidity(self):
        """Read relative humidity in %"""
        self.write_command(CMD_MEASURE_RELATIVE_HUMIDITY)
        sleep_ms(25)
        rh = self.i2c.readfrom(self.addr, 3)
        rh2 = rh[0] << 8
        rh2 = rh2 | rh[1]
        return (125 * rh2 / 65536) - 6