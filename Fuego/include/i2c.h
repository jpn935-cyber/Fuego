#pragma once

#include <stdint.h>
#include <driver/i2c.h>
#include <esp_err.h>

class i2c {
    public:
        i2c(i2c_port_t port_num, gpio_num_t sda_pin, gpio_num_t scl_pin, uint32_t clk_speed = 100000);
        ~i2c();
        esp_err_t readBytes(uint8_t address, uint8_t* data, size_t len);
        esp_err_t writeBytes(uint8_t address, const uint8_t* data, size_t len);
    private:
        i2c_port_t port;
        i2c_config_t config;
};