#pragma once

#include <stdint.h>
#include <driver/uart.h>
#include <soc/gpio_num.h>
#include <esp_err.h>

class uart {
    public:
        uart(uart_port_t port_num, gpio_num_t tx_pin, gpio_num_t rx_pin, int baud_rate = 115200);
        ~uart();
        esp_err_t tx(const uint8_t* data, size_t len);
        esp_err_t rx(uint8_t* data, size_t len);
    private:
        uart_port_t port;
        uart_config_t config;
};
