#include "../include/Lora.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <string.h>

Lora::Lora() {
    uart_driver_install(UART_NUM, BUF_SIZE, BUF_SIZE, 0, NULL, 0);
    uart_param_config(UART_NUM, &uart_config);
}

void Lora::loraReset(void)
{
    // Configure GPIO as output
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LORA_RST),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);

    // Drive reset LOW
    gpio_set_level((gpio_num_t)LORA_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(10));   // 10 ms reset pulse

    // Release reset
    gpio_set_level((gpio_num_t)LORA_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(10));   // allow LoRa to boot
}

void Lora::powerOnLora(){
    gpio_set_direction((gpio_num_t)LORA_PWR, GPIO_MODE_OUTPUT);
    gpio_set_level((gpio_num_t)LORA_PWR, 1);
}

void Lora::uartWriteBytes(const char* data) {
    uart_write_bytes(UART_NUM, data, strlen(data));
}

void Lora::uartReadBytes(uint8_t* buffer) {
     int len = uart_read_bytes(UART_NUM, buffer, BUF_SIZE - 1,
                                  1000 / portTICK_PERIOD_MS);
        if (len > 0) {
            buffer[len] = '\0';
            ESP_LOGI(TAG, "Received: %s", (char*)buffer);
        }

        vTaskDelay(pdMS_TO_TICKS(1000)); 
}