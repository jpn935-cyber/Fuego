#include "driver/uart.h"
#include "esp_log.h"

static const char *TAG = "LORA_UART";

#define LORA_RST 4
#define LORA_PWR 5
#define UART_NUM UART_NUM_0
#define BUF_SIZE 256

uart_config_t uart_config = {
    .baud_rate = 115200,
    .data_bits = UART_DATA_8_BITS,
    .parity    = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
};

class Lora {
    public:

    uint8_t data[BUF_SIZE];
    Lora();

    void powerOnLora();
    void uartWriteBytes(const char* data);
    void uartReadBytes(uint8_t* buffer);
    void loraReset();
    private:
};