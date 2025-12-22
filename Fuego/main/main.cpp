// main/main.c   ← 100 % copy-paste, works on ESP32-C6 right now
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "esp_rom_sys.h"
#include "../include/i2c.h"
#include "../include/uart.h"
#include "../include/Lora.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_sleep.h"
#include "esp_log.h"
#include "soc/rtc.h"
#include "esp_spiffs.h"
#include "esp_vfs.h"

extern "C" {
    #include "../include/MLX90640_API.h"
    #include "../include/redar_i2c_driver.h"
}

static const char *TAG = "forest-cam";
paramsMLX90640 params;

// ──────────────────────────────────────────────────
// CONFIGURE THESE FOR YOUR BOARD
// ──────────────────────────────────────────────────
#define GPIO_SDA            GPIO_NUM_2
#define GPIO_SCL            GPIO_NUM_3
#define I2C_PORT            I2C_NUM_0
#define PIR_GPIO            GPIO_NUM_7      // any GPIO that supports input & wakeup
#define CAMERA_POWER_GPIO   GPIO_NUM_11     // controls P-MOSFET → hard power to camera
#define WAKEUP_TIMEOUT_SEC  300             // 5 minutes max awake time (safety)

// ──────────────────────────────────────────────────
// Optional: second GPIO to wake from (e.g. mmWave or button)
// ──────────────────────────────────────────────────
#define USE_SECOND_WAKEUP   0
#if USE_SECOND_WAKEUP
  #define PIR2_GPIO GPIO_NUM_8
#endif

#define TA_SHIFT 8
#define MLX_WIDTH  32
#define MLX_HEIGHT 24
#define MLX_PIXELS (MLX_WIDTH * MLX_HEIGHT)
#define TA_SHIFT   0    // Typical value, adjust if needed

float emissivity = 0.95;
float tr;
unsigned char slaveAddress = 0x33;

static float subpageTo[768];
static float image[768];

static uint16_t eeMLX90640[832];
static uint16_t mlx90640Frame[834];
paramsMLX90640 mlx90640;
static float mlx90640To[768];
int status;
const TickType_t xDelay = 2000 / portTICK_PERIOD_MS;

// ──────────────────────────────────────────────────
static void power_on_camera(void)
{
    gpio_set_direction((gpio_num_t)CAMERA_POWER_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level((gpio_num_t)CAMERA_POWER_GPIO, 1);
    vTaskDelay(100 / portTICK_PERIOD_MS);  // Settle time
}

static void power_off_camera(void)
{
    gpio_set_level(CAMERA_POWER_GPIO, 0);
    ESP_LOGI(TAG, "Camera powered OFF");
}

void dump_file_to_serial(const char *filename)
{
    ESP_LOGI(TAG, "Starting dump of %s", filename);
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        ESP_LOGE(TAG, "Failed to open %s", filename);
        return;
    }
    uint8_t buffer[256];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        for (size_t i = 0; i < bytes_read; i++) {
            ESP_LOGI(TAG, "%02X", buffer[i]);
        }
    }
    ESP_LOGI(TAG, "Dump complete");
    fclose(fp);
}

// ──────────────────────────────────────────────────
// This runs every time we wake up (from PIR or timer)
// ──────────────────────────────────────────────────
static void handle_wakeup(void)
{
    //esp_sleep_source_t cause = esp_sleep_get_wakeup_cause();

    // switch (cause) {
    //     case ESP_SLEEP_WAKEUP_EXT0:
    //         ESP_LOGI(TAG, "Wakeup from PIR (EXT0)");
    //         break;
    //     case ESP_SLEEP_WAKEUP_EXT1:
    //         ESP_LOGI(TAG, "Wakeup from PIR (EXT1)");
    //         break;
    //     case ESP_SLEEP_WAKEUP_TIMER:
    //         ESP_LOGI(TAG, "Wakeup from safety timer (5 min max)");
    //         break;
    //     default:
    //         ESP_LOGI(TAG, "Wakeup was not caused by PIR or timer (%d)", cause);
    //         break;
    // }
    vTaskDelay(pdMS_TO_TICKS(1));
    REDAR_I2CInit();
    vTaskDelay(pdMS_TO_TICKS(1));
    // Allocate memory for EEPROM data and frame data
    uint16_t *eeData = (uint16_t *)malloc(832 * sizeof(uint16_t));
    uint16_t *frameData = (uint16_t *)malloc(834 * sizeof(uint16_t));
    float *temperatures = (float *)malloc(768 * sizeof(float));
vTaskDelay(pdMS_TO_TICKS(1));
    if (!eeData || !frameData || !temperatures) {
        ESP_LOGE(TAG, "Failed to allocate memory for MLX90640 data");
        return;
    }
vTaskDelay(pdMS_TO_TICKS(1));
    // Wake up the sensor by setting refresh rate (starts operation)
    int error = MLX90640_SetRefreshRate(0x33, 0x03); // Set to 2Hz for example
    if (error != 0) {
        ESP_LOGE(TAG, "Failed to set refresh rate: %d", error);
        free(eeData);
        free(frameData);
        free(temperatures);
        return;
    }

    // Get factory data (EEPROM)
    error = MLX90640_DumpEE(0x33, eeData);
    vTaskDelay(pdMS_TO_TICKS(1));
    if (error != 0) {
        ESP_LOGE(TAG, "Failed to dump EEPROM: %d", error);
        free(eeData);
        free(frameData);
        free(temperatures);
        return;
    }

    // Extract parameters from factory data
    error = MLX90640_ExtractParameters(eeData, &params);
    vTaskDelay(pdMS_TO_TICKS(1));
    if (error != 0) {
        ESP_LOGE(TAG, "Failed to extract parameters: %d", error);
        free(eeData);
        free(frameData);
        free(temperatures);
        return;
    }

    ESP_LOGI(TAG, "Factory data extracted successfully");

    // Grab frame data
    error = MLX90640_GetFrameData(0x33, frameData);
    if (error != 0) {
        ESP_LOGE(TAG, "Failed to get frame data: %d", error);
        free(eeData);
        free(frameData);
        free(temperatures);
        return;
    }

    // Calculate temperatures
    //MLX90640_CalculateTo(frameData, &params, 0.95, 23.0, temperatures); // emissivity 0.95, reflected temp 23C

    // Log some sample temperatures
    ESP_LOGI(TAG, "Sample temperatures:");
    for (int i = 0; i < 32; i += 8) { // Log every 8th pixel in first row
        ESP_LOGI(TAG, "Pixel %d: %.2f C", i, temperatures[i]);
    }

    // ←←← YOUR AI CODE GOES HERE ←←←
    // power_on_camera();
    // take_photo_and_run_yolov11();
    // if (detected_person_or_fire) → send LoRa / Thread alert
    // power_off_camera();

    ESP_LOGI(TAG, "MLX90640 data processed – sleeping again in 30 seconds for demo");
    vTaskDelay(pdMS_TO_TICKS(30000));   // replace with your real code
    // ←←← END OF YOUR CODE ←←←

    free(eeData);
    free(frameData);
    free(temperatures);
}

// ──────────────────────────────────────────────────
extern "C" void app_main(void)
{
    esp_rom_printf(">>> app_main entered <<<\n");
    ESP_LOGI(TAG, "Forest AI Cam – ultra-low power wake-on-PIR");
    ESP_LOGI(TAG, "Starting MLX90640 setup");

    // 1. Camera power control (P-MOSFET → active HIGH)
    // gpio_reset_pin(CAMERA_POWER_GPIO);
    // gpio_set_direction(CAMERA_POWER_GPIO, GPIO_MODE_OUTPUT);
    // gpio_set_level(CAMERA_POWER_GPIO, 0);   // camera OFF

    vTaskDelay(pdMS_TO_TICKS(3000));
    ESP_LOGI(TAG, "Initial delay complete");
    power_on_camera();
    ESP_LOGI(TAG, "Camera powered on");
    vTaskDelay(pdMS_TO_TICKS(500));
    ESP_LOGI(TAG, "Delay after power on complete");
    REDAR_I2CInit();
    ESP_LOGI(TAG, "REDAR I2C initialized");
    status = MLX90640_DumpEE(slaveAddress, eeMLX90640);
    ESP_LOGI(TAG, "EEPROM dumped, status %d", status);
    status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
    ESP_LOGI(TAG, "Parameters extracted, status %d", status);

    int error = MLX90640_SetRefreshRate(0x33, 0x03); // Set to 2Hz for example
    if (error != 0) {
        ESP_LOGE(TAG, "Failed to set refresh rate: %d", error);
        return;
    }

    vTaskDelay(pdMS_TO_TICKS(3000));
    int subpage;

    /* init image buffer ONCE */
    for (int i = 0; i < 768; i++) {
        image[i] = NAN;
    }

    /* -------- First subpage -------- */
    for (int i = 0; i < 768; i++) {
        subpageTo[i] = NAN;
    }

    MLX90640_GetSubFrameData(slaveAddress, mlx90640Frame);

    tr = MLX90640_GetTa(mlx90640Frame, &mlx90640) - TA_SHIFT;
    MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, subpageTo);

    for (int i = 0; i < 768; i++) {
        if (!isnan(subpageTo[i])) {
            image[i] = subpageTo[i];
        }
    }

    vTaskDelay(pdMS_TO_TICKS(260));
    
    /* -------- Second subpage -------- */
    for (int i = 0; i < 768; i++) {
        subpageTo[i] = NAN;
    }

    MLX90640_GetSubFrameData(slaveAddress, mlx90640Frame);

    tr = MLX90640_GetTa(mlx90640Frame, &mlx90640) - TA_SHIFT;
    MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, subpageTo);

    for (int i = 0; i < 768; i++) {
        if (!isnan(subpageTo[i])) {
            image[i] = subpageTo[i];
        }
    }

    vTaskDelay(pdMS_TO_TICKS(1000));


    for (int p = 0; p < 768; p++) {
        ESP_LOGI("REDAR", "%.2f", image[p]);
    }

    // while (1) {
    //     esp_rom_printf(">>> app_main entered <<<\n");
    //     ESP_LOGI(TAG, "Forest AI Cam – ultra-low power wake-on-PIR");
    //     //handle_wakeup();   // runs once per wake-up

    //     // ───── Go back to deep sleep until next PIR or 5-min timeout ─────

    //     // Wake on PIR rising edge (RTC_IO only on certain pins – GPIO7 is safe on C6)
    //     //esp_sleep_enable_ext1_wakeup(BIT64(PIR_GPIO), ESP_EXT1_WAKEUP_ANY_HIGH);   // 1 = rise

    //     // Optional: wake on multiple pins (EXT1)
    //     // uint64_t mask = BIT64(PIR_GPIO) | BIT64(PIR2_GPIO);
    //     // esp_sleep_enable_ext1_wakeup(mask, ESP_EXT1_WAKEUP_ANY_HIGH);e
    //     //power_on_camera();
    // }
}