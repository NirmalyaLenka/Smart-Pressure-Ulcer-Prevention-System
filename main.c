/**
 * Smart Pressure Ulcer Prevention System
 * STM32H7 Main Controller
 *
 * Responsibilities:
 *   - Read risk scores from FPGA over UART
 *   - Run PID controller for pneumatic bladder pressure
 *   - Enforce hardware safety interlocks
 *   - Forward telemetry to ESP32 for MQTT publishing
 */

#include "stm32h7xx_hal.h"
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "pid_controller.h"
#include "sensor_read.h"
#include "safety_interlock.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* -------------------------------------------------------------------------- */
/* Configuration                                                               */
/* -------------------------------------------------------------------------- */

#define NUM_ZONES               64        /* 8x8 bladder grid */
#define RISK_UART               huart1    /* FPGA -> STM32 */
#define TELEMETRY_UART          huart2    /* STM32 -> ESP32 */
#define VALVE_PWM_FREQ_HZ       1000
#define TARGET_PRESSURE_MMHG    40.0f
#define MAX_PRESSURE_MMHG       200.0f
#define ACTUATION_DURATION_MS   180000
#define COOLDOWN_MS             600000
#define TASK_STACK_SIZE         512

/* -------------------------------------------------------------------------- */
/* Risk score thresholds                                                       */
/* -------------------------------------------------------------------------- */

#define RISK_SAFE               0.30f
#define RISK_CAUTION            0.60f
#define RISK_HIGH               0.80f
#define RISK_CRITICAL           0.90f

/* -------------------------------------------------------------------------- */
/* Data structures                                                             */
/* -------------------------------------------------------------------------- */

typedef enum {
    ZONE_SAFE = 0,
    ZONE_CAUTION,
    ZONE_HIGH_RISK,
    ZONE_CRITICAL
} ZoneState;

typedef struct {
    uint8_t  zone_id;
    float    risk_score;
    float    pressure_kpa;
    float    skin_temp_delta;
    float    spo2_percent;
    uint32_t duration_ms;
    ZoneState state;
} ZoneData;

typedef struct {
    ZoneData zones[NUM_ZONES];
    uint32_t timestamp_ms;
    uint8_t  actuation_active[NUM_ZONES];
    uint32_t actuation_start_ms[NUM_ZONES];
} SystemState;

/* -------------------------------------------------------------------------- */
/* Global state and handles                                                    */
/* -------------------------------------------------------------------------- */

static SystemState g_state;
static PID_Handle  g_pid[NUM_ZONES];
static QueueHandle_t xRiskScoreQueue;
static QueueHandle_t xTelemetryQueue;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;
TIM_HandleTypeDef  htim1;  /* PWM for solenoid valves */

/* -------------------------------------------------------------------------- */
/* Helper: parse FPGA UART frame                                               */
/* -------------------------------------------------------------------------- */

/* FPGA sends 64-byte frames: [zone_id(1)] [risk_score_fixed(2)] [checksum(1)] */
static int parse_fpga_frame(uint8_t *buf, uint8_t len,
                             uint8_t *zone_out, float *risk_out)
{
    if (len < 4) return -1;
    uint8_t zone = buf[0];
    if (zone >= NUM_ZONES) return -1;

    /* risk score is Q8.8 fixed point */
    uint16_t raw = ((uint16_t)buf[1] << 8) | buf[2];
    float risk = (float)raw / 256.0f;

    uint8_t checksum = buf[0] ^ buf[1] ^ buf[2];
    if (checksum != buf[3]) return -1;

    *zone_out = zone;
    *risk_out = risk;
    return 0;
}

/* -------------------------------------------------------------------------- */
/* Helper: update zone state from risk score                                   */
/* -------------------------------------------------------------------------- */

static ZoneState classify_risk(float score)
{
    if (score >= RISK_CRITICAL) return ZONE_CRITICAL;
    if (score >= RISK_HIGH)     return ZONE_HIGH_RISK;
    if (score >= RISK_CAUTION)  return ZONE_CAUTION;
    return ZONE_SAFE;
}

/* -------------------------------------------------------------------------- */
/* Task: read risk scores from FPGA                                            */
/* -------------------------------------------------------------------------- */

static void task_read_fpga(void *pvParam)
{
    uint8_t rx_buf[4];
    uint8_t zone;
    float   risk;

    for (;;) {
        /* Blocking receive of one complete frame */
        if (HAL_UART_Receive(&RISK_UART, rx_buf, sizeof(rx_buf), 100) == HAL_OK) {
            if (parse_fpga_frame(rx_buf, sizeof(rx_buf), &zone, &risk) == 0) {
                /* Update state */
                g_state.zones[zone].risk_score = risk;
                g_state.zones[zone].state      = classify_risk(risk);

                /* Push to telemetry queue (non-blocking) */
                xQueueSendToBack(xRiskScoreQueue, &zone, 0);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/* -------------------------------------------------------------------------- */
/* Task: PID actuation control                                                 */
/* -------------------------------------------------------------------------- */

static void task_actuate(void *pvParam)
{
    uint8_t zone;
    uint32_t now;

    for (;;) {
        if (xQueueReceive(xRiskScoreQueue, &zone, portMAX_DELAY) == pdTRUE) {
            now = HAL_GetTick();
            ZoneData *z = &g_state.zones[zone];

            /* Check cooldown — do not re-actuate too soon */
            if (g_state.actuation_active[zone]) {
                uint32_t elapsed = now - g_state.actuation_start_ms[zone];
                if (elapsed >= ACTUATION_DURATION_MS) {
                    /* Deflate bladder */
                    valve_close(zone);
                    g_state.actuation_active[zone] = 0;
                }
                continue;
            }

            if (z->state >= ZONE_HIGH_RISK) {
                /* Safety check before any actuation */
                float current_pressure = read_bladder_pressure(zone);
                if (!safety_interlock_check(current_pressure, MAX_PRESSURE_MMHG)) {
                    /* Hardware interlock: close all valves */
                    for (int i = 0; i < NUM_ZONES; i++) valve_close(i);
                    continue;
                }

                /* Run PID to reach target pressure */
                float pid_output = PID_Update(&g_pid[zone],
                                              TARGET_PRESSURE_MMHG,
                                              current_pressure);
                valve_set_pwm(zone, (uint16_t)pid_output);
                g_state.actuation_active[zone]    = 1;
                g_state.actuation_start_ms[zone]  = now;
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/* Task: send telemetry to ESP32                                               */
/* -------------------------------------------------------------------------- */

static void task_telemetry(void *pvParam)
{
    char buf[256];
    int  len;

    for (;;) {
        for (uint8_t z = 0; z < NUM_ZONES; z++) {
            ZoneData *zd = &g_state.zones[z];
            len = snprintf(buf, sizeof(buf),
                "{\"zone\":%d,\"risk\":%.3f,\"state\":%d,"
                "\"pressure\":%.2f,\"temp_delta\":%.2f,"
                "\"spo2\":%.1f,\"actuating\":%d}\n",
                z, zd->risk_score, (int)zd->state,
                zd->pressure_kpa, zd->skin_temp_delta,
                zd->spo2_percent,
                (int)g_state.actuation_active[z]);

            HAL_UART_Transmit(&TELEMETRY_UART,
                              (uint8_t *)buf, len, 50);
        }
        vTaskDelay(pdMS_TO_TICKS(6000)); /* 6-second telemetry interval */
    }
}

/* -------------------------------------------------------------------------- */
/* Initialization                                                              */
/* -------------------------------------------------------------------------- */

static void system_init(void)
{
    HAL_Init();
    SystemClock_Config();

    MX_UART1_Init();   /* FPGA link */
    MX_UART2_Init();   /* ESP32 link */
    MX_TIM1_Init();    /* Solenoid PWM */

    /* Initialize PID controllers for each zone */
    for (int i = 0; i < NUM_ZONES; i++) {
        PID_Init(&g_pid[i],
                 /* Kp */ 2.0f,
                 /* Ki */ 0.5f,
                 /* Kd */ 0.1f,
                 /* output_min */ 0.0f,
                 /* output_max */ 65535.0f);  /* 16-bit PWM */
    }

    /* Initialize state */
    memset(&g_state, 0, sizeof(g_state));

    /* Create queues */
    xRiskScoreQueue = xQueueCreate(32, sizeof(uint8_t));
    xTelemetryQueue = xQueueCreate(8,  sizeof(uint8_t));
}

/* -------------------------------------------------------------------------- */
/* Entry point                                                                 */
/* -------------------------------------------------------------------------- */

int main(void)
{
    system_init();

    /* Create FreeRTOS tasks */
    xTaskCreate(task_read_fpga,  "FPGA_RX",   TASK_STACK_SIZE, NULL, 3, NULL);
    xTaskCreate(task_actuate,    "ACTUATE",    TASK_STACK_SIZE, NULL, 2, NULL);
    xTaskCreate(task_telemetry,  "TELEMETRY",  TASK_STACK_SIZE, NULL, 1, NULL);

    /* Start scheduler — does not return */
    vTaskStartScheduler();

    for (;;) {}
    return 0;
}
