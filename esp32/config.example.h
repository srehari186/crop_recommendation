#pragma once
/*
 * ESP32 Crop Prediction - CONFIGURATION TEMPLATE
 * -------------------------------------------------
 * 1. Copy this file to "config.h" in the same folder:
 *        cp config.example.h config.h
 * 2. Fill in YOUR real values below.
 * 3. config.h is gitignored, so your Wi-Fi/Blynk secrets are NEVER committed.
 *
 * The ESP32 ONLY reads sensors and calls the Render API. It does NOT run the
 * model locally - the .pkl model stays on the Render server.
 */

// ----- Wi-Fi -----
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASS     "YOUR_WIFI_PASSWORD"

// ----- Render API (HTTPS, no trailing slash on host) -----
#define RENDER_HOST   "https://YOUR-RENDER-URL"   // e.g. https://crop-prediction-api.onrender.com
#define RENDER_PATH   "/predict"

// ----- Blynk IoT -----
#define BLYNK_TEMPLATE_ID    "YOUR_TEMPLATE_ID"
#define BLYNK_TEMPLATE_NAME  "Crop Prediction"
#define BLYNK_AUTH_TOKEN     "YOUR_AUTH_TOKEN"

// ----- Optional features (comment out to disable) -----
#define USE_OLED      // SSD1306 128x64 I2C display
#define USE_BLYNK     // Blynk IoT telemetry

// Debug ONLY: 1 = send a fixed bench sample instead of reading sensors.
// NEVER leave this on in the final deployment.
#define TEST_MODE 0

// How often to predict (ms). Blynk stays responsive between cycles.
#define PREDICTION_INTERVAL  30000

// ----- Sensor pins (change to match your wiring) -----
#define DHT_PIN     4     // DHT22 / DHT11 data pin
#define PH_PIN      34    // analog pH (voltage divider to 3.3V)
#define RAIN_PIN    35    // analog rain sensor
#define NPK_RX      16    // NPK sensor UART RX
#define NPK_TX      17    // NPK sensor UART TX
#define OLED_ADDR   0x3C  // SSD1306 I2C address
#define OLED_SDA    21
#define OLED_SCL    22
