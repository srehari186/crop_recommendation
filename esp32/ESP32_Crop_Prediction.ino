/*
 * ESP32 Smart Crop Recommendation - Render API + OLED + Blynk
 * ===========================================================
 * Architecture (NO model on the ESP32):
 *   Sensors -> ESP32 -> Wi-Fi -> Render /predict -> Top 3 crops
 *                                                      |
 *                                    +-----------------+------------------+
 *                                    v                                  v
 *                                OLED display                      Blynk IoT
 *
 * The trained Random Forest (.pkl) lives ONLY on the Render server.
 * This sketch only reads sensors, sends them as JSON, and displays/forwards
 * the prediction it receives. It never computes a crop prediction itself.
 *
 * Libraries (Arduino IDE Library Manager):
 *   - esp32 board support
 *   - ArduinoJson (v7)            -> build/parse JSON
 *   - Adafruit SSD1306 + Adafruit GFX (USE_OLED)
 *   - Blynk (USE_BLYNK)
 *   - DHT sensor library (for temperature/humidity)
 *
 * Config lives in config.h (gitignored). Copy config.example.h -> config.h.
 */

#include "config.h"

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

#ifdef USE_OLED
  #include <Wire.h>
  #include <Adafruit_GFX.h>
  #include <Adafruit_SSD1306.h>
#endif

#ifdef USE_BLYNK
  #include <BlynkSimpleEsp32.h>
#endif

#include <DHTesp.h>

// ---------------------------------------------------------------------------
// Sensor readings (filled by readSensors())
// ---------------------------------------------------------------------------
float nitrogen, phosphorus, potassium;
float temperature, humidity, ph, rainfall;

// ---------------------------------------------------------------------------
// Last prediction received from Render (top 3)
// ---------------------------------------------------------------------------
struct CropReco { char name[16]; float prob; };
CropReco reco[3];
bool haveResult = false;
String mlStatus = "BOOT";          // OLED/Blynk status string
bool sensorError = false;

// ---------------------------------------------------------------------------
// OLED
// ---------------------------------------------------------------------------
#ifdef USE_OLED
  Adafruit_SSD1306 display(128, 64, &Wire, -1);
  bool oledOk = false;
#endif
unsigned long lastPageSwitch = 0;
bool showSensorsPage = false;

// ---------------------------------------------------------------------------
// Non-blocking timing
// ---------------------------------------------------------------------------
unsigned long lastPrediction = 0;

DHTesp dht;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
void drawText(const char* line) {
  Serial.println(line);
#ifdef USE_OLED
  if (oledOk) display.println(line);
#endif
}

// ---------------------------------------------------------------------------
// Wi-Fi
// ---------------------------------------------------------------------------
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting Wi-Fi");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(300); Serial.print('.');
  }
  Serial.println(WiFi.status() == WL_CONNECTED ? "\nWi-Fi connected"
                                               : "\nWi-Fi FAILED");
}

// ---------------------------------------------------------------------------
// Read real sensors (replace calibration constants with your own).
// ---------------------------------------------------------------------------
void readSensors() {
  sensorError = false;

#if TEST_MODE
  // Debug-only fixed sample. Do NOT enable in production.
  nitrogen = 90; phosphorus = 42; potassium = 43;
  temperature = 28.5; humidity = 80; ph = 6.5; rainfall = 200;
  return;
#endif

  // Temperature + humidity (DHT22/DHT11)
  TempAndHumidity th = dht.getTempAndHumidity();
  if (dht.getStatus() == 0) {
    temperature = th.temperature;
    humidity = th.humidity;
  } else {
    sensorError = true;
    Serial.println("[SENSOR] DHT read failed");
  }

  // pH: analog probe through a voltage divider (CALIBRATE for your probe).
  int phRaw = analogRead(PH_PIN);
  float phVolt = phRaw / 4095.0f * 3.3f;
  ph = 7.0f + (2.5f - phVolt) * 2.0f;   // <-- adjust slope/offset to calibrate

  // Rainfall: analog gauge (CALIBRATE mm range for your sensor).
  int rainRaw = analogRead(RAIN_PIN);
  rainfall = (rainRaw / 4095.0f) * 300.0f;  // <-- calibrate to mm

  // NPK sensor over UART (adapt to your sensor's protocol/frame).
  nitrogen = phosphorus = potassium = 0;
  if (Serial2.available()) {
    String line = Serial2.readStringUntil('\n');
    int iN = line.indexOf("N:"); int iP = line.indexOf("P:");
    int iK = line.indexOf("K:");
    if (iN >= 0) nitrogen    = line.substring(iN + 2).toFloat();
    if (iP >= 0) phosphorus  = line.substring(iP + 2).toFloat();
    if (iK >= 0) potassium   = line.substring(iK + 2).toFloat();
  } else {
    sensorError = true;
    Serial.println("[SENSOR] NPK not available");
  }
}

// ---------------------------------------------------------------------------
// POST sensor values to Render /predict and parse the Top-3.
// On ANY failure: setmlStatus, keep last result, do NOT crash/reboot.
// ---------------------------------------------------------------------------
void sendToRender() {
  if (WiFi.status() != WL_CONNECTED) {
    mlStatus = "WIFI DOWN";
    Serial.println("[Render] Wi-Fi not connected - will retry.");
    return;
  }

  WiFiClientSecure client;
  client.setInsecure();   // Render provides a valid cert; setInsecure avoids
                          // shipping a CA bundle on the ESP32 (college demo).
  HTTPClient http;
  String url = String(RENDER_HOST) + String(RENDER_PATH);
  if (!http.begin(client, url)) {
    mlStatus = "ML SERVER ERROR";
    Serial.println("[Render] HTTP begin failed - will retry.");
    return;
  }
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(10000);

  JsonDocument req;
  req["N"] = nitrogen;
  req["P"] = phosphorus;
  req["K"] = potassium;
  req["temperature"] = temperature;
  req["humidity"] = humidity;
  req["ph"] = ph;
  req["rainfall"] = rainfall;
  String body;
  serializeJson(req, body);

  Serial.printf("\n[Render] POST %s\n", url.c_str());
  int code = http.POST(body);

  if (code <= 0) {
    mlStatus = "ML SERVER ERROR";
    Serial.printf("[Render] HTTP error (%d) - will retry.\n", code);
    http.end();
    return;
  }
  if (code >= 400) {
    mlStatus = "ML SERVER ERROR";
    Serial.printf("[Render] HTTP %d - will retry.\n", code);
    http.end();
    return;
  }

  String payload = http.getString();
  http.end();

  JsonDocument res;
  if (deserializeJson(res, payload) != DeserializationError::Ok) {
    mlStatus = "INVALID JSON";
    Serial.println("[Render] Invalid JSON response - will retry.");
    return;
  }

  if (res["error"]) {
    mlStatus = "ERR: " + res["error"].as<String>();
    Serial.printf("[Render] API error: %s - will retry.\n", mlStatus.c_str());
    return;
  }

  JsonArray recs = res["recommendations"];
  if (!recs || recs.size() == 0) {
    mlStatus = "NO PREDICTION";
    Serial.println("[Render] Empty recommendations - will retry.");
    return;
  }

  int n = min((int)recs.size(), 3);
  for (int i = 0; i < n; i++) {
    const char* c = recs[i]["crop"] | "?";
    strncpy(reco[i].name, c, sizeof(reco[i].name) - 1);
    reco[i].name[sizeof(reco[i].name) - 1] = '\0';
    reco[i].prob = recs[i]["probability"] | 0.0f;
  }
  haveResult = true;
  mlStatus = "OK";
  Serial.println("[Render] Top-3 received:");
  for (int i = 0; i < n; i++)
    Serial.printf("  %d. %s %.0f%%\n", i + 1, reco[i].name, reco[i].prob * 100);
}

// ---------------------------------------------------------------------------
// OLED render (two pages, auto-switched)
// ---------------------------------------------------------------------------
#ifdef USE_OLED
void renderOLED() {
  if (!oledOk) return;
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  char buf[22];
  if (showSensorsPage) {
    display.println("SENSORS");
    snprintf(buf, sizeof(buf), "N%g P%g K%g", nitrogen, phosphorus, potassium);
    display.println(buf);
    snprintf(buf, sizeof(buf), "T%gC H%g%%", temperature, humidity);
    display.println(buf);
    snprintf(buf, sizeof(buf), "pH%g R%gmm", ph, rainfall);
    display.println(buf);
    display.println(mlStatus);
  } else {
    display.println("TOP 3 CROPS");
    if (haveResult) {
      for (int i = 0; i < 3; i++) {
        snprintf(buf, sizeof(buf), "%d.%s %d%%", i + 1, reco[i].name,
                 (int)(reco[i].prob * 100));
        display.println(buf);
      }
    } else {
      display.println(mlStatus);
    }
  }
  display.display();
}
#endif

// ---------------------------------------------------------------------------
// Blynk telemetry (Virtual Pin mapping)
// ---------------------------------------------------------------------------
#ifdef USE_BLYNK
void sendBlynk() {
  if (!Blynk.connected()) return;
  Blynk.virtualWrite(V0,  nitrogen);
  Blynk.virtualWrite(V1,  phosphorus);
  Blynk.virtualWrite(V2,  potassium);
  Blynk.virtualWrite(V3,  temperature);
  Blynk.virtualWrite(V4,  humidity);
  Blynk.virtualWrite(V5,  ph);
  Blynk.virtualWrite(V6,  rainfall);
  if (haveResult) {
    Blynk.virtualWrite(V10, String(reco[0].name));
    Blynk.virtualWrite(V11, (int)(reco[0].prob * 100));
    Blynk.virtualWrite(V12, String(reco[1].name));
    Blynk.virtualWrite(V13, (int)(reco[1].prob * 100));
    Blynk.virtualWrite(V14, String(reco[2].name));
    Blynk.virtualWrite(V15, (int)(reco[2].prob * 100));
  }
  Blynk.virtualWrite(V20, mlStatus);
}
#endif

// ---------------------------------------------------------------------------
// One prediction cycle (called on interval, non-blocking elsewhere)
// ---------------------------------------------------------------------------
void predictCycle() {
  readSensors();
  if (sensorError) {
    mlStatus = "SENSOR ERROR";
    Serial.println("[SENSOR] error - skipping prediction this cycle.");
  } else {
    sendToRender();
  }
#ifdef USE_BLYNK
  sendBlynk();
#endif
#ifdef USE_OLED
  renderOLED();
#endif
}

// ---------------------------------------------------------------------------
// Setup / loop
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1500);

  dht.setup(DHT_PIN, DHTesp::DHT22);

#ifdef USE_OLED
  Wire.begin(OLED_SDA, OLED_SCL);
  if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    oledOk = true;
    display.clearDisplay();
    display.display();
  } else {
    Serial.println("[OLED] init failed");
  }
#endif

  Serial.println("=== ESP32 Crop Prediction (Render API) ===");

  connectWiFi();
#ifdef USE_BLYNK
  Blynk.begin(BLYNK_AUTH_TOKEN, WIFI_SSID, WIFI_PASS);
#endif

  lastPrediction = -PREDICTION_INTERVAL;   // run immediately on first loop
}

void loop() {
#ifdef USE_BLYNK
  Blynk.run();
#endif

  // Non-blocking prediction on interval.
  if (millis() - lastPrediction >= PREDICTION_INTERVAL) {
    lastPrediction = millis();
    predictCycle();
  }

  // Auto-switch OLED pages every 4 s.
#ifdef USE_OLED
  if (oledOk && millis() - lastPageSwitch >= 4000) {
    lastPageSwitch = millis();
    showSensorsPage = !showSensorsPage;
    renderOLED();
  }
#endif

  // Keep Wi-Fi alive if it drops.
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
}
