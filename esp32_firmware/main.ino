#include <WiFi.h>
#include <HTTPClient.h>

// -------------------------------------------------------------
// CONFIGURATION
// -------------------------------------------------------------
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Replace with the IP address of the machine running the FastAPI server
const char* serverName = "http://192.168.1.XXX:8000/api/telemetry"; 

const char* machineID = "ESP32-NODE-01";
const char* machineType = "M";

// -------------------------------------------------------------
// SENSOR PINS (Example)
// -------------------------------------------------------------
#define TEMP_SENSOR_PIN 34 // Example Analog Pin
#define RPM_SENSOR_PIN  35 // Example Analog Pin

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Connect to Wi-Fi
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected.");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    
    // Start connection and send HTTP header
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    // ---------------------------------------------------------
    // SENSOR READINGS (Or Dummy Data generation)
    // ---------------------------------------------------------
    // In a real scenario, use analogRead(pin) and convert to physical units
    // float rawTemp = analogRead(TEMP_SENSOR_PIN);
    
    // Here we generate realistic simulated data for the backend
    float airTemp = random(2900, 3100) / 10.0;     // 290.0 to 310.0 K
    float processTemp = airTemp + random(50, 150) / 10.0; // 5 to 15 K higher
    int rpm = random(1200, 2500);
    float torque = random(200, 600) / 10.0;        // 20.0 to 60.0 Nm
    int toolWear = random(10, 250);                // 10 to 250 min

    // Create JSON Payload manually (or use ArduinoJson library)
    String jsonPayload = "{";
    jsonPayload += "\"machine_id\":\"" + String(machineID) + "\",";
    jsonPayload += "\"machine_type\":\"" + String(machineType) + "\",";
    jsonPayload += "\"air_temp\":" + String(airTemp) + ",";
    jsonPayload += "\"process_temp\":" + String(processTemp) + ",";
    jsonPayload += "\"rotational_speed\":" + String(rpm) + ",";
    jsonPayload += "\"torque\":" + String(torque) + ",";
    jsonPayload += "\"tool_wear\":" + String(toolWear);
    jsonPayload += "}";

    Serial.println("Publishing Telemetry:");
    Serial.println(jsonPayload);

    // Send HTTP POST request
    int httpResponseCode = http.POST(jsonPayload);

    if (httpResponseCode > 0) {
      Serial.print("HTTP Response code: ");
      Serial.println(httpResponseCode);
      String response = http.getString();
      Serial.println(response);
    } else {
      Serial.print("Error code: ");
      Serial.println(httpResponseCode);
    }
    
    // Free resources
    http.end();
  } else {
    Serial.println("WiFi Disconnected");
  }

  // Poll every 5 seconds
  delay(5000);
}
