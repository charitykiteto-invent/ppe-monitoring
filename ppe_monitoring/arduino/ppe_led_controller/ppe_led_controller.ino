// PPE LED controller. Change these constants to match your wiring.
const uint8_t RED_LED_PIN = 8;
const uint8_t BLUE_LED_PIN = 9;
const uint8_t GREEN_LED_PIN = 10;
const uint8_t BUZZER_PIN = 11;
const bool BUZZER_ENABLED = false;
const unsigned long SERIAL_FAILSAFE_MS = 6000;

unsigned long lastValidCommandMs = 0;

void allOff() {
  digitalWrite(RED_LED_PIN, LOW);
  digitalWrite(BLUE_LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);
  if (BUZZER_ENABLED) digitalWrite(BUZZER_PIN, LOW);
}

void setState(uint8_t pin, bool alarm) {
  allOff();
  digitalWrite(pin, HIGH);
  if (BUZZER_ENABLED && alarm) digitalWrite(BUZZER_PIN, HIGH);
}

void setup() {
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BLUE_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  allOff();
  Serial.begin(115200);
  Serial.setTimeout(30);
  lastValidCommandMs = millis();
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "RED") {
      setState(RED_LED_PIN, true); Serial.println("ACK:RED");
    } else if (command == "BLUE") {
      setState(BLUE_LED_PIN, true); Serial.println("ACK:BLUE");
    } else if (command == "GREEN") {
      setState(GREEN_LED_PIN, false); Serial.println("ACK:GREEN");
    } else if (command == "OFF") {
      allOff(); Serial.println("ACK:OFF");
    } else if (command == "PING") {
      Serial.println("PONG");
    } else {
      Serial.println("ERR:UNKNOWN");
      return;
    }
    lastValidCommandMs = millis();
  }
  if (millis() - lastValidCommandMs > SERIAL_FAILSAFE_MS) {
    allOff();
  }
}

