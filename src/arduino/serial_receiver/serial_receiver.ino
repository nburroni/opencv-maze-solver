#include <DCMotor.h>
#include <VirtualWire.h>

int cnst = 5;
char buf[5] = "";

DCMotor motor0(M0_EN, M0_D0, M0_D1);
DCMotor motor1(M1_EN, M1_D0, M1_D1);

const int moveSpeed = 140;
const int turnSpeed = 130;

const int waitTime = 200;
const int waitTimeTurn = 75;

boolean hasTurned = false;

void setup() {
  // initialize serial communication:
  Serial.begin(9600);
  vw_setup(2000);
  vw_set_rx_pin(16);
  vw_rx_start();
  pinMode(LED_BUILTIN, OUTPUT);


}

void loop() {
  char value = receive1();

  if(value == 'f'){
        digitalWrite(LED_BUILTIN, HIGH);

    moveFw();
  } else if(value == 'r'){
        digitalWrite(LED_BUILTIN, HIGH);

    turnRight();
  } else if(value == 'l'){
        digitalWrite(LED_BUILTIN, HIGH);

    turnLeft();
  } 

  stop1();
}
char receive1(){
  Serial.begin(9600);
  vw_setup(2000);
  vw_set_rx_pin(16);
  vw_rx_start();
  byte message[VW_MAX_MESSAGE_LEN];
  byte messageLength = (byte) 1;
  if (vw_get_message(message, &messageLength)) {
  //      digitalWrite(LED_BUILTIN, HIGH);
        digitalWrite(LED_BUILTIN, HIGH);



  }
  else{
//    digitalWrite(LED_BUILTIN, LOW);
  }
  return message[0];
}

long microsecondsToCentimeters(long microseconds)
{
  // The speed of sound is 340 m/s or 29 microseconds per centimeter.
  // The ping travels out and back, so to find the distance of the
  // object we take half of the distance travelled.
  return microseconds / 29 / 2;
}

void moveFw() {
  motor0.setSpeed(moveSpeed);
  motor1.setSpeed(moveSpeed);
  delay(waitTime); //waits
}

void moveBw() {
  motor0.setSpeed(-moveSpeed);
  motor1.setSpeed(-moveSpeed);
  delay(waitTime); //waits
}

void turnRight() {
  motor0.setClockwise(false);
  motor0.setSpeed(turnSpeed);
  motor1.setSpeed(turnSpeed);
  delay(waitTimeTurn); //waits
  motor0.setClockwise(true);
}

void turnLeft() {
  motor1.setClockwise(false);
  motor0.setSpeed(turnSpeed);
  motor1.setSpeed(turnSpeed);
  delay(waitTimeTurn); //waits
  motor1.setClockwise(true);

}
void stop1(){
  motor0.setSpeed(0);
  motor1.setSpeed(0);
  delay(1250);
  digitalWrite(LED_BUILTIN, LOW);
  delay(1250);
  
}
