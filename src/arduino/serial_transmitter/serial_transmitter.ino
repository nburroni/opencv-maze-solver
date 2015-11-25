/*
 * Código cargado en el arduino para transmitir la señal rc recibida por el Serial Port
 */

#include <VirtualWire.h>

char buffer[2] = "";

void setup(){
  Serial.begin(9600);
  vw_setup(2000);
  vw_set_tx_pin(4);
}

void loop(){
  if(Serial.available() > 0) {
    Serial.readBytes(buffer, 2);
  }
  Serial.print(&buffer[0]);
  //char c = char(buffer[0]);
  send(&buffer[0]);
  //send("f");
}
void send (char *message){
  vw_send((uint8_t *)message, strlen(message));
  vw_wait_tx(); // Wait until the whole message is gone
}
