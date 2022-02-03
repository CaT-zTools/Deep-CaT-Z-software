
///////////////////////////////////////////////////////////////////////////////////////////////
//                                       GLOBAL VARIABLES                                    //
///////////////////////////////////////////////////////////////////////////////////////////////

// Define 4 input pins and 4 output pins
// Sensors
#define DI_Pin_0 2
#define DI_Pin_1 3
#define DI_Pin_2 4
#define DI_Pin_3 5

// Actuators
#define DO_Pin_0 8
#define DO_Pin_1 9
#define DO_Pin_2 10
#define DO_Pin_3 11

// Maximum number of characters for the input sequence
#define maxChars  25 // ATTENTION: include '\n'

// Global variables
bool stringComplete = false; // Flag
String temp_str;
char inputCommand[  maxChars ]; // Input command from the PC to the Arduino
char outputCommand[ maxChars ]; // Output command to the PC, from the Arduino

// Variables for the communication: PC -> Arduino
char state = '.'; // Behavior state
int pos_X = -1; // Coordinate: X
int pos_Y = -1; // Coordinate: Y
int pos_Z = -1; // Coordinate: Z
int zone_L = -1; // ROI - zone
int keyPressed = -1; // if any key is pressed

// // Variables for the communication: Arduino -> PC
// Define initial states of the input/output
bool DO_0 = LOW; //state of output actuator 0 - OFF
bool DO_1 = LOW; //state of output actuator 1 - OFF
bool DO_2 = LOW; //state of output actuator 2 - OFF
bool DO_3 = LOW; //state of output actuator 3 - OFF

volatile int DI_0 = 0; //state of input sensor 0 - OFF
volatile int DI_1 = 0; //state of input sensor 1 - OFF
volatile int DI_2 = 0; //state of input sensor 2 - OFF
volatile int DI_3 = 0; //state of input sensor 3 - OFF


///////////////////////////////////////////////////////////////////////////////////////////////
//                                        USER DEFINED FUNCTION                              //
///////////////////////////////////////////////////////////////////////////////////////////////

// Define all signals from input command to the output actuators.
void User_Defined_Control_Fun() {
  
  // EXAMPLE:
  // Everytime the input behavioral state is 
  // - "Standstill" (S), the output actuator 0 is turned ON; if not: turn OFF
  // - "Walking" (W), the output actuator 1 is turned ON; if not: turn OFF
  // - "Rearing" (R), the output actuator 2 is turned ON; if not: turn OFF
  // - "Grooming" (G), the output actuator 3 is turned ON; if not: turn OFF
  
  // Actuator 0 - DO_0
  // Define if needed
  if( state == 'S' ) {
    DO_0 = HIGH;
  }
  else {
    DO_0 = LOW;
  }
  
  // Actuator 1 - DO_1 
  if( state == 'W' ) {
    DO_1 = HIGH;
  }
  else {
    DO_1 = LOW;
  }
  
  // Actuator 2 - DO_2  
  if( state == 'R' ) {
    DO_2 = HIGH;
  }
  else {
    DO_2 = LOW;
  }

  // Actuator 3 - DO_3
  // Define if needed
  if( state == 'G' ) {
    DO_3 = HIGH;
  }
  else {
    DO_3 = LOW;
  }
  
}

////////////////////////////
//  !!! DO NOT TOUCH !!!  //
////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////
//                                        S E T U P                                          //
///////////////////////////////////////////////////////////////////////////////////////////////

void setup() {  
  // digital outputs
  pinMode( DO_Pin_0, OUTPUT );
  pinMode( DO_Pin_1, OUTPUT );
  pinMode( DO_Pin_2, OUTPUT );
  pinMode( DO_Pin_3, OUTPUT );
  
  digitalWrite( DO_Pin_0, DO_0 );
  digitalWrite( DO_Pin_1, DO_1 );
  digitalWrite( DO_Pin_2, DO_2 );
  digitalWrite( DO_Pin_3, DO_3 );

  // digital inputs
  pinMode( DI_Pin_0, INPUT  );
  pinMode( DI_Pin_1, INPUT  );
  pinMode( DI_Pin_2, INPUT  );
  pinMode( DI_Pin_3, INPUT  );
  
  attachInterrupt( digitalPinToInterrupt( DI_Pin_0), Callback_DI_Pin_0, RISING);
  attachInterrupt( digitalPinToInterrupt( DI_Pin_1), Callback_DI_Pin_1, RISING);
  attachInterrupt( digitalPinToInterrupt( DI_Pin_2), Callback_DI_Pin_2, RISING);
  attachInterrupt( digitalPinToInterrupt( DI_Pin_3), Callback_DI_Pin_3, RISING);
 
  //Serial.begin(115200);
  Serial.begin(1000000);
  Serial.println( "Ready to go!" );
}

void loop() {

  if( stringComplete ) {
    // GET INFORMATION FROM PC
    sscanf( inputCommand, "%c,%d,%d,%d,%d", &state, &pos_X, &pos_Y, &pos_Z, &zone_L, &keyPressed );
    stringComplete = false;
    
    // CALL USER-DEFINED CONTROL FUNCTION
    User_Defined_Control_Fun();
    digitalWrite( DO_Pin_0, DO_0 );
    digitalWrite( DO_Pin_1, DO_1 );
    digitalWrite( DO_Pin_2, DO_2 );
    digitalWrite( DO_Pin_3, DO_3 );

    // SEND INFORMATION BACK TO PC
    sprintf( outputCommand, "%d,%d,%d,%d,%d,%d,%d,%d", DI_0, DI_1, DI_2, DI_3, DO_0, DO_1, DO_2, DO_3 );
    Serial.println( outputCommand );
	
    // RESET DI BUFFERS
    DI_0 = 0;
    DI_1 = 0; 
    DI_2 = 0;
    DI_3 = 0;   
  }    
}

void serialEvent() { // Read serial port and get stuff into global variables
  if( Serial.available() )
  {
    temp_str = Serial.readStringUntil('\n');
    temp_str.toCharArray(inputCommand, maxChars);

    stringComplete = true;
  }
}

void Callback_DI_Pin_0() {
  DI_0++;
}

void Callback_DI_Pin_1() {
  DI_1++;
}

void Callback_DI_Pin_2() {
  DI_2++;
}

void Callback_DI_Pin_3() {
  DI_3++;
}
