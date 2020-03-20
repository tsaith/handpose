#include <math.h>
#include <vector>
#include <iostream>
#include "madgwick_ahrs.h"


using namespace std;
using std::vector;
 
int main(void) {
  
    float samplePeriod = 0.01; // seconds
    float beta = 0.041;

    Madgwick madgwick(samplePeriod, beta);

    madgwick.update()

    cout << "Start" << endl;
     
    return 0;
}
 
