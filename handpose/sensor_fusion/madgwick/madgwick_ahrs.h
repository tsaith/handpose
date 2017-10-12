//=============================================================================================
// MadgwickAHRS.h
//=============================================================================================
//
// Implementation of Madgwick's IMU and AHRS algorithms.
// See: http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/
//
// From the x-io website "Open-source resources available on this website are
// provided under the GNU General Public Licence unless an alternative licence
// is provided in source."
//
// Date			Author          Notes
// 29/09/2011	SOH Madgwick    Initial release
// 02/10/2011	SOH Madgwick	Optimised for reduced CPU load
//
//=============================================================================================
#ifndef MadgwickAHRS_h
#define MadgwickAHRS_h
#include <math.h>
#include <vector>

using std::vector;

//--------------------------------------------------------------------------------------------
// Variable declaration
class Madgwick{
private:
    static float invSqrt(float x);
    float beta;				// algorithm gain
    float q0 = 1.0f;
    float q1 = 0.0f;
    float q2 = 0.0f;
    float q3 = 0.0f;	// quaternion of sensor frame relative to auxiliary frame
	vector<float> quat = {1.0f, 0.0f, 0.0f, 0.0f};
    float invSampleFreq;
    float roll;
    float pitch;
    float yaw;
    char anglesComputed;
    void computeAngles();

//-------------------------------------------------------------------------------------------
// Function declarations
public:
    Madgwick(float samplePeriod, float beta);
    //Madgwick(float samplePeriod, float beta, float q0, float q1, float q2, float q3);
    Madgwick(void);
    void begin(float sampleFrequency) { invSampleFreq = 1.0f / sampleFrequency; }
    void update(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz);
    void updateIMU(float gx, float gy, float gz, float ax, float ay, float az);
    float getRoll() {
        if (!anglesComputed) computeAngles();
        return roll * 57.29578f;
    }
    float getPitch() {
        if (!anglesComputed) computeAngles();
        return pitch * 57.29578f;
    }
    float getYaw() {
        if (!anglesComputed) computeAngles();
        return yaw * 57.29578f + 180.0f;
    }
    float getRollRadians() {
        if (!anglesComputed) computeAngles();
        return roll;
    }
    float getPitchRadians() {
        if (!anglesComputed) computeAngles();
        return pitch;
    }
    float getYawRadians() {
        if (!anglesComputed) computeAngles();
        return yaw;
    }
    float get_q0() { return q0; }
    float get_q1() { return q1; }
    float get_q2() { return q2; }
    float get_q3() { return q3; }

    vector<float> get_quat_array() {
	    quat[0] = q0;
	    quat[1] = q1;
	    quat[2] = q2;
	    quat[3] = q3;
	    return quat;
	}
};
#endif

