#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <sstream>
#include "libwebsockets.h"
#include <signal.h>
#include "Fusion.h"
#include "mouse.h"

#define SCREEN_WIDTH 1366
#define SCREEN_HIGHT 768

struct  timeval    begin_tv;
struct  timeval    cur_tv;
struct  timeval    end_tv;
int mGyroTime;
int mAccTime;
float dT;
Fusion *mFusion = NULL;
bool mEnabled[NUM_FUSION_MODE];
float mEstimatedGyroRate;
vec4_t mAttitudes[NUM_FUSION_MODE];


using namespace std;
#define MAX_PAYLOAD_SIZE  10 * 1024
#define WS_SERVER "192.168.43.88"
static volatile int exit_sig = 0;
static int reset_flag = 0;
 
void sighdl( int sig ) {
    lwsl_notice( "%d traped", sig );
    exit_sig = 1;
}
 
struct session_data {
    int msg_count;
    unsigned char buf[LWS_PRE + MAX_PAYLOAD_SIZE];
    int len;
};

void getRotationMatrixFromVector(float* R, vec4_t q) {
    float q0 = q.w;
    float q1 = q.x;
    float q2 = q.y;
    float q3 = q.z;
    float sq_q1 = 2 * q1 * q1;
    float sq_q2 = 2 * q2 * q2;
    float sq_q3 = 2 * q3 * q3;
    float q1_q2 = 2 * q1 * q2;
    float q3_q0 = 2 * q3 * q0;
    float q1_q3 = 2 * q1 * q3;
    float q2_q0 = 2 * q2 * q0;
    float q2_q3 = 2 * q2 * q3;
    float q1_q0 = 2 * q1 * q0;
    R[0] = 1 - sq_q2 - sq_q3;
    R[1] = q1_q2 - q3_q0;
    R[2] = q1_q3 + q2_q0;
    R[3] = 0.0f;
    R[4] = q1_q2 + q3_q0;
    R[5] = 1 - sq_q1 - sq_q3;
    R[6] = q2_q3 - q1_q0;
    R[7] = 0.0f;
    R[8] = q1_q3 - q2_q0;
    R[9] = q2_q3 + q1_q0;
    R[10] = 1 - sq_q1 - sq_q2;
    R[11] = 0.0f;
    R[12] = R[13] = R[14] = 0.0f;
    R[15] = 1.0f;
}

 
int connected = 0;
int callback( struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len ) {
    struct session_data *data = (struct session_data *) user;
    switch ( reason ) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED: {  // 连接到服务器后的回调
            lwsl_notice( "Connected to server\n" );
            break;
        }
 
        case LWS_CALLBACK_CLIENT_RECEIVE: {
            string input = (char *) in;
            istringstream ss(input);
            string token;
            vec3_t accData;
            vec3_t magData;
            vec3_t gyroData;
            float rotationMatrix[16];
            int leftClick = 0;
            int rightClick = 0;
            int resetClick = 0;
            int i = 0;
            //436f6479,watch_imu,2175,-0.02566,-0.08788,0.96874,-0.00143,-0.00159,-0.00052,
            while(std::getline(ss, token, ',')) {
                i++;
                try {
                    if (i>6 & i<10) {
                        gyroData[i-7] = stof(token);
                    }else if (i>3 & i <7){
                        accData[i-4] = stof(token);
                    }else if (i>9 & i <13){
                        magData[i-10] = stof(token);
                    }else if (i == 13){
                        leftClick = stoi(token);
                    }else if (i == 14){
                        rightClick = stoi(token);
                    }else if (i == 15){
                        resetClick = stoi(token);
                    }
                }catch(...) {
                    std::cout << token << '\n';
                    exit_sig = 1;
                    reset_flag = 1;
                }
            }
            if (leftClick==1 || rightClick==1)
            	printf("left:[%d],right=[%d]\n",leftClick,rightClick);
            if (resetClick == 1) {
                resetClick = 0;
                reset_flag = 1;
                printf("reset\n");
                break;
            }
            float dT = 20000 / 1000000.0f;
            const float freq = 1 / dT;
            mFusion->handleGyro(gyroData, dT);
            mFusion->handleAcc(accData, dT);
            //mFusion->handleMag(magData);
            if (mFusion->hasEstimate()) {
                const vec4_t q(mFusion->getAttitude());
                getRotationMatrixFromVector(rotationMatrix, q);
                //lwsl_notice("%2f,%2f,%2f,%2f\n",q.x,q.y,q.z,q.w);
                lwsl_notice("\n 0[%2.3f]\n 1[%1.3f]\n 2[%1.3f]\n 3[%1.3f]\n 4[%1.3f]\n 5[%1.3f]\n 6[%1.3f]\n 7[%1.3f]\n 8[%1.3f]\n 9[%1.3f]\n10[%1.3f]\n",                        rotationMatrix[0], rotationMatrix[1], rotationMatrix[2], rotationMatrix[3], rotationMatrix[4], rotationMatrix[5],rotationMatrix[6], rotationMatrix[7], rotationMatrix[8], rotationMatrix[9], rotationMatrix[10]);
            }
#define SCREEN_WIDTH 1366
#define SCREEN_HIGHT 768
            int x = (SCREEN_WIDTH/2) - (SCREEN_WIDTH * rotationMatrix[0]);
            int y = (SCREEN_HIGHT/2) - (SCREEN_HIGHT * rotationMatrix[9]);
            x =(x<0)?0:x;
            x =(x>SCREEN_WIDTH)?SCREEN_WIDTH:x;
            y =(y<0)?0:y;
            y =(y>SCREEN_HIGHT)?SCREEN_HIGHT:y;

            move_mouse_pointer(x,y);
            if (leftClick)
                mouseClick(Button1);
            if (rightClick)
                mouseClick(Button3);
            //printf("x:[%d], y[%d]\n",x,y);
            //lwsl_notice( "Rx: %s\n", in );
            //lwsl_notice("%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,%1.5f,\n",accData.x,accData.y,accData.z,gyroData.x,gyroData.y,gyroData.z,magData.x,magData.y,magData.z);
            break;
        }
        case LWS_CALLBACK_CLIENT_WRITEABLE:  {   // 当此客户端可以发送数据时的回调
            lwsl_notice("%s, %d\n",__func__,__LINE__);
            if ( data->msg_count < 3 ) {
                // 前面LWS_PRE个字节必须留给LWS
                memset( data->buf, 0, sizeof( data->buf ));
                char *msg = (char *) &data->buf[ LWS_PRE ];
                data->len = sprintf( msg, "==pos==");
                lwsl_notice( "Tx: %s\n", msg );
                // 通过WebSocket发送文本消息
                lws_write( wsi, &data->buf[ LWS_PRE ], data->len, LWS_WRITE_TEXT );
                connected = 1;
            }
            break;
        }
    }
    return 0;
}
 
struct lws_protocols protocols[] = {
    {
        //协议名称，协议回调，接收缓冲区大小
        "", callback, sizeof( struct session_data ), MAX_PAYLOAD_SIZE,
    },
    {
        NULL, NULL,   0 // 最后一个元素固定为此格式
    }
};

int main(int argc, char const *argv[]) {
    // 信号处理函数
    signal( SIGTERM, sighdl );
 
    // 用于创建vhost或者context的参数
    struct lws_context_creation_info ctx_info = { 0 };
    ctx_info.port = CONTEXT_PORT_NO_LISTEN;
    ctx_info.iface = NULL;
    ctx_info.protocols = protocols;
    ctx_info.gid = -1;
    ctx_info.uid = -1;
 
    // 创建一个WebSocket处理器
    struct lws_context *context = lws_create_context( &ctx_info );
 
    int port =5000;
    char addr_port[256] = { 0 };
    sprintf( addr_port, "%s:%u", WS_SERVER, port & 65535 );
 
    // 客户端连接参数
    struct lws_client_connect_info conn_info = { 0 };
    conn_info.context = context;
    conn_info.address = WS_SERVER;
    conn_info.port = port;
    conn_info.ssl_connection = 0;
    conn_info.path = "/ws";
    conn_info.host = addr_port;
    conn_info.origin = addr_port;
    conn_info.protocol = protocols[ 0 ].name;
    mFusion = new Fusion();
    mFusion->init(FUSION_NOMAG); // normal, no_mag, no_gyro
    //mFusion->init(FUSION_9AXIS); // normal, no_mag, no_gyro
    mEnabled[FUSION_9AXIS] = true;
    mEnabled[FUSION_NOMAG] = true;
    mEnabled[FUSION_NOGYRO] = true;

    gettimeofday(&begin_tv, NULL);
    gettimeofday(&cur_tv, NULL);
    mGyroTime = cur_tv.tv_usec;
    mAccTime = cur_tv.tv_usec;

    GetGlobalMousePosition();  //get the global position of the pointer

    sleep(1);
    struct lws *wsi = lws_client_connect_via_info( &conn_info );
    while ( !exit_sig ) {
        // 执行一次事件循环（Poll），最长等待1000毫秒
        lws_service( context, 1000 );
        /**
         * 下面的调用的意义是：当连接可以接受新数据时，触发一次WRITEABLE事件回调
         * 当连接正在后台发送数据时，它不能接受新的数据写入请求，所有WRITEABLE事件回调不会执行
         */
        if (connected == 0) {
            lws_callback_on_writable( wsi );
        }
        if (reset_flag == 1) {
            reset_flag = 0;
            delete mFusion;
            mFusion = NULL; 
            mFusion = new Fusion();
            printf("Fusion reset\n");
            mFusion->init(FUSION_NOMAG); // normal, no_mag, no_gyro
        }
    }
    // 销毁上下文对象
    lws_context_destroy( context );
 
    gettimeofday(&end_tv, NULL);
	//printf("%2f,%2f,%2f,%2f\n",fuse.get_q0(),fuse.get_q1(),fuse.get_q2(),fuse.get_q3()); 
	printf("%ld,%ld\n",begin_tv.tv_sec,end_tv.tv_sec);
	return 0;
}
