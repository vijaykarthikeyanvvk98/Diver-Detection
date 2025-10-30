#ifndef VIDEOSTREAMER_H
#define VIDEOSTREAMER_H

#include <QDateTime>
#include <QImage>
#include <QObject>
#include <QThread>
#include <QTimer>
#include <QVariant>
#include "qthread.h"
#include <deque>
#include <opencv2/highgui.hpp>
#include <vector>
#include <qthread.h>

#include <QImage>
#include <QObject>
#include <QThread>
#include "qthread.h"
#include <opencv2/core.hpp> // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp> // Video write
#include <opencv2/dnn.hpp>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>


using namespace cv;
static cv::VideoCapture cap;
using namespace std;

// Detection struct
struct ServoPLL {
    int servo_center = 1500;   // neutral microseconds
    int servo_signal = 1500;   // current output
    float Kp_area = 0.05f;     // proportional gain for area
    float Kp_pos = 0.05f;      // proportional gain for position
    float deadband_area = 500.0f;
    float deadband_pos = 10.0f;

    Rect referenceRect;         // target lock reference
    bool initialized = false;

    void setReference(const Rect& ref);

    int update(const Rect& detected, Size frameSize);

    /*void drawReference(Mat& frame) {
        if (initialized) {
            rectangle(frame, referenceRect, Scalar(0, 255, 255), 2);
            putText(frame, "Reference", Point(referenceRect.x, referenceRect.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
        }
    }*/
};

struct Detection {
    int class_id;
    float confidence;
    Rect box;
};
class KalmanBoxTracker
{
public:
    cv::KalmanFilter kf;
    cv::Mat state;
    KalmanBoxTracker(cv::Rect2f bbox);
    cv::Rect2f predict();
    void update(cv::Rect2f bbox);
};


struct Track
{
    int id;
    cv::Rect2f bbox;
    float conf;
    int last_seen;
    KalmanBoxTracker kalman;
    int hits;
    int age;
    Track(); // <-- ADD THIS LINE
    Track(int tid, cv::Rect2f b, float c, int frame_idx);
};
class IOUTracker
{
public:
    std::map<int, Track> tracks;
    int next_id = 1;
    float iou_thresh = 0.3f;
    int max_age = 30;

    void update(std::vector<std::pair<cv::Rect2f,float>>& detections, int frame_idx);
    std::vector<Track> get_active_tracks();
};

class VideoStreamer: public QObject
{
    Q_OBJECT
    Q_PROPERTY(QRectF rectangle READ rectangle NOTIFY rectangleChanged)
    Q_PROPERTY(bool detect READ detect NOTIFY detectChanged)

public:
    explicit VideoStreamer(QObject *parent = nullptr);
    ~VideoStreamer();
    QRectF rectangle() const;
    void setRectangle(const QRectF &rect);
    bool detect();
    void setdetect(bool);
    bool nearlyEqual(float a, float b, float epsilon = 1e-6f);

public:
    void streamVideo();
    QThread* threadStreamer;
    void catchFrame(cv::Mat emittedFrame);
    void draw_label(Mat& input_image, String label, int left, int top, int right, int bottom);
    // Post-processing for [1, 5, 8400] output
    vector<Detection> post_process(Mat& frame, Mat& output, Size original_size);
    void grabFrame();
    void catchFrame2(cv::Mat emittedFrame);
    QThread *threadStreamer2;
    void grabImage();
    void grabImage2();
    QRectF m_rectangle;
    bool isdetected=false;
    IOUTracker iouTracker;
    int frame_idx = 0;

public slots:
    void openVideoCamera(QString path);
    void streamerThreadSlot();


private:
    Mat simg;
    Mat sharpened;
    Mat blurred;
    cv::Mat frame, frame2, imageContrastHigh4, drawing, lowContrastMask;
    QTimer tUpdate,*timer;
    int FPS_count=0;
    double fps;
    qint64 lastTimeStamp;
    qint64 currentTime=0, elapsedTime;
    QString FPS;
    int thickness = 4;
    int frame_width = 0;
    int frame_height = 0;

    // Rotation angle in degrees
    double angle = 15.0;

    QString m_canny1 = "80";

    QString m_canny2 = "240";

    int enhance_value = 1;
    std::atomic_bool is_processing=false;
    Mutex mutex1,mutex2,mutex3,mutex4;
    float display_fps=0.0;
    const double Kp_x = 2.0; // horizontal gain
    const double Kp_y = 2.0; // vertical gain
    const double DEADZONE = 10.0;      // pixels
    int servo_center = 1500;  // neutral
    int servo_signal = 1500;  // output
signals:
    void newImage(QImage &);
    void newImage2(QImage &);
    void emitThreadImage(cv::Mat frameThread);
    void emit_dimensions();
    void emitThreadImage2(cv::Mat frameThread);
    void interrupt_request();
    void rectangleChanged();
    void rectangleUpdated(const QRectF &rect);
    void isdetect(bool);
    void detectChanged();
};





class FPSMeter
{
    std::chrono::steady_clock::time_point last;
    std::deque<double> times;
    int maxlen;
public:
    FPSMeter(int avg_over=30);
    void tick();
    double fps();
};


#endif // VIDEOSTREAMER_H
