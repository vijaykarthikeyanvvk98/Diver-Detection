#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <qdebug.h>
#include <qlogging.h>
#include <vector>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QWindow>
#include "opencvimageprovider.h"
#include "videostreamer.h"
#include <opencv2/opencv.hpp>
#include <QObject>

/*using namespace cv;
using namespace dnn;
using namespace std;

// ----------- CONFIG -------------
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
const float CONFIDENCE_THRESHOLD = 0.25f;
const string MODEL_PATH = "best.onnx";
const string VIDEO_PATH = "C:/Users/vijay/Videos/test2.mp4";
// --------------------------------

// Detection struct

struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

// Function to draw a box and label
void draw_label(Mat& input_image, string label, int left, int top, int right, int bottom)
{
    rectangle(input_image, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
    int baseLine;
    Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, label_size.height);
    rectangle(input_image, Point(left, top - label_size.height),
              Point(left + label_size.width, top + baseLine),
              Scalar(0, 255, 0), FILLED);
    putText(input_image, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
}

// Post-processing for [1, 5, 8400] output
vector<Detection> post_process(Mat& frame, Mat& output, Size original_size)
{
    vector<Detection> detections;
    const int rows = output.size[2];

    float x_factor = (float)original_size.width / INPUT_WIDTH;
    float y_factor = (float)original_size.height / INPUT_HEIGHT;

    for (int i = 0; i < rows; ++i)
    {
        float x_center = output.at<float>(0, 0, i);
        float y_center = output.at<float>(0, 1, i);
        float w = output.at<float>(0, 2, i);
        float h = output.at<float>(0, 3, i);
        float conf = output.at<float>(0, 4, i);

        if (conf >= CONFIDENCE_THRESHOLD)
        {
            int left = int((x_center - 0.5f * w) * x_factor);
            int top = int((y_center - 0.5f * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            Detection det;
            det.class_id = 0;  // single class
            det.confidence = conf;
            det.box = Rect(left, top, width, height);
            detections.push_back(det);
        }
    }

    // Apply Non-Maximum Suppression
    vector<int> indices;
    vector<Rect> boxes;
    vector<float> confidences;

    for (auto& det : detections) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }

    dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    vector<Detection> nms_detections;
    for (int idx : indices) {
        nms_detections.push_back(detections[idx]);
    }

    return nms_detections;
}
std::vector<Mat> outputs={};
std::vector<String> names={};*/

int main(int argc, char *argv[])
{
    /*std::cout << "CUDA available: " << cv::cuda::getCudaEnabledDeviceCount() << "\n";
    std::cout << "OpenCV CUDA version: " << CV_VERSION << "\n";

    std::cout << cv::getBuildInformation() << std::endl;

    //_putenv("OPENCV_DNN_CUDA_DISABLE_CUDNN=1");
    //std::cout << "Disable cuDNN = " << getenv("OPENCV_DNN_CUDA_DISABLE_CUDNN") << std::endl;

    cv::cuda::setDevice(0);
    //cv::cuda::resetDevice(); // then setDevice again if needed
    //cv::cuda::setDevice(0);

    // Load model
    Net net = readNetFromONNX(MODEL_PATH);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Open video
    VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        cerr << "❌ Error opening video file: " << VIDEO_PATH << endl;
        return -1;
    }

    Mat frame;
    while (cap.read(frame))
    {
        Mat blob;
        blobFromImage(frame, blob, 1.0 / 255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

        net.setInput(blob);
        names= net.getUnconnectedOutLayersNames();
        //qDebug()<<names;
        std::vector<cv::Mat> _outputs={};
        //net.forward(net.getUnconnectedOutLayersNames());

        /*Mat output = outputs[0];  // [1, 5, 8400]

        vector<Detection> detections = post_process(frame, output, frame.size());

        for (auto& det : detections)
        {
            draw_label(frame, format("object %.2f", det.confidence),
                       det.box.x, det.box.y,
                       det.box.x + det.box.width, det.box.y + det.box.height);
        }*/

        /*imshow("YOLOv8 Detection", frame);
        if (waitKey(1) == 27) break; // ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;*/
    QApplication app(argc, argv);
    app.setOrganizationName("Vikra");
    app.setOrganizationDomain("votpl.com");
    qRegisterMetaType<cv::Mat>("cv::Mat");

    QQmlApplicationEngine engine;

    // std::cout << cv::getBuildInformation() << std::endl;
    VideoStreamer videoStreamer;

    OpencvImageProvider *liveImageProvider(new OpencvImageProvider);

    engine.rootContext()->setContextProperty("VideoStreamer", &videoStreamer);

    engine.rootContext()->setContextProperty("liveImageProvider", liveImageProvider);

    engine.addImageProvider("live", liveImageProvider);

    const QUrl url(QStringLiteral("qrc:/Resources/test.qml"));

    QObject::connect(&videoStreamer,
                     &VideoStreamer::newImage,
                     liveImageProvider,
                     &OpencvImageProvider::updateImage);

    engine.load(url);
    QObject *rootObject = engine.rootObjects().first();
    QWindow *window = qobject_cast<QWindow *>(rootObject);

    if (window) {
        window->setTitle("SEASIGHT OVERLAY & VIDEO RECORDER");

        window->showMaximized();
        window->setMinimumSize(QSize(670, 470));
    }

    return app.exec();
}
