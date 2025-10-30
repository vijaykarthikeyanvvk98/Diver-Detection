import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.3
import QtQuick.Window 2.15


ApplicationWindow {
    id: root
    visible: true
    color: "black" //"#011026"


    /*background: Rectangle {
              gradient: Gradient {
                  GradientStop {
                      position: 0.00;
                      color: light_dark?"#18396b":"#ffffff";
                  }
                  GradientStop {
                      position: 0.3;
                      color: light_dark?dark_theme:"#ffffff";
                  }
                  GradientStop {
                      position: 1.00;
                      color: light_dark?"black":"#ffffff";
                  }
              }
    }*/
    signal map_image_Captured
    property string dark_theme: "#011026"
    property string light_theme: "#FFFFFF"
    property bool light_dark: true
    property string img_path: ""
    property bool load_not: false
    property string vid_path: ""
    property int threshold_1: 0
    property int threshold_2: 0
    property bool img_vid: false
    property bool front_visible: false
    property bool home_visible: false
    property bool connect_pop_status: false
    property string msg_text: ""
    property string msg_text2: ""
    property string msg_title: ""
    property string filePath: ""
    property color neonGreen: "#39FF14"

    //RGB values
    property int red_value: 0
    property int green_value: 0
    property int blue_value: 0
    property real scaleFactor: 1.5

    //property bool name: value
    //parameters
    property real bright: 10
    property real contrast: 1

    //Filter Parameters
    property int filter_type: 0
    property bool is_filter: false
    property int f_size: 0
    property int spat_x: 0
    property int spat_y: 0
    property int color_dev: 0

    //Labels
    property string filter_size: "Kernel Size:"
    property string spatial_dist1: "\u03C3<sub>x</sub>" + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + ":"
    property string spatial_dist2: "\u03C3<sub>y</sub>" + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + ":"
    property string color_sigma: "\u03C3<sub>color</sub>" + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + ":"

    property string saveDirectory: ""
    property string logFileDir: ""
    property string current_theme_color2: ""
    property int file_type: -1
    property int exp_type: -1

    Component.onCompleted: {
        VideoStreamer.openVideoCamera(0)
    }

    Connections {
        target: liveImageProvider

        function onImageChanged() {
            //capture_ss.visible = true
            //console.log("Image")
            opencvImage.reload()
            //imageRect2.visible=false

        }

        /*function onImageChanged2() {
            capture_ss.visible = true
            opencvImage2.reload()
        }*/
    }

    Connections {
        target: VideoStreamer

        /*function onWriting_success() {
            video_view.open_msg()
        }*/
        function onRectangleChanged()
        {
            /*imageRect2.x= VideoStreamer.rectangle.x* root.width / 1000
            imageRect2.y= VideoStreamer.rectangle.y* root.height / 1000
            imageRect2.width= VideoStreamer.rectangle.width * root.width / 1000
            imageRect2.height= VideoStreamer.rectangle.height* root.height / 1000*/
            //imageRect2.visible=true
            //console.log("VideoStreamer.rectangle.x");
        }

        function onDetectChanged()
        {
            imageRect2.visible=VideoStreamer.detect;

        }
    }

    Rectangle {
        id: imageRect


        /*anchors.horizontalCenter: parent.horizontalCenter
        //anchors.horizontalCenterOffset: -0.1*parent.width
        anchors.verticalCenter: parent.verticalCenter
        anchors.verticalCenterOffset: 0.02 * parent.width
        width: 0.5 * parent.width
        height: 0.7 * parent.height*/
        anchors {
            left: parent.left
            right: parent.right
            top: parent.top
            bottom: parent.bottom
            margins: 0.0025 * parent.width
        }

        //anchors.bottom: parent.bottom
        color: "transparent"
        border.color: "white"
        //border.width: 3
        //visible: front_visible
        z: 0
        Image {
            id: opencvImage
            width:parent.width
            height: parent.height
            anchors.centerIn: parent
            fillMode: Image.NoOption
            property bool counter: false
            visible: true
            //source: "qrc:/resources/images/dummy_template3.jpg" // "image://live/image"
            asynchronous: false
            cache: false

            function reload() {
                counter = !counter
                source = "image://live/image?id=" + counter
            }
        }

        MouseArea {
            anchors.fill: parent

            onClicked: {
                //VideoStreamer.pause_streaming()
                //model.visible = true
            }

            onDoubleClicked: {
                //VideoStreamer.resume_streaming()
                //model.visible = false
            }
        }

        Rectangle {
            id: imageRect2


            /*x: VideoStreamer.rectangle.x* root.width / 1000
            y: VideoStreamer.rectangle.y* root.height / 1000
            width: VideoStreamer.rectangle.width * root.width / 1000
            height: VideoStreamer.rectangle.height* root.height / 1000*/
            x: VideoStreamer.rectangle.x
            y: VideoStreamer.rectangle.y
            width: VideoStreamer.rectangle.width
            height: VideoStreamer.rectangle.height

            // This Behavior block will automatically animate
            // the changes to the 'x' and 'y' properties
            Behavior on x {
                SmoothedAnimation { velocity: 400 }
            }
            Behavior on y {
                SmoothedAnimation { velocity: 400 }
            }
            color:"transparent"
            border.color: "red"
            border.width: 0.0025*root.width
            visible: false
        }
    }



}
