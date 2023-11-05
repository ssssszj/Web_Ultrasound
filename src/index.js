import * as tf from '@tensorflow/tfjs';

const cameraFeed = document.getElementById('cameraFeed');
const capturedCanvas = document.createElement('canvas');
capturedCanvas.id = 'capturedCanvas';
let canvas1 = document.getElementById('canvasOne');
// const fullscreen = document.getElementById('fullscreen');

var model;
var classification_model;

// 初始化
async function init() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {facingMode:{ exact: "environment" }}, // 后置摄像头
            // video: {facingMode: "user"}, // 前置摄像头
        });
        //console.log(stream.getVideoTracks()[0].getSettings().width,stream.getVideoTracks()[0].getSettings().height);
        cameraFeed.width = stream.getVideoTracks()[0].getSettings().width*2.0;
        cameraFeed.height = stream.getVideoTracks()[0].getSettings().height*2.0;
        cameraFeed.srcObject = stream;
    } catch (error) {
        console.error('Error accessing the camera:', error);
    }


    model = await tf.loadGraphModel('./segment_model/model.json');
    classification_model = await tf.loadGraphModel('./classification_model/model.json');

    console.log(tf.getBackend());
    var intervalID = setInterval(predict_segment,2000);
    console.log(intervalID);
    // predict_segment();
}



async function predict_segment(){
    let ctx1 = canvas1.getContext('2d');
    //ctx1.clearRect(0, 0, canvas1.width, canvas1.height);

    capturedCanvas.width = cameraFeed.height;
    capturedCanvas.height = cameraFeed.width;
    let capturectx = capturedCanvas.getContext('2d');
    capturectx.save();
    capturectx.translate(capturedCanvas.width/2,capturedCanvas.height/2);
    capturectx.rotate(270*Math.PI/180);
    capturectx.drawImage(cameraFeed, -cameraFeed.width/2, -cameraFeed.height/2, cameraFeed.width, cameraFeed.height);
    capturectx.restore();
    var example = tf.browser.fromPixels(capturedCanvas,3).cast('float32');

    //rotate reverse clockwise 90 degree(example)
    let input_canvas = document.createElement('canvas');
    input_canvas.width = cameraFeed.width;
    input_canvas.height = cameraFeed.height;
    let input_ctx = input_canvas.getContext('2d');
    input_ctx.drawImage(cameraFeed, 0, 0, input_canvas.width, input_canvas.height);
    let input = tf.browser.fromPixels(input_canvas,3).cast('float32');
    let input_data = await input.data();
    //console.log(input_data);

    //console.log(cameraFeed.width,cameraFeed.height);
    //console.log(example.shape);

    var width = 128; 
    var height = 128;
    console.log(example.shape);
    example = example.expandDims();

    // *-----classification-----*
    // console.time('classification');
    // let class_prediction = await classification_model.predict(example);
    // let class_scores = await class_prediction.data();
    // let max_score_id = class_scores.indexOf(Math.max(...class_scores));
    // let classes = ["others","ultrasound image"];
    // console.log(classes[max_score_id]);
    // console.timeEnd('classification');

    // if(classes[max_score_id] === "others"){
    //     let canvas = document.getElementById('canvasOne');
    //     let ctx = canvas.getContext('2d');
    //     ctx.clearRect(0, 0, canvas.width, canvas.height);
    //     const outputcanvas = document.getElementById('outputcanvas');
    //     if(outputcanvas){
    //         outputcanvas.remove();
    //     }
    //     return;
    // }

    console.time('segmentation');
    example = tf.image.resizeNearestNeighbor(example, [256, 256]);

    // *-----prediction-----*
    let prediction = await model.predict(example);
    prediction = prediction.reshape([128,128,-1])
    prediction = tf.argMax (prediction, -1)
    // prediction = prediction.expandDims(2);
    // prediction = tf.image.resizeNearestNeighbor(prediction,[width,height])
    let prediction_data = await prediction.data();

    // now lets generate the visualization image

    var predicted_result = prediction_data;
    const out_bytes = new Uint8ClampedArray(width * height * 4);

    for (let i = 0; i < width * height; ++i) {
        const j = i * 4;
        out_bytes[j + 0] = (predicted_result[i]>0 ? 255 : 0);
        out_bytes[j + 1] = (predicted_result[i]>0 ? 255 : 0);
        out_bytes[j + 2] = (predicted_result[i]>0 ? 255 : 0);
        out_bytes[j + 3] = 255;
    }


    const out = new ImageData(out_bytes, width, height);
    var tmpcanvas = document.createElement('canvas');
    tmpcanvas.width = height;
    tmpcanvas.height = width;
    var tmpctx = tmpcanvas.getContext('2d');
    tmpctx.putImageData(out, 0, 0);
    //rotate clockwise 90 degree(tmpcanvas)

    let canvas = document.createElement('canvas');
    let ctxc = canvas.getContext('2d');
    canvas.width = tmpcanvas.width;
    canvas.height = tmpcanvas.height;
    ctxc.save();
    ctxc.translate(canvas.width/2,canvas.height/2);
    ctxc.rotate(90*Math.PI/180);
    ctxc.drawImage(tmpcanvas, -tmpcanvas.width/2,-tmpcanvas.height/2, tmpcanvas.width, tmpcanvas.height);
    ctxc.restore();
    console.log(canvas.width,canvas.height);

    // canvas1.width = cameraFeed.width;
    // canvas1.height = cameraFeed.height;
    // ctx1.drawImage(canvas, 0, 0, canvas1.width, canvas1.height);
    console.timeEnd('segmentation');

    // console.time('Feature Extraction');
    window.getMat(input,input_data,canvas);
    // console.timeEnd('Feature Extraction');
}

init();