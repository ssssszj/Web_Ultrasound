import * as tf from '@tensorflow/tfjs';
//const cameraFeed = document.getElementById('cameraFeed');
const result = document.getElementById('result');
var model;
var classification_model;
// cameraFeed.width = 350;
// cameraFeed.height = 300;\\

const capturedCanvas1 = document.getElementById('capturedCanvas1');
const ctx1 = capturedCanvas1.getContext('2d');
const capturedCanvas2 = document.getElementById('capturedCanvas2');
const ctx2 = capturedCanvas2.getContext('2d');

const img = new Image();
img.src = './cs.jpg';
var index = 1.2;
// 在图像加载完成后执行绘制操作
img.onload = function() {
    capturedCanvas1.width = img.width / index;
    capturedCanvas1.height = img.height / index;
    //ctx.drawImage(img, 0, 0);
    ctx1.drawImage(img, 0, 0, capturedCanvas1.width, capturedCanvas1.height);
    capturedCanvas2.width = img.width / index;
    capturedCanvas2.height = img.height / index;
    //ctx.drawImage(img, 0, 0);
    ctx2.drawImage(img, 0, 0, capturedCanvas2.width, capturedCanvas2.height);
};
// 初始化
async function init() {

    //model = await tf.loadLayersModel('./resunet_segment_model/model.json');
    model = await tf.loadGraphModel('./segment_model/model.json');
    classification_model = await tf.loadGraphModel('./classification_model/model.json');

    var intervalID = setInterval(predict_segment, 2000);
    console.log(intervalID);
    //predict_segment();
}

async function predict_segment() {
    var example = tf.browser.fromPixels(capturedCanvas1, 3).cast('float32');
    let input = example;
    let input_data = await input.data();
    //console.log(input_data);

    //console.log(cameraFeed.width,cameraFeed.height);
    //console.log(example.shape);

    var width = 128;
    var height = 128;
    var w = example.shape[1];
    var h = example.shape[0];
    example = example.expandDims();

    console.time('classification');
    let class_prediction = await classification_model.predict(example);
    let class_scores = await class_prediction.data();
    let max_score_id = class_scores.indexOf(Math.max(...class_scores));
    let classes = ["others", "ultrasound image"];
    result.innerHTML = classes[max_score_id].toString();
    console.log(classes[max_score_id]);
    console.timeEnd('classification');

    // if(classes[max_score_id] === "others"){
    //     let canvas = document.getElementById('canvasOne');
    //     let ctx = canvas.getContext('2d');
    //     ctx.clearRect(0, 0, canvas.width, canvas.height);
    //     return;
    // }

    console.time('segmentation');
    example = tf.image.resizeNearestNeighbor(example, [256, 256]);
    //prediction
    let prediction = await model.predict(example);
    prediction = prediction.reshape([128, 128, -1])
    prediction = tf.argMax(prediction, -1)
        // prediction = prediction.expandDims(2);
        // prediction = tf.image.resizeNearestNeighbor(prediction,[width,height])
    let prediction_data = await prediction.data();

    // now lets generate the visualization image

    var predicted_result = prediction_data;
    const out_bytes = new Uint8ClampedArray(width * height * 4);

    for (let i = 0; i < width * height; ++i) {
        const j = i * 4;
        out_bytes[j + 0] = (predicted_result[i] > 0 ? 255 : 0);
        out_bytes[j + 1] = (predicted_result[i] > 0 ? 255 : 0);
        out_bytes[j + 2] = (predicted_result[i] > 0 ? 255 : 0);
        out_bytes[j + 3] = 255;
    }


    const out = new ImageData(out_bytes, width, height);
    let canvas = document.getElementById('canvasOne');
    var tmpcanvas = document.createElement('canvas');
    tmpcanvas.width = width;
    tmpcanvas.height = height;
    var tmpctx = tmpcanvas.getContext('2d');
    tmpctx.putImageData(out, 0, 0);



    canvas.width = w;
    canvas.height = h;
    let ctx = canvas.getContext('2d');
    ctx.drawImage(tmpcanvas, 0, 0, w, h);
    console.timeEnd('segmentation');

    console.time('Feature Extraction');
    window.getMat(input, input_data);
    console.timeEnd('Feature Extraction');
}

init();