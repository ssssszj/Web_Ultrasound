import * as tf from '@tensorflow/tfjs';
import * as THREE from 'three';
const {XRWebGLLayer} = window;
const cameraFeed = document.getElementById('cameraFeed');
const capturedCanvas = document.getElementById('capturedCanvas');
const result = document.getElementById('result');

var model;
var classification_model;

let camera, scene, renderer, xrSession, videoTexture;
let xrReferenceSpace;

// cameraFeed.width = 350;
// cameraFeed.height = 300;

/*
//指定图片在canvas上显示
function showCanvas(dataUrl) {
    console.info(dataUrl);
    var canvas_ = document.getElementById('canvasOne');
    var ctx_ = canvas_.getContext('2d');
    //加载图片
    var img = new Image();
    img.onload = function() {
        ctx_.drawImage(img, 0, 0, canvas_.width, canvas_.height);
    }
    img.src = dataUrl;
    // document.body.appendChild(img);
}

function showCanvas2(dataUrl){
    console.info(dataUrl);
    var canvas_ = document.getElementById('canvasTwo');
    var ctx_ = canvas_.getContext('2d');
    //加载图片
    var img = new Image();
    img.onload = function() {
        ctx_.drawImage(img, 0, 0, canvas_.width, canvas_.height);
    }
    img.src = dataUrl;
}
*/
/*async function Getap(){
    var Area = document.getElementById("Area");
    var Perimeter = document.getElementById("Perimeter");
    var BA = document.getElementById("BA");
    var BP = document.getElementById("BP");
    var ConC = document.getElementById("ConC");

    var ret = eel.get_features()();
    let result = await ret;
    console.log(result);
    Area.innerHTML = result[0].toFixed(2); 
    Perimeter.innerHTML = result[1].toFixed(2);
    BP.innerHTML = result[2].toFixed(2)*100+'%';
    BA.innerHTML = result[3].toFixed(2);
    ConC.innerHTML = result[4].toFixed(2);
}*/

// 拍照并保存未标记图像到本地
// captureButton.addEventListener('click', async() => {
//     //eel.clear_downloads()();
//     capturedCanvas.width = 350;
//     capturedCanvas.height = 300;
//     capturedCanvas.getContext('2d').drawImage(cameraFeed, 0, 0, capturedCanvas.width, capturedCanvas.height);

//     // 将图像保存为链接
//     const capturedImageURL = capturedCanvas.toDataURL('image/png');
//     downloadLink.href = capturedImageURL;
//     downloadLink.style.display = 'block';

//     // var input_img = new Image();
//     // input_img.src = capturedImageURL;
//     // const res = await runner.predict(input_img);
//     // var output_canvas = document.getElementById('canvasOne');
//     // var output_ctx = output_canvas.getContext('2d');
//     // output_ctx.drawImage(res, 0, 0, output_canvas.width, output_canvas.height);
//     // 自动下载图像
//     //downloadLink.click();
//     //图片地址：C:\Users\杜佳琪\Desktop\weblab\webdemo\react_basic\camera\my-app\public\original_img.png
//     //2. 调用函数 model_predict
//     //eel.model_predict()();
//     //经过图像分割预测模型以及mask.py程序之后得到标记后的图像为./res.png
//     //读取res.png图像显示在web网页上
//     //Getap();
//     //showCanvas('./added_prediction/original_img.png') //3. 这里添加的是mask.py输出的图像位置；
//     //showCanvas2('./black/black_img.png')
// });

// async function loadOpenCV() {
//     return new Promise((resolve) => {
//         global.Module = {
//             onRuntimeInitialized: resolve
//         };
//         global.cv = require('./opencv.js');
//     });
// }



// 初始化
async function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera()
    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Setup XR
    xrSession = await navigator.xr.requestSession('immersive-vr',{
        requiredFeatures: ['local', 'viewer']
    });

    // Setup XRLayer
    const xrLayer = new XRWebGLLayer(xrSession, renderer);
    renderer.xr.setCompatibleXRDevice(xrLayer);

    // Setup XR Ref Space
    xrReferenceSpace = await xrSession.requestReferenceSpace('local');
    const xrViewerSpace = await xrSession.requestReferenceSpace('viewer');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {facingMode:{ exact: "environment" }}, // 后置摄像头
            //video: {facingMode: "user"}, // 前置摄像头
        });
        //console.log(stream.getVideoTracks()[0].getSettings().width,stream.getVideoTracks()[0].getSettings().height);
        cameraFeed.width = stream.getVideoTracks()[0].getSettings().width*0.8;
        cameraFeed.height = stream.getVideoTracks()[0].getSettings().height*0.8;
        cameraFeed.srcObject = stream;
        cameraFeed.play();
    } catch (error) {
        console.error('Error accessing the camera:', error);
    }

    videoTexture = new THREE.VideoTexture(cameraFeed);
    const geometry = new THREE.PlaneGeometry(1,1);
    const material = new THREE.MeshBasicMaterial({side: THREE.DoubleSide, map: videoTexture});
    const plane = new THREE.Mesh(geometry, material);
    scene.add(plane);

    renderer.setAnimationLoop(render);

    model = await tf.loadGraphModel('./segment_model/model.json');
    classification_model = await tf.loadGraphModel('./classification_model/model.json');

    console.log(tf.getBackend());
    // var intervalID = setInterval(predict_segment,2000);
    // console.log(intervalID);

}

function render(time,xrFrame){
    time *= 0.001;

    videoTexture.needsUpdate = true;

    const viwerPose = xrFrame.getViewerPose(xrReferenceSpace);
}

async function predict_segment(){
    capturedCanvas.width = cameraFeed.width;
    capturedCanvas.height = cameraFeed.height;
    capturedCanvas.getContext('2d').drawImage(cameraFeed, 0, 0, capturedCanvas.width, capturedCanvas.height);
    var example = tf.browser.fromPixels(capturedCanvas,3).cast('float32');
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

    // *-----classification-----*
    console.time('classification');
    let class_prediction = await classification_model.predict(example);
    let class_scores = await class_prediction.data();
    let max_score_id = class_scores.indexOf(Math.max(...class_scores));
    let classes = ["others","ultrasound image"];
    result.innerHTML = classes[max_score_id].toString();
    console.log(classes[max_score_id]);
    console.timeEnd('classification');

    if(classes[max_score_id] === "others"){
        let canvas = document.getElementById('canvasOne');
        let ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const outputcanvas = document.getElementById('outputcanvas');
        if(outputcanvas){
            outputcanvas.remove();
        }
        return;
    }

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
    window.getMat(input,input_data);
    console.timeEnd('Feature Extraction');
}

init();
