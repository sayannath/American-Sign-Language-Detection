import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class Classifier {
  Interpreter _interpreter;
  List<String> _labelList;

  Classifier() {
    _loadModel();
    _loadLabel();
  }

  void _loadModel() async {
    _interpreter = await Interpreter.fromAsset('asl.tflite');

    var inputShape = _interpreter.getInputTensor(0).shape;
    var outputShape = _interpreter.getOutputTensor(0).shape;
    print(inputShape);
    print(outputShape);
    print('Load Model - $inputShape / $outputShape');
  }

  void _loadLabel() async {
    final labelData = await rootBundle.loadString('assets/labels.txt');
    final labelList = labelData.split('\n');
    _labelList = labelList;
    print(labelData);
    print(labelList);
    print('Load Label');
  }

  Future<img.Image> loadImage(String imagePath) async {
    var originData = File(imagePath).readAsBytesSync();
    var originImage = img.decodeImage(originData);
    print("Loading Image");
    return originImage;
  }

  Future<dynamic> runModel(img.Image loadImage) async {
    var modelImage = img.copyResize(loadImage, width: 224, height: 224);
    var modelInput = imageToByteListFloat32(modelImage, 224);
    print("Run Model");

    //[1, 29]
    var outputsForPrediction = [List.generate(29, (index) => 0.0)];
    print("Before $outputsForPrediction");
    Stopwatch stopwatch = new Stopwatch()..start();
    _interpreter.run(modelInput.buffer, outputsForPrediction);
    print('Inference Time ${stopwatch.elapsed.inMilliseconds}');
    Map<int, double> map = outputsForPrediction[0].asMap();
    var sortedKeys = map.keys.toList()
      ..sort((k1, k2) => map[k2].compareTo(map[k1]));

    List<dynamic> result = [];
    for (var i = 0; i < 3; i++) {
      result.add({
        'label': _labelList[sortedKeys[i]],
        'value': map[sortedKeys[i]],
      });
    }
    print("Result $result");
    return result;
  }

  //Convert Image to Float32
  Float32List imageToByteListFloat32(img.Image image, int inputSize) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);

        buffer[pixelIndex++] = (img.getRed(pixel) - 128) / 128;
        buffer[pixelIndex++] = (img.getGreen(pixel) - 128) / 128;
        buffer[pixelIndex++] = (img.getBlue(pixel) - 128) / 128;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }
}
