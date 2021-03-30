import 'package:asl_app/services/classifier_service.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class ClassifierPage extends StatefulWidget {
  final CameraDescription camera;

  const ClassifierPage({Key key, this.camera}) : super(key: key);

  @override
  _ClassifierPageState createState() => _ClassifierPageState();
}

class _ClassifierPageState extends State<ClassifierPage> {
  Classifier _classifier;
  CameraController _controller;
  Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _classifier = Classifier();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0.0,
        title: Text(
          'American Sign Language TF-Lite',
          style: TextStyle(
            color: Colors.black,
          ),
        ),
      ),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Column(
              children: [
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(10),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: CameraPreview(_controller),
                    ),
                  ),
                ),
                Container(
                  height: 200,
                  child: IconButton(
                    iconSize: 60,
                    onPressed: () {
                      _runModel(context);
                    },
                    icon: Icon(Icons.camera),
                  ),
                ),
              ],
            );
          } else {
            return Center(
              child: CircularProgressIndicator(),
            );
          }
        },
      ),
    );
  }

  void _runModel(context) async {
    await _initializeControllerFuture;
    final path = join(
      (await getTemporaryDirectory()).path,
      '${DateTime.now()}.png',
    );

    await _controller.takePicture(path);

    var loadImage = await _classifier.loadImage(path);
    var loadResult = await _classifier.runModel(loadImage);

    _showModalBottomSheet(context, loadImage, loadResult);
  }

  void _showModalBottomSheet(context, loadImage, loadResult) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
      ),
      builder: (BuildContext bc) {
        return Container(
          height: MediaQuery.of(context).size.height * 0.75,
          padding: const EdgeInsets.only(top: 10, left: 5, right: 5),
          child: Column(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Image.memory(
                  img.encodeJpg(loadImage),
                  height: 300,
                ),
              ),
              Expanded(
                child: Container(
                  child: ListView.builder(
                    padding: const EdgeInsets.all(10),
                    itemBuilder: (context, index) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 8.0),
                        child: Column(
                          children: [
                            Row(
                              children: [
                                Text(
                                  '${index + 1}. ',
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: 20,
                                  ),
                                ),
                                Flexible(
                                  child: Text(
                                    loadResult[index]['label'],
                                    style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: 15,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            Text(
                              loadResult[index]['value'].toString(),
                              style: TextStyle(
                                color: Colors.orange,
                                fontWeight: FontWeight.bold,
                                fontSize: (20 - index).toDouble(),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                    itemCount: loadResult.length,
                  ),
                ),
              )
            ],
          ),
        );
      },
    );
  }
}
