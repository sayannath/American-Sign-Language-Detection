package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import android.util.Log;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;
//import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier.ImageClassifierOptions;

import static java.lang.Math.min;

/** A classifier specialized to label images using TensorFlow Lite. */
public abstract class Classifier {
  private static final Logger LOGGER = new Logger();

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    GPU
  }

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 3;

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel;

  /** Image size along the x axis. */
  private final int imageSizeX;

  /** Image size along the y axis. */
  private final int imageSizeY;

  /** Optional GPU delegate for acceleration. */
//  private GpuDelegate gpuDelegate = null;


  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;
  protected ImageClassifier imageClassifier;

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;

  /** Input image TensorBuffer. */
  private TensorImage inputImageBuffer;


  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, Device device, int numThreads)
      throws IOException {

    return new ClassifierFloatMobileNet(activity, device, numThreads);
  }

  /** An immutable result returned by a Classifier describing what was recognized. */
  public static class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
        final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  /** Initializes a {@code Classifier}. */
  protected Classifier(Activity activity, Device device, int numThreads) throws IOException {
    if (device != Device.CPU) {
      throw new IllegalArgumentException(
              "Manipulating the hardware accelerators is not allowed in the Task"
                      + " library currently. Only CPU is allowed.");
    }

    // Create the ImageClassifier instance.
    ImageClassifierOptions options =
            ImageClassifierOptions.builder()
                    .setMaxResults(MAX_RESULTS)
                    .setNumThreads(numThreads)
                    .build();
    imageClassifier = ImageClassifier.createFromFileAndOptions(activity, getModelPath(), options);
    LOGGER.d("Created a Tensorflow Lite Image Classifier.");

    // Get the input image size information of the underlying tflite model.
    tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
    MetadataExtractor metadataExtractor = new MetadataExtractor(tfliteModel);

    // Image shape is in the format of {1, height, width, 3}.
    int[] imageShape = metadataExtractor.getInputTensorShape(/*inputIndex=*/ 0);
    imageSizeY = imageShape[1];
    imageSizeX = imageShape[2];
  }

  /** Runs inference and returns the classification results. */
  public List<Recognition> recognizeImage(final Bitmap bitmap, int sensorOrientation) {
    // Logs this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    TensorImage inputImage = TensorImage.fromBitmap(bitmap);
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    int cropSize = min(width, height);

    ImageProcessingOptions imageOptions =
            ImageProcessingOptions.builder()
                    .setOrientation(getOrientation(sensorOrientation))
                    // Set the ROI to the center of the image.
                    .setRoi(
                            new Rect(
                                    /*left=*/ (width - cropSize) / 2,
                                    /*top=*/ (height - cropSize) / 2,
                                    /*right=*/ (width + cropSize) / 2,
                                    /*bottom=*/ (height + cropSize) / 2))
                    .build();

    // Runs the inference call.
    Trace.beginSection("runInference");
    long startTimeForReference = SystemClock.uptimeMillis();
    List<Classifications> results = imageClassifier.classify(inputImage, imageOptions);
    long endTimeForReference = SystemClock.uptimeMillis();
    Trace.endSection();
    LOGGER.v("Time-Cost to run model inference: " + (endTimeForReference - startTimeForReference));

    Trace.endSection();
    // Gets top-k results.
    return getTopKProbability(results);
  }


    /** Closes the interpreter and model to release resources. */
    public void close() {
      if (imageClassifier != null) {
        imageClassifier.close();
      }
    }


  /** Get the image size along the x axis. */
  public int getImageSizeX() {
    return imageSizeX;
  }

  /** Get the image size along the y axis. */
  public int getImageSizeY() {
    return imageSizeY;
  }

  /** Loads input image, and applies preprocessing. */
  private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
    // Loads bitmap into a TensorImage.
    inputImageBuffer.load(bitmap);

    // Creates processor for the TensorImage.
    int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
    int numRoration = sensorOrientation / 90;


    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                    .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
                    .add(new Rot90Op(numRoration))
                    .add(getPreprocessNormalizeOp())
                    .build();
    return imageProcessor.process(inputImageBuffer);
  }

  /** Gets the top-k results. */
  private static List<Recognition> getTopKProbability(List<Classifications> classifications) {
    // Find the best classifications.
    final ArrayList<Recognition> recognitions = new ArrayList<>();
    // All the demo models are single head models. Get the first Classifications in the results.
    for (Category category : classifications.get(0).getCategories()) {
      recognitions.add(
              new Recognition(
                      "" + category.getLabel(), category.getLabel(), category.getScore(), null));
    }
    return recognitions;
  }

  /** Gets the name of the model file stored in Assets. */
  protected abstract String getModelPath();

  /** Gets the name of the label file stored in Assets. */
  protected abstract String getLabelPath();

  /** Gets the TensorOperator to nomalize the input image in preprocessing. */
  protected abstract TensorOperator getPreprocessNormalizeOp();

  /* Convert the camera orientation in degree into {@link ImageProcessingOptions#Orientation}.*/
  private static ImageProcessingOptions.Orientation getOrientation(int cameraOrientation) {
    switch (cameraOrientation / 90) {
      case 3:
        return ImageProcessingOptions.Orientation.BOTTOM_LEFT;
      case 2:
        return ImageProcessingOptions.Orientation.BOTTOM_RIGHT;
      case 1:
        return ImageProcessingOptions.Orientation.TOP_RIGHT;
      default:
        return ImageProcessingOptions.Orientation.TOP_LEFT;
    }
  }
  /**
   * Gets the TensorOperator to dequantize the output probability in post processing.
   *
   * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
   * essentially linear transformation). For float model, de-quantize is not required. But to
   * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
   * 1.0f, respectively.
   */
  protected abstract TensorOperator getPostprocessNormalizeOp();
}
