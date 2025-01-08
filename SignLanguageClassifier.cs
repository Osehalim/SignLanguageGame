using UnityEngine;
using Unity.Barracuda;

public class SignLanguageClassifier : MonoBehaviour
{
    public NNModel onnxModel;  // Drag and drop your ONNX model in the Inspector
    private IWorker worker;

    void Start()
    {
        // Create a worker to run the ONNX model
        Model model = ModelLoader.Load(onnxModel);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }

    public void Classify(Texture2D image)
    {
        // Convert the input image to a Tensor
        Tensor inputTensor = TransformInput(image);

        // Run the model on the input tensor
        worker.Execute(inputTensor);

        // Get the output tensor (prediction)
        Tensor outputTensor = worker.PeekOutput();

        // Find the predicted class
        int predictedClass = outputTensor.ArgMax()[1];  // Get index of the max output value

        // Display or log the predicted class
        Debug.Log("Predicted sign class: " + predictedClass);

        // Dispose of tensors to free memory
        inputTensor.Dispose();
        outputTensor.Dispose();
    }

    // Converts a Texture2D image to a Tensor (1x1x28x28 for grayscale)
    Tensor TransformInput(Texture2D image)
    {
        Color32[] pixels = image.GetPixels32();
        float[] floatValues = new float[pixels.Length];

        for (int i = 0; i < pixels.Length; i++)
        {
            floatValues[i] = pixels[i].grayscale;  // Convert to grayscale
        }

        return new Tensor(1, 28, 28, 1, floatValues);
    }

    void OnDestroy()
    {
        // Dispose of the worker when the game object is destroyed
        worker.Dispose();
    }
}
