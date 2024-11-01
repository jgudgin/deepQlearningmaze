package neuralnetwork;

//the input layer simply feeds the received data forward into the first hidden layer
//it standardizes the format and size of the data going into the network
public class Input extends Layer {

    //constructor initializes input layer with specified input and output sizes
    public Input(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    @Override
    public double[] forward(double[] inputs) {
        //input and outputs should match since no transformation is applied yet
        this.inputs = inputs;
        this.outputs = inputs;
        //forward the input data to the next layer
        return outputs;
    }
    
    @Override
    public double[] calcNextGradients(double[] layerGradients) {
        //return 0 array matching the input size since there are no layers before the input
        return new double[inputs.length];
    }
}
