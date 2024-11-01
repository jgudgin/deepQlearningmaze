package neuralnetwork;

public class Output extends Layer {

    public Output(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    @Override
    public double[] forward(double[] inputs) {
        this.inputs = inputs;
        for (int j = 0; j < outputs.length; j++) {
            outputs[j] = biases[j];
            for (int i = 0; i < inputs.length; i++) {
                outputs[j] += inputs[i] * weights[i][j];
            }
            //linear activation (no transformation for Q-values)
        }
        return outputs;
    }
    
    @Override
    public double[] calcNextGradients(double[] layerGradients) {
        //the output layer computes the gradients based on the loss
        double[] gradients = new double[inputs.length];
        for (int i = 0; i < gradients.length; i++){
            gradients[i] = layerGradients[i] * inputs[i];
        }
        
        return gradients;
    }
}
