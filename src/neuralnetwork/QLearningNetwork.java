package neuralnetwork;

import java.util.Arrays;

public class QLearningNetwork {

    private Layer input;    //create input layer
    private Layer[] hidden; //create hidden layers
    private Layer output;   //create output layer
    private double alpha;   //learning rate
    private double gamma;   //discount factor, reduces reward for every time step

    //constructor for the network
    public QLearningNetwork(int inputSize, int outputSize, int[] hiddenSizes, double alpha, double gamma) {
        this.alpha = alpha;
        this.gamma = gamma;

//        System.out.println("initializing layers...");
        //initialize the layers
        //input layer
//        System.out.println("creating input layer with inputSize = " + inputSize + " and outputSize = " + hiddenSizes[0]);
        input = new Input(inputSize, hiddenSizes[0]);

        //hidden layers
        hidden = new Layer[hiddenSizes.length];
        for (int i = 0; i < hiddenSizes.length; i++) {
            int inputSizeForLayer = i == 0 ? inputSize : hiddenSizes[i - 1];
            int outputSizeForLayer = hiddenSizes[i];

//            System.out.println("creating hidden layer with inputSize = " + inputSizeForLayer + " and outputSize = " + outputSizeForLayer);
            hidden[i] = new Hidden(inputSizeForLayer, outputSizeForLayer);
        }

        //output layer
//        System.out.println("creating output layer with inputSize = " + hiddenSizes[hiddenSizes.length - 1]);
        output = new Output(hiddenSizes[hiddenSizes.length - 1], outputSize);
    }

    //forward pass through the network
    //predicted Q-values = Q(s,a)
    public double[] predict(Experience experience) {
        double[] inputs = experience.getCurrentState().convertToInput(experience.getAction()); //encode details for easier input and management
        double[] outputs = input.forward(inputs);   //forward encoded inputs
        int i = 1;
        for (Layer layer : hidden) {
//            System.out.println("forwarding the output through hidden layer " + i);
            outputs = layer.forward(outputs);   //forward through hidden layers, using the outputs of previous layer neurons as inputs for the new layer ones
            i++;
        }
        return output.forward(outputs);     //return output through output layer with Q-value prediction
    }

    //train network based on Q-learning update formula:
    //Qt(s,a) = Qt-1(s,a) + alpha* (R(s,a) + gamma * maxa' * Q(s',a') - Qt-1(s,a))
    public void train(Experience experience) {

        //predict the current Q-values for current state-action pair
        //Qt-1(s,a)
        
        //current predicted Q-values for calculating updated ones
        double[] qValuesCurrent = predict(experience);
        
        //original for calculating error
        double[] initialQValues = qValuesCurrent.clone();
//        System.out.println("predicted current Q-values (Qt-1) = " + Arrays.toString(initialQValues));

        //predict the Q-values for the next state if it exists, otherwise initialize an array
        //those Q-values correspond to Q(s',a') for all possible actions a' in the next state
        double[] qValuesNext = new double[qValuesCurrent.length];
        if (experience.getNextState() != null) {
            Experience nextExperience = new Experience(experience.getNextState(), experience.getAction(), experience.getRewardReceived(), experience.getNextState());
            qValuesNext = predict(nextExperience);
        } else {
            Arrays.fill(qValuesNext, 0);
        }
//        System.out.println("predicted next Q-values (Q(s',a')) = " + Arrays.toString(qValuesNext));

        //calculate maximum Q-value for the next state (s') across all possible actions (a')
        //maxa' * Q(s',a')
        double maxQNext = Arrays.stream(qValuesNext).max().orElse(0);

//        System.out.println("Max Q-value for next state (maxa' * Q(s',a')) = " + maxQNext);

        //calculate the target Q-value using Q-learning update formula
        //targetValue = R(s,a) + gamma * maxQNext
        double targetQValue = experience.getRewardReceived() + gamma * maxQNext;
//        System.out.println("Target Q-value (R(s,a) + gamma * maxQNext): " + targetQValue);

        //update Q-value for current action in current state
        //update Qt(s,a) with: alpha * (targetQValue - Qt-1(s,a))
        int actionIndex = experience.getAction().index();
//        System.out.println("Updating Q-value for action index " + (actionIndex + 1) + " from " + qValuesCurrent[actionIndex]);
        qValuesCurrent[actionIndex] += (alpha * (targetQValue - qValuesCurrent[actionIndex]));
//        System.out.println("Updated Q-value for action index " + (actionIndex + 1) + ": " + qValuesCurrent[actionIndex]);

        //backpropagate the updated Q-values through the neural network, using the predicted ones to calculate error
        backpropagate(experience, qValuesCurrent, initialQValues);
//        System.out.println("Backpropagation completed for experience: " + experience);
    }

    public void backpropagate(Experience experience, double[] updatedQValues, double[] predictedQValues) {
        //convert state-action pair to one-hot encoded input before putting into neural network
        double[] stateInput = experience.getCurrentState().convertToInput(experience.getAction());
        double[] inputs = new double[stateInput.length];

        //array to store the gradients of each neuron in a hidden layer
        double[] nextLayerGradients = new double[0];

        //array to store the gradients of each neuron in the output layer
        double[] outputGradients = new double[0];

//        System.out.println("updatedQValues = " + Arrays.toString(updatedQValues));
//        System.out.println("predictedQValues = " + Arrays.toString(predictedQValues));

        System.arraycopy(stateInput, 0, inputs, 0, stateInput.length);

        //calculate the error by finding the difference between the predicted and updated Q-values
        //the error is used to calculate the gradients for backpropagation
        double[] error = new double[updatedQValues.length];
        for (int i = 0; i < updatedQValues.length; i++) {
//            System.out.println("calculating the error between predicted Q value = " + predictedQValues[i] + " and updated Q value = " + updatedQValues[i]);
            error[i] = updatedQValues[i] - predictedQValues[i];
//            System.out.println(error[i]);
        }

        //backpropagation through the output layer
        //calculate gradient for output layer: (∂L / ∂z) = error * σ'(z)
        //where z is the output of the neural network and σ'(z) is the derivative of ReLU activation function
        //create an array for each neuron in the output layer
        outputGradients = new double[predictedQValues.length];
        for (int i = 0; i < outputGradients.length; i++) {
//            System.out.println("calculating the gradient for neuron " + (i + 1) + " in output layer");
//            System.out.println("gradient = " + error[i] + " * " + reluDerivative(predictedQValues[i]));
            outputGradients[i] = error[i] * reluDerivative(predictedQValues[i]);   //use the ReLU derivative to calculate the gradient
//            System.out.println("gradient = " + outputGradients[i]);
        }

        //update output layer weights and biases using calculated
        //weight gradients and output gradients
        output.updateWeights(calcWGradients(outputGradients, inputs), outputGradients, alpha);

        //backpropagation through hidden layers
        //this is for calculating the gradients for weight updates
        for (int i = hidden.length - 1; i >= 0; i--) {

            Layer currentLayer = hidden[i];

            //stores the gradients of the neurons on the next layer
            //the array should be equal in size to the amount of neurons in the current hidden layer
            nextLayerGradients = new double[currentLayer.getOutputSize()];

//            System.out.println("amount of neurons in next layer = " + nextLayerGradients.length);

            //represents the output values of neurons in the current layer
            //when i == 0, currentOutputs is set to inputs
            //for subsequent hidden layers i > 0, currentOutputs is set to the outputs of the 
            //previous hidden layer (hidden[i - 1].getOutputs();)
//            double[] currentOutputs = (i == hidden.length) ? inputs : hidden[i - 1].getOutputs();

            double[] currentOutputs;
            currentOutputs = hidden[i].getOutputs();
//            System.out.println(Arrays.toString(currentOutputs));
            double[] layerGradients = new double[currentLayer.getOutputSize()];

            //calcuate the gradients of each neuron in the current hidden layer l,
            //using the gradients of the layer above (l + 1)
            //δ^(l) = ((W^(l+1)^T * δ^(l + 1) * σ'(z^(l))
            //where W^(l + 1) is the weight matrix from layer l to layer l + 1
            //δ^(l + 1) is the gradient from the next layer, and 
            //σ'(z^(l)) is the derivative of the activation function at layer l.
            
//            System.out.println("iterating through " + layerGradients.length + " neurons");
//            System.out.println("amount of neurons in hidden layer = " + nextLayerGradients.length);
//            System.out.println("Processing hidden layer " + (i+1) + " with currentOutputs size = " + currentOutputs.length);
//            System.out.println("Expected neuron count = " + layerGradients.length);

            for (int j = 0; j < layerGradients.length; j++) {
                //calculate the gradient for each neuron in the current layer by using gradients
                //from the next layer and the derivative of the ReLU activation func
                
//                System.out.println("calculating gradient for neuron = " + (j + 1) + " in hidden layer " + (i + 1));
//                System.out.println("gradient = " + nextLayerGradients[j] + " * " + reluDerivative(currentOutputs[j]));
//                System.out.println("gradient = " + nextLayerGradients[j] * reluDerivative(currentOutputs[j]));
                layerGradients[j] = nextLayerGradients[j] * reluDerivative(currentOutputs[j]);
            }

            //update weights for the current layer using the calculated weight gradients
            //and layer gradients found from backpropagation
            
//            System.out.println("update the weights with inputs: "
//                    + "\nupdateWeights:"
//                    + "\nweightGradients = calcWGradients"
//                    + "\n biasGradients = " + Arrays.toString(layerGradients)
//                    + "\n alpha = " + alpha);
            currentLayer.updateWeights(calcWGradients(layerGradients, currentOutputs), layerGradients, alpha);

            //calculate next gradients for the previous layer using layer gradients calculated for current layer
            //this prepares for backpropagation to the previous layer
            nextLayerGradients = currentLayer.calcNextGradients(layerGradients);
        }
    }

    //find the derivative of the ReLU activation function
    //the derivative indicates how much the output of the ReLU function changes
    //with respect to changes in its input
    //             1 if x > 0
    //ReLU'(x) = { 
    //             0 if x <= 0
    private double reluDerivative(double output) {
        return (output > 0) ? 1 : 0;
    }

    //method to calculate weight gradients
    //using (∂L / ∂wij) = (∂L / ∂oj) * xi
    //where (∂L / ∂wij) is the weight gradient with respect to loss
    //(∂L / ∂oj) is the gradient of the loss with respect to the ouput of neuron j
    //and xi = the input value
    private double[][] calcWGradients(double[] gradients, double[] inputs) {
        double[][] weightGradients = new double[inputs.length][gradients.length];

        //multiply (∂L / ∂oj) * xi to get weight gradient of each neuron
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < gradients.length; j++) {
                weightGradients[i][j] = gradients[j] * inputs[i];
            }
        }
        
//        System.out.println("inputs length = " + inputs.length);
//        System.out.println("weight gradients length = " + weightGradients.length);
        return weightGradients;
    }
}
