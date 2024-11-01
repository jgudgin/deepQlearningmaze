package neuralnetwork;

import java.util.List;
import java.util.ArrayList;

public class Agent {

    private State currentState;
    private ExperienceReplay experienceReplay;
    private double totalReward;
    private EpsilonSoft epsilonSoft;
    private double[] qValues;
    private MazeApp mazeApp;
    private QLearningNetwork qLearningNetwork;
    
    
    private int batchSize = 32;
    int inputSize = 1024;
    int outputSize = 4;
    int[] hiddenSizes = {10, 5};
    double learningRate = 0.1;
    double discountFactor = 0.9;
    double epsilon = 1.0;
    double tau = 0.5;

    public Agent(State initialState, MazeApp mazeApp) {
        this.currentState = initialState;
        this.experienceReplay = new ExperienceReplay(batchSize);
        this.epsilonSoft = new EpsilonSoft(epsilon, tau);
        this.qLearningNetwork = new QLearningNetwork(inputSize, outputSize, hiddenSizes, learningRate, discountFactor);
        this.totalReward = 0.0;
        this.qValues = new double[4];
        this.mazeApp = mazeApp;
    }

    public State getCurrentState() {
        return currentState;
    }

    public void move(List<Action> actions) {
        //use epsilon soft policy to select an action
        Action action = epsilonSoft.selectAction(qValues, actions);
        double reward = calculateReward(action);

        State nextState = currentState.getNextState(action);

        if (nextState != null) {
            //create a new experience
            Experience experience = new Experience(currentState, action, reward, nextState);

            //add experience to the replay buffer
            experienceReplay.addExperience(experience);

            //update the total reward
            totalReward += reward;

            //update the current state and surroundings
            currentState = nextState;

            //train the neural network with a batch of experiences
            trainWithBatch();
        }
    }

    private void trainWithBatch() {
        //check if there are enough experiences in the replay buffer
        if (experienceReplay.getBufferSize() < batchSize) {
            return; //not enough experiences to train
        }

        List<Experience> batch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            Experience experience = experienceReplay.sampleExperience();
            if (experience != null) {
                batch.add(experience);
            }
        }

        //train the q learning network with the sampled batch of experiences
        for (Experience experience : batch) {
            qLearningNetwork.train(experience);
        }
    }

    public double calculateReward(Action action) {

        int[] goalPos = mazeApp.getEndPosition();
        int blockedDirections = currentState.countBlockedDirections();

        //assign negative reward if agent reaches dead end
        if (blockedDirections >= 3) {
            return -5;
        }

        //assign positive reward if the goal coordinates are reached
        if (currentState.getX() == goalPos[0] && currentState.getY() == goalPos[1]) {
            return 10;
        }

        //zero points for normal movement
        return 0;
    }

}
