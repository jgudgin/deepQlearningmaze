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

    private int batchSize = 50;
    int inputSize = 1024;
    int outputSize = 4;
    int[] hiddenSizes = {10, 5};
    double learningRate = 0.5;
    double discountFactor = 0.9;
    double epsilon = 1.0;
    double minEpsilon = 0.1;
    double decayRate = 0.999;
    double tau = 0.3;

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
        // Use epsilon soft policy to select an action
        Action action = epsilonSoft.selectAction(qValues, actions);

        // Get the next state based on the action taken
        State nextState = currentState.getNextState(action);

        int[][] maze = mazeApp.getMaze();

        // Check if the next state is valid (i.e., not a wall)
        if (nextState != null && maze[nextState.getX()][nextState.getY()] == 0) { // Ensure the next cell is a path

            // Update the current state to the new valid state
            currentState = nextState;

            // Calculate the reward based on the move
            double reward = calculateReward(action);
//            System.out.println("Reward for moving " + action.toString() + " = " + reward);

            // Create a new experience
            Experience experience = new Experience(currentState, action, reward, nextState);
            experienceReplay.addExperience(experience);

            // Update the total reward
            totalReward += reward;

            // Train the neural network with a batch of experiences
            trainWithBatch();

            // Decay epsilon after each action
            if (this.epsilonSoft.getEpsilon() > minEpsilon) {
                this.epsilonSoft.setEpsilon(this.epsilonSoft.getEpsilon() * decayRate);
            }
        } else {

//            System.out.println("Invalid move");
            double reward = -1;    //-1 point for hitting a wall

//            System.out.println("Reward for moving " + action.toString() + " = " + reward);
            // Create a new experience
            Experience experience = new Experience(currentState, action, reward, nextState);
            experienceReplay.addExperience(experience);

            // Update the total reward
            totalReward += reward;

            // Train the neural network with a batch of experiences
            trainWithBatch();

            // Decay epsilon after each action
            if (this.epsilonSoft.getEpsilon() > minEpsilon) {
                this.epsilonSoft.setEpsilon(this.epsilonSoft.getEpsilon() * decayRate);
            }

        }
//        System.out.println("Current Epsilon: " + this.epsilonSoft.getEpsilon());
        System.out.println("total reward: " + totalReward);
    }

    private void trainWithBatch() {
        //check if there are enough experiences in the replay buffer
        if (experienceReplay.getBufferSize() < batchSize) {
            System.out.println("Not enough experiences to train from");
            return; //not enough experiences to train
        }

        List<Experience> batch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            Experience experience = experienceReplay.sampleExperience();
            if (experience != null) {
                batch.add(experience);
            }
        }
//        System.out.println("training with current batch size: " + batch.size());

        //train the q learning network with the sampled batch of experiences
        int i = 1;
        for (Experience experience : batch) {
//            System.out.println("training experience number: " + i + " from the batch");
            qLearningNetwork.train(experience);
            i++;
        }
    }

    public double calculateReward(Action action) {
        int[] goalPos = mazeApp.getEndPosition();
        int currentX = currentState.getX();
        int currentY = currentState.getY();

        // Calculate current Manhattan distance
        int currentDistance = Math.abs(currentX - goalPos[0]) + Math.abs(currentY - goalPos[1]);

        // Get next state based on action
        State nextState = currentState.getNextState(action);
        if (nextState != null) {
            int nextX = nextState.getX();
            int nextY = nextState.getY();
            int nextDistance = Math.abs(nextX - goalPos[0]) + Math.abs(nextY - goalPos[1]);

            // Reward for moving closer and penalty for moving away
            double reward = (currentDistance - nextDistance) * 5; // Scale as needed
            return reward; // Return calculated reward
        }

        // High reward for reaching the goal
        if (currentX == goalPos[0] && currentY == goalPos[1]) {
            return 50; // Moderate reward
        }

        // Small penalty for each action taken
        return -0.01; // Small penalty for normal movement
    }
}
