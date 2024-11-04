package neuralnetwork;

import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;

public class Agent {

    private State currentState;
    private ExperienceReplay experienceReplay;
    private double totalReward;
    private EpsilonSoft epsilonSoft;
    private double[] qValues;
    private MazeApp mazeApp;
    private QLearningNetwork qLearningNetwork;
    private Scanner input;

    private int batchSize = 1000;
    int inputSize = 100;
    int outputSize = 4;
    int[] hiddenSizes = {10, 5};
    double learningRate = 0.1;
    double discountFactor = 0.9;
    double epsilon = 1.0;
    double minEpsilon = 0.1;
    double decayRate = 0.999;
    double tau = 0.5;

    //amount of valid moves before epsilon is adjusted
    int movesBeforeDecay = 1000;
    
    //initial state of move counter
    int moveCounter = 0;

    public Agent(State initialState, MazeApp mazeApp) {
        this.currentState = initialState;
        this.experienceReplay = new ExperienceReplay(batchSize);
        this.epsilonSoft = new EpsilonSoft(epsilon, tau);
        this.qLearningNetwork = new QLearningNetwork(inputSize, outputSize, hiddenSizes, learningRate, discountFactor);
        this.totalReward = 0.0;
        this.mazeApp = mazeApp;
    }

    public State getCurrentState() {
        return currentState;
    }

    public void move(List<Action> actions) throws InterruptedException {

//        input = new Scanner(System.in);
        //create a new array to store q values for available actions
        qValues = new double[actions.size()];

        // Use epsilon soft policy to select an action
        Action action = epsilonSoft.selectAction(qValues, actions);

//        System.out.println("Agent has chosen to go: " + action.toString());
//        System.out.println("Agents next state should be: " + this.getCurrentState().getNextState(action).toString());
        int[][] maze = mazeApp.getMaze();

        //flag for checking if a certain move is valid
        boolean isNextActionValid = currentState.isPath(action);

        // Check if the next state is valid (i.e., not a wall)
        if (isNextActionValid) { // Ensure the next cell is a path
//            System.out.println("currentState.isPath(" + action.toString() + ") = " + isNextActionValid);

//            System.out.println("move is valid, making...");
            // Update the current state to the new valid state
            currentState = getCurrentState().getNextState(action);

            //also update the surroundings for the updated current state
            currentState.updateSurroundings(mazeApp.getMaze(), currentState.getX(), currentState.getY());

            //update the surroundings of the agent after updating the state
//            nextState.updateSurroundings(mazeApp.getMaze(), getCurrentState().getX(), getCurrentState().getY());
//            System.out.println("Agents should be surroundings after moving: " + nextState.getSurroundings().toString());
//            System.out.println("Agent true state after moving: " + this.getCurrentState().toString());
            // Calculate the reward based on the move
            double reward = calculateReward(action);
//            System.out.println("Reward for moving " + action.toString() + " = " + reward);

            // Create a new experience
            Experience experience = new Experience(currentState, action, reward, currentState);
            experienceReplay.addExperience(experience);

            // Update the total reward
            totalReward += reward;

            // Train the neural network with a batch of experiences
            trainWithBatch();

            // Increment move counter and check if it's time to decay epsilon
            moveCounter++;
            if (moveCounter >= movesBeforeDecay) {
                if (this.epsilonSoft.getEpsilon() > minEpsilon) {
                    this.epsilonSoft.setEpsilon(this.epsilonSoft.getEpsilon() * decayRate);
                }
                moveCounter = 0; // Reset the counter
            }
        } else {
            System.out.println("Invalid move, " + action.toString() + " is blocked");
        }

//        input.nextLine();
//        Thread.sleep(10000);
//        System.out.println("Current Epsilon: " + this.epsilonSoft.getEpsilon());
//        System.out.println("total reward: " + totalReward);
    }

    private void trainWithBatch() {
        //check if there are enough experiences in the replay buffer
        if (experienceReplay.getBufferSize() < batchSize) {
//            System.out.println("Not enough experiences to train from");
            return; //not enough experiences to train
        }

        List<Experience> batch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            Experience experience = experienceReplay.sampleExperience();
            if (experience != null && !batch.contains(experience)) {
                batch.add(experience);
            }
        }
//        System.out.println("training with current batch size: " + batch.size());

          System.out.println("batch contents: " + batch.toString());
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

        // Calculate current Manhattan distance to the goal
        int currentDistance = Math.abs(currentX - goalPos[0]) + Math.abs(currentY - goalPos[1]);

        // High reward for reaching the goal
        if (currentX == goalPos[0] && currentY == goalPos[1]) {
            return 5000; // High reward for reaching the goal
        }

        //if the agent is at a dead end then deduct points
        if (currentState.countBlockedDirections() == 3) {
            return 1.0 / (currentDistance + 1) * 2;
        }

        // Intermediate reward based on distance
        double distanceReward = 1.0 / (currentDistance + 1); // Reward increases as distance decreases

        // Small penalty for each action taken
        double stepPenalty = -10;

        return distanceReward + stepPenalty;
    }

    public void setTotalReward(int totalReward){
        this.totalReward = totalReward;
    }
}
