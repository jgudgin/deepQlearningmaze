package neuralnetwork;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

//state object holding information about the agents current co-ordinates and surroundings
//includes method for getting the next state after an action
public class State {

    private int x;
    private int y;
    private Map<Action, Surrounding> surroundings; //maps each action to either WALL or PATH
    private MazeApp maze;

    //constructor for the coordinates and the surroundings
    public State(int x, int y, MazeApp maze) {
        this.x = x;
        this.y = y;
        this.maze = maze;
        this.surroundings = new HashMap<>();
    }

    public enum Surrounding {
        WALL, PATH
    }

    //getters
    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public Map<Action, Surrounding> getSurroundings() {
        return surroundings;
    }

    public void updateSurroundings(int[][] maze, int x, int y) {
        surroundings.clear(); // Clear previous surroundings

        // Check for each direction and update surroundings based on the maze structure
        if (y > 0 && maze[y - 1][x] == 0) { // North
//            System.out.println("Setting North as PATH: x=" + x + ", y=" + (y - 1));
            surroundings.put(Action.NORTH, Surrounding.PATH);
        } else {
//            System.out.println("Setting North as WALL: x=" + x + ", y=" + (y - 1));
            surroundings.put(Action.NORTH, Surrounding.WALL);
        }

        if (y < maze.length - 1 && maze[y + 1][x] == 0) { // South
//            System.out.println("Setting South as PATH: x=" + x + ", y=" + (y + 1));
            surroundings.put(Action.SOUTH, Surrounding.PATH);
        } else {
//            System.out.println("Setting South as WALL: x=" + x + ", y=" + (y + 1));
            surroundings.put(Action.SOUTH, Surrounding.WALL);
        }

        if (x < maze[0].length - 1 && maze[y][x + 1] == 0) { // East
//            System.out.println("Setting East as PATH: x=" + (x + 1) + ", y=" + y);
            surroundings.put(Action.EAST, Surrounding.PATH);
        } else {
//            System.out.println("Setting East as WALL: x=" + (x + 1) + ", y=" + y);
            surroundings.put(Action.EAST, Surrounding.WALL);
        }

        if (x > 0 && maze[y][x - 1] == 0) { // West
//            System.out.println("Setting West as PATH: x=" + (x - 1) + ", y=" + y);
            surroundings.put(Action.WEST, Surrounding.PATH);
        } else {
//            System.out.println("Setting West as WALL: x=" + (x - 1) + ", y=" + y);
            surroundings.put(Action.WEST, Surrounding.WALL);
        }
    }

    //check if certain direction is path
    public boolean isPath(Action action) {
        return surroundings.get(action) == Surrounding.PATH;
    }

    //get the next state based on the action
    public State getNextState(Action action) {
        if (!isPath(action)) {
            System.out.println("moving " + action.toString() + " in state " + this.toString() + "\nleads to a wall");
            return null;
        }

        int newX = x + action.getDeltaX();
        int newY = y + action.getDeltaY();

        updateSurroundings(maze.getMaze(), newX, newY);
        return new State(newX, newY, maze);
    }

    //method for encoding State coordinates and Action direction values into an array
    public double[] convertToInput(Action action) {
        double[] stateInput = new double[]{this.getX(), this.getY()};
        double[] actionInput = action.convertToInput();

        double[] combinedInput = new double[stateInput.length + actionInput.length];
        System.arraycopy(stateInput, 0, combinedInput, 0, stateInput.length);
        System.arraycopy(actionInput, 0, combinedInput, stateInput.length, actionInput.length);

        return combinedInput;
    }

    //check how many directions are blocked based on the current state
    public int countBlockedDirections() {
        int count = 0;

        //check each direction
        for (Action action : Action.values()) {
            if (!isPath(action)) {
                count++;
            }
        }

        return count;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }

        State state = (State) obj;
        return x == state.x && y == state.y && surroundings.equals(state.surroundings);
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, surroundings);
    }

    @Override
    public String toString() {
        return "\nx = " + this.getX() + "\ny = " + this.getY();
    }
}
