package neuralnetwork;

//action class holding information about the types of actions available for the game
public class Action {

    public static final Action NORTH = new Action("NORTH", 0, -1);
    public static final Action SOUTH = new Action("SOUTH", 0, 1);
    public static final Action EAST = new Action("EAST", 1, 0);
    public static final Action WEST = new Action("WEST", -1, 0);

    private final String name;
    private final int deltaX;
    private final int deltaY;

    private Action(String name, int deltaX, int deltaY) {
        this.name = name;
        this.deltaX = deltaX;
        this.deltaY = deltaY;
    }

    public int getDeltaX() {
        return deltaX;
    }

    public int getDeltaY() {
        return deltaY;
    }

    @Override
    public String toString() {
        return name;
    }

    //method for neural network input
    public double[] convertToInput() {
        switch (this.name) {
            case "NORTH":
                return new double[]{1, 0, 0, 0};
            case "SOUTH":
                return new double[]{0, 1, 0, 0};
            case "EAST":
                return new double[]{0, 0, 1, 0};
            case "WEST":
                return new double[]{0, 0, 0, 1};
            default:
                throw new IllegalArgumentException("Unknown action: " + this.name);
        }

    }
    
    //method for direct indexing (for updating Q-values)
    public int index() {
        switch (this.name) {
            case "NORTH":
                return 0;
            case "SOUTH":
                return 1;
            case "EAST":
                return 2;
            case "WEST":
                return 3;
            default:
                throw new IllegalArgumentException("Unknown action: " + this.name);
        }
    }

}
