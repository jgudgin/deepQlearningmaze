package neuralnetwork;

import javax.swing.*;
import java.awt.*;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

// Class for generating a maze
public class MazeApp {

    private static final int GRID_SIZE = 32; // Change to 32x32 grid
    private static final int CELL_SIZE = 30; // Pixel size for each cell
    private int[][] maze; // Keep the maze as int[][]
    private JPanel gridPanel; // Panel containing the grid

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new MazeApp().createAndShowGUI());
    }

    private void createAndShowGUI() {
        JFrame frame = new JFrame("Maze Game");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        gridPanel = new JPanel(new GridLayout(GRID_SIZE, GRID_SIZE));
        gridPanel.setPreferredSize(new Dimension(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE));

        // Initialize the maze with walls
        maze = generateMaze();

        // Create cells based on the maze array
        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                JPanel cell = new JPanel();
                cell.setPreferredSize(new Dimension(CELL_SIZE, CELL_SIZE));

                // Determine wall or path based on the maze array
                if (maze[row][col] == 1) {
                    cell.setBackground(Color.BLACK); // Wall
                } else {
                    cell.setBackground(Color.WHITE); // Path
                }

                // Set start and end point colors
                if (row == 1 && col == 1) {
                    cell.setBackground(Color.GREEN); // Start point
                } else if (row == GRID_SIZE - 3 && col == GRID_SIZE - 3) {
                    cell.setBackground(Color.RED); // End point
                }

                gridPanel.add(cell);
            }
        }

        // Add the grid panel to the frame
        frame.add(gridPanel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private int[][] generateMaze() {
        int[][] maze = new int[GRID_SIZE][GRID_SIZE];

        // Initialize all cells as walls
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                maze[i][j] = 1; // Wall
            }
        }

        // Start carving paths from the first cell
        carvePath(maze, 1, 1);

        // Set start and end points
        maze[1][1] = 0; // Start point
        maze[GRID_SIZE - 3][GRID_SIZE - 3] = 0; // Goal point

        return maze;
    }

    // DFS maze generation
    private void carvePath(int[][] maze, int x, int y) {
        // Define directions: (dx, dy)
        int[][] directions = {
            {2, 0}, // East
            {-2, 0}, // West
            {0, 2}, // South
            {0, -2} // North
        };

        // Shuffle the directions to ensure random carving
        Random random = new Random();
        for (int i = 0; i < directions.length; i++) {
            int j = random.nextInt(directions.length);
            // Swap
            int[] temp = directions[i];
            directions[i] = directions[j];
            directions[j] = temp;
        }

        // Carve paths in each direction
        for (int[] dir : directions) {
            int newX = x + dir[0];
            int newY = y + dir[1];

            // Check if the new coordinates are within bounds and if the cell has not been visited
            if (newX > 0 && newX < GRID_SIZE - 1 && newY > 0 && newY < GRID_SIZE - 1 && maze[newX][newY] == 1) {
                // Carve a path between the current cell and the new cell
                maze[x + dir[0] / 2][y + dir[1] / 2] = 0; // Remove wall between
                maze[newX][newY] = 0; // Carve the new cell

                // Recursively carve from the new cell
                carvePath(maze, newX, newY);
            }
        }
    }
}
