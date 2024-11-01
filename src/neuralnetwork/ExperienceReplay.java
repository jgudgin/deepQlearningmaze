package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

//store intial experiences before using the neural network and select one to start learning from
public class ExperienceReplay {
    
    private final List<Experience> replayBuffer;
    private final int bufferSize;
    private final Random random;
    
    public ExperienceReplay(int bufferSize){
        this.bufferSize = bufferSize;
        this.replayBuffer = new ArrayList<>();
        this.random = new Random();
    }
    
    //add an experience to the buffer
    public void addExperience(Experience experience){
        if (replayBuffer.size() < bufferSize) {
            replayBuffer.add(experience);
        } else {
            //replace the oldest experience with a new one (circular buffer)
            replayBuffer.set(random.nextInt(bufferSize), experience);
        }
    }
    
    //sample an experience from the buffer
    public Experience sampleExperience(){
        if (replayBuffer.isEmpty()){
            return null;
        }
        
        int index = random.nextInt(replayBuffer.size());
        return replayBuffer.get(index);
    }

    //create the 'experience' objects from the agents initial movements and store them
    public void createExperiences(List<Experience> experiences) {
        for (Experience experience : experiences){
            addExperience(experience);
        }
    }
}