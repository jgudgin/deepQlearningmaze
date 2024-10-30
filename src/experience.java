//used to store initial experiences for use of experience replay


public class experience {
    
    state currentState; //the current given state
    action actionTaken; //the current action about to be taken
    reward rewardReceived;  //the reward received from that action in that state
    state nextState;    //the next state
    boolean done;   //boolean flag for when to stop storing experiences
    
    public experience(state currentState, action actionTaken, reward rewardReceived, state nextState){
        this.done = false;
        this.currentState = currentState;
        this.actionTaken = actionTaken;
        this.rewardReceived = rewardReceived;
        this.nextState = nextState;
    }
    
    
    //store getters and setters in here so they can be easily tracked and modified
    //setters
    public void setCurrentState(state currentState){
        this.currentState = currentState;
    }
    
    public void setActionTaken(action actionTaken){
        this.actionTaken = actionTaken;
    }
    
    public void setRewardReceived(reward rewardReceived){
        this.rewardReceived = rewardReceived;
    }
    
    public void setNextState(state nextState){
        this.nextState = nextState;
    }
    
    public void setDone(boolean done){
        this.done = done;
    }
    
    //getters
    public int getCurrentState(){
        return currentState;
    }
    
    public int getActionTaken(){
        return actionTaken;
    }
    
    public int getRewardReceived(){
        return rewardReceived;
    }
    
    public int getNextState(){
        return nextState;
    }
}
