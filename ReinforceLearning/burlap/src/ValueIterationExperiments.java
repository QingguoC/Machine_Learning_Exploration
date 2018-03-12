import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.mdp.singleagent.common.*;
import java.util.ArrayList;

public class ValueIterationExperiments {

    public static void main(String[] args){

        testViIterationsSmallMDP();
        testViIterationsLargeMDP();
        testIncentiveLargeMDP();
        testIncentiveSmallMDP();
    }

    public static void testViIterationsSmallMDP(){
        int repeat = 10;
        int[] iterations = {1,3,5,7,9,11};
        double[] gammas = {0.8,0.9,0.99};
        String[][] evalues = new String[iterations.length * gammas.length * repeat][8];

        for (int i = 0; i < iterations.length; i++){
            for (int j = 0; j < gammas.length; j++){
                for (int r = 0; r < repeat; r++){
                    int maxIteration = iterations[i];
                    double gamma = gammas[j];
                    int mapSize = GridWorlds.smallMap.length;
                    GridWorldDomain gwd = new GridWorldDomain(mapSize, mapSize);
                    gwd.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                    gwd.setMap(GridWorlds.smallMap);
                    //only go in intended directon 80% of the time
                    gwd.setProbSucceedTransitionDynamics(0.8);
                    SADomain domain = gwd.generateDomain();
                    //get initial state with agent in 0,0
                    State initialState = new GridWorldState(new GridAgent(0, 0));
                    SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
                    double start = System.nanoTime();

                    ValueIteration viPlanner = new ValueIteration(domain, gamma,hashingFactory , maxIteration);
                    Policy p =  viPlanner.planFromState(initialState);

                    double runningTime = System.nanoTime() - start;
                    Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());

                    ArrayList<Double> maxDeltas = viPlanner.getMaxDeltaInEachIteration();
                    double lastMaxDelta = maxDeltas.get(maxDeltas.size() - 1);
                    int rewards = Utitlities.calRewards(ep);
                    int numOfActions = ep.actionSequence.size();

                    String[] eval = {"small","valueIteration", maxIteration + "", gamma + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                    evalues[i * gammas.length * repeat + j * repeat +  r]  = eval;

                    System.out.println("iterations "+maxIteration + " gamma " + gamma + " repeat " + r);
                    System.out.println( "reward seq size " + ep.rewardSequence.size());
                    System.out.println( "action seq size " + ep.actionSequence.size());
                    System.out.println( "Value model rewards: " + Utitlities.calRewards(ep));
                    System.out.println("total value iteration is : " + viPlanner.getTotalValueIterations());
                    System.out.println("maxDeltas size is : " + maxDeltas.size());
                    if (r == 0 && gamma == 0.99){
                        Utitlities.manualValueFunctionVis((ValueFunction)viPlanner, p, mapSize, "Value Iteration with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
                    }

                }
            }
        }
        String[] header = {"gridWorld","strategy", "maxIteration", "gamma", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"viSmallIterationExp");

    }
    public static void testViIterationsLargeMDP(){
        int repeat = 10;
        int[] iterations = {15,30,45,60,75,90};
        double[] gammas = {0.8,0.9,0.99};
        String[][] evalues = new String[iterations.length * gammas.length * repeat][8];

        for (int i = 0; i < iterations.length; i++){
            for (int j = 0; j < gammas.length; j++){
                for (int r = 0; r < repeat; r++){
                    int maxIteration = iterations[i];
                    double gamma = gammas[j];
                    int mapSize = GridWorlds.largeMap.length;
                    int[] exit = new int[]{0, mapSize - 1};
                    GridWorldDomain gwd = new GridWorldDomain(mapSize, mapSize);
                    gwd.setTf(new GridWorldTerminalFunction(exit[0],exit[1]));

                    gwd.setMap(GridWorlds.largeMap);
                    //only go in intended directon 80% of the time
                    gwd.setProbSucceedTransitionDynamics(0.8);
                    SADomain domain = gwd.generateDomain();
                    //get initial state with agent in 0,0
                    State initialState = new GridWorldState(new GridAgent(0, 0));
                    SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

                    double start = System.nanoTime();

                    ValueIteration viPlanner = new ValueIteration(domain, gamma,hashingFactory , maxIteration);
                    Policy p =  viPlanner.planFromState(initialState);

                    double runningTime = System.nanoTime() - start;
                    Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());

                    ArrayList<Double> maxDeltas = viPlanner.getMaxDeltaInEachIteration();
                    double lastMaxDelta = maxDeltas.get(maxDeltas.size() - 1);
                    int rewards = Utitlities.calRewards(ep);
                    int numOfActions = ep.actionSequence.size();

                    String[] eval = {"large","valueIteration", maxIteration + "", gamma + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                    evalues[i * gammas.length * repeat + j * repeat +  r]  = eval;

                    System.out.println("iterations "+maxIteration + " gamma " + gamma + " repeat " + r);
                    System.out.println( "reward seq size " + ep.rewardSequence.size());
                    System.out.println( "action seq size " + ep.actionSequence.size());
                    System.out.println( "Value model rewards: " + Utitlities.calRewards(ep));
                    System.out.println("total value iteration is : " + viPlanner.getTotalValueIterations());
                    System.out.println("maxDeltas size is : " + maxDeltas.size());
                    if (r == 0 && gamma == 0.99){
                        Utitlities.manualValueFunctionVis((ValueFunction)viPlanner, p, mapSize, "Value Iteration with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
                    }

                }
            }
        }
        String[] header = {"gridWorld","strategy", "maxIteration", "gamma", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"viLargeIterationExp");

    }
    public static void testIncentiveLargeMDP(){
        int repeat = 10;
        //int[] iterations = {10,30,50,70,90,110,130,150,170,190};
        int mapSize = GridWorlds.largeMap.length;
        int[] exit = new int[]{0, mapSize - 1};
        int[][] incentiveLocs = {{mapSize - 1, 0},{mapSize - 1, mapSize - 1},{mapSize/2 - 1, mapSize/2 - 1}};
        int[] numIncentiveLocs = {1,2,3};
        double[] incentives = {0,0.5,1,1.15,1.16,1.162,1.163};
        String[][] evalues = new String[numIncentiveLocs.length * incentives.length * repeat][8];

        for (int i = 0; i < numIncentiveLocs.length; i++){
            for (int j = 0; j < incentives.length; j++){
                for (int r = 0; r < repeat; r++){
                    int numIncentiveLoc = numIncentiveLocs[i];
                    double incentive = incentives[j];
                    GridWorldDomain gwd = new GridWorldDomain(mapSize, mapSize);
                    gwd.setTf(new GridWorldTerminalFunction(exit[0],exit[1]));

                    GridWorldRewardFunction grf = new GridWorldRewardFunction(mapSize,mapSize,-1);
                    for (int n = 0; n < numIncentiveLoc; n++){
                        grf.setReward(incentiveLocs[n][0], incentiveLocs[n][1], -1 + incentive);
                    }
                    //grf.setReward(mapSize - 1, 0, -1 + incentive);
                    gwd.setRf(grf);

                    gwd.setMap(GridWorlds.largeMap);
                    //only go in intended directon 80% of the time
                    gwd.setProbSucceedTransitionDynamics(0.8);
                    SADomain domain = gwd.generateDomain();
                    //get initial state with agent in 0,0
                    State initialState = new GridWorldState(new GridAgent(0, 0));
                    SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

                    double start = System.nanoTime();

                    ValueIteration viPlanner = new ValueIteration(domain, 0.99,hashingFactory , 400);
                    Policy p =  viPlanner.planFromState(initialState);

                    double runningTime = System.nanoTime() - start;
                    Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());

                    ArrayList<Double> maxDeltas = viPlanner.getMaxDeltaInEachIteration();
                    double lastMaxDelta = maxDeltas.get(maxDeltas.size() - 1);
                    int rewards = Utitlities.calRewards(ep);
                    int numOfActions = ep.actionSequence.size();

                    String[] eval = {"large","valueIteration", numIncentiveLoc + "", incentive + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                    evalues[i * incentives.length * repeat + j * repeat +  r]  = eval;

                    System.out.println("iterations "+numIncentiveLoc + " incentive " + incentive + " repeat " + r);
                    System.out.println( "reward seq size " + ep.rewardSequence.size());
                    System.out.println( "action seq size " + ep.actionSequence.size());
                    System.out.println( "Value model rewards: " + Utitlities.calRewards(ep));
                    System.out.println("total value iteration is : " + viPlanner.getTotalValueIterations());
                    System.out.println("maxDeltas size is : " + maxDeltas.size());
                    System.out.println( "exit reward " + grf.getRewardForTransitionsTo(exit[0],exit[1]));
                    if (r == 0 ){
                        Utitlities.manualValueFunctionVis((ValueFunction)viPlanner, p, mapSize, "Value Iteration with numIncentiveLoc " + numIncentiveLoc + "; incentive " + incentive,initialState,domain,hashingFactory);
                    }

                }
            }
        }
        String[] header = {"gridWorld","strategy", "numIncentiveLoc", "incentive", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"viLargeIncentiveExp");

    }
    public static void testIncentiveSmallMDP(){
        int repeat = 10;
        //int[] iterations = {10,30,50,70,90,110,130,150,170,190};
        int mapSize = GridWorlds.smallMap.length;
        int[] exit = new int[]{0, mapSize - 1};
        double[] incentives = {0, 1, 1.1, 1.2,1.22,1.24};
        String[][] evalues = new String[ incentives.length * repeat][6];

            for (int j = 0; j < incentives.length; j++){
                for (int r = 0; r < repeat; r++){
                    double incentive = incentives[j];
                    GridWorldDomain gwd = new GridWorldDomain(mapSize, mapSize);
                    gwd.setTf(new GridWorldTerminalFunction(exit[0],exit[1]));

                    GridWorldRewardFunction grf = new GridWorldRewardFunction(mapSize,mapSize,-1);

                    grf.setReward(mapSize - 1, 0, -1 + incentive);
                    gwd.setRf(grf);

                    gwd.setMap(GridWorlds.smallMap);
                    //only go in intended directon 80% of the time
                    gwd.setProbSucceedTransitionDynamics(0.8);
                    SADomain domain = gwd.generateDomain();
                    //get initial state with agent in 0,0
                    State initialState = new GridWorldState(new GridAgent(0, 0));
                    SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();



                    ValueIteration viPlanner = new ValueIteration(domain, 0.99,hashingFactory , 0.001);
                    Policy p =  viPlanner.planFromState(initialState);

                    double start = System.nanoTime();
                    Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());
                    double rollOutTime = System.nanoTime() - start;
                    ArrayList<Double> maxDeltas = viPlanner.getMaxDeltaInEachIteration();
                    double lastMaxDelta = maxDeltas.get(maxDeltas.size() - 1);
                    int rewards = Utitlities.calRewards(ep);
                    int numOfActions = ep.actionSequence.size();

                    String[] eval = {"Small","valueIteration",  incentive + "",  rewards + "", numOfActions + "", rollOutTime + ""};
                    evalues[j * repeat +  r]  = eval;

                    System.out.println( " incentive " + incentive + " repeat " + r);
                    System.out.println( "reward seq size " + ep.rewardSequence.size());
                    System.out.println( "action seq size " + ep.actionSequence.size());
                    System.out.println( "Value model rewards: " + Utitlities.calRewards(ep));
                    System.out.println("total value iteration is : " + viPlanner.getTotalValueIterations());

                    if (r == 0 ){
                        Utitlities.manualValueFunctionVis((ValueFunction)viPlanner, p, mapSize, "Value Iteration with incentive " + incentive,initialState,domain,hashingFactory);
                    }

                }
        }
        String[] header = {"gridWorld","strategy",  "incentive", "rewards", "numOfActions", "rollOutTime"};
        Utitlities.writeEvalToFile(evalues,header,"viSmallIncentiveExp");

    }
}
