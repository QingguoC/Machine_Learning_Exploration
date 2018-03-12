import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
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


public class PolicyIterationExperiments {
    public static void main(String[] args){

        testPiIterationsSmallMDP();
        testPiIterationsLargeMDP();
        testIncentiveSmallMDP();
    }

    public static void testPiIterationsSmallMDP(){
        int repeat = 10;
        int[] iterations = {1,2,3,4,5,6,7,8};
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

                    PolicyIteration piPlanner = new PolicyIteration(domain, gamma, hashingFactory,  4, maxIteration);
                    Policy p =  piPlanner.planFromState(initialState);

                    double runningTime = System.nanoTime() - start;
                    Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());

                    ArrayList<Double> maxDeltas = piPlanner.getDeltaInEachPIteration();
                    double lastMaxDelta = maxDeltas.get(maxDeltas.size() - 1);
                    int rewards = Utitlities.calRewards(ep);
                    int numOfActions = ep.actionSequence.size();

                    String[] eval = {"small","PolicyIteration", maxIteration + "", gamma + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                    evalues[i * gammas.length * repeat + j * repeat +  r]  = eval;

                    System.out.println("iterations "+maxIteration + " gamma " + gamma + " repeat " + r);
                    System.out.println( "reward seq size " + ep.rewardSequence.size());
                    System.out.println( "action seq size " + ep.actionSequence.size());
                    System.out.println( "Value model rewards: " + Utitlities.calRewards(ep));
                    System.out.println("total value iteration is : " + piPlanner.getTotalValueIterations());
                    System.out.println("maxDeltas size is : " + maxDeltas.size());
                    if (r == 0 && gamma == 0.99){
                        Utitlities.manualValueFunctionVis((ValueFunction)piPlanner, p, mapSize, "Policy Iteration with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
                    }

                }
            }
        }
        String[] header = {"gridWorld","strategy", "maxIteration", "gamma", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"piSmallIterationExp");

    }
    public static void testPiIterationsLargeMDP(){
        int repeat = 10;
        int[] iterations = {5,10,15,20,25,30};
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

                    PolicyIteration piPlanner = new PolicyIteration(domain, gamma, hashingFactory,  4, maxIteration);
                    Policy p =  piPlanner.planFromState(initialState);

                    double runningTime = System.nanoTime() - start;
                    Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());

                    ArrayList<Double> maxDeltas = piPlanner.getDeltaInEachPIteration();
                    double lastMaxDelta = maxDeltas.get(maxDeltas.size() - 1);
                    int rewards = Utitlities.calRewards(ep);
                    int numOfActions = ep.actionSequence.size();

                    String[] eval = {"large","valueIteration", maxIteration + "", gamma + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                    evalues[i * gammas.length * repeat + j * repeat +  r]  = eval;

                    System.out.println("iterations "+maxIteration + " gamma " + gamma + " repeat " + r);
                    System.out.println( "reward seq size " + ep.rewardSequence.size());
                    System.out.println( "action seq size " + ep.actionSequence.size());
                    System.out.println( "Value model rewards: " + Utitlities.calRewards(ep));
                    System.out.println("total value iteration is : " + piPlanner.getTotalValueIterations());
                    System.out.println("maxDeltas size is : " + maxDeltas.size());
                    if (r == 0 && gamma == 0.99){
                        Utitlities.manualValueFunctionVis((ValueFunction)piPlanner, p, mapSize, "Policy Iteration with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
                    }

                }
            }
        }
        String[] header = {"gridWorld","strategy", "maxIteration", "gamma", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"piLargeIterationExp");

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



                PolicyIteration piPlanner = new PolicyIteration(domain, 0.99, hashingFactory,  0.001,10,30);
                Policy p =  piPlanner.planFromState(initialState);

                double start = System.nanoTime();
                Episode ep = PolicyUtils.rollout(p, initialState, domain.getModel());
                double rollOutTime = System.nanoTime() - start;

                int rewards = Utitlities.calRewards(ep);
                int numOfActions = ep.actionSequence.size();

                String[] eval = {"Small","policyIteration",  incentive + "",  rewards + "", numOfActions + "", rollOutTime + ""};
                evalues[j * repeat +  r]  = eval;

                System.out.println( " incentive " + incentive + " repeat " + r);
                System.out.println( "reward seq size " + ep.rewardSequence.size());
                System.out.println( "action seq size " + ep.actionSequence.size());
                System.out.println( "Value model rewards: " + Utitlities.calRewards(ep));
                System.out.println("total value iteration is : " + piPlanner.getTotalValueIterations());

                if (r == 0 ){
                    Utitlities.manualValueFunctionVis((ValueFunction)piPlanner, p, mapSize, "Policy Iteration with incentive " + incentive,initialState,domain,hashingFactory);
                }

            }
        }
        String[] header = {"gridWorld","strategy",  "incentive", "rewards", "numOfActions", "rollOutTime"};
        Utitlities.writeEvalToFile(evalues,header,"piSmallIncentiveExp");

    }
}

