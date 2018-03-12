import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.util.ArrayList;

public class QLearningExperiments {
    public static void main(String[] args){

        testQIterationsGammaSmallMDP();
        testQIterationsGammaLargeMDP();
        testQIterationsEpsilonSmallMDP();
        testQIterationsEpsilonLargeMDP();
        testQIterationsLearningRateSmallMDP();
        testQIterationsLearningRateLargeMDP();
        testQIterationsInitQSmallMDP();
        testQIterationsInitQLargeMDP();
    }

    public static void testQIterationsGammaSmallMDP(){
        int repeat = 20;
        int[] iterations = {1,31,61,91,121,151};
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

                    SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);
                    QLearning agent = new QLearning(domain, gamma, new SimpleHashableStateFactory(),0., 0.1);
                    EpsilonGreedy p = (EpsilonGreedy) agent.getLearningPolicy();
                    p.setEpsilon(0.1);
                    Episode lastEp = null;
                    for(int m = 0; m < maxIteration; m++){
                        Episode e = agent.runLearningEpisode(env);

                        System.out.println(maxIteration + ": " + m + ": " + Utitlities.calRewards(e));
                        //System.out.println(i + ": " + e.action(0));
                        //reset environment for next learning episode
                        env.resetEnvironment();
                        if (m == maxIteration - 1){
                            lastEp = e;
                        }
                    }
                    double runningTime = System.nanoTime() - start;

                    double lastMaxDelta = agent.getMaxQChangeInLastEpisode();
                    int rewards = Utitlities.calRewards(lastEp);
                    int numOfActions = lastEp.actionSequence.size();

                    String[] eval = {"small","Q", maxIteration + "", gamma + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                    evalues[i * gammas.length * repeat + j * repeat +  r]  = eval;

                    System.out.println("iterations "+maxIteration + " gamma " + gamma + " repeat " + r);
                    System.out.println( "reward seq size " + lastEp.rewardSequence.size());
                    System.out.println( "getLastNumSteps " + agent.getLastNumSteps());
                    System.out.println( "Value model rewards: " + Utitlities.calRewards(lastEp));
                    System.out.println("TotalNumberOfSteps is : " + agent.getTotalNumberOfSteps());
                    if (r == 0 && gamma == 0.99){
                        Utitlities.manualValueFunctionVis((ValueFunction)agent, p, mapSize, "Q Learning with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
                    }

                }
            }
        }
        String[] header = {"gridWorld","strategy", "maxIteration", "gamma", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"qSmallIterationGammaExp");

    }
    public static void testQIterationsGammaLargeMDP(){
        int repeat = 20;
        int[] iterations = {100,200,300,400,500,600};
        double[] gammas = {0.8,0.9,0.99};
        String[][] evalues = new String[iterations.length * gammas.length * repeat][8];

        for (int i = 0; i < iterations.length; i++){
            for (int j = 0; j < gammas.length; j++){
                for (int r = 0; r < repeat; r++){
                    int maxIteration = iterations[i];
                    double gamma = gammas[j];
                    int mapSize = GridWorlds.largeMap.length;
                    GridWorldDomain gwd = new GridWorldDomain(mapSize, mapSize);
                    gwd.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                    gwd.setMap(GridWorlds.largeMap);
                    //only go in intended directon 80% of the time
                    gwd.setProbSucceedTransitionDynamics(0.8);
                    SADomain domain = gwd.generateDomain();
                    //get initial state with agent in 0,0
                    State initialState = new GridWorldState(new GridAgent(0, 0));
                    SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
                    double start = System.nanoTime();

                    SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);
                    QLearning agent = new QLearning(domain, gamma, new SimpleHashableStateFactory(),0., 0.1);
                    EpsilonGreedy p = (EpsilonGreedy) agent.getLearningPolicy();
                    p.setEpsilon(0.1);
                    Episode lastEp = null;
                    for(int m = 0; m < maxIteration; m++){
                        Episode e = agent.runLearningEpisode(env);

                        //System.out.println(maxIteration + ": " + m + ": " + e.maxTimeStep());
                        System.out.println(maxIteration + ": " + m + ": " + Utitlities.calRewards(e));
                        //System.out.println(i + ": " + e.action(0));
                        //reset environment for next learning episode
                        env.resetEnvironment();
                        if (m == maxIteration - 1){
                            lastEp = e;
                        }
                    }
                    double runningTime = System.nanoTime() - start;

                    double lastMaxDelta = agent.getMaxQChangeInLastEpisode();
                    int rewards = Utitlities.calRewards(lastEp);
                    int numOfActions = lastEp.actionSequence.size();

                    String[] eval = {"large","Q", maxIteration + "", gamma + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                    evalues[i * gammas.length * repeat + j * repeat +  r]  = eval;

                    System.out.println("iterations "+maxIteration + " gamma " + gamma + " repeat " + r);
                    System.out.println( "reward seq size " + lastEp.rewardSequence.size());
                    System.out.println( "getLastNumSteps " + agent.getLastNumSteps());
                    System.out.println( "Value model rewards: " + Utitlities.calRewards(lastEp));
                    System.out.println("TotalNumberOfSteps is : " + agent.getTotalNumberOfSteps());
                    if (r == 0 && gamma == 0.99){
                        Utitlities.manualValueFunctionVis((ValueFunction)agent, p, mapSize, "Q Learning with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
                    }

                }
            }
        }
        String[] header = {"gridWorld","strategy", "maxIteration", "gamma", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"qSmallIterationGammaExp");

    }

    public static void testQIterationsEpsilonSmallMDP(){
        int repeat = 20;
        //double[] learningRates = {0.1,0.2,0.3};
        double[] epsilons = {0.1,0.11,0.15, 0.2 ,0.3};
        int maxIteration = 160;
        double learningRate = 0.1;
        String[][] evalues = new String[ epsilons.length * repeat * maxIteration/4][7];


            for (int j = 0; j < epsilons.length; j++){
                for (int r = 0; r < repeat; r++){
                    //double learningRate = learningRates[i];
                    double epsilon = epsilons[j];
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

                    SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);
                    QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(),0., learningRate);
                    EpsilonGreedy p = (EpsilonGreedy) agent.getLearningPolicy();
                    p.setEpsilon(epsilon);
                    //Episode lastEp = null;
                    for(int m = 0; m < maxIteration; m++){
                        Episode e = agent.runLearningEpisode(env);

                        System.out.println( learningRate + "-learningrate " +  epsilon + "-epsilon " + m  + "-iteration " + r + "-repeat " + Utitlities.calRewards(e));
                        //System.out.println(i + ": " + e.action(0));
                        //reset environment for next learning episode
                        env.resetEnvironment();
                        if (m % 4 == 0){
                            double runningTime = System.nanoTime() - start;

                            double lastMaxDelta = agent.getMaxQChangeInLastEpisode();
                            int rewards = Utitlities.calRewards(e);
                            int numOfActions = e.actionSequence.size();
                            String[] eval = {"small",  (m + 1) + "", epsilon + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                            evalues[j * repeat * maxIteration/4 + r * maxIteration/4 + m/4] = eval;
                        }


                    }




//                    System.out.println("iterations "+maxIteration + " epsilon " + epsilon + " repeat " + r);
//                    System.out.println( "reward seq size " + lastEp.rewardSequence.size());
//                    System.out.println( "getLastNumSteps " + agent.getLastNumSteps());
//                    System.out.println( "Value model rewards: " + Utitlities.calRewards(lastEp));
//                    System.out.println("TotalNumberOfSteps is : " + agent.getTotalNumberOfSteps());
//                    if (r == 0 && epsilon == 0.99){
//                        Utitlities.manualValueFunctionVis((ValueFunction)agent, p, mapSize, "Value Iteration with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
//                    }

                }
            }

        String[] header = {"gridWorld", "iteration", "epsilon", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"qSmallIterationEpsilonExp");

    }
    public static void testQIterationsEpsilonLargeMDP(){
        int repeat = 20;
        //double[] learningRates = {0.1,0.2,0.3};
        double[] epsilons = {0.1,0.2,0.25, 0.3 ,0.4};
        int maxIteration = 600;
        double learningRate = 0.1;
        String[][] evalues = new String[ epsilons.length * repeat * maxIteration/5][7];


        for (int j = 0; j < epsilons.length; j++){
            for (int r = 0; r < repeat; r++){
                //double learningRate = learningRates[i];
                double epsilon = epsilons[j];
                int mapSize = GridWorlds.largeMap.length;
                GridWorldDomain gwd = new GridWorldDomain(mapSize, mapSize);
                gwd.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd.setProbSucceedTransitionDynamics(0.8);
                SADomain domain = gwd.generateDomain();
                //get initial state with agent in 0,0
                State initialState = new GridWorldState(new GridAgent(0, 0));
                SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
                double start = System.nanoTime();

                SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);
                QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(),0., learningRate);
                EpsilonGreedy p = (EpsilonGreedy) agent.getLearningPolicy();
                p.setEpsilon(epsilon);
                //Episode lastEp = null;
                for(int m = 0; m < maxIteration; m++){
                    Episode e = agent.runLearningEpisode(env);

                    System.out.println( learningRate + "-learningrate " +  epsilon + "-epsilon " + m  + "-iteration " + r + "-repeat " + Utitlities.calRewards(e));
                    //System.out.println(i + ": " + e.action(0));
                    //reset environment for next learning episode
                    env.resetEnvironment();
                    if (m % 5 == 0){
                        double runningTime = System.nanoTime() - start;

                        double lastMaxDelta = agent.getMaxQChangeInLastEpisode();
                        int rewards = Utitlities.calRewards(e);
                        int numOfActions = e.actionSequence.size();
                        String[] eval = {"large",  (m + 1) + "", epsilon + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                        evalues[j * repeat * maxIteration/5 + r * maxIteration/5 + m/5] = eval;
                    }


                }




//                    System.out.println("iterations "+maxIteration + " epsilon " + epsilon + " repeat " + r);
//                    System.out.println( "reward seq size " + lastEp.rewardSequence.size());
//                    System.out.println( "getLastNumSteps " + agent.getLastNumSteps());
//                    System.out.println( "Value model rewards: " + Utitlities.calRewards(lastEp));
//                    System.out.println("TotalNumberOfSteps is : " + agent.getTotalNumberOfSteps());
//                    if (r == 0 && epsilon == 0.99){
//                        Utitlities.manualValueFunctionVis((ValueFunction)agent, p, mapSize, "Value Iteration with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
//                    }

            }
        }

        String[] header = {"gridWorld", "iteration", "epsilon", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"qLargeIterationEpsilonExp");

    }
    public static void testQIterationsLearningRateSmallMDP(){
        int repeat = 20;
        double[] learningRates = {0.1,0.2,0.3};
        //double[] epsilons = {0.1,0.11,0.15, 0.2 ,0.3};
        int maxIteration = 160;
        double epsilon = 0.2;
        String[][] evalues = new String[ learningRates.length * repeat * maxIteration/4][7];


        for (int j = 0; j < learningRates.length; j++){
            for (int r = 0; r < repeat; r++){
                //double learningRate = learningRates[i];
                double learningRate = learningRates[j];
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

                SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);

                QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(),0., learningRate);
                EpsilonGreedy p = (EpsilonGreedy) agent.getLearningPolicy();
                p.setEpsilon(epsilon);
                //Episode lastEp = null;
                for(int m = 0; m < maxIteration; m++){
                    Episode e = agent.runLearningEpisode(env);

                    System.out.println( learningRate + "-learningrate " +  epsilon + "-epsilon " + m  + "-iteration " + r + "-repeat " + Utitlities.calRewards(e));
                    //System.out.println(i + ": " + e.action(0));
                    //reset environment for next learning episode
                    env.resetEnvironment();
                    if (m % 4 == 0){
                        double runningTime = System.nanoTime() - start;

                        double lastMaxDelta = agent.getMaxQChangeInLastEpisode();
                        int rewards = Utitlities.calRewards(e);
                        int numOfActions = e.actionSequence.size();
                        String[] eval = {"small",  (m + 1) + "", learningRate + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                        evalues[j * repeat * maxIteration/4 + r * maxIteration/4 + m/4] = eval;
                    }


                }




//                    System.out.println("iterations "+maxIteration + " epsilon " + epsilon + " repeat " + r);
//                    System.out.println( "reward seq size " + lastEp.rewardSequence.size());
//                    System.out.println( "getLastNumSteps " + agent.getLastNumSteps());
//                    System.out.println( "Value model rewards: " + Utitlities.calRewards(lastEp));
//                    System.out.println("TotalNumberOfSteps is : " + agent.getTotalNumberOfSteps());
//                    if (r == 0 && epsilon == 0.99){
//                        Utitlities.manualValueFunctionVis((ValueFunction)agent, p, mapSize, "Value Iteration with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
//                    }

            }
        }

        String[] header = {"gridWorld", "iteration", "learningRate", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"qSmallIterationLearningRateExp");

    }
    public static void testQIterationsLearningRateLargeMDP(){
        int repeat = 20;
        double[] learningRates = {0.1,0.2,0.3};
        //double[] epsilons = {0.1,0.11,0.15, 0.2 ,0.3};
        int maxIteration = 600;
        double epsilon = 0.2;
        String[][] evalues = new String[ learningRates.length * repeat * maxIteration/5][7];


        for (int j = 0; j < learningRates.length; j++){
            for (int r = 0; r < repeat; r++){
                //double learningRate = learningRates[i];
                double learningRate = learningRates[j];
                int mapSize = GridWorlds.largeMap.length;
                GridWorldDomain gwd = new GridWorldDomain(mapSize, mapSize);
                gwd.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd.setProbSucceedTransitionDynamics(0.8);
                SADomain domain = gwd.generateDomain();
                //get initial state with agent in 0,0
                State initialState = new GridWorldState(new GridAgent(0, 0));
                SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
                double start = System.nanoTime();

                SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);

                QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(),0., learningRate);
                EpsilonGreedy p = (EpsilonGreedy) agent.getLearningPolicy();
                p.setEpsilon(epsilon);
                //Episode lastEp = null;
                for(int m = 0; m < maxIteration; m++){
                    Episode e = agent.runLearningEpisode(env);

                    System.out.println( learningRate + "-learningrate " +  epsilon + "-epsilon " + m  + "-iteration " + r + "-repeat " + Utitlities.calRewards(e));
                    //System.out.println(i + ": " + e.action(0));
                    //reset environment for next learning episode
                    env.resetEnvironment();
                    if (m % 5 == 0){
                        double runningTime = System.nanoTime() - start;

                        double lastMaxDelta = agent.getMaxQChangeInLastEpisode();
                        int rewards = Utitlities.calRewards(e);
                        int numOfActions = e.actionSequence.size();
                        String[] eval = {"small",  (m + 1) + "", learningRate + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                        evalues[j * repeat * maxIteration/5 + r * maxIteration/5 + m/5] = eval;
                    }


                }




//                    System.out.println("iterations "+maxIteration + " epsilon " + epsilon + " repeat " + r);
//                    System.out.println( "reward seq size " + lastEp.rewardSequence.size());
//                    System.out.println( "getLastNumSteps " + agent.getLastNumSteps());
//                    System.out.println( "Value model rewards: " + Utitlities.calRewards(lastEp));
//                    System.out.println("TotalNumberOfSteps is : " + agent.getTotalNumberOfSteps());
//                    if (r == 0 && epsilon == 0.99){
//                        Utitlities.manualValueFunctionVis((ValueFunction)agent, p, mapSize, "Value Iteration with iteration " + maxIteration + "; gamma " + gamma,initialState,domain,hashingFactory);
//                    }

            }
        }

        String[] header = {"gridWorld", "iteration", "learningRate", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"qLargeIterationLearningRateExp");

    }
    public static void testQIterationsInitQSmallMDP(){
        int repeat = 20;
        double[] initQs = {-0.5, 0,0.5};
        //double[] epsilons = {0.1,0.11,0.15, 0.2 ,0.3};
        int maxIteration = 160;
        double epsilon = 0.2;
        double learningRate = 0.3;
        String[][] evalues = new String[ initQs.length * repeat * maxIteration/4][7];


        for (int j = 0; j < initQs.length; j++){
            for (int r = 0; r < repeat; r++){
                //double learningRate = learningRates[i];
                double initQ = initQs[j];
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

                SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);

                QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(),initQ, learningRate);
                EpsilonGreedy p = (EpsilonGreedy) agent.getLearningPolicy();
                p.setEpsilon(epsilon);
                //Episode lastEp = null;
                for(int m = 0; m < maxIteration; m++){
                    Episode e = agent.runLearningEpisode(env);

                    System.out.println( initQ + "-initQ " +  epsilon + "-epsilon " + m  + "-iteration " + r + "-repeat " + Utitlities.calRewards(e));
                    //System.out.println(i + ": " + e.action(0));
                    //reset environment for next learning episode
                    env.resetEnvironment();
                    if (m % 4 == 0){
                        double runningTime = System.nanoTime() - start;

                        double lastMaxDelta = agent.getMaxQChangeInLastEpisode();
                        int rewards = Utitlities.calRewards(e);
                        int numOfActions = e.actionSequence.size();
                        String[] eval = {"small",  (m + 1) + "", initQ + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                        evalues[j * repeat * maxIteration/4 + r * maxIteration/4 + m/4] = eval;
                    }


                }




//                    System.out.println("iterations "+maxIteration + " epsilon " + epsilon + " repeat " + r);
//                    System.out.println( "reward seq size " + lastEp.rewardSequence.size());
//                    System.out.println( "getLastNumSteps " + agent.getLastNumSteps());
//                    System.out.println( "Value model rewards: " + Utitlities.calRewards(lastEp));
//                    System.out.println("TotalNumberOfSteps is : " + agent.getTotalNumberOfSteps());
                    if (r == 0 ){
                        Utitlities.manualValueFunctionVis((ValueFunction)agent, p, mapSize, "Q Learning with iteration " + maxIteration + "; initQ " + initQ,initialState,domain,hashingFactory);
                    }

            }
        }

        String[] header = {"gridWorld", "iteration", "initQ", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"qSmallIterationInitQExp");

    }
    public static void testQIterationsInitQLargeMDP(){
        int repeat = 20;
        double[] initQs = {-0.5, 0,0.5};
        //double[] epsilons = {0.1,0.11,0.15, 0.2 ,0.3};
        int maxIteration = 600;
        double epsilon = 0.2;
        double learningRate = 0.3;
        String[][] evalues = new String[ initQs.length * repeat * maxIteration/5][7];


        for (int j = 0; j < initQs.length; j++){
            for (int r = 0; r < repeat; r++){
                //double learningRate = learningRates[i];
                double initQ = initQs[j];
                int mapSize = GridWorlds.largeMap.length;
                GridWorldDomain gwd = new GridWorldDomain(mapSize, mapSize);
                gwd.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd.setProbSucceedTransitionDynamics(0.8);
                SADomain domain = gwd.generateDomain();
                //get initial state with agent in 0,0
                State initialState = new GridWorldState(new GridAgent(0, 0));
                SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
                double start = System.nanoTime();

                SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);

                QLearning agent = new QLearning(domain, 0.99, new SimpleHashableStateFactory(),initQ, learningRate);
                EpsilonGreedy p = (EpsilonGreedy) agent.getLearningPolicy();
                p.setEpsilon(epsilon);
                //Episode lastEp = null;
                for(int m = 0; m < maxIteration; m++){
                    Episode e = agent.runLearningEpisode(env);

                    System.out.println( initQ + "-initQ " +  epsilon + "-epsilon " + m  + "-iteration " + r + "-repeat " + Utitlities.calRewards(e));
                    //System.out.println(i + ": " + e.action(0));
                    //reset environment for next learning episode
                    env.resetEnvironment();
                    if (m % 5 == 0){
                        double runningTime = System.nanoTime() - start;

                        double lastMaxDelta = agent.getMaxQChangeInLastEpisode();
                        int rewards = Utitlities.calRewards(e);
                        int numOfActions = e.actionSequence.size();
                        String[] eval = {"large",  (m + 1) + "", initQ + "", lastMaxDelta + "", rewards + "", numOfActions + "", runningTime + ""};
                        evalues[j * repeat * maxIteration/5 + r * maxIteration/5 + m/5] = eval;
                    }


                }




//                    System.out.println("iterations "+maxIteration + " epsilon " + epsilon + " repeat " + r);
//                    System.out.println( "reward seq size " + lastEp.rewardSequence.size());
//                    System.out.println( "getLastNumSteps " + agent.getLastNumSteps());
//                    System.out.println( "Value model rewards: " + Utitlities.calRewards(lastEp));
//                    System.out.println("TotalNumberOfSteps is : " + agent.getTotalNumberOfSteps());
                if (r == 0 ){
                    Utitlities.manualValueFunctionVis((ValueFunction)agent, p, mapSize, "Q Learning with iteration " + maxIteration + "; initQ " + initQ,initialState,domain,hashingFactory);
                }

            }
        }

        String[] header = {"gridWorld", "iteration", "initQ", "lastMaxDelta", "rewards", "numOfActions", "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"qLargeIterationInitQExp");

    }
}
