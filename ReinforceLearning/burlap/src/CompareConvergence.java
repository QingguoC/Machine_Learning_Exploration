import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.util.ArrayList;

public class CompareConvergence {
    public static void main(String[] args){

        testConvergenceSmallMDP();
        testConvergenceLargeMDP();
        testTConvergenceSmallMDP();
        testTConvergenceLargeMDP();
    }
    public static void testConvergenceSmallMDP(){
        int repeat = 20;
        double[] maxDeltas = {0.01,0.001,0.0001};

        String[][] evalues = new String[ maxDeltas.length * repeat * 3][6];
        int mapSize = GridWorlds.smallMap.length;
        State initialState = new GridWorldState(new GridAgent(0, 0));
        for (int i = 0; i < maxDeltas.length; i++){
            for (int r = 0; r < repeat; r++){
                double maxDelta = maxDeltas[i];

                //Value Iteration
                GridWorldDomain gwd0 = new GridWorldDomain(mapSize, mapSize);
                gwd0.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd0.setMap(GridWorlds.smallMap);
                //only go in intended directon 80% of the time
                gwd0.setProbSucceedTransitionDynamics(0.8);
                SADomain domain0 = gwd0.generateDomain();

                SimpleHashableStateFactory hashingFactory0 = new SimpleHashableStateFactory();
                double start = System.nanoTime();

                ValueIteration viPlanner = new ValueIteration(domain0, 0.8,hashingFactory0 , maxDelta);
                Policy p0 =  viPlanner.planFromState(initialState);

                double runningTime = System.nanoTime() - start;
                Episode ep0 = PolicyUtils.rollout(p0, initialState, domain0.getModel());
                int maxIteration = viPlanner.getTotalValueIterations();
                int rewards = Utitlities.calRewards(ep0);

                String[] eval = {"small","valueIteration", maxDelta + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  0]  = eval;
                System.out.println(  "value iteration " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards");

                //Policy Iteration
                GridWorldDomain gwd1 = new GridWorldDomain(mapSize, mapSize);
                gwd1.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd1.setMap(GridWorlds.smallMap);
                //only go in intended directon 80% of the time
                gwd1.setProbSucceedTransitionDynamics(0.8);
                SADomain domain1 = gwd1.generateDomain();

                SimpleHashableStateFactory hashingFactory1 = new SimpleHashableStateFactory();
                start = System.nanoTime();

                PolicyIteration piPlanner = new PolicyIteration(domain1, 0.8,hashingFactory1 , maxDelta, 4, 30);
                Policy p1 =  piPlanner.planFromState(initialState);

                runningTime = System.nanoTime() - start;
                Episode ep1 = PolicyUtils.rollout(p1, initialState, domain1.getModel());
                maxIteration = piPlanner.getTotalValueIterations();
                rewards = Utitlities.calRewards(ep1);

                eval = new String[]{"small","policyIteration", maxDelta + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  1]  = eval;
                System.out.println(  "policy iteration " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards");

                //Q Learning
                GridWorldDomain gwd2 = new GridWorldDomain(mapSize, mapSize);
                gwd2.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd2.setMap(GridWorlds.smallMap);
                //only go in intended directon 80% of the time
                gwd2.setProbSucceedTransitionDynamics(0.8);
                SADomain domain2 = gwd2.generateDomain();

                SimpleHashableStateFactory hashingFactory2 = new SimpleHashableStateFactory();
                start = System.nanoTime();

                QLearning qAgent = new QLearning(domain2, 0.8, hashingFactory2,0., 0.3);
                EpsilonGreedy p2 = (EpsilonGreedy) qAgent.getLearningPolicy();
                p2.setEpsilon(0.2);
                qAgent.setMaximumEpisodesForPlanning(200);
                qAgent.setMaxQChangeForPlanningTerminaiton(maxDelta);
                //qAgent.setMaxRewardsChangeForPlanningTerminaiton(maxDelta);
                qAgent.setTrackLastNmaxQChange(new double[1]);
                //qAgent.setTrackLastNRewardsChange(new double[1]);
                Policy pq = qAgent.planFromState(initialState);
                runningTime = System.nanoTime() - start;
                Episode ep2 = PolicyUtils.rollout(pq, initialState, domain2.getModel());
                maxIteration = qAgent.getNumEpisodes();
                rewards = Utitlities.calRewards(ep2);

                eval = new String[]{"small","QLearning", maxDelta + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  2]  = eval;
                System.out.println(  "QLearning " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards " + qAgent.getTotalNumberOfSteps()+"-totalSteps" );
                if (r == 0 && maxDelta == 0.0001){
                    Utitlities.manualValueFunctionVis((ValueFunction)viPlanner, p0, mapSize, "Value Iteration converged with maxDelta " + maxDelta ,initialState,domain0,hashingFactory0);
                    Utitlities.manualValueFunctionVis((ValueFunction)piPlanner, p1, mapSize, "Policy Iteration converged with maxDelta " + maxDelta ,initialState,domain1,hashingFactory1);
                    Utitlities.manualValueFunctionVis((ValueFunction)qAgent, p2, mapSize, "Q Learning at end of iteration " + maxIteration ,initialState,domain2,hashingFactory2);
                }
            }

        }
        String[] header = {"gridWorld","strategy","maxDelta", "maxIteration",  "rewards",  "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"convergenceSmallWorldExp");

    }
    public static void testConvergenceLargeMDP(){
        int repeat = 20;
        double[] maxDeltas = {0.01,0.001,0.0001};

        String[][] evalues = new String[ maxDeltas.length * repeat * 3][6];
        int mapSize = GridWorlds.largeMap.length;
        State initialState = new GridWorldState(new GridAgent(0, 0));
        for (int i = 0; i < maxDeltas.length; i++){
            for (int r = 0; r < repeat; r++){
                double maxDelta = maxDeltas[i];

                //Value Iteration
                GridWorldDomain gwd0 = new GridWorldDomain(mapSize, mapSize);
                gwd0.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd0.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd0.setProbSucceedTransitionDynamics(0.8);
                SADomain domain0 = gwd0.generateDomain();

                SimpleHashableStateFactory hashingFactory0 = new SimpleHashableStateFactory();
                double start = System.nanoTime();

                ValueIteration viPlanner = new ValueIteration(domain0, 0.8,hashingFactory0 , maxDelta);
                Policy p0 =  viPlanner.planFromState(initialState);

                double runningTime = System.nanoTime() - start;
                Episode ep0 = PolicyUtils.rollout(p0, initialState, domain0.getModel());
                int maxIteration = viPlanner.getTotalValueIterations();
                int rewards = Utitlities.calRewards(ep0);

                String[] eval = {"small","valueIteration", maxDelta + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  0]  = eval;
                System.out.println(  "value iteration " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards");

                //Policy Iteration
                GridWorldDomain gwd1 = new GridWorldDomain(mapSize, mapSize);
                gwd1.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd1.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd1.setProbSucceedTransitionDynamics(0.8);
                SADomain domain1 = gwd1.generateDomain();

                SimpleHashableStateFactory hashingFactory1 = new SimpleHashableStateFactory();
                start = System.nanoTime();

                PolicyIteration piPlanner = new PolicyIteration(domain1, 0.8,hashingFactory1 , maxDelta, 4, 30);
                Policy p1 =  piPlanner.planFromState(initialState);

                runningTime = System.nanoTime() - start;
                Episode ep1 = PolicyUtils.rollout(p1, initialState, domain1.getModel());
                maxIteration = piPlanner.getTotalValueIterations();
                rewards = Utitlities.calRewards(ep1);

                eval = new String[]{"small","policyIteration", maxDelta + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  1]  = eval;
                System.out.println(  "policy iteration " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards");

                //Q Learning
                GridWorldDomain gwd2 = new GridWorldDomain(mapSize, mapSize);
                gwd2.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd2.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd2.setProbSucceedTransitionDynamics(0.8);
                SADomain domain2 = gwd2.generateDomain();

                SimpleHashableStateFactory hashingFactory2 = new SimpleHashableStateFactory();
                start = System.nanoTime();

                QLearning qAgent = new QLearning(domain2, 0.8, hashingFactory2,0., 0.3);
                EpsilonGreedy p2 = (EpsilonGreedy) qAgent.getLearningPolicy();
                p2.setEpsilon(0.2);
                qAgent.setMaximumEpisodesForPlanning(600);
                qAgent.setMaxQChangeForPlanningTerminaiton(maxDelta);
                qAgent.setTrackLastNmaxQChange(new double[1]);
                Policy pq = qAgent.planFromState(initialState);
                runningTime = System.nanoTime() - start;
                Episode ep2 = PolicyUtils.rollout(pq, initialState, domain2.getModel());
                maxIteration = qAgent.getNumEpisodes();
                rewards = Utitlities.calRewards(ep2);

                eval = new String[]{"large","QLearning", maxDelta + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  2]  = eval;
                System.out.println(  "QLearning " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards " + qAgent.getTotalNumberOfSteps()+"-totalSteps" );
                if (r == 0 && maxDelta == 0.0001){
                    Utitlities.manualValueFunctionVis((ValueFunction)viPlanner, p0, mapSize, "Value Iteration converged with maxDelta " + maxDelta ,initialState,domain0,hashingFactory0);
                    Utitlities.manualValueFunctionVis((ValueFunction)piPlanner, p1, mapSize, "Policy Iteration converged with maxDelta " + maxDelta ,initialState,domain1,hashingFactory1);
                    Utitlities.manualValueFunctionVis((ValueFunction)qAgent, p2, mapSize, "Q Learning at end of iteration " + maxIteration ,initialState,domain2,hashingFactory2);
                }
            }

        }
        String[] header = {"gridWorld","strategy","maxDelta", "maxIteration",  "rewards",  "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"convergenceLargeWorldExp");

    }
    public static void testTConvergenceSmallMDP(){
        int repeat = 20;
        double[] Ts = {0.99,0.9,0.8,0.7};
        double maxDelta = 0.001;
        String[][] evalues = new String[ Ts.length * repeat * 3][6];
        int mapSize = GridWorlds.smallMap.length;
        State initialState = new GridWorldState(new GridAgent(0, 0));
        for (int i = 0; i < Ts.length; i++){
            for (int r = 0; r < repeat; r++){
                 double T = Ts[i];

                //Value Iteration
                GridWorldDomain gwd0 = new GridWorldDomain(mapSize, mapSize);
                gwd0.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd0.setMap(GridWorlds.smallMap);
                //only go in intended directon 80% of the time
                gwd0.setProbSucceedTransitionDynamics(T);
                SADomain domain0 = gwd0.generateDomain();

                SimpleHashableStateFactory hashingFactory0 = new SimpleHashableStateFactory();
                double start = System.nanoTime();

                ValueIteration viPlanner = new ValueIteration(domain0, 0.8,hashingFactory0 , maxDelta);
                Policy p0 =  viPlanner.planFromState(initialState);

                double runningTime = System.nanoTime() - start;
                Episode ep0 = PolicyUtils.rollout(p0, initialState, domain0.getModel());
                int maxIteration = viPlanner.getTotalValueIterations();
                int rewards = Utitlities.calRewards(ep0);

                String[] eval = {"small","valueIteration", T + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  0]  = eval;
                System.out.println(  "value iteration " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards");

                //Policy Iteration
                GridWorldDomain gwd1 = new GridWorldDomain(mapSize, mapSize);
                gwd1.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd1.setMap(GridWorlds.smallMap);
                //only go in intended directon 80% of the time
                gwd1.setProbSucceedTransitionDynamics(T);
                SADomain domain1 = gwd1.generateDomain();

                SimpleHashableStateFactory hashingFactory1 = new SimpleHashableStateFactory();
                start = System.nanoTime();

                PolicyIteration piPlanner = new PolicyIteration(domain1, 0.8,hashingFactory1 , maxDelta, 4, 300);
                Policy p1 =  piPlanner.planFromState(initialState);

                runningTime = System.nanoTime() - start;
                Episode ep1 = PolicyUtils.rollout(p1, initialState, domain1.getModel());
                maxIteration = piPlanner.getTotalValueIterations();
                rewards = Utitlities.calRewards(ep1);

                eval = new String[]{"small","policyIteration", T + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  1]  = eval;
                System.out.println(  "policy iteration " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards");

                //Q Learning
                GridWorldDomain gwd2 = new GridWorldDomain(mapSize, mapSize);
                gwd2.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd2.setMap(GridWorlds.smallMap);
                //only go in intended directon 80% of the time
                gwd2.setProbSucceedTransitionDynamics(T);
                SADomain domain2 = gwd2.generateDomain();

                SimpleHashableStateFactory hashingFactory2 = new SimpleHashableStateFactory();
                start = System.nanoTime();

                QLearning qAgent = new QLearning(domain2, 0.8, hashingFactory2,0., 0.3);
                EpsilonGreedy p2 = (EpsilonGreedy) qAgent.getLearningPolicy();
                p2.setEpsilon(0.2);
                qAgent.setMaximumEpisodesForPlanning(200);
                qAgent.setMaxQChangeForPlanningTerminaiton(maxDelta);
                //qAgent.setMaxRewardsChangeForPlanningTerminaiton(maxDelta);
                qAgent.setTrackLastNmaxQChange(new double[1]);
                //qAgent.setTrackLastNRewardsChange(new double[1]);
                Policy pq = qAgent.planFromState(initialState);
                runningTime = System.nanoTime() - start;
                Episode ep2 = PolicyUtils.rollout(pq, initialState, domain2.getModel());
                maxIteration = qAgent.getNumEpisodes();
                rewards = Utitlities.calRewards(ep2);

                eval = new String[]{"small","QLearning", T + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  2]  = eval;
                System.out.println(  "QLearning " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards " + qAgent.getTotalNumberOfSteps()+"-totalSteps" );
                if (r == 0 ){
                    Utitlities.manualValueFunctionVis((ValueFunction)viPlanner, p0, mapSize, "Value Iteration converged with T " + T ,initialState,domain0,hashingFactory0);
                    Utitlities.manualValueFunctionVis((ValueFunction)piPlanner, p1, mapSize, "Policy Iteration converged with T " + T ,initialState,domain1,hashingFactory1);
                    Utitlities.manualValueFunctionVis((ValueFunction)qAgent, p2, mapSize, "Q Learning with T " + T ,initialState,domain2,hashingFactory2);
                }
            }

        }
        String[] header = {"gridWorld","strategy","T", "maxIteration",  "rewards",  "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"TConvergenceSmallWorldExp");

    }
    public static void testTConvergenceLargeMDP(){
        int repeat = 20;
        double[] Ts = {0.99,0.9,0.8,0.7};
        double maxDelta = 0.001;
        String[][] evalues = new String[ Ts.length * repeat * 3][6];
        int mapSize = GridWorlds.largeMap.length;
        State initialState = new GridWorldState(new GridAgent(0, 0));
        for (int i = 0; i < Ts.length; i++){
            for (int r = 0; r < repeat; r++){
                double T = Ts[i];

                //Value Iteration
                GridWorldDomain gwd0 = new GridWorldDomain(mapSize, mapSize);
                gwd0.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd0.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd0.setProbSucceedTransitionDynamics(T);
                SADomain domain0 = gwd0.generateDomain();

                SimpleHashableStateFactory hashingFactory0 = new SimpleHashableStateFactory();
                double start = System.nanoTime();

                ValueIteration viPlanner = new ValueIteration(domain0, 0.8,hashingFactory0 , maxDelta);
                Policy p0 =  viPlanner.planFromState(initialState);

                double runningTime = System.nanoTime() - start;
                Episode ep0 = PolicyUtils.rollout(p0, initialState, domain0.getModel());
                int maxIteration = viPlanner.getTotalValueIterations();
                int rewards = Utitlities.calRewards(ep0);

                String[] eval = {"small","valueIteration", T + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  0]  = eval;
                System.out.println(  "value iteration " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards");

                //Policy Iteration
                GridWorldDomain gwd1 = new GridWorldDomain(mapSize, mapSize);
                gwd1.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd1.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd1.setProbSucceedTransitionDynamics(T);
                SADomain domain1 = gwd1.generateDomain();

                SimpleHashableStateFactory hashingFactory1 = new SimpleHashableStateFactory();
                start = System.nanoTime();

                PolicyIteration piPlanner = new PolicyIteration(domain1, 0.8,hashingFactory1 , maxDelta, 4, 1000);
                Policy p1 =  piPlanner.planFromState(initialState);

                runningTime = System.nanoTime() - start;
                Episode ep1 = PolicyUtils.rollout(p1, initialState, domain1.getModel());
                maxIteration = piPlanner.getTotalValueIterations();
                rewards = Utitlities.calRewards(ep1);

                eval = new String[]{"small","policyIteration", T + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  1]  = eval;
                System.out.println(  "policy iteration " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards");

                //Q Learning
                GridWorldDomain gwd2 = new GridWorldDomain(mapSize, mapSize);
                gwd2.setTf(new GridWorldTerminalFunction(0, mapSize - 1));

                gwd2.setMap(GridWorlds.largeMap);
                //only go in intended directon 80% of the time
                gwd2.setProbSucceedTransitionDynamics(T);
                SADomain domain2 = gwd2.generateDomain();

                SimpleHashableStateFactory hashingFactory2 = new SimpleHashableStateFactory();
                start = System.nanoTime();

                QLearning qAgent = new QLearning(domain2, 0.8, hashingFactory2,0., 0.3);
                EpsilonGreedy p2 = (EpsilonGreedy) qAgent.getLearningPolicy();
                p2.setEpsilon(0.2);
                qAgent.setMaximumEpisodesForPlanning(600);
                qAgent.setMaxQChangeForPlanningTerminaiton(maxDelta);
                //qAgent.setMaxRewardsChangeForPlanningTerminaiton(maxDelta);
                qAgent.setTrackLastNmaxQChange(new double[1]);
                //qAgent.setTrackLastNRewardsChange(new double[1]);
                Policy pq = qAgent.planFromState(initialState);
                runningTime = System.nanoTime() - start;
                Episode ep2 = PolicyUtils.rollout(pq, initialState, domain2.getModel());
                maxIteration = qAgent.getNumEpisodes();
                rewards = Utitlities.calRewards(ep2);

                eval = new String[]{"large","QLearning", T + "", maxIteration + "",   rewards + "", runningTime + ""};
                evalues[i * repeat * 3 + r * 3 +  2]  = eval;
                System.out.println(  "QLearning " + maxIteration  + "-maxIteration " + r + "-repeat " + runningTime + "-runningTime " + rewards + "-rewards " + qAgent.getTotalNumberOfSteps()+"-totalSteps" );
                if (r == 0 ){
                    Utitlities.manualValueFunctionVis((ValueFunction)viPlanner, p0, mapSize, "Value Iteration converged with T " + T ,initialState,domain0,hashingFactory0);
                    Utitlities.manualValueFunctionVis((ValueFunction)piPlanner, p1, mapSize, "Policy Iteration converged with T " + T ,initialState,domain1,hashingFactory1);
                    Utitlities.manualValueFunctionVis((ValueFunction)qAgent, p2, mapSize, "Q Learning with T " + T ,initialState,domain2,hashingFactory2);
                }
            }

        }
        String[] header = {"gridWorld","strategy","T", "maxIteration",  "rewards",  "runningTime"};
        Utitlities.writeEvalToFile(evalues,header,"TConvergenceLargeWorldExp");

    }
}
