
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

public class CountOnes {

    public static void main(String[] args) {
        evalCOs();
        evalCOsTime();
    }
    private static void evalCOs(){
        int repeat = 30;
        int[] iterations = {10,30,60,100,200,400};
        int[] Ns = {100,200,300,400,500,600,700};
        String[][] evalues = new String[repeat * iterations.length * Ns.length * 4][5];
        for(int i = 0;i < repeat; i++){
            for(int j = 0; j < iterations.length; j++) {
                for(int k = 0; k <Ns.length; k++) {
                    int iteration = iterations[j];
                    int N = Ns[k];
                    int[] ranges = new int[N];
                    Arrays.fill(ranges, 2);
                    System.out.println("Starting the " + (i+1) + " repeat, max iterations " + iteration +", Number of Points is " + N + ".");
                    EvaluationFunction ef = new CountOnesEvaluationFunction();
                    Distribution odd = new DiscreteUniformDistribution(ranges);
                    NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
                    MutationFunction mf = new DiscreteChangeOneMutation(ranges);
                    CrossoverFunction cf = new UniformCrossOver();
                    Distribution df = new DiscreteDependencyTree(.1, ranges);
                    HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                    GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                    ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);


                    double start = System.nanoTime();
                    RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                    FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iteration *100);
                    fit.train();
                    //System.out.println(ef.value(rhc.getOptimal()));
                    double end = System.nanoTime();
                    double trainingTime = end - start;
                    String[] rhcEval = {"RHC", iteration + "", N + "", ef.value(rhc.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 0] = rhcEval;

                    start = System.nanoTime();
                    SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
                    fit = new FixedIterationTrainer(sa, iteration *100);
                    fit.train();
                    //System.out.println(ef.value(sa.getOptimal()));
                    end = System.nanoTime();
                    trainingTime = end - start;
                    String[] saEval = {"SA", iteration + "", N + "", ef.value(sa.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 1] = saEval;

                    start = System.nanoTime();
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
                    fit = new FixedIterationTrainer(ga, iteration);
                    fit.train();
                    //System.out.println(ef.value(ga.getOptimal()));
                    end = System.nanoTime();
                    trainingTime = end - start;
                    String[] gaEval = {"GA", iteration + "", N + "", ef.value(ga.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 2] = gaEval;


                    start = System.nanoTime();
                    MIMIC mimic = new MIMIC(50, 10, pop);
                    fit = new FixedIterationTrainer(mimic, iteration);
                    fit.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    //System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
                    String[] mimicEval = {"MIMIC", iteration + "", N + "", ef.value(mimic.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 3] = mimicEval;
                }
            }
        }
        String[] headerEval = {"Algorithm","Iteration","N","Evaluation","TrainingTime"};

        writeEvalToFile(evalues,headerEval,"evaluesOfFourROsForCountOnes");
    }

    private static void evalCOsTime(){
        int repeat = 4;
        int[] iterations = {10,20,30,40,50,60};
        int[] Ns = {10,20,30,40,50,60};
        String[][] evalues = new String[repeat * iterations.length * Ns.length * 4][5];
        for(int i = 0;i < repeat; i++){
            for(int j = 0; j < iterations.length; j++) {
                for(int k = 0; k <Ns.length; k++) {
                    int iteration = iterations[j];
                    int N = Ns[k];
                    int[] ranges = new int[N];
                    Arrays.fill(ranges, 2);
                    System.out.println("Starting the " + (i+1) + " repeat, max iterations " + iteration +", Number of Points is " + N + ".");
                    EvaluationFunction ef = new CountOnesEvaluationFunction();
                    Distribution odd = new DiscreteUniformDistribution(ranges);
                    NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
                    MutationFunction mf = new DiscreteChangeOneMutation(ranges);
                    CrossoverFunction cf = new UniformCrossOver();
                    Distribution df = new DiscreteDependencyTree(.1, ranges);
                    HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                    GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                    ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);


                    double start = System.nanoTime();
                    RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                    FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iteration *100);
                    fit.train();
                    //System.out.println(ef.value(rhc.getOptimal()));
                    double end = System.nanoTime();
                    double trainingTime = end - start;
                    String[] rhcEval = {"RHC", iteration + "", N + "", ef.value(rhc.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 0] = rhcEval;

                    start = System.nanoTime();
                    SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
                    fit = new FixedIterationTrainer(sa, iteration *100);
                    fit.train();
                    //System.out.println(ef.value(sa.getOptimal()));
                    end = System.nanoTime();
                    trainingTime = end - start;
                    String[] saEval = {"SA", iteration + "", N + "", ef.value(sa.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 1] = saEval;

                    start = System.nanoTime();
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
                    fit = new FixedIterationTrainer(ga, iteration);
                    fit.train();
                    //System.out.println(ef.value(ga.getOptimal()));
                    end = System.nanoTime();
                    trainingTime = end - start;
                    String[] gaEval = {"GA", iteration + "", N + "", ef.value(ga.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 2] = gaEval;


                    start = System.nanoTime();
                    MIMIC mimic = new MIMIC(50, 10, pop);
                    fit = new FixedIterationTrainer(mimic, iteration);
                    fit.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    //System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
                    String[] mimicEval = {"MIMIC", iteration + "", N + "", ef.value(mimic.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 3] = mimicEval;
                }
            }
        }
        String[] headerEval = {"Algorithm","Iteration","N","Evaluation","TrainingTime"};

        writeEvalToFile(evalues,headerEval,"evaluesOfFourROsForCountOnesTime");

    }
    private static void writeEvalToFile(String[][] logs,String[] header, String fileName){

        StringBuilder builder = new StringBuilder();
        for(int i = 0; i < header.length; i++)//for each column
        {
            builder.append(header[i]);//append to the output string
            if(i < header.length - 1)//if this is not the last row element
                builder.append(",");//then add comma
        }
        builder.append("\n");

        for(int i = 0; i < logs.length; i++)//for each iteration
        {
            for(int j = 0; j < logs[0].length; j++)//for each RO
            {
                builder.append(logs[i][j]);//append to the output string
                if(j < logs[0].length - 1)//if this is not the last row element
                    builder.append(",");//then add comma
            }
            builder.append("\n");//append new line at the end of the row
        }
        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new FileWriter("src/data/" + fileName + ".txt"));
            writer.write(builder.toString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
