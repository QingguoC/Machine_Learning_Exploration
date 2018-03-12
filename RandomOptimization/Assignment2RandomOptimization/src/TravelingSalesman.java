
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.*;
import opt.example.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
public class TravelingSalesman {


    public static void main(String[] args) {
        fourROsForTSP();
        fourROsForTSPTime();
    }

    private static void fourROsForTSP(){

        int repeat = 20;
        int[] iterations = {10,30,60,100,200,400};
        int[] Ns = {10,20,40,60,80};
        String[][] evalues = new String[repeat * iterations.length * Ns.length * 4][5];
        for(int i = 0;i < repeat; i++){
            for(int j = 0; j < iterations.length; j++) {
                for(int k = 0; k <Ns.length; k++) {
                    int iteration = iterations[j];
                    int N = Ns[k];
                    Random random = new Random();
                    // create the random points
                    double[][] points = new double[N][2];
                    for (int m = 0; m < points.length; m++) {
                        points[m][0] = random.nextDouble();
                        points[m][1] = random.nextDouble();
                    }
                    System.out.println("Starting the " + (i+1) + " repeat, max iterations " + iteration +", Number of Points is " + N + ".");
                    // for rhc, sa, and ga we use a permutation based encoding
                    TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
                    Distribution odd = new DiscretePermutationDistribution(N);
                    NeighborFunction nf = new SwapNeighbor();
                    MutationFunction mf = new SwapMutation();
                    CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
                    HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                    GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

                    double start = System.nanoTime();
                    RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                    FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iteration * 10);
                    fit.train();
                    double end = System.nanoTime();
                    double trainingTime = end - start;
                    //System.out.println("RHC: " + ef.value(rhc.getOptimal()));
                    String[] rhcEval = {"RHC", iteration + "", N + "", ef.value(rhc.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 0] = rhcEval;

                    start = System.nanoTime();
                    SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
                    fit = new FixedIterationTrainer(sa, iteration * 10);
                    fit.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    //System.out.println("SA: " + ef.value(sa.getOptimal()));
                    String[] saEval = {"SA", iteration + "", N + "", ef.value(sa.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 1] = saEval;

                    start = System.nanoTime();
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
                    fit = new FixedIterationTrainer(ga, iteration);
                    fit.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    //System.out.println("GA: " + ef.value(ga.getOptimal()));
                    String[] gaEval = {"GA", iteration + "", N + "", ef.value(ga.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 2] = gaEval;

                    start = System.nanoTime();
                    // for mimic we use a sort encoding
                    ef = new TravelingSalesmanSortEvaluationFunction(points);
                    int[] ranges = new int[N];
                    Arrays.fill(ranges, N);
                    odd = new DiscreteUniformDistribution(ranges);
                    Distribution df = new DiscreteDependencyTree(.1, ranges);
                    ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                    MIMIC mimic = new MIMIC(200, 20, pop);
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
        String[] headerEval = {"Algorithm","Iteration","NumberOfPoints","Evaluation","TrainingTime"};

        writeEvalToFile(evalues,headerEval,"evaluesOfFourROsForTSP");
    }
    private static void fourROsForTSPTime(){

        int repeat = 4;
        int[] iterations = {10,20,30,40,50,60};
        int[] Ns = {10,20,30,40,50,60};
        String[][] evalues = new String[repeat * iterations.length * Ns.length * 4][5];
        for(int i = 0;i < repeat; i++){
            for(int j = 0; j < iterations.length; j++) {
                for(int k = 0; k <Ns.length; k++) {
                    int iteration = iterations[j];
                    int N = Ns[k];
                    Random random = new Random();
                    // create the random points
                    double[][] points = new double[N][2];
                    for (int m = 0; m < points.length; m++) {
                        points[m][0] = random.nextDouble();
                        points[m][1] = random.nextDouble();
                    }
                    System.out.println("Starting the " + (i+1) + " repeat, max iterations " + iteration +", Number of Points is " + N + ".");
                    // for rhc, sa, and ga we use a permutation based encoding
                    TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
                    Distribution odd = new DiscretePermutationDistribution(N);
                    NeighborFunction nf = new SwapNeighbor();
                    MutationFunction mf = new SwapMutation();
                    CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
                    HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                    GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

                    double start = System.nanoTime();
                    RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                    FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iteration * 10);
                    fit.train();
                    double end = System.nanoTime();
                    double trainingTime = end - start;
                    //System.out.println("RHC: " + ef.value(rhc.getOptimal()));
                    String[] rhcEval = {"RHC", iteration + "", N + "", ef.value(rhc.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 0] = rhcEval;

                    start = System.nanoTime();
                    SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
                    fit = new FixedIterationTrainer(sa, iteration * 10);
                    fit.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    //System.out.println("SA: " + ef.value(sa.getOptimal()));
                    String[] saEval = {"SA", iteration + "", N + "", ef.value(sa.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 1] = saEval;

                    start = System.nanoTime();
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
                    fit = new FixedIterationTrainer(ga, iteration);
                    fit.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    //System.out.println("GA: " + ef.value(ga.getOptimal()));
                    String[] gaEval = {"GA", iteration + "", N + "", ef.value(ga.getOptimal()) + "", trainingTime + ""};
                    evalues[i * iterations.length * Ns.length* 4 + j * Ns.length * 4 + k * 4 + 2] = gaEval;

                    start = System.nanoTime();
                    // for mimic we use a sort encoding
                    ef = new TravelingSalesmanSortEvaluationFunction(points);
                    int[] ranges = new int[N];
                    Arrays.fill(ranges, N);
                    odd = new DiscreteUniformDistribution(ranges);
                    Distribution df = new DiscreteDependencyTree(.1, ranges);
                    ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                    MIMIC mimic = new MIMIC(200, 20, pop);
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
        String[] headerEval = {"Algorithm","Iteration","NumberOfPoints","Evaluation","TrainingTime"};

        writeEvalToFile(evalues,headerEval,"evaluesOfFourROsForTSPTime");
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
