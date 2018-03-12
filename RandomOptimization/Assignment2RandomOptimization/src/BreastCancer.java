
import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying breast cancer benign or malignant
 *
 * @author Qingguo Chen
 * @version 1.0
 */
public class BreastCancer {

    private static Instance[] instances_Train = initializeInstances(455,"src/data/breast_cancer_train.txt");
    private static Instance[] instances_Test = initializeInstances(114,"src/data/breast_cancer_test.txt");
    private static int inputLayer = 30, hiddenLayer = 80, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances_Train);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        initialRun();

        findBestTForSA();
        findBestCoolingForSA();
        findBestMateRateForGA();
        findBestMutationRateForGA();
        runLearningCurve();
        runLearningCurveLessNodes();
        OptimizedROsForCancer();
        TimeComplexROsForCancer();
    }

    // Run three RO Algorithms with default settings
    private static void initialRun(){
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);
        String[][] errorLogs = new String[oa.length][]; // logs for sumofsquaresErrors during training iterations
        String[][] performances = new String[oa.length][6]; // evaluation on training and test set
        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            performances[i][0] = oaNames[i];
            String[] errorLogByIteration = train(oa[i], networks[i], oaNames[i],trainingIterations); //trainer.train();

            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            errorLogs[i] = errorLogByIteration;
            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances_Train.length; j++) {
                networks[i].setInputValues(instances_Train[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances_Train[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            String trainAccu = df.format(correct/(correct+incorrect)*100);

            performances[i][1] = trainAccu;
            performances[i][3] = df.format(trainingTime);
            performances[i][4] = df.format(testingTime);
            results +=  "\nResults for Training set " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
            "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
            + trainAccu + "%\nTraining time: " + df.format(trainingTime)
            + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            double predicted_Test, actual_Test;
            correct = 0;
            incorrect = 0;
            start = System.nanoTime();
            for(int j = 0; j < instances_Test.length; j++) {
                networks[i].setInputValues(instances_Test[j].getData());
                networks[i].run();

                predicted_Test = Double.parseDouble(instances_Test[j].getLabel().toString());
                actual_Test = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            String testAccu = df.format(correct/(correct+incorrect)*100);

            performances[i][2] = testAccu;
            performances[i][5] = df.format(testingTime);
            results +=  "\nResults for Testing set " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + testAccu + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        }
        System.out.println(results);
        String[] headerLogs = new String[oa.length+1];
        headerLogs[0] = "Iteration";
        for(int i = 1; i < headerLogs.length; i++){
            headerLogs[i] = oaNames[i-1];
        }
        writeLogToFile(errorLogs,headerLogs,"initialRunTrainingLogs");

        String[] headerPerformance = {"Algorithm","TrainingAccuracy","TestAccuracy","TrainingTime","TestingTimeOnTrain","TestingTimeOnTest"};

        writePerformanceToFile(performances,headerPerformance,"initialRunPerformances");
    }

    // Run SA Algorithms with Different T
    private static void findBestTForSA(){
        trainingIterations = 1000;
        double[] sa_Ts = {10,100,1E3,1E4,1E5,1E6,1E7,1E8,1E9,1E10,1E11};
        int repeat = 10;
        String[][] performances = new String[repeat * sa_Ts.length][3]; // evaluation on training and test set
        for(int m = 0; m < repeat; m++) {
            OptimizationAlgorithm[] sa = new OptimizationAlgorithm[sa_Ts.length];
            BackPropagationNetwork networks_sa[] = new BackPropagationNetwork[sa_Ts.length];
            NeuralNetworkOptimizationProblem[] nnop_sa = new NeuralNetworkOptimizationProblem[sa_Ts.length];

            for (int i = 0; i < sa.length; i++) {
                networks_sa[i] = factory.createClassificationNetwork(
                        new int[]{inputLayer, hiddenLayer, outputLayer});
                nnop_sa[i] = new NeuralNetworkOptimizationProblem(set, networks_sa[i], measure);
                sa[i] = new SimulatedAnnealing(sa_Ts[i], .95, nnop_sa[i]);
            }


            for (int i = 0; i < sa.length; i++) {
                performances[m * sa.length + i][0] = sa_Ts[i] + "";
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;

                train(sa[i], networks_sa[i], oaNames[1],trainingIterations); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
                Instance optimalInstance = sa[i].getOptimal();
                networks_sa[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < instances_Train.length; j++) {
                    networks_sa[i].setInputValues(instances_Train[j].getData());
                    networks_sa[i].run();

                    predicted = Double.parseDouble(instances_Train[j].getLabel().toString());
                    actual = Double.parseDouble(networks_sa[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                String trainAccu = df.format(correct / (correct + incorrect) * 100);

                performances[m * sa.length + i][1] = trainAccu;

                results += "\nResults for Training set with T = " + sa_Ts[i] + oaNames[1] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + trainAccu + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                double predicted_Test, actual_Test;
                correct = 0;
                incorrect = 0;
                start = System.nanoTime();
                for (int j = 0; j < instances_Test.length; j++) {
                    networks_sa[i].setInputValues(instances_Test[j].getData());
                    networks_sa[i].run();

                    predicted_Test = Double.parseDouble(instances_Test[j].getLabel().toString());
                    actual_Test = Double.parseDouble(networks_sa[i].getOutputValues().toString());

                    double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                String testAccu = df.format(correct / (correct + incorrect) * 100);

                performances[m * sa.length + i][2] = testAccu;

                results += "\nResults for Testing set with T = " + sa_Ts[i] + oaNames[1] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + testAccu + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
            }
            System.out.println(results);
        }

        String[] headerPerformance = {"T","TrainingAccuracy","TestAccuracy"};

        writePerformanceToFile(performances,headerPerformance,"PerformancesOfSAwithTs");

    }

    // Run SA Algorithms with Different T
    private static void findBestCoolingForSA(){
        trainingIterations = 1000;
        double[] sa_Cools = {0.99,0.98,0.97,0.95,0.93,0.9,0.85,0.8};
        int repeat = 10;
        String[][] performances = new String[repeat * sa_Cools.length][3]; // evaluation on training and test set
        for(int m = 0; m < repeat; m++) {
            OptimizationAlgorithm[] sa = new OptimizationAlgorithm[sa_Cools.length];
            BackPropagationNetwork networks_sa[] = new BackPropagationNetwork[sa_Cools.length];
            NeuralNetworkOptimizationProblem[] nnop_sa = new NeuralNetworkOptimizationProblem[sa_Cools.length];

            for (int i = 0; i < sa.length; i++) {
                networks_sa[i] = factory.createClassificationNetwork(
                        new int[]{inputLayer, hiddenLayer, outputLayer});
                nnop_sa[i] = new NeuralNetworkOptimizationProblem(set, networks_sa[i], measure);
                sa[i] = new SimulatedAnnealing(10, sa_Cools[i], nnop_sa[i]);
            }


            for (int i = 0; i < sa.length; i++) {
                performances[m * sa.length + i][0] = sa_Cools[i] + "";
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;

                train(sa[i], networks_sa[i], oaNames[1],trainingIterations); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
                Instance optimalInstance = sa[i].getOptimal();
                networks_sa[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < instances_Train.length; j++) {
                    networks_sa[i].setInputValues(instances_Train[j].getData());
                    networks_sa[i].run();

                    predicted = Double.parseDouble(instances_Train[j].getLabel().toString());
                    actual = Double.parseDouble(networks_sa[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                String trainAccu = df.format(correct / (correct + incorrect) * 100);

                performances[m * sa.length + i][1] = trainAccu;

                results += "\nResults for Training set with T = 10, Cooling =  " + sa_Cools[i] + oaNames[1] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + trainAccu + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                double predicted_Test, actual_Test;
                correct = 0;
                incorrect = 0;
                start = System.nanoTime();
                for (int j = 0; j < instances_Test.length; j++) {
                    networks_sa[i].setInputValues(instances_Test[j].getData());
                    networks_sa[i].run();

                    predicted_Test = Double.parseDouble(instances_Test[j].getLabel().toString());
                    actual_Test = Double.parseDouble(networks_sa[i].getOutputValues().toString());

                    double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                String testAccu = df.format(correct / (correct + incorrect) * 100);

                performances[m * sa.length + i][2] = testAccu;

                results += "\nResults for Testing set with T = 10, Cooling = " + sa_Cools[i] + oaNames[1] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + testAccu + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
            }
            System.out.println(results);
        }

        String[] headerPerformance = {"Cooling","TrainingAccuracy","TestAccuracy"};

        writePerformanceToFile(performances,headerPerformance,"PerformancesOfSAwithCoolings");

    }

    private static void findBestMateRateForGA(){
        trainingIterations = 20;
        int[] gaMateRates = {10,30,60,100,150,200};
        int repeat = 10;
        String[][] performances = new String[repeat * gaMateRates.length][3]; // evaluation on training and test set
        for(int m = 0; m < repeat; m++) {
            OptimizationAlgorithm[] ga = new OptimizationAlgorithm[gaMateRates.length];
            BackPropagationNetwork networks_ga[] = new BackPropagationNetwork[gaMateRates.length];
            NeuralNetworkOptimizationProblem[] nnop_ga = new NeuralNetworkOptimizationProblem[gaMateRates.length];

            for (int i = 0; i < ga.length; i++) {
                networks_ga[i] = factory.createClassificationNetwork(
                        new int[]{inputLayer, hiddenLayer, outputLayer});
                nnop_ga[i] = new NeuralNetworkOptimizationProblem(set, networks_ga[i], measure);
                ga[i] = new StandardGeneticAlgorithm(200, gaMateRates[i], 10, nnop_ga[i]);
            }


            for (int i = 0; i < ga.length; i++) {
                performances[m * ga.length + i][0] = gaMateRates[i] + "";
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;

                train(ga[i], networks_ga[i], oaNames[2],trainingIterations); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
                Instance optimalInstance = ga[i].getOptimal();
                networks_ga[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < instances_Train.length; j++) {
                    networks_ga[i].setInputValues(instances_Train[j].getData());
                    networks_ga[i].run();

                    predicted = Double.parseDouble(instances_Train[j].getLabel().toString());
                    actual = Double.parseDouble(networks_ga[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                String trainAccu = df.format(correct / (correct + incorrect) * 100);

                performances[m * ga.length + i][1] = trainAccu;

                results += "\nResults for Training set with Mating rate =  " + gaMateRates[i] + oaNames[1] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + trainAccu + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                double predicted_Test, actual_Test;
                correct = 0;
                incorrect = 0;
                start = System.nanoTime();
                for (int j = 0; j < instances_Test.length; j++) {
                    networks_ga[i].setInputValues(instances_Test[j].getData());
                    networks_ga[i].run();

                    predicted_Test = Double.parseDouble(instances_Test[j].getLabel().toString());
                    actual_Test = Double.parseDouble(networks_ga[i].getOutputValues().toString());

                    double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                String testAccu = df.format(correct / (correct + incorrect) * 100);

                performances[m * ga.length + i][2] = testAccu;

                results += "\nResults for Testing set with Mating rate = " + gaMateRates[i] + oaNames[1] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + testAccu + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
            }
            System.out.println(results);
        }

        String[] headerPerformance = {"MatingRate","TrainingAccuracy","TestAccuracy"};

        writePerformanceToFile(performances,headerPerformance,"PerformancesOfGAwithMatingRates");

    }

    private static void findBestMutationRateForGA(){
        trainingIterations = 20;
        int[] gaMutationRates = {3,10,30,60,80,90,100};
        int repeat = 10;
        String[][] performances = new String[repeat * gaMutationRates.length][3]; // evaluation on training and test set
        for(int m = 0; m < repeat; m++) {
            OptimizationAlgorithm[] ga = new OptimizationAlgorithm[gaMutationRates.length];
            BackPropagationNetwork networks_ga[] = new BackPropagationNetwork[gaMutationRates.length];
            NeuralNetworkOptimizationProblem[] nnop_ga = new NeuralNetworkOptimizationProblem[gaMutationRates.length];

            for (int i = 0; i < ga.length; i++) {
                networks_ga[i] = factory.createClassificationNetwork(
                        new int[]{inputLayer, hiddenLayer, outputLayer});
                nnop_ga[i] = new NeuralNetworkOptimizationProblem(set, networks_ga[i], measure);
                ga[i] = new StandardGeneticAlgorithm(200, 100,gaMutationRates[i], nnop_ga[i]);
            }


            for (int i = 0; i < ga.length; i++) {
                performances[m * ga.length + i][0] = gaMutationRates[i] + "";
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;

                train(ga[i], networks_ga[i], oaNames[2],trainingIterations); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);
                Instance optimalInstance = ga[i].getOptimal();
                networks_ga[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < instances_Train.length; j++) {
                    networks_ga[i].setInputValues(instances_Train[j].getData());
                    networks_ga[i].run();

                    predicted = Double.parseDouble(instances_Train[j].getLabel().toString());
                    actual = Double.parseDouble(networks_ga[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                String trainAccu = df.format(correct / (correct + incorrect) * 100);

                performances[m * ga.length + i][1] = trainAccu;

                results += "\nResults for Training set with Mutation rate =  " + gaMutationRates[i] + oaNames[1] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + trainAccu + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                double predicted_Test, actual_Test;
                correct = 0;
                incorrect = 0;
                start = System.nanoTime();
                for (int j = 0; j < instances_Test.length; j++) {
                    networks_ga[i].setInputValues(instances_Test[j].getData());
                    networks_ga[i].run();

                    predicted_Test = Double.parseDouble(instances_Test[j].getLabel().toString());
                    actual_Test = Double.parseDouble(networks_ga[i].getOutputValues().toString());

                    double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                String testAccu = df.format(correct / (correct + incorrect) * 100);

                performances[m * ga.length + i][2] = testAccu;

                results += "\nResults for Testing set with Mutation rate = " + gaMutationRates[i] + oaNames[1] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + testAccu + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
            }
            System.out.println(results);
        }

        String[] headerPerformance = {"MutationRate","TrainingAccuracy","TestAccuracy"};

        writePerformanceToFile(performances,headerPerformance,"PerformancesOfGAwithMutationRates");

    }

    // Run three RO Algorithms with optimized settings
    private static void runLearningCurve(){
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(10, .8, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 60, nnop[2]);
        String[][] errorLogs = new String[oa.length][]; // logs for sumofsquaresErrors during training iterations
        String[][] performances = new String[oa.length][6]; // evaluation on training and test set
        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            performances[i][0] = oaNames[i];
            String[] errorLogByIteration = train(oa[i], networks[i], oaNames[i],trainingIterations); //trainer.train();

            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            errorLogs[i] = errorLogByIteration;
            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances_Train.length; j++) {
                networks[i].setInputValues(instances_Train[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances_Train[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            String trainAccu = df.format(correct/(correct+incorrect)*100);

            performances[i][1] = trainAccu;
            performances[i][3] = df.format(trainingTime);
            performances[i][4] = df.format(testingTime);
            results +=  "\nResults for Training set " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + trainAccu + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            double predicted_Test, actual_Test;
            correct = 0;
            incorrect = 0;
            start = System.nanoTime();
            for(int j = 0; j < instances_Test.length; j++) {
                networks[i].setInputValues(instances_Test[j].getData());
                networks[i].run();

                predicted_Test = Double.parseDouble(instances_Test[j].getLabel().toString());
                actual_Test = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            String testAccu = df.format(correct/(correct+incorrect)*100);

            performances[i][2] = testAccu;
            performances[i][5] = df.format(testingTime);
            results +=  "\nResults for Testing set " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + testAccu + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        }
        System.out.println(results);
        String[] headerLogs = new String[oa.length+1];
        headerLogs[0] = "Iteration";
        for(int i = 1; i < headerLogs.length; i++){
            headerLogs[i] = oaNames[i-1];
        }
        writeLogToFile(errorLogs,headerLogs,"RunTrainingLogsBreastCancer");

        String[] headerPerformance = {"Algorithm","TrainingAccuracy","TestAccuracy","TrainingTime","TestingTimeOnTrain","TestingTimeOnTest"};

        writePerformanceToFile(performances,headerPerformance,"PerformancesOfROBreastCancer");
    }



    // Run three RO Algorithms with less Nodes for breast cancer problem.
    private static void runLearningCurveLessNodes(){
        hiddenLayer = 10;
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(10, .8, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 60, nnop[2]);
        String[][] errorLogs = new String[oa.length][]; // logs for sumofsquaresErrors during training iterations
        String[][] performances = new String[oa.length][6]; // evaluation on training and test set
        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            performances[i][0] = oaNames[i];
            String[] errorLogByIteration = train(oa[i], networks[i], oaNames[i],trainingIterations); //trainer.train();

            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            errorLogs[i] = errorLogByIteration;
            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances_Train.length; j++) {
                networks[i].setInputValues(instances_Train[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances_Train[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            String trainAccu = df.format(correct/(correct+incorrect)*100);

            performances[i][1] = trainAccu;
            performances[i][3] = df.format(trainingTime);
            performances[i][4] = df.format(testingTime);
            results +=  "\nResults for Training set " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + trainAccu + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            double predicted_Test, actual_Test;
            correct = 0;
            incorrect = 0;
            start = System.nanoTime();
            for(int j = 0; j < instances_Test.length; j++) {
                networks[i].setInputValues(instances_Test[j].getData());
                networks[i].run();

                predicted_Test = Double.parseDouble(instances_Test[j].getLabel().toString());
                actual_Test = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            String testAccu = df.format(correct/(correct+incorrect)*100);

            performances[i][2] = testAccu;
            performances[i][5] = df.format(testingTime);
            results +=  "\nResults for Testing set " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + testAccu + "%\nTesting time: " + df.format(testingTime) + " seconds\n";
        }
        System.out.println(results);
        String[] headerLogs = new String[oa.length+1];
        headerLogs[0] = "Iteration";
        for(int i = 1; i < headerLogs.length; i++){
            headerLogs[i] = oaNames[i-1];
        }
        writeLogToFile(errorLogs,headerLogs,"RunTrainingLogsBreastCancerWithLessNodes");

        String[] headerPerformance = {"Algorithm","TrainingAccuracy","TestAccuracy","TrainingTime","TestingTimeOnTrain","TestingTimeOnTest"};

        writePerformanceToFile(performances,headerPerformance,"PerformancesOfROBreastCancerWithLessNodes");
    }


    private static void OptimizedROsForCancer(){
        int repeat = 5;
        int[] iterations = {10,30,100,300,1000};
        int[] Nodes = {10,80};
        String[][] evalues = new String[repeat * iterations.length * Nodes.length * oaNames.length * 2][6];
        for(int i = 0;i < repeat; i++){
            for(int j = 0; j < iterations.length; j++) {
                for(int k = 0; k < Nodes.length; k++) {

                    int iteration = iterations[j];
                    int N = Nodes[k];
                    System.out.println("Starting the " + i + " repeat, max iterations " + iteration +", with " + N + " Nodes in Hidden Layer.");
                    networks = new BackPropagationNetwork[3];
                    nnop = new NeuralNetworkOptimizationProblem[3];

                    oa = new OptimizationAlgorithm[3];
                    for(int m = 0; m < oa.length; m++) {
                        networks[m] = factory.createClassificationNetwork(
                                new int[] {inputLayer, N, outputLayer});
                        nnop[m] = new NeuralNetworkOptimizationProblem(set, networks[m], measure);
                    }

                    oa[0] = new RandomizedHillClimbing(nnop[0]);
                    oa[1] = new SimulatedAnnealing(10, .8, nnop[1]);
                    oa[2] = new StandardGeneticAlgorithm(200, 100, 60, nnop[2]);

                    for(int m = 0; m < oa.length; m++) {
                        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                        train(oa[m], networks[m], oaNames[m],iteration); //trainer.train();

                        end = System.nanoTime();
                        trainingTime = end - start;
                        trainingTime /= Math.pow(10,9);

                        Instance optimalInstance = oa[m].getOptimal();
                        networks[m].setWeights(optimalInstance.getData());

                        double predicted, actual;
                        start = System.nanoTime();
                        for(int t = 0; t < instances_Train.length; t++) {
                            networks[m].setInputValues(instances_Train[t].getData());
                            networks[m].run();

                            predicted = Double.parseDouble(instances_Train[t].getLabel().toString());
                            actual = Double.parseDouble(networks[m].getOutputValues().toString());

                            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                        }
                        end = System.nanoTime();
                        testingTime = end - start;
                        testingTime /= Math.pow(10,9);

                        String trainAccu = df.format(correct/(correct+incorrect)*100);


                        String[] eval = {oaNames[m], iteration + "", N + "", "train" ,trainAccu+ "", df.format(trainingTime)};
                        evalues[i * iterations.length * Nodes.length* oaNames.length * 2 + j * Nodes.length* oaNames.length * 2 + k * oaNames.length * 2 + m * 2+ 0] = eval;

                        double predicted_Test, actual_Test;
                        correct = 0;
                        incorrect = 0;
                        start = System.nanoTime();
                        for(int t = 0; t < instances_Test.length; t++) {
                            networks[m].setInputValues(instances_Test[t].getData());
                            networks[m].run();

                            predicted_Test = Double.parseDouble(instances_Test[t].getLabel().toString());
                            actual_Test = Double.parseDouble(networks[m].getOutputValues().toString());

                            double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

                        }
                        end = System.nanoTime();
                        testingTime = end - start;
                        testingTime /= Math.pow(10,9);

                        String testAccu = df.format(correct/(correct+incorrect)*100);

                        String[] evaltest = {oaNames[m], iteration + "", N + "", "test" ,testAccu+ "", df.format(testingTime)};
                        evalues[i * iterations.length * Nodes.length* oaNames.length * 2 +
                                j * Nodes.length* oaNames.length * 2 + k * oaNames.length * 2 + m * 2+ 1] = evaltest;

                    }

                }
            }
        }
        String[] headerEval = {"Algorithm","Iteration","Nodes","TrainTest","Evaluation","Time"};

        writePerformanceToFile(evalues,headerEval,"evaluesOfThreeROsForCancers");
    }
    private static void TimeComplexROsForCancer(){
        int repeat = 5;
        int[] iterations = {10,20,30,40,50,60};
        int[] Nodes = {10,20,30,40,50,60};
        String[][] evalues = new String[repeat * iterations.length * Nodes.length * oaNames.length * 2][6];
        for(int i = 0;i < repeat; i++){
            for(int j = 0; j < iterations.length; j++) {
                for(int k = 0; k < Nodes.length; k++) {

                    int iteration = iterations[j];
                    int N = Nodes[k];
                    System.out.println("Starting the " + i + " repeat, max iterations " + iteration +", with " + N + " Nodes in Hidden Layer.");
                    networks = new BackPropagationNetwork[3];
                    nnop = new NeuralNetworkOptimizationProblem[3];

                    oa = new OptimizationAlgorithm[3];
                    for(int m = 0; m < oa.length; m++) {
                        networks[m] = factory.createClassificationNetwork(
                                new int[] {inputLayer, N, outputLayer});
                        nnop[m] = new NeuralNetworkOptimizationProblem(set, networks[m], measure);
                    }

                    oa[0] = new RandomizedHillClimbing(nnop[0]);
                    oa[1] = new SimulatedAnnealing(10, .8, nnop[1]);
                    oa[2] = new StandardGeneticAlgorithm(200, 100, 60, nnop[2]);

                    for(int m = 0; m < oa.length; m++) {
                        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                        train(oa[m], networks[m], oaNames[m],iteration); //trainer.train();

                        end = System.nanoTime();
                        trainingTime = end - start;
                        trainingTime /= Math.pow(10,9);

                        Instance optimalInstance = oa[m].getOptimal();
                        networks[m].setWeights(optimalInstance.getData());

                        double predicted, actual;
                        start = System.nanoTime();
                        for(int t = 0; t < instances_Train.length; t++) {
                            networks[m].setInputValues(instances_Train[t].getData());
                            networks[m].run();

                            predicted = Double.parseDouble(instances_Train[t].getLabel().toString());
                            actual = Double.parseDouble(networks[m].getOutputValues().toString());

                            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                        }
                        end = System.nanoTime();
                        testingTime = end - start;
                        testingTime /= Math.pow(10,9);

                        String trainAccu = df.format(correct/(correct+incorrect)*100);


                        String[] eval = {oaNames[m], iteration + "", N + "", "train" ,trainAccu+ "", df.format(trainingTime)};
                        evalues[i * iterations.length * Nodes.length* oaNames.length * 2 + j * Nodes.length* oaNames.length * 2 + k * oaNames.length * 2 + m * 2+ 0] = eval;

                        double predicted_Test, actual_Test;
                        correct = 0;
                        incorrect = 0;
                        start = System.nanoTime();
                        for(int t = 0; t < instances_Test.length; t++) {
                            networks[m].setInputValues(instances_Test[t].getData());
                            networks[m].run();

                            predicted_Test = Double.parseDouble(instances_Test[t].getLabel().toString());
                            actual_Test = Double.parseDouble(networks[m].getOutputValues().toString());

                            double trash = Math.abs(predicted_Test - actual_Test) < 0.5 ? correct++ : incorrect++;

                        }
                        end = System.nanoTime();
                        testingTime = end - start;
                        testingTime /= Math.pow(10,9);

                        String testAccu = df.format(correct/(correct+incorrect)*100);

                        String[] evaltest = {oaNames[m], iteration + "", N + "", "test" ,testAccu+ "", df.format(testingTime)};
                        evalues[i * iterations.length * Nodes.length* oaNames.length * 2 +
                                j * Nodes.length* oaNames.length * 2 + k * oaNames.length * 2 + m * 2+ 1] = evaltest;

                    }

                }
            }
        }
        String[] headerEval = {"Algorithm","Iteration","Nodes","TrainTest","Evaluation","Time"};

        writePerformanceToFile(evalues,headerEval,"evaluesOfThreeROsForCancersTime");
    }
    private static void writeLogToFile(String[][] logs,String[] header, String fileName){

        StringBuilder builder = new StringBuilder();
        for(int i = 0; i < header.length; i++)//for each column
        {
            builder.append(header[i]);//append to the output string
            if(i < header.length - 1)//if this is not the last row element
                builder.append(",");//then add comma
        }
        builder.append("\n");

        for(int i = 0; i < logs[0].length; i++)//for each iteration
        {
            builder.append(i+",");
            for(int j = 0; j < logs.length; j++)//for each RO
            {
                builder.append(logs[j][i]);//append to the output string
                if(j < logs.length - 1)//if this is not the last row element
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

    private static void writePerformanceToFile(String[][] logs,String[] header, String fileName){

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

    private static String[] train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int Iterations) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");
        String[] errorList = new String[Iterations];
        for(int i = 0; i < Iterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances_Train.length; j++) {
                network.setInputValues(instances_Train[j].getData());
                network.run();

                Instance output = instances_Train[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
            errorList[i] = df.format(error);
        }
        return  errorList;
    }

    private static Instance[] initializeInstances(int sample_size, String file_name) {

        double[][][] attributes = new double[sample_size][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(file_name)));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[30]; // 30 attributes
                attributes[i][1] = new double[1];
                attributes[i][1][0] = Integer.parseInt(scan.next());
                for(int j = 0; j < 30; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());


            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel( new Instance(attributes[i][1][0]) );

        }

        return instances;
    }
}
