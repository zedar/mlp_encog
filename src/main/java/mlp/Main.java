package mlp;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.mathutil.randomize.ConsistentRandomizer;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizeArray;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.Normalizer;

public class Main {
  private static double[][] trainIn;
  private static double[][] trainOut;
  private static double[][] testIn;
  private static double[][] testOut;

  public static void main(String[] args) throws Exception {
    Config cfg = Config.build(args);
    if (!(cfg.getInLayerN() > 0 && cfg.getInLayerN() <= 4)) {
      System.out.println("Input layer must be 1-4 neurons");
      return;
    }
    if (!(cfg.getOutLayerN() > 0 && cfg.getOutLayerN() <= 4)) {
      System.out.println("Output layer must have 1-3 neurons.");
      return;
    }
    if (!(cfg.getHiddenLayerN() > 0)) {
      System.out.println("Hidden layer must have at least 1 neuron.");
      return;
    }
    if (cfg.getTrainDSPath() == null) {
      System.out.println("Undefined train data file.");
      return;
    }
    double[][][] trainDS = Reader.loadDS(cfg.getTrainDSPath(), cfg.getInLayerN(), cfg.getInLayerF(), cfg.getOutLayerN(), cfg.getOutLayerF(), false);
    double[][] inDS = trainDS[0];
    double[][] outDS = trainDS[1];
    double[][] inTestDS = null;
    double[][] outTestDS = null;

    if (cfg.getTestDSPath() != null) {
      double[][][] testDS = Reader.loadDS(cfg.getTestDSPath(), cfg.getInLayerN(), cfg.getInLayerF(), cfg.getOutLayerN(), cfg.getOutLayerF(), false);
      inTestDS = testDS[0];
      outTestDS = testDS[1];
    }

    Reader.printDS("INPUT LAYER", inDS, 6);
    Reader.printDS("OUTPUT LAYER", outDS, 6);
    System.out.printf("INPUT LAYER OBSERVATIONS: %d\n", inDS.length);

    Reader.normalizeZeroOne(inDS);
    Reader.normalizeZeroOne(outDS);
    trainIn = inDS;
    trainOut = outDS;

    if (cfg.getTestDSPath() != null) {
      Reader.normalizeZeroOne(inTestDS);
      Reader.normalizeZeroOne(outTestDS);
      testIn = inTestDS;
      testOut = outTestDS;
    }

    BasicNetwork network = new BasicNetwork();
    network.addLayer(new BasicLayer(null, true, cfg.getInLayerN()));
    network.addLayer(new BasicLayer(new ActivationSigmoid(), true, cfg.getHiddenLayerN()));
    network.addLayer(new BasicLayer(new ActivationSigmoid(), true, cfg.getOutLayerN()));
    network.getStructure().finalizeStructure();
    network.reset();
    //(new ConsistentRandomizer(-1,1)).randomize(network);

    // training data
    MLDataSet trainingSet = new BasicMLDataSet(trainIn, trainOut);

    // train the neural network
    final Backpropagation train = new Backpropagation(network, trainingSet, 0.07, 0);
    int epoch = 1;

    do {
      train.iteration();
      //System.out.printf("Epoch # %d Error: %f\n", epoch, train.getError());
      epoch++;
    } while (train.getError() > 0.00001 && epoch <= 50000);

    train.finishTraining();

    System.out.printf("Error %f \n", train.getError());

    PrintWriter fActual = null;
    PrintWriter fIdeal = null;
    try {
      fActual = new PrintWriter(new FileWriter("plot_actual.dat"));
      fIdeal = new PrintWriter(new FileWriter("plot_ideal.dat"));

      System.out.printf("Neural Network Results\n");

      if (cfg.getTestDSPath() != null) {
        MLDataSet testSet = new BasicMLDataSet(testIn, testOut);
        trainingSet = testSet;
      }

      for (MLDataPair pair : trainingSet) {
        final MLData output = network.compute(pair.getInput());
        double x = pair.getInput().getData(0);
        double actual = output.getData(0);
        double ideal = pair.getIdeal().getData(0);
        System.out.printf("%f, %f, %f, %f, actual=%f %f %f %f\n",
            pair.getInput().getData(0),
            pair.getInput().getData(1),
            pair.getInput().getData(2),
            pair.getInput().getData(3),
            output.getData(0),
            output.getData(1),
            output.getData(2),
            output.getData(3));
        //fActual.println("\t" + x + "\t" + actual);
        //fIdeal.println("\t" + x + "\t" + ideal);
      }
    } catch (Exception ex) {
      ex.printStackTrace();
    } finally {
      if (fActual != null) {
        fActual.close();
      }
      if (fIdeal != null) {
        fIdeal.close();
      }
    }

    Encog.getInstance().shutdown();
  }
}
