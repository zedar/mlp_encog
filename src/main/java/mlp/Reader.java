package mlp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

public class Reader {
  public static double[][][] loadDS(String trainDSPath, int inN, int[] inFs, int outN) throws Exception {
    ArrayList<double[]> in = new ArrayList<>();
    ArrayList<double[]> out = new ArrayList<>();

    // we assume that input file is in csv format with space as delimiter. First 4 values are features while 5th is label.
    BufferedReader br = new BufferedReader(new FileReader(trainDSPath));
    String line;
    while ((line = br.readLine()) != null) {
      String[] features = line.split(" ");
//      if (features.length < inN+1) {
//        throw new IllegalArgumentException("train DS has invalid number of features");
//      }
      double[] inf = new double[inN];
      for (int i=0; i<inN; i++) {
        int pos = i;
        if (inFs != null && i < inFs.length) {
          pos = inFs[i]-1;
        }
        inf[i] = Double.valueOf(features[pos]);
      }
      in.add(inf);

//      int outv = Integer.valueOf(features[features.length-1]);
//      if (outv < 0 || outv > outN) {
//        throw new IllegalArgumentException("Invalid output feature value. Not inline with number of output neurons");
//      }
//      double[] outf = new double[outN];
//      Arrays.fill(outf, 0.0);
//      outf[outv-1] = 1.0;

      // ONE OUTPUT ATTRIBUTE
//      double[] outf = new double[outN];
//      Arrays.fill(outf, 0.0);
//      outf[0] = Double.valueOf(features[features.length-1]);
//      out.add(outf);

      // OUTPUT THE SAME AS INPUT
      double[] outf = new double[inN];
      for (int i=0; i<inN; i++) {
        int pos = i;
        if (inFs != null && i < inFs.length) {
          pos = inFs[i]-1;
        }
        outf[i] = Double.valueOf(features[pos]);
      }
      out.add(outf);
    }
    return new double[][][] {in.toArray(new double[][]{{}}), out.toArray(new double[][]{{}})};
  }

  public static void printDS(String prompt, double[][] ds, Integer head) {
    System.out.println("-------------------------");
    System.out.println(prompt);
    if (head == null) head = ds.length;
    for (int i=0; i<ds.length && i<head; i++) {
      StringBuilder sb = new StringBuilder();
      sb.append(i).append(" : ");
      for (int j=0; j<ds[i].length; j++) {
        sb.append(ds[i][j]).append(" ");
      }
      System.out.println(sb.toString());
    }
  }

  public static void normalizeZeroOne(double[][] data) {
    double min = 0.0;
    double max = 0.0;
    for (int i = 0; i < data.length; i++) {
      double d = data[i][0];
      if (d < min) {
        min = d;
      }
      if (d > max) {
        max = d;
      }

    }
    System.out.printf("NORMALIZE: MIN: %f, MAX: %f\n", min, max);
    for (int i = 0; i < data.length; i++) {
      //data[i][0] = (data[i][0] - min) / (max - min);
      // (b-a)(x-min)/(max-min) + a
      double a = 0.0;
      double b = 1.0;
      data[i][0] = (b-a)*(data[i][0] - min) / (max - min) + a;
    }
  }
}
