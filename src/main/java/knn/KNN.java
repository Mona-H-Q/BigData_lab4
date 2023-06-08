package knn;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.file.FileStore;
import java.nio.file.PathMatcher;
import java.nio.file.WatchService;
import java.nio.file.attribute.UserPrincipalLookupService;
import java.nio.file.spi.FileSystemProvider;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.HashMap;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Iterator;
import javafx.util.Pair;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class KNN {
    private static final int K = 10;

    public static class KNNMapper extends Mapper<LongWritable, Text, IntWritable, Pair<Float, String>> {
        private ArrayList<float[]> testData;
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // 将 distributed cache file 中的测试数据装入本地的内存数据 testData 中
            try {
                Path[] cacheFiles = context.getLocalCacheFiles();
                if (cacheFiles != null && cacheFiles.length > 0) {
                    String line;
                    String[] tokens;
                    float[] temp = new float[4];
                    BufferedReader joinReader = new BufferedReader(new FileReader(cacheFiles[0].toString()));
                    try {
                        while ((line = joinReader.readLine()) != null) {
                            tokens = line.split(",", 4);
                            for(int i = 0; i < 4; ++i)
                                temp[i] = Float.parseFloat(tokens[i]);
                            testData.add(temp);
                        }
                    } finally {
                        joinReader.close();
                    }
                }
            } catch (IOException e) {
                System.err.println("Exception reading DistributedCache: "+e);
            }
        }

        private float distance(float[] A, float[] B){
            float ret = 0;
            for(int i = 0; i < 4; ++i)
                ret += (A[i] - B[i]) * (A[i] - B[i]);
            return ret;
        }
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",", 5);
            float[] values = new float[4];
            for(int i = 0; i < 4; ++i)
                values[i] = Float.parseFloat(tokens[i]);
            int id = 0;
            // 枚举所有训练数据，并且写入
            // key = 测试数据id; value = <距离，类>
            for(float[] test : testData) {
                float dist = distance(values, test);
                context.write(new IntWritable(id), new Pair<Float, String>(dist, tokens[4]));
                id ++;
            }
        }
    }

    public static class KNNCombiner extends Reducer<IntWritable, Pair<Float, String>, IntWritable, Pair<Float, String>> {
        @Override
        protected void reduce(IntWritable key, Iterable<Pair<Float, String>> values, Context context) throws IOException, InterruptedException {
            int count = 0;
            for(Pair<Float, String> val : values){
                context.write(key, val);
                ++ count;
                if(count == K) break ;
            }
        }
    }

    public static class KNNReducer extends Reducer<IntWritable, Pair<Float, String>, IntWritable, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Pair<Float, String>> values, Context context) throws IOException, InterruptedException {
            Map<String, Integer> m = new HashMap<String, Integer>();
            String mxType = "";
            int mxSum = 0;

            int count = 0;
            for(Pair<Float, String> val : values){
                ++ count;
                String str = val.getValue();
                int temp = 1;
                if(m.containsKey(str)){
                    temp = m.get(str) + 1;
                    m.put(str, temp);
                } else
                    m.put(str, 1);
                if(temp > mxSum) {
                    mxType = str;
                    mxSum = temp;
                }
                if(count == K) break ;
            }
            context.write(key, new Text(mxType));
        }

    }

    public static void main(String[] args) {
        try {
            Job job = Job.getInstance(new Configuration(), "knn");
            job.setJarByClass(KNN.class);

            job.setMapperClass(KNNMapper.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Pair.class);

            job.setCombinerClass(KNNCombiner.class);
            job.setReducerClass(KNNReducer.class);

            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Text.class);

            job.addCacheFile(new Path(args[0]).toUri());
            FileInputFormat.addInputPath(job, new Path(args[1]));
            FileOutputFormat.setOutputPath(job, new Path(args[2]));
            System.exit(job.waitForCompletion(true) ? 0 : 1);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
