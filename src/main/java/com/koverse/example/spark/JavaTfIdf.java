package com.koverse.example.spark;

import com.koverse.com.google.common.collect.Lists;

import com.koverse.sdk.data.SimpleRecord;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

import scala.Tuple2;

import java.util.List;

public class JavaTfIdf implements java.io.Serializable {

  private static final long serialVersionUID = 8741666028339586272L;
  private final String textFieldName;
  private final String tokenizationString;
  
  public JavaTfIdf(String textFieldName, String tokenizationString) {
    this.textFieldName = textFieldName;
    this.tokenizationString = tokenizationString;
  }

  /**
   * Divides text records into lists of n adjacent words
   * @param inputRecordsRdd input RDD of SimpleRecords
   * @param n size of n-grams 
   * @return a JavaRDD of lists of n-grams from the input records
   */
  public JavaRDD<List<String>> getNgrams(JavaRDD<SimpleRecord> inputRecordsRdd, int n) {

    // split the text in the records into lowercase words
    JavaRDD<List<String>> words = inputRecordsRdd.map(record -> {
      return Lists.newArrayList(record.get(textFieldName).toString().toLowerCase()
          .split(tokenizationString));
    });

    // divide lists of words into ngrams
    JavaRDD<List<String>> ngrams = words.map(line -> {
      List<String> ngram_list = Lists.newArrayList();
      for(int i = 0; i < (line.size() - n); i++){
        ngram_list.add(String.join(" ", line.subList(i, i+n))); 
      }
      return Lists.newArrayList(ngram_list);
    });

    return ngrams;
  }

    // combine the lower casing of the string with generating the pairs.
    // JavaPairRDD<String, Integer> ones = ngrams.mapToPair(word -> {
    //   return new Tuple2<String, Integer>(word.toLowerCase().trim(), 1);
    // });

    // sum up the counts for each word
    // JavaPairRDD<String, Integer> wordCountRdd = ones
    //     .reduceByKey((count, amount) -> count + amount);

    // turn each tuple into an output Record with a "word" and "count" field
    // JavaRDD<SimpleRecord> outputRdd = wordCountRdd.map(wordCountTuple -> {
    //   SimpleRecord record = new SimpleRecord();
    //   record.put("word", wordCountTuple._1);
    //   record.put("count", wordCountTuple._2);
    //   return record;
    // });

    // return outputRdd;
    // return inputRecordsRdd;

  // }
}
