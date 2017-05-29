package com.koverse.example.spark;

import com.koverse.com.google.common.collect.Lists;

import com.koverse.sdk.data.SimpleRecord;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

import scala.Tuple2;

import java.util.List;
import java.util.HashMap;

public class JavaTfIdf implements java.io.Serializable {

  private static final long serialVersionUID = 8741666028339586272L;
  private final String textFieldName;
  private final String tokenizationString;
  private final Integer n;
  
  public JavaTfIdf(String textFieldName, String tokenizationString, Integer n) {
    this.textFieldName = textFieldName;
    this.tokenizationString = tokenizationString;
    this.n = n;
  }

  /**
   * Divides text records into lists of n adjacent words
   * @param inputRecordsRdd input RDD of SimpleRecords
   * @return a JavaRDD of lists of n-grams from the input records
   */
  public JavaRDD<List<String>> getNgrams(JavaRDD<SimpleRecord> inputRecordsRdd) {
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

  /**
   * Pairs ngrams with their term frequencies weighted by document length
   * @param inputNgrams input RDD of lists of ngrams, where each list matches a single document
   * @return a JavaRDD of lists of ngrams paired with their TF values for the document represented by that list
   */
  public JavaRDD<List<Tuple2<String, Integer>>> getNgramTFs(JavaRDD<List<String>> inputNgrams){
    // sum up the counts for each ngram and divide by document length
    JavaRDD<List<Tuple2<String, Integer>>> ngramCountRdd = inputNgrams.map(ngram_list -> {
      // count terms by incrementing values in a hashmap
      HashMap<String, Integer> count_map = new HashMap<String, Integer>();
      for(String ngram : ngram_list){
        int count = count_map.getOrDefault(ngram, 0) + 1;
        count_map.put(ngram, count);
      } 

      // compile list of terms and pair them with counts
      int terms_in_document = ngram_list.size();
      List<Tuple2<String, Integer>> pairs = Lists.newArrayList();
      for(HashMap.Entry<String, Integer> pair : count_map.entrySet()){
        pairs.add(new Tuple2<String, Integer>(pair.getKey(), pair.getValue() / terms_in_document)); 
      }
      return pairs;
    });

    return ngramCountRdd;
  }


  /**
   * Pairs ngrams with their inverse document frequencies across the entire corpus
   * @param inputNgrams input RDD of lists of ngrams, where each list matches a single document
   * @return a JavaPairRDD of ngrams paired with their IDF values
   */
  public JavaPairRDD<String, Double> getNgramIDFs(JavaRDD<List<String>> inputNgrams){

    // generate an RDD containing the unique ngrams in the input lists
    JavaRDD<String> ngrams = inputNgrams.flatMap(ngram_list -> {return ngram_list;}).distinct();

    // append to each ngram the inverse document frequency of that ngram in inputNgrams
    JavaPairRDD<String, Double> ngram_idfs = ngrams.mapToPair(ngram -> {
      Long ngram_count = inputNgrams.filter(ngram_list -> {return ngram_list.contains(ngram);}).count();
      Double idf = Math.log(inputNgrams.count()/ngram_count);
      return new Tuple2<String, Double>(ngram, idf);
    });

    return ngram_idfs;
  }

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
