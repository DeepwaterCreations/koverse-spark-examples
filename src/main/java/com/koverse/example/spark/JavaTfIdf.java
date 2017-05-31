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
  private final Integer ngramSize;
  
  /**
   * A TF-IDF spark transformation for ngrams.
   * @param textFieldName the name of the record field containing the document text
   * @param tokenizationString the symbols to split document text on
   * @param ngramSize the number of words to use in ngrams
   */
  public JavaTfIdf(String textFieldName, String tokenizationString, Integer ngramSize) {
    this.textFieldName = textFieldName;
    this.tokenizationString = tokenizationString;
    this.ngramSize = ngramSize;
  }

  /**
   * Divides text records into lists of n adjacent words.
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
      List<String> ngramList = Lists.newArrayList();
      for (int i = 0; i < (line.size() - ngramSize); i++) {
        ngramList.add(String.join(" ", line.subList(i, i + ngramSize))); 
      }
      return Lists.newArrayList(ngramList);
    });

    return ngrams;
  }

  /**
   * Pairs ngrams with their term frequencies weighted by document length.
   * @param inputNgrams input RDD of lists of ngrams, where each list 
   *     matches a single document
   * @return a JavaRDD of lists of ngrams paired with their TF values for the 
   *     document represented by that list
   */
  public JavaRDD<List<Tuple2<String, Integer>>> getNgramTFs(JavaRDD<List<String>> inputNgrams) {
    // sum up the counts for each ngram and divide by document length
    JavaRDD<List<Tuple2<String, Integer>>> ngramCountRdd = inputNgrams.map(ngramList -> {
      // count terms by incrementing values in a hashmap
      HashMap<String, Integer> countMap = new HashMap<String, Integer>();
      for (String ngram : ngramList) {
        int count = countMap.getOrDefault(ngram, 0) + 1;
        countMap.put(ngram, count);
      } 

      // compile list of terms and pair them with counts
      int termsInDocument = ngramList.size();
      List<Tuple2<String, Integer>> pairs = Lists.newArrayList();
      for (HashMap.Entry<String, Integer> pair : countMap.entrySet()) {
        pairs.add(new Tuple2<String, Integer>(pair.getKey(), pair.getValue() / termsInDocument)); 
      }
      return pairs;
    });

    return ngramCountRdd;
  }

  /**
   * Pairs ngrams with their inverse document frequencies across the entire corpus.
   * @param inputNgrams input RDD of lists of ngrams, where each list matches a single document
   * @return a JavaPairRDD of ngrams paired with their IDF values
   */
  public JavaPairRDD<String, Double> getNgramIdfs(JavaRDD<List<String>> inputNgrams) {

    // generate an RDD containing the unique ngrams in the input lists
    JavaRDD<String> ngrams = inputNgrams.flatMap(ngramList -> { 
      return ngramList; 
    })
        .distinct();

    // append to each ngram the inverse document frequency of that ngram in 
    // inputNgrams
    JavaPairRDD<String, Double> ngramIdfs = ngrams.mapToPair(ngram -> {
      Long ngramCount = inputNgrams.filter(ngramList -> {
        return ngramList.contains(ngram);
      })
          .count();
      Double idf = Math.log(inputNgrams.count() / ngramCount);
      return new Tuple2<String, Double>(ngram, idf);
    });

    return ngramIdfs;
  }

  /**
   * Gets TF and IDF values for ngrams, multiplies them together, and puts them into a SimpleRecord.
   * @param inputRecordsRdd input RDD of SimpleRecords
   * @return a JavaRDD of SimpleRecords that have "ngram" and "tfidf" fields in each record
   */
  public JavaRDD<SimpleRecord> getTfIdfs(JavaRDD<SimpleRecord> inputRecordsRdd) {
    // get values via other functions
    JavaRDD<List<String>> ngramLists = getNgrams(inputRecordsRdd);
    JavaRDD<List<Tuple2<String, Integer>>> ngramTFs = getNgramTFs(ngramLists);
    JavaPairRDD<String, Double> ngramIDFs = getNgramIDFs(ngram_lists);

    // combine tfs and idfs into a SimpleRecord RDD
    JavaRDD<SimpleRecord> ngramTfIdfs = ngramTFs.flatMap(ngramList -> {
      List<SimpleRecord> output = Lists.newArrayList();
      for (Tuple2<String, Integer> tfpair : ngramList) {
        SimpleRecord record = new SimpleRecord();
        String ngram = tfpair._1;
        Integer tf = tfpair._2;
        Double idf = ngramIdfs.lookup(ngram).get(0);
        Double tfidf = tf * idf;
        record.put("ngram", ngram);
        record.put("tfidf", tfidf);
        output.add(record);
      }
      return output;
    }); 

    return ngramTfIdfs;
  }
}
