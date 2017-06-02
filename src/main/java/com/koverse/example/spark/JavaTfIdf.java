package com.koverse.example.spark;

import com.koverse.com.google.common.collect.Lists;

import com.koverse.sdk.data.SimpleRecord;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

import scala.Tuple2;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

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
   * Divides text records into strings of n adjacent words, labeled by document.
   * @param inputRecordsRdd input RDD of SimpleRecords
   * @return a JavaPairRDD of n-grams from the input records keyed by document id
   */
  public JavaPairRDD<Long, String> getNgrams(JavaRDD<SimpleRecord> inputRecordsRdd) {
    // add document ids to records
    JavaPairRDD<SimpleRecord, Long> inputRecordsWithIds = inputRecordsRdd.zipWithIndex();

    JavaPairRDD<Long, String> ngrams = inputRecordsWithIds.flatMapToPair(recordPair -> {
      SimpleRecord record = recordPair._1;
      Long id = recordPair._2;
      // Split document string to words and make lowercase
      String[] words = record.get(textFieldName).toString().toLowerCase().split(tokenizationString);
      ArrayList<String> document = new ArrayList<String>(Arrays.asList(words));
      // Parse ngrams out of list of strings
      List<Tuple2<Long, String>> pairs = Lists.newArrayList();
      for (int i = 0; i < (document.size() - ngramSize); i++) {
        String ngram = String.join(" ", document.subList(i, i + ngramSize));
        pairs.add(new Tuple2<Long, String>(id, ngram));
      }
      return pairs;
    });

    return ngrams;
  }

  /**
   * Pairs ngrams with their term frequencies weighted by document length.
   * @param inputNgrams input JavaPairRDD of labeled ngrams
   * @return a JavaPairRDD of document id+ngram terms paired with their tf values
   */
  public JavaPairRDD<Tuple2<Long, String>, Double> getTFs(JavaPairRDD<Long, String> inputNgrams) {
    // get the number of terms in each document
    Map<Long, Object> documentLengths = inputNgrams.countByKey();

    JavaPairRDD<Tuple2<Long, String>, Double> ngramCounts = inputNgrams
        .groupByKey()
        .flatMapToPair(document -> {
          Long id = document._1;
          Iterable<String> ngramList = document._2;
          // count terms per document by incrementing values in a hashmap
          HashMap<String, Integer> countMap = new HashMap<String, Integer>();
          for (String ngram : ngramList) {
            int count = countMap.getOrDefault(ngram, 0) + 1;
            countMap.put(ngram, count);
          } 

          // compile list of terms and pair them with counts divided by document lengths
          // int termsInDocument = ngramList.size();
          long termsInDocument = (long)documentLengths.get(id);
          List<Tuple2<Tuple2<Long, String>, Double>> pairs = Lists.newArrayList();
          for (HashMap.Entry<String, Integer> pair : countMap.entrySet()) {
            Tuple2<Long, String> term = new Tuple2<Long, String>(id, pair.getKey());
            double tf = pair.getValue() / (double)termsInDocument;
            pairs.add(new Tuple2<Tuple2<Long, String>, Double>(term, tf)); 
          }
          return pairs;
        });

    return ngramCounts;
  }

  /**
   * Pairs ngrams with their inverse document frequencies across the entire corpus.
   * @param inputNgrams input JavaPairRDD of document ids paired to ngrams
   * @return a JavaPairRDD of ngrams paired with their IDF values
   */
  public JavaPairRDD<String, Double> getIdfs(JavaPairRDD<Long, String> inputNgrams) {

    // generate an RDD containing the unique ngrams in the input lists
    JavaRDD<String> ngrams = inputNgrams.values().distinct();
    long totalNgramCount = ngrams.count();

    // generate a map containing the ngrams paired with their document counts
    JavaPairRDD<String, Long> reverseInputNgrams = inputNgrams.mapToPair(pair -> {
      return new Tuple2<String, Long>(pair._2, pair._1);
    });
    Map<String, Object> ngramDocCounts = reverseInputNgrams.countByKey();


    // append to each ngram the inverse document frequency of that ngram in 
    // inputNgrams
    JavaPairRDD<String, Double> ngramIdfs = ngrams.mapToPair(ngram -> {
      long ngramCount = (long)ngramDocCounts.get(ngram);
      Double idf = Math.log(totalNgramCount / ngramCount);
      return new Tuple2<String, Double>(ngram, idf);
    });

    return ngramIdfs;
  }

  /**
   * Gets TF and IDF values for ngrams, multiplies them together, and puts them into a SimpleRecord.
   * @param inputRecordsRdd input RDD of SimpleRecords
   * @return a JavaRDD of SimpleRecords with "document_id", "ngram" and "tfidf" fields
   */
  public JavaRDD<SimpleRecord> getTfIdfs(JavaRDD<SimpleRecord> inputRecordsRdd) {
    // get values via other functions
    JavaPairRDD<Long, String> ngrams = getNgrams(inputRecordsRdd);
    JavaPairRDD<Tuple2<Long, String>, Double> ngramTFs = getTFs(ngrams);
    Map<String, Double> ngramIdfs = getIdfs(ngrams).collectAsMap();

    // combine tfs and idfs into a SimpleRecord RDD
    JavaRDD<SimpleRecord> ngramTfIdfs = ngramTFs.map(ngramPair -> {
      // consolidate information
      Tuple2<Long, String> document = ngramPair._1;
      Long id = document._1;
      String ngram = document._2;
      Double tf = ngramPair._2;
      Double idf = ngramIdfs.get(ngram);

      // calculate tfidf
      Double tfidf = tf * idf;

      // build record
      SimpleRecord record = new SimpleRecord();
      record.put("document_id", id);
      record.put("ngram", ngram);
      record.put("tfidf", tfidf);
      return record;
    }); 

    return ngramTfIdfs;
  }
}
