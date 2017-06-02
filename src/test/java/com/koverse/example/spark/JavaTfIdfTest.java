package com.koverse.example.spark;

import static org.junit.Assert.*; 
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.junit.Test;

import com.holdenkarau.spark.testing.SharedJavaSparkContext;
import com.koverse.com.google.common.collect.Lists;
import com.koverse.sdk.data.SimpleRecord;
import java.util.Optional;

import scala.Tuple2;

import java.util.Map;
/**
 * These tests leverage the great work at https://github.com/holdenk/spark-testing-base
 */
public class JavaTfIdfTest  extends SharedJavaSparkContext {

  @Test
  public void rddTest() {
    // Create the SimpleRecords we will put in our input RDD
    SimpleRecord record0 = new SimpleRecord();
    SimpleRecord record1 = new SimpleRecord();
    SimpleRecord record2 = new SimpleRecord();
    SimpleRecord record3 = new SimpleRecord();
    record0.put("text", "these words are to be counted");
    record0.put("id", 0);
    record1.put("text", "more words that are worth counting");
    record1.put("id", 1);
    record2.put("text", "this three gram this three gram this three gram this three gram");
    record2.put("id", 2);
    record3.put("text", "are worth counting");
    record3.put("id", 3);
    
    // Create the input RDD
    JavaRDD<SimpleRecord> inputRecordsRdd = jsc().parallelize(Lists.newArrayList(record0, record1, record2, record3));
    
    // Create tests for 2-grams and 3-grams
    JavaTfIdf tfidf2gram = new JavaTfIdf("text", "['\".?!,:;\\s]", 2);
    JavaTfIdf tfidf3gram = new JavaTfIdf("text", "['\".?!,:;\\s]", 3);

    // Test 2gram ngrams
    // Should return rows for all of the (non-unique) ngrams in the input text
    JavaPairRDD<Long, String> output2grams = tfidf2gram.getNgrams(inputRecordsRdd);
    assertEquals(23, output2grams.count());
    // Check that ngrams at the beginnings and ends of documents are included
    List<String> output2gramsVals = output2grams.values().collect();
    assertTrue(output2gramsVals.contains("these words"));
    assertTrue(output2gramsVals.contains("worth counting"));

    // Test 3gram ngrams
    // Should return rows for all of the (non-unique) ngrams in the input text
    JavaPairRDD<Long, String> output3grams = tfidf3gram.getNgrams(inputRecordsRdd);
    assertEquals(19, output3grams.count());
    // Check that ngrams at the beginnings and ends of documents are included
    List<String> output3gramsVals = output3grams.values().collect();
    assertTrue(output3gramsVals.contains("these words are"));
    assertTrue(output3gramsVals.contains("are worth counting"));

    // Test 2gram TFs
    // Should return rows for all of the (unique within a document) ngrams
    JavaPairRDD<Tuple2<Long, String>, Double> outputTfs2g = tfidf2gram.getTFs(output2grams);
    assertEquals(15, outputTfs2g.count());
    // Should calculate Tf as (term count within the document) / (term length of document)
    Map<Tuple2<Long, String>, Double> outputTfs2gvals = outputTfs2g.collectAsMap();
    Double expectedtf_2g_doc0 = 1.0 / 5.0;
    Double expectedtf_2g_doc2 = 4.0 / 11.0;
    assertEquals(expectedtf_2g_doc0, outputTfs2gvals.get(new Tuple2<Long, String>(0l, "these words")));
    assertEquals(expectedtf_2g_doc2, outputTfs2gvals.get(new Tuple2<Long, String>(2l, "this three")));

    // Test 3gram TFs
    // Should return rows for all of the (unique within a document) ngrams
    JavaPairRDD<Tuple2<Long, String>, Double> outputTfs3g = tfidf3gram.getTFs(output3grams);
    assertEquals(12, outputTfs3g.count());
    // Should calculate Tf as (term count within the document) / (term length of document)
    Map<Tuple2<Long, String>, Double> outputTfs3gvals = outputTfs3g.collectAsMap();
    Double expectedtf_3g_doc0 = 1.0 / 4.0;
    Double expectedtf_3g_doc2 = 4.0 / 10.0;
    assertEquals(expectedtf_3g_doc0, outputTfs3gvals.get(new Tuple2<Long, String>(0l, "these words are")));
    assertEquals(expectedtf_3g_doc2, outputTfs3gvals.get(new Tuple2<Long, String>(2l, "this three gram")));

    //Test 2gram IDFs
    // Should return rows for all of the (unique across the corpus) ngrams
    JavaPairRDD<String, Double> outputIdfs2g = tfidf2gram.getIdfs(output2grams);
    assertEquals(13, outputIdfs2g.count());
    // Should calculate Idf as log(total documents in corpus / number of documents containing at least 
    // one instance of term)
    Map<String, Double> outputIdfs2gvals = outputIdfs2g.collectAsMap();
    Double expectedidf_2g_these_words = Math.log(4.0 / 1.0);
    Double expectedidf_2g_are_worth = Math.log(4.0 / 2.0);
    Double expectedidf_2g_this_three = Math.log(4.0 / 1.0);
    assertEquals(expectedidf_2g_these_words, outputIdfs2gvals.get("these words"));
    assertEquals(expectedidf_2g_are_worth, outputIdfs2gvals.get("are worth"));
    assertEquals(expectedidf_2g_this_three, outputIdfs2gvals.get("this three"));
    
    //Test 3gram IDFs
    // Should return rows for all of the (unique across the corpus) ngrams
    JavaPairRDD<String, Double> outputIdfs3g = tfidf3gram.getIdfs(output3grams);
    assertEquals(11, outputIdfs3g.count());
    // Should calculate Idf as log(total documents in corpus / number of documents containing at least 
    // one instance of term)
    Map<String, Double> outputIdfs3gvals = outputIdfs3g.collectAsMap();
    Double expectedidf_3g_these_words_are = Math.log(4.0 / 1.0);
    Double expectedidf_3g_are_worth_counting = Math.log(4.0 / 2.0);
    Double expectedidf_3g_this_three_gram = Math.log(4.0 / 1.0);
    assertEquals(expectedidf_3g_these_words_are, outputIdfs3gvals.get("these words are"));
    assertEquals(expectedidf_3g_are_worth_counting, outputIdfs3gvals.get("are worth counting"));
    assertEquals(expectedidf_3g_this_three_gram, outputIdfs3gvals.get("this three gram"));

    //Test 2gram TfIdfs
    // Should return rows for all of the (unique within a document) ngrams
    JavaRDD<SimpleRecord> outputTfIdfs2g = tfidf2gram.getTfIdfs(inputRecordsRdd);
    assertEquals(15, outputTfIdfs2g.count());
    // Check for inclusion of an ngram that should appear
    List<SimpleRecord> outputRecords2gram = outputTfIdfs2g.collect();
    Optional<SimpleRecord> countRecordOptional2gram = outputRecords2gram.stream()
     .filter(record -> record.get("ngram").equals("are to"))
     .findFirst();
    assertTrue(countRecordOptional2gram.isPresent());
    // Should calculate TfIdf as Tf * Idf
    assertEquals((1.0/5.0) * Math.log(4.0/1.0), countRecordOptional2gram.get().get("tfidf"));

    //Test 3gram TfIdfs
    // Should return rows for all of the (unique within a document) ngrams
    JavaRDD<SimpleRecord> outputTfIdfs3g = tfidf3gram.getTfIdfs(inputRecordsRdd);
    assertEquals(12, outputTfIdfs3g.count());
    // Check for inclusion of an ngram that should appear
    List<SimpleRecord> outputRecords3gram = outputTfIdfs3g.collect();
    Optional<SimpleRecord> countRecordOptional3gram = outputRecords3gram.stream()
     .filter(record -> record.get("ngram").equals("are to be"))
     .findFirst();
    assertTrue(countRecordOptional3gram.isPresent());
    // Should calculate TfIdf as Tf * Idf
    assertEquals((1.0/4.0) * Math.log(4.0/1.0), countRecordOptional3gram.get().get("tfidf"));
  }
}
