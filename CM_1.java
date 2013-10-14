package weka.filters.supervised.attribute;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.Map.Entry;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

import net.sf.json.JSONArray;
import net.sf.json.JSONObject;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.converters.CSVSaver;
import weka.core.converters.JSONLoader;
import weka.core.json.JSONNode;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.Remove;


public class CM_1 extends SimpleBatchFilter{
	
	private int m_folds = 10;
	private int toprange = 5;
	private int bottomrange = 5;
	
	private Map<String, Double> RankingSums = new HashMap<String, Double>();
	private Map<String, Double> rangedRankings = new HashMap<String, Double>();
	
	public String globalInfo() {
	
		return "A supervised attribute filter is used to compute " 
		  + "the CM_1 score for different datasets . It uses a 10 fold cross validation"
		  + "and creates a CM_1 score for each fold finally combined to one big ranking.";
	}
	
	  /**
	   * Returns an enumeration describing the available options.
	   *
	   * @return an enumeration of all the available options.
	   */
//	  public Enumeration listOptions() {
//		
//	    Vector newVector = new Vector(1);
//		
//	    newVector.addElement(new Option("\tSpecify the seed of randomization\n"
//					    + "\tused to randomize the class\n"
//					    + "\torder (default: 1)",
//					    "R", 1, "-R <seed>"));
//		
//	    newVector.
//		addElement(new Option("\tSet number of folds for reduced error\n" +
//				      "\tpruning. One fold is used as pruning set.\n" +
//				      "\t(default 3)",
//				      "N", 1, "-N <number of folds>"));
//		
//	    return newVector.elements();
//	  }
//	  
//	  public void setOptions(String[] options) throws Exception {
//			
//		    String seedString = Utils.getOption('N', options);
//		    if (seedString.length() != 0)
//		    	m_folds = Integer.parseInt(seedString);
//		    else 
//		    	m_folds = 1;  
//		  }
	
	public Capabilities getCapabilities(){
		Capabilities result = super.getCapabilities();
		result.enableAllAttributes();
		result.enableAllClasses(); //// filter doesn't need class to be set//
	    return result;
	}
	 
	protected Instances determineOutputFormat(Instances inputFormat) {
		 Instances result = new Instances(inputFormat);	//output format same like input without any new attributes, all instances copied
		 return result;
	 }
	 
	 protected Instances process(Instances inst) throws Exception {
		 Instances result = new Instances(determineOutputFormat(inst), 0);
	     for (int i = 0; i < inst.numInstances(); i++) {
	       double[] values = new double[result.numAttributes()];
	       for (int n = 0; n < inst.numAttributes(); n++)
	         values[n] = inst.instance(i).value(n);
	       result.add(new DenseInstance(1, values)); //denseinstance able to store every value, not only numerical
	     }
	     setInputFormat(result);
	     createFolds(result);
	     Instances finalresult = adjustInstances(result);
	     setOutputFormat(finalresult);
	     return finalresult;
	 }

	 public void createFolds(Instances inputdata) throws Exception{
		 StratifiedRemoveFolds remove = new StratifiedRemoveFolds();    	// new instance of filter
		 remove.setNumFolds(m_folds);  											// set options, folds defined by user
		 remove.setSeed(1);
		 remove.setInvertSelection(true);									//divide in n-folds and get the first until n-1 fold to compute CM_1 score 
		 
		 for(int i = 1; i <= m_folds; i++){
			 remove.setFold(i);
			 remove.setInputFormat(inputdata);                          		// inform filter about dataset **AFTER** setting options
			 Instances newData = Filter.useFilter(inputdata, remove);		 	// apply filter
//			 CSVSaver saver = new CSVSaver();
//			 saver.setInstances(newData);
//			 saver.setFile(new File("./test" + i + ".csv"));
//			 saver.writeBatch();
			 applyCM_1(newData);
		 }
		 
		 //get Ranking from all folds and if flag set compute json

		 Map<String, Double> sortedbyRanking = sortByValues(RankingSums); 
		 writeRankingtoFile(sortedbyRanking);
	 }
	 
	 public void applyCM_1(Instances mergedFolds) throws IOException{
		 
		 double wantedClass = 1.0;												//class value as double, if there are three different classes the values are 0.0, 1.0 and 2.0
		 
		 int numAttributes = mergedFolds.numAttributes();
		 int num_specificClass = 0;
		 int num_otherClasses = 0;
		 double min = -1.0;														//initialize with -1 making the assumption that there are only positive values in the dataset
		 double max = 0.0;
		 
		 double sum_specificClass = 0.0;											
		 double sum_otherClasses = 0.0;	
		 
		 Map<String, Double> CM_1Scores = new HashMap<String, Double>();		//create a HashMap to store the CM_1 Score for each fold
		 
		 int instances = mergedFolds.numInstances();									//get the number of instances  in each fold
		 
		 for(int attribute = 0; attribute < numAttributes-1; attribute++){		//get sum for each attribute column
			 
			 for(int instance = 0 ; instance < instances; instance++){			//for each instance get the data
				 Instance inst = mergedFolds.get(instance);
				 
				 if(inst.classValue() == wantedClass){
					 sum_specificClass= sum_specificClass + inst.value(attribute);						//add attributes value to sum array
					 num_specificClass++;
				 }
				 else{
					 double value = inst.value(attribute);
					 sum_otherClasses  = sum_otherClasses + value;
				 	 num_otherClasses++;
				 	 
				 	 if(value < min || min == -1)											//calculate max and min value
				 		 min = value;
				 	 else if (value > max)
				 		 max = value;
				 	}
				 
			 	}
//			 
//			 System.out.println("Min: "+ min );
//			 System.out.println("Max: "+ max);
//			 System.out.println("SumClass: "+ sum_specificClass);
//			 System.out.println("SumOther: "+ sum_otherClasses);
//			 System.out.println("NumClasses: "+ num_specificClass);
//			 System.out.println("NumOther: "+ num_otherClasses);
//			
//			 System.out.println("Wanted Class: "+mergedFolds.classAttribute().value(1));
			 
			 double CM_1Score = ((sum_specificClass/num_specificClass) - (sum_otherClasses/num_otherClasses))/(1+(max-min));
//			 System.out.println("CM_1 Score for attribute: " +mergedFolds.attribute(attribute).toString() + " -> "+ CM_1Score);
			 CM_1Scores.put(mergedFolds.attribute(attribute).name(), CM_1Score); // put CM_1 score for each colum of attribute
	 		} //all attributes computed
		 compute_Ranking(CM_1Scores);
		 }
	 
	 public void compute_Ranking(Map<String, Double> CM_1Scores) throws IOException{
		 
		 Map<String, Double> sorted = sortByValues(CM_1Scores);
		 List<String> sortedAsArray = new ArrayList<String>(sorted.keySet());		//convert Keys to array, CM_1 scores no longer needed
		 for(int i = 0; i< sortedAsArray.size(); i++){
			 if(!RankingSums.containsKey(sortedAsArray.get(i))){
				 RankingSums.put(sortedAsArray.get(i), (double)i+1);
			 }
			 else{
				 RankingSums.put(sortedAsArray.get(i), RankingSums.get(sortedAsArray.get(i)) + (double)i+1) ;	//update attributes ranking sum; ranking[attribute] = previous sum + index
			 }
		 }
	 }
	 
	 public Instances adjustInstances( Instances input) throws Exception{
		 System.out.println(rangedRankings);
		 String IndicesToBeRemoved = "";
		 for (int i = 0; i < input.numAttributes(); i++) {
		      Attribute att = getInputFormat().attribute(i);
		      if(!(rangedRankings.containsKey(att.name())) && att.index()!= input.classIndex()){
		    	  IndicesToBeRemoved += String.valueOf(i) + ",";
		      }
		 	}
		 if(!IndicesToBeRemoved.isEmpty()){
			 IndicesToBeRemoved = IndicesToBeRemoved.substring(0, IndicesToBeRemoved.length() - 1); //remove additional comma at end
			 Remove remove = new Remove();
			 remove.setAttributeIndices(IndicesToBeRemoved);
		   	 remove.setInputFormat(input);
		   	 input = Filter.useFilter(input, remove);
		 }
		 
		 return input;
	}
		 
		
	 public void writeRankingtoFile (Map<String, Double> sortedbyRanking) throws IOException{
		 
		 int index = 1;
		 String jsontopattributes = "[{\"key\": \"topattributes\", \"color\": \"#d62728\"  , \"values\": [";
		 String jsonleastattributes = "{\"key\": \"bottomattributes\", \"color\": \"#1f77b4\",  \"values\": [";
		 
		  for (Map.Entry pairs : sortedbyRanking.entrySet()) {
		        if(index < bottomrange)
		        {
		        	jsonleastattributes = jsonleastattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  pairs.getValue().toString() + "} , ";
		        	rangedRankings.put(pairs.getKey().toString(), (Double) pairs.getValue());
		        }
		        if(index == bottomrange)
		        {
		        	jsonleastattributes = jsonleastattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  pairs.getValue().toString() + "}]}]";
		        	rangedRankings.put(pairs.getKey().toString(), (Double) pairs.getValue());
		        }
		        if (index > sortedbyRanking.size() - toprange)
		        {
		        	jsontopattributes = jsontopattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  pairs.getValue().toString() + "} , ";
		        	rangedRankings.put(pairs.getKey().toString(), (Double) pairs.getValue());
		        }
		        if(index == sortedbyRanking.size())
		        {
		        	jsontopattributes = jsontopattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  pairs.getValue().toString() + "}]},";
		        	rangedRankings.put(pairs.getKey().toString(), (Double) pairs.getValue());
		        }
		        index++;
		        
		  }
		  
		  String finaljson = "CM_1data = " + jsontopattributes + jsonleastattributes;
		  String datajson = jsontopattributes + jsonleastattributes;
		  
		  try {
			  
				FileWriter file = new FileWriter("CM_1.json");
				file.write(finaljson);
				file.flush();
				file.close();

			} catch (IOException e) {
				e.printStackTrace();
			}

	 }
		 
	 public static <K extends Comparable,V extends Comparable> Map<K,V> sortByValues(Map<K,V> map){
	        List<Map.Entry<K,V>> entries = new LinkedList<Map.Entry<K,V>>(map.entrySet());
	      
	        Collections.sort(entries, new Comparator<Map.Entry<K,V>>() {

	            @Override
	            public int compare(Entry<K, V> o1, Entry<K, V> o2) {
	                return o2.getValue().compareTo(o1.getValue());
	            }
	        });
	      
	        //LinkedHashMap will keep the keys in the order they are inserted
	        //which is currently sorted on natural ordering
	        Map<K,V> sortedMap = new LinkedHashMap<K,V>();
	      
	        for(Map.Entry<K,V> entry: entries){
	            sortedMap.put(entry.getKey(), entry.getValue());
	        }
	      
	        return sortedMap;
	    }
	 
	 
	 public static void main(String[] args) {
		 runFilter(new CM_1(), args);
	 }
	

}