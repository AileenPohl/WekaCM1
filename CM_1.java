package weka.filters.supervised.attribute;


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
import java.util.Map.Entry;
import java.util.Vector;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.Remove;


public class CM_1 extends SimpleBatchFilter{
	
	private static final long serialVersionUID = 1L;
	protected int m_folds;
	private int m_toprange;
	private int m_bottomrange;
	private int m_targetclass = 1;
	private Map<String, Double> RankingSums;
	private Map<String, Double> rangedRankings;
	private boolean m_graphcomputed;
	
	
	public CM_1(){
		m_folds = 10;
		m_toprange = 5;
		m_bottomrange = 5;
		
		RankingSums = new HashMap<String, Double>();
		rangedRankings = new HashMap<String, Double>();
		
	}

	
	public String globalInfo() {
	
		return "A supervised attribute filter is used to compute " 
		  + "the CM_1 score for different datasets . It uses a 10 fold cross validation"
		  + "and creates a CM_1 score for each fold finally combined to one big ranking.";
	}
	
	public Enumeration listOptions() {
//		
	    Vector<Option> result;
    	Enumeration enm;
//		
		result = new Vector<Option>();
	
	    enm = super.listOptions();
	    while (enm.hasMoreElements())
	      result.addElement((Option) enm.nextElement());
	
	    result.addElement(new Option("\tThe number of folds (default: 10).\n",
	        "-f", 1, "-f <int>"));
	    
	    result.addElement(new Option("\tThe number of elements taken from the top for ranking (default: 5).\n",
		        "-T", 1, "-T <int>"));
	    
	    result.addElement(new Option("\tThe number of elements taken from the bottom for ranking (default: 5).\n",
		        "-B", 1, "-B <int>"));
	    
	    result.addElement(new Option("\tThe target Class (default: 1).\n",
		        "-C", 1, "-C <int>"));
	    
	    return result.elements();
	  }
	  
	public void setOptions(String[] options) throws Exception {
			
		 String numberofFoldsString = Utils.getOption('k', options);
			if (numberofFoldsString.length() != 0) {
			  setNumberofFolds((Integer.parseInt(numberofFoldsString)));
			} else {
				setNumberofFolds(10);
			} 
		    
	    String topRangeString = Utils.getOption('T', options);
	    if (topRangeString.length() != 0) {
	      setTopRange((Integer.parseInt(topRangeString)));
	    } else {
	    	setTopRange(5);
	    }
	    
	    String bottomRangeString = Utils.getOption('B', options);
	    if (bottomRangeString.length() != 0) {
	      setBottomRange((Integer.parseInt(bottomRangeString)));
	    } else {
	    	setBottomRange(5);
	    }
	    
	    String targetClassString = Utils.getOption('C', options);
	    if (targetClassString.length() != 0) {
		      setTargetClass(Integer.parseInt(targetClassString));
		    }
	    else {
	    	setTargetClass(1);
	    }
	    String graphComputedString = Utils.getOption('G', options);
	    if (graphComputedString.length() != 0) {
		      setGraphComputed(Boolean.parseBoolean(targetClassString));
		    }
	    else {
	    	setGraphComputed(false);
	    }
	    
		    
		    if (getInputFormat() != null)
		        setInputFormat(getInputFormat());
	  }


	public String[] getOptions(){
		 Vector<String> result = new Vector<String>();
	    String[] options = super.getOptions();
	    for (int i = 0; i < options.length; i++) {
	      result.add(options[i]);
	    }

	    result.add("-k");
	    result.add("" + getNumberofFolds());
	    result.add("-T");
	    result.add("" + getTopRange());
	    result.add("-B");
	    result.add("" + getBottomRange());
	    result.add("-C");
	    result.add("" + getTargetClass());
	    result.add("-G");
	    result.add("" + isGraphComputed());
	    
	    
	    return result.toArray(new String[result.size()]);
	}
	public boolean isGraphComputed() {
		
		return m_graphcomputed;
	}
	
	public void setGraphComputed(boolean graphcomputed){
		m_graphcomputed = graphcomputed;
	}
	
	public String graphcomputedTipText(){
		return "Specifies if a optional graph of the results should be computed";
	}
	
	public int getNumberofFolds() {
		return m_folds;
	}
	
	public void setNumberofFolds(int folds) {
		m_folds = folds;
	  }
	
	public String numberofFoldsTipText(){
		return "The Number of Folds which are used for Cross-Validation";
	}
	
	public int getTopRange() {
		return m_toprange;
	}
	
	public void setTopRange(int toprange) {
		m_toprange = toprange;
		
	}
	
	public String TopRangeTipText(){
		return "The Number of topattributes taken for Ranking";
	}
	
	public int getBottomRange() {
		return m_bottomrange;
	}
	
	public void setBottomRange(int bottomrange) {
		m_bottomrange = bottomrange;
		
	}
	
	public String BottomRangeTipText(){
		return "The Number of bottomattributes taken for Ranking";
	}

	public int getTargetClass(){
		return m_targetclass;
	}
	
	public void setTargetClass(int targetclass) {
		m_targetclass = targetclass;
	  }
	
	public String targetClassTipText(){
		return "Sets the target class for the attribute selection as an numerical value";
	}
	
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
	     
	     
	     applyRangesandComputeJSON(RankingSums);
	     
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
			 Instances modifiedData = Filter.useFilter(inputdata, remove);		 	// apply filter
			 
			 computeCM1(modifiedData);
		 }
	 }
	 
	public void computeCM1(Instances mergedFolds) throws IOException{
		 
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

				 if(inst.classValue() == (double) m_targetclass - 1.0){								// zero based index of values, but input > 0
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
			 
			 double CM_1Score = ((sum_specificClass/num_specificClass) - (sum_otherClasses/num_otherClasses))/(1+(max-min));
			 CM_1Scores.put(mergedFolds.attribute(attribute).name(), CM_1Score); // put CM_1 score for each column of attribute
	 		} //all attributes computed
		 compute_Ranking(CM_1Scores);
		 }
	 
	public void compute_Ranking(Map<String, Double> CM_1Scores){
		 
		 Map<String, Double> sorted = sortByValues(CM_1Scores);
		 List<String> sortedAsArray = new ArrayList<String>(sorted.keySet());		//convert Keys to array, CM_1 scores no longer needed
		 
		 for(int i = 0; i< sortedAsArray.size(); i++){
			 if(!RankingSums.containsKey(sortedAsArray.get(i))){
				 RankingSums.put(sortedAsArray.get(i), ((double)i+1));
			 }
			 else{
				 RankingSums.put(sortedAsArray.get(i), RankingSums.get(sortedAsArray.get(i)) + ((double)i+1)) ;	//update attributes ranking sum; ranking[attribute] = previous sum + index
			 }
		 }
	 }
	 
	public Instances adjustInstances( Instances input) throws Exception{
		 String IndicesToBeRemoved = "";
		 System.out.println("Ranged Rankings: " + rangedRankings);
		 for (int i = 0; i < input.numAttributes(); i++) {
		      Attribute att = input.attribute(i);
		      if(!(rangedRankings.containsKey(att.name())) && att.index()!= input.classIndex()){
		    	  IndicesToBeRemoved += String.valueOf(i+1) + ",";
		      }
		 }
		 System.out.println("Indices to be removed: " + IndicesToBeRemoved);
		 if(!IndicesToBeRemoved.isEmpty()){
			 IndicesToBeRemoved = IndicesToBeRemoved.substring(0, IndicesToBeRemoved.length() - 1); //remove additional comma at end
			 Remove remove = new Remove();
			 remove.setAttributeIndices(IndicesToBeRemoved);
		   	 remove.setInputFormat(input);
		   	 input = Filter.useFilter(input, remove);
		 }
		 
		 return input;
	}
		 
	public void applyRangesandComputeJSON (Map<String, Double> sortedbyRanking) throws IOException{
		 
		 int index = 1;
		 String jsontopattributes = "[{\"key\": \"topattributes\", \"color\": \"#d62728\"  , \"values\": [";
		 String jsonleastattributes = "{\"key\": \"bottomattributes\", \"color\": \"#1f77b4\",  \"values\": [";
		 
		  for (Map.Entry pairs : sortedbyRanking.entrySet()) {
		        if(index < m_bottomrange)
		        {
		        	jsonleastattributes = jsonleastattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  pairs.getValue().toString() + "} , ";	
					rangedRankings.put(pairs.getKey().toString(), (Double) pairs.getValue());
		        }
		        if(index == m_bottomrange)
		        {
		        	jsonleastattributes = jsonleastattributes + "{ \"label\" : " + "\"" + pairs.getKey().toString() + "\"" + ", \"value\": " +  pairs.getValue().toString() + "}]}]";
					rangedRankings.put(pairs.getKey().toString(), (Double) pairs.getValue());
		        }
		        if (index > sortedbyRanking.size() - m_toprange)
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
		  
		  //use Json to compute Graph if user set the option
		  if(isGraphComputed()){
			  String finaljson = "CM_1data = " + jsontopattributes + jsonleastattributes;
			  try {
				  generateGraph(finaljson);
			  } catch (IOException e) {
					e.printStackTrace();
			  }
		  }
			  
			  
				  
		  }
	
	 private void generateGraph(String finaljson) throws IOException {
			  
		 FileWriter file = new FileWriter("CM1.json");
		 file.write(finaljson);
		 file.flush();
		 file.close();
		
		 String operatingSystem = System.getProperty("os.name");
		  
		 if(operatingSystem.startsWith("Windows"))
			 Runtime.getRuntime().exec("cmd start index.html");
		 if(operatingSystem.startsWith("Mac"))
			 Runtime.getRuntime().exec("open index.html");
		 if(operatingSystem.startsWith("Linux")){
			 Runtime.getRuntime().exec("xdg-open index.html");
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