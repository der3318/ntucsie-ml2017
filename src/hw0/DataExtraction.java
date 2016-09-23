package hw0;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Comparator;
import java.util.PriorityQueue;

public class DataExtraction {
	
	public void extractCol2File(String _inPath, int _col, String _outPath) {
		try {
			PriorityQueue<Double> queue = new PriorityQueue<Double>( new DoubleComparator() );
			BufferedReader br;
			br = new BufferedReader( new FileReader(_inPath) );
			String line = null;
			while( ( line = br.readLine() ) != null )
				queue.add(Double.parseDouble( line.trim().split(" ")[_col]) );
			br.close();
			BufferedWriter bw = new BufferedWriter( new FileWriter( new File(_outPath) ) );
			if(queue.isEmpty() == false)	bw.write( "" + queue.poll() );
			while(queue.isEmpty() == false)	bw.write( "," + queue.poll() );
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private class DoubleComparator implements Comparator<Double>
	{
	    @Override
	    public int compare(Double x, Double y)
	    {
	        if(x < y)	return -1;
	        if(x > y)	return 1;	
	        return 0;
	    }
	}
	
}
