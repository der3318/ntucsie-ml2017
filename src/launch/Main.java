package launch;

import hw0.DataExtraction;
import hw0.ImgConverter;

public class Main {
	
	public static void main(String[] args) throws Exception {
		if( args[0].equals("1") )
			new DataExtraction().extractCol2File(args[2], Integer.parseInt(args[1]), "ans1.txt");
		else if( args[0].equals("2") )
			new ImgConverter().convert(args[1], "ans2.png");
	}
	
}
