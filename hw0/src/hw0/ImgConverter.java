package hw0;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class ImgConverter {

	public void convert(String _inPath, String _outPath) {
		BufferedImage img = null, imgOut = null;
		try {
			img = ImageIO.read( new File(_inPath) );
			imgOut = new BufferedImage(img.getColorModel(), img.copyData(null), img.getColorModel().isAlphaPremultiplied(), null);
		} catch (IOException e) {
			System.out.println(e);
		}
		int width = img.getWidth(), height = img.getHeight();
	    for(int x = 0 ; x < width ; x++)
	    	for(int y = 0 ; y < height ; y++)
	    	    imgOut.setRGB( x, y, img.getRGB(width - x - 1, height - y - 1) );
	    try{
	      ImageIO.write( imgOut, "png", new File(_outPath) );
	    }catch(IOException e){
	      System.out.println(e);
	    }

	}
	
}
