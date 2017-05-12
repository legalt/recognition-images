#ifndef BMP_H
#define BMP_H

#include <iostream>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <math.h>

class ReadBMP
{
	private:
		std::string mFileName;
		int mWidth = 0;
		int mHeight = 0;
		int ** mData;

	public:
		ReadBMP ( std::string );
		int getWidth ();
		int getHeight ();		
		int ** getPixelData ();
		~ReadBMP ();
};

#endif