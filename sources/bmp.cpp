#include "../includes/bmp.h"
#include <stdlib.h>
ReadBMP::ReadBMP ( std::string fileName ) :mFileName(fileName) 
{ 
	const int HEADER_SIZE = 54;
			
	FILE * fImage = fopen(mFileName.c_str(), "rb");
	unsigned char info[HEADER_SIZE];

	if ( fImage == NULL )
	{
		std::cerr << "File can't read" << std::endl;
		exit(0);
	}

	fread(info, sizeof(unsigned char), HEADER_SIZE, fImage);
	
	mWidth  = *(int*)&info[18];
	mHeight = *(int*)&info[22];

	int row_padded = (mWidth * 3 + 3) & (~3);	
	unsigned char* data = new unsigned char[row_padded];    
	std::vector<int> pixels;    

    for( int i = 0; i < mHeight; i++ )
    {
        fread(data, sizeof(unsigned char), row_padded, fImage);

        for ( int j = 0; j < mWidth * 3; j += 3 )
        {            
            unsigned char tmp = data[j];
            data[j] = data[j+2];
            data[j+2] = tmp;
            
            int sum = (int)data[j] + data[j + 1] + data[j + 2];
			int value = (sum < 100 ? 0 : 1);

			pixels.push_back(value);
        }
    }

    delete [] data;
    fclose(fImage);
    
    mData = new int*[mWidth];    
    for ( int x = 0; x < mWidth; x++ )
    {
    	mData[x] = new int[mHeight];
    	for (int y = 0; y < mHeight; y++ )
    	{    		
    		mData[x][y] = pixels.at(x + y);		
    	}
    }

    pixels.clear();	        	
}

ReadBMP::~ReadBMP ()
{
	if ( mData )
	{
		for ( int x = 0; x < mWidth; x++ )
		{
			delete [] mData[x];
		}

		delete [] mData;
	}	
}

int ReadBMP::getWidth () { return mWidth; }

int ReadBMP::getHeight () { return mHeight; }

int ** ReadBMP::getPixelData () { return mData; }
