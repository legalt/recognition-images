#include <iostream>
#include <string.h>
#include "../includes/bmp.h"
#include "../includes/annlib.h"
#include <dirent.h>
#include <sys/types.h>
#include <vector>

#define NETWORK_INPUTNEURONS 1
#define NETWORK_OUTPUT 900
#define HIDDEN_LAYERS 0
#define EPOCHS 20000

typedef ReadBMP parse_bmp;

std::vector<std::string> findTestImages ( const char * directory )
{
    DIR * folder_in = opendir(directory);
    std::vector<std::string> files;
    // char ** files;
    // int counter = 0;

    // files = new char*[32];
    while ( folder_in )
    {
        dirent * dp = readdir(folder_in);

        if ( dp != NULL ) 
        {
            if ( strlen(dp->d_name) > 2 )
            {
                // files[counter] = dp->d_name;
                // counter++;
                files.push_back(std::string(dp->d_name));
            }
        } else {
            closedir(folder_in);
            break;
        }
    }
    return files;
}

using namespace std;

int main ()
{	
    parse_bmp imageFrs("./test_data/t.bmp");
    parse_bmp imageSec("./test_data/t.bmp");
    double **learnSample;
    double **sample;
    	
	// init sample data and etalon
    {
        int counter = 0;
		learnSample = new double*[1];
		sample = new double*[1];
        learnSample[0] = new double[imageFrs.getHeight() * imageFrs.getWidth()];
		sample[0] = new double[imageSec.getHeight() * imageSec.getWidth()];
	
        for ( int y = 0; y < imageFrs.getHeight(); y++ )
        {
            for ( int x = 0; x < imageFrs.getWidth(); x++ )
            {
                learnSample[0][counter] = imageFrs.getPixelData()[y][x];
				sample[0][counter] = imageFrs.getPixelData()[y][x];
                counter++;
            }
        }
    }
	
    const int PATTERN_COUNT = 1;
    /*double pattern[PATTERN_COUNT][NETWORK_OUTPUT]=
    {
        {1, 1},
        {0, 1}
    };*/

    //desired output values
    //double desiredout[2][NETWORK_OUTPUT] = sample;

    BackpropogationError net(PATTERN_COUNT, NETWORK_INPUTNEURONS, NETWORK_OUTPUT, HIDDEN_LAYERS, HIDDEN_LAYERS);

    for( int i = 0; i < EPOCHS; i++ )
    {
        double error = 0;

        for( int j = 0; j < PATTERN_COUNT; j++ )
        {
            error += net.train(sample[j], learnSample[j], 0.2, 0.1);
        }

        error /= PATTERN_COUNT;
                
        std::cout << "ERROR:" << error << "\r";

    }

    //once trained test all patterns
    for ( int i = 0; i < PATTERN_COUNT; i++ )
    {
        net.recognize(learnSample[i]);    	
        std::cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *sample[i] << " NET RESULT: "<< net.getOutput().m_aNeurons[0]->m_nOutput << std::endl;
    }

	return 0;
}
