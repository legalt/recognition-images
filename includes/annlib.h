#ifndef ANNLIB_H
#define ANNLIB_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

class Neuron 
{
public:
    Neuron () {}
	Neuron ( int nInputsCount );
	~Neuron();	

	double * m_aWeights;
	double * m_aDelta;
	double 	m_nOutput = 0;
	double 	m_nInput = 0;
	double 	m_nWeight = 0;

    inline double getWeight ( int index ) { return m_aWeights[index]; }
    inline double getInput () { return m_nInput; }
    inline double getInputWeight () { return m_nWeight; }
    inline double getDeltaValue ( int index ) { return m_aDelta[index]; }

	inline void setOutput ( double nOutput ) { m_nOutput = nOutput;}
    inline void setDeltaValue ( int index, double delta ) { m_aDelta[index] = delta; }
};

class LayerPerceptron
{
public:
    LayerPerceptron (): m_aLayerInput(0) {}
	void init ( int nInputsSize, int nNeuronsCount );
	~LayerPerceptron();


	LayerPerceptron & operator= ( const LayerPerceptron & layer );

	void calculate_outputs();

	Neuron ** m_aNeurons;
    double *  m_aLayerInput;
	int 	  m_nNeuronsCount = 0;
	int 	  m_nInputsCount = 0;
};

class BackpropogationError {
private:
	LayerPerceptron m_inputLayer;
	LayerPerceptron m_outputLayer;
	LayerPerceptron ** m_aHiddenLayers;
	int m_nHiddenLayerCount = 0;

public:	
	BackpropogationError ( int nInputCount, int nInputNeuronsCount, int nOutputCount, int * aHiddenLayers, int nHiddenLayerCount );
	~BackpropogationError ();	

	void recognize ( const double * aInput );
	void update ( int nLayerIndex );
	double train ( const double *desiredoutput, const double *input, double alpha, double momentum );

	inline LayerPerceptron & getOutput () { return m_outputLayer; }
};

#endif
