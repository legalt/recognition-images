#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "../includes/annlib.h"

// sigmoid act function
double activate ( double sum )
{           
    return 1 / ( 1 + exp(-sum));    
}

Neuron::Neuron ( int nInputsCount )
{
	double sign = -1;
	double weight;
	m_aDelta = new double(nInputsCount);
	m_aWeights = new double (nInputsCount);

	for ( int i = 0; i < nInputsCount; i++ )
	{
		weight = (double(rand()) / double(RAND_MAX)) / 2.f;
		weight *= sign;
		sign *= -1;

		m_aWeights[i] = weight;
		m_aDelta[i] = 0;
	}

	m_nInput = 1;

	weight = (double(rand()) / double(RAND_MAX)) / 2.f;
	weight *= sign;
	sign *= -1;
	m_nWeight = weight;
} 

Neuron::~Neuron ()
{
	// delete [] m_aWeights;
	// delete [] m_aDelta;
}

void LayerPerceptron::init ( int nInputsCount, int nNeuronsCount )
{
	m_nNeuronsCount = nNeuronsCount;
	m_nInputsCount = nInputsCount;
	m_aNeurons = new Neuron*[nNeuronsCount];
	m_aLayerInput = new double[nInputsCount];

	for ( int i = 0; i < nNeuronsCount; i++ )
	{
		m_aNeurons[i] = new Neuron(nInputsCount);
	}
}

LayerPerceptron::~LayerPerceptron () 
{
	// delete [] m_aNeurons;
	// delete [] m_aLayerInput;
}

void LayerPerceptron::calculate_outputs ()
{
	double sum = 0;

	for ( int i = 0; i < m_nNeuronsCount; i++ )
	{
		for ( int j = 0; j < m_nInputsCount; j++ )
		{
			sum += m_aNeurons[i]->getWeight(j) * m_aLayerInput[j];            
		}

		sum += m_aNeurons[i]->getInputWeight() * m_aNeurons[i]->getInput();
        // printf("%d\n", activate(sum));
		m_aNeurons[i]->setOutput(activate(sum));
	}
}

LayerPerceptron & LayerPerceptron::operator= ( const LayerPerceptron & layer ) 
{
		m_aNeurons = new Neuron*[layer.m_nNeuronsCount];

		for ( int i = 0; i < layer.m_nNeuronsCount; i++ )
		{
			m_aNeurons[i] = layer.m_aNeurons[i];			
		}

		m_aLayerInput = layer.m_aLayerInput;
		m_nNeuronsCount = layer.m_nNeuronsCount;
		m_nInputsCount = layer.m_nInputsCount;

		return * this;
}

BackpropogationError::BackpropogationError ( int nInputCount, int nInputNeuronsCount, int nOutputCount, int * aHiddenLayers, int nHiddenLayerCount )
{	
	m_inputLayer.init(nInputCount, nInputNeuronsCount);	
	LayerPerceptron output_layer;    

    if ( aHiddenLayers && nHiddenLayerCount ) 
	{        
		m_aHiddenLayers = new LayerPerceptron*[nHiddenLayerCount];
        m_nHiddenLayerCount = nHiddenLayerCount;
                
        for ( int i = 0; i < nHiddenLayerCount; i++ )
        {
        	if ( i == 0 )
        	{
        		m_aHiddenLayers[i]->init(nInputNeuronsCount, aHiddenLayers[i]);
        	} else {
        		m_aHiddenLayers[i]->init(aHiddenLayers[i - 1], aHiddenLayers[i]);
        	} 	
        }

        output_layer.init(aHiddenLayers[nHiddenLayerCount - 1], nOutputCount);	
	} else {		
		output_layer.init(nInputNeuronsCount, nOutputCount);
	}

	m_outputLayer = output_layer;
}

BackpropogationError::~BackpropogationError ()
{
	if ( m_nHiddenLayerCount )
	{
		for ( int i = 0; i < m_nHiddenLayerCount; i++ )
	    {
	        delete m_aHiddenLayers[i];
	    }

	    delete [] m_aHiddenLayers;
	}	
}

void BackpropogationError::update ( int nLayerIndex ) 
{    
	if ( nLayerIndex == -1 )
	{        
		for ( int i = 0; i < m_inputLayer.m_nNeuronsCount; i++ )
        {                            
            if ( m_nHiddenLayerCount )
            {
                m_aHiddenLayers[0]->m_aLayerInput[i] = m_inputLayer.m_aNeurons[i]->m_nOutput;
            } else {
                m_outputLayer.m_aLayerInput[i] = m_inputLayer.m_aNeurons[i]->m_nOutput;
            }
        }
	} else {
		for ( int i = 0; i < m_aHiddenLayers[nLayerIndex]->m_nNeuronsCount; i++ )
        {            
            if ( nLayerIndex < m_nHiddenLayerCount -1 )
            {
                m_aHiddenLayers[nLayerIndex + 1]->m_aLayerInput[i] = m_aHiddenLayers[nLayerIndex]->m_aNeurons[i]->m_nOutput;
            } else {
                m_outputLayer.m_aLayerInput[i] = m_aHiddenLayers[nLayerIndex]->m_aNeurons[i]->m_nOutput;
            }
        }
	}
}

void BackpropogationError::recognize ( const double * aInput )
{
	memcpy(m_inputLayer.m_aLayerInput, aInput, m_inputLayer.m_nInputsCount * sizeof(double));    
    m_inputLayer.calculate_outputs();
    update(-1);

    if ( m_aHiddenLayers )
    {
        for( int i = 0; i < m_nHiddenLayerCount; i++ )
        {
            m_aHiddenLayers[i]->calculate_outputs();
            update(i);
        }
    }
    
    m_outputLayer.calculate_outputs();
} 

double BackpropogationError::train ( const double *desiredoutput, const double *input, double alpha, double momentum )
{
	double errorg = 0;
    double errorc;
    double sum = 0,csum = 0;
    double delta, udelta;
    double output;
    
    recognize(input);
    int i,j,k;


    for ( i = 0; i < m_outputLayer.m_nNeuronsCount; i++ )
    {        
        output = m_outputLayer.m_aNeurons[i]->m_nOutput;        
        errorc = (desiredoutput[i] - output) * output * (1 - output);
        errorg += (desiredoutput[i] - output) * (desiredoutput[i] - output);
        
        for ( j = 0; j < m_outputLayer.m_nInputsCount; j++ )
        {            
            delta = m_outputLayer.m_aNeurons[i]->getDeltaValue(j);         
            udelta = alpha * errorc * m_outputLayer.m_aLayerInput[j] + delta * momentum;            
            m_outputLayer.m_aNeurons[i]->m_aWeights[j] += udelta;
            m_outputLayer.m_aNeurons[i]->setDeltaValue(j, udelta);            
            sum += m_outputLayer.m_aNeurons[i]->m_aWeights[j] * errorc;
        }
        
        m_outputLayer.m_aNeurons[i]->m_nWeight += alpha * errorc * m_outputLayer.m_aNeurons[i]->m_nInput;
    }
        
    for ( i = (m_nHiddenLayerCount - 1); i >= 0; i-- )
    {
        for( j = 0; j < m_aHiddenLayers[i]->m_nNeuronsCount; j++ )
        {
            output = m_aHiddenLayers[i]->m_aNeurons[j]->m_nOutput;            
            errorc = output * ( 1 - output) * sum;
            
            for ( k=0; k < m_aHiddenLayers[i]->m_nInputsCount; k++ )
            {
                delta = m_aHiddenLayers[i]->m_aNeurons[j]->getDeltaValue(k);
                udelta = alpha * errorc * m_aHiddenLayers[i]->m_aLayerInput[k] + delta * momentum;
                m_aHiddenLayers[i]->m_aNeurons[j]->m_aWeights[k] += udelta;
                m_aHiddenLayers[i]->m_aNeurons[j]->setDeltaValue(k, udelta);
                csum += m_aHiddenLayers[i]->m_aNeurons[j]->m_aWeights[k] * errorc;
            }

            m_aHiddenLayers[i]->m_aNeurons[j]->m_nWeight += alpha * errorc * m_aHiddenLayers[i]->m_aNeurons[j]->m_nInput;
        }

        sum = csum;
        csum = 0;
    }
    
    for ( i = 0; i < m_inputLayer.m_nNeuronsCount; i++ )
    {
        output = m_inputLayer.m_aNeurons[i]->m_nOutput;
        errorc = output * (1 - output) * sum;

        for ( j = 0; j < m_inputLayer.m_nInputsCount; j++ )
        {
            delta = m_inputLayer.m_aNeurons[i]->getDeltaValue(j);
            udelta = alpha * errorc * m_inputLayer.m_aLayerInput[j] + delta * momentum;            
            m_inputLayer.m_aNeurons[i]->m_aWeights[j] += udelta;
            m_inputLayer.m_aNeurons[i]->setDeltaValue(j, udelta);
        }        

        m_inputLayer.m_aNeurons[i]->m_nWeight += alpha * errorc * m_inputLayer.m_aNeurons[i]->m_nInput;
    }

    return errorg / 2;
}