#################################################################################################
# author: alexanderchf1@gmail.com - Alexander Y. Choquenaira Florez
# note: in many places, the code could be shorter, but that would just make it less comprehensible
# comments are not revised, have them with caution
#################################################################################################
import tensorflow as tf
import numpy as np
import math
import random
import time
import os
import argparse
import pickle
from sklearn import metrics
from MyEmbedding import MyEmbedding
from MyMultiHeadAttention import MyMultiHeadAttention
from MyDecoderLayer import MyDecoderLayer
from MyModel import MyModel
import json
from os.path import isfile, join


global ARGS

ARGS = argparse.ArgumentParser

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()    
		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps
  
	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)    
		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('jsonFile', type =str, help = 'JSON file with parameters.')
	ARGStemp = parser.parse_args()	

	return ARGStemp

def load_json(jsonFile, path_input, path_model):	
	global ARGS

	#with open(ARGS.jsonFile) as f:
			#params = json.load(f)

	with open(jsonFile) as f:
		params = json.load(f)

	

	
	ARGS.inputFileRadical = path_input + params['inputFileRadical']
	ARGS.outFile = path_model #+ '/' + params['outFile']
	ARGS.maxConsecutiveNonImprovements = params['maxConsecutiveNonImprovements']
	ARGS.batchSize = params['batchSize']
	ARGS.nEpochs = params['nEpochs']
	ARGS.dimensionEmbedding = params['dimensionEmbedding']
	ARGS.dff = params['dff']
	ARGS.numberDecoderLayers = params['numberDecoderLayers']
	ARGS.numberProjections = params['numberProjections']
	ARGS.headDimension = params['headDimension']


	#f.write('JSON_param: ' + JSON_param)
	f = ARGS.f	
	f.write('inputFileRadical: ' + params['inputFileRadical'] + '\n')
	f.write('ARGS.inputFileRadical: '+ ARGS.inputFileRadical+ '\n')
	f.write('ARGS.outFile: '+ ARGS.outFile+ '\n')
	f.write('ARGS.maxConsecutiveNonImprovements: '+ str(ARGS.maxConsecutiveNonImprovements)+ '\n')
	f.write('ARGS.batchSize: '+str(ARGS.batchSize)+ '\n')
	f.write('ARGS.nEpochs: '+str(ARGS.nEpochs)+ '\n')
	f.write('ARGS.dimensionEmbedding: '+ str(ARGS.dimensionEmbedding)+ '\n')
	f.write('ARGS.dff: '+ str(ARGS.dff)+ '\n')
	f.write('ARGS.numberDecoderLayers: '+ str(ARGS.numberDecoderLayers)+ '\n')
	f.write('ARGS.numberProjections: '+ str(ARGS.numberProjections)+ '\n')
	f.write('ARGS.headDimension: '+ str(ARGS.headDimension)+ '\n')

def prepareHotVectors(train_tensor, numberOfInputCodes):
	nVisitsOfEachPatient_List = np.array([len(seq) for seq in train_tensor]) - 1
	numberOfPatients = len(train_tensor)
	maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

	#print('maxNumberOfAdmissions1: ', maxNumberOfAdmissions1)
	x_hotvectors_tensorf = np.zeros((maxNumberOfAdmissions, numberOfPatients, numberOfInputCodes))
	y_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, numberOfInputCodes))
	mask = np.zeros((maxNumberOfAdmissions, numberOfPatients))

	for idx, train_patient_matrix in enumerate(train_tensor):
		for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last admission, which is not part of the training
			for code in visit_line:
				x_hotvectors_tensorf[i_th_visit, idx, code] = 1
		
		for i_th_visit, visit_line in enumerate(train_patient_matrix[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
			for code in visit_line:
				y_hotvectors_tensor[i_th_visit, idx, code] = 1
		
		mask[:nVisitsOfEachPatient_List[idx], idx] = 1.    

	nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List)    
	  
	return x_hotvectors_tensorf, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List, maxNumberOfAdmissions


def getNumberOfCodes(sets):
	highestCode = 0
	for set in sets:
		for pat in set:
			for adm in pat:
				for code in adm:
					if code > highestCode:
						highestCode = code
	return (highestCode + 1)

def load_data():	
	main_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
	main_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))
	
	f = ARGS.f	
	
	f.write("-> " + str(len(main_trainSet)) + " patients at dimension 0 for file: .train dimensions "+ '\n')	
	f.write("-> " + str(len(main_testSet)) + " patients at dimension 0 for file: .test dimensions "+ '\n')

	f.write("Note: these files carry 3D tensor data; the above numbers refer to dimension 0, dimensions 1 and 2 have irregular sizes."+ '\n')

	numberOfInputCodes = getNumberOfCodes([main_trainSet,main_testSet])
	f.write('Number of diagnosis input codes: ' + str(numberOfInputCodes)+ '\n')

	train_sorted_index = sorted(range(len(main_trainSet)), key=lambda x: len(main_trainSet[x]))  #lambda x: len(seq[x]) --> f(x) return len(seq[x])
	main_trainSet = [main_trainSet[i] for i in train_sorted_index]  

	test_sorted_index = sorted(range(len(main_testSet)), key=lambda x: len(main_testSet[x]))
	main_testSet = [main_testSet[i] for i in test_sorted_index]

	return main_trainSet, main_testSet, numberOfInputCodes

def train_step(temp_mm, cce, optimizer, train_loss, inp, target, target_mask):
	# es necesario definir mask con el input actual
	with tf.GradientTape() as tape:
		tape.watch(inp)
		predictions = temp_mm(inp, True, target_mask)    		
		loss        = cce(target, predictions)
    

	gradients = tape.gradient(loss, temp_mm.trainable_variables)        
	optimizer.apply_gradients(zip(gradients, temp_mm.trainable_variables))
	            
	train_loss(loss)    
	#train_accuracy(target, predictions)
	return predictions

def build_model():
	global ARGS
	# Defining model
	n_codes  = ARGS.numberOfInputCodes#846#270#my_transformer['n_codes']#846#847

	d_emb    = ARGS.dimensionEmbedding#960#480#960#my_transformer['d_emb']#120#240#512#200#424
	d_model  = ARGS.dimensionEmbedding#960#480#960#my_transformer['d_model']#120#240#512#200#424
	dff      = ARGS.dff#120#60#120#480#my_transformer['dff']#240#480#1200#400#1692

	n_layers = ARGS.numberDecoderLayers#1#my_transformer['n_layers']#6#2#1#6
	n_proj   = ARGS.numberProjections#5#my_transformer['n_proj']#8
	d_transf = ARGS.headDimension#15#my_transformer['d_transf']#15#64


	temp_mm = MyModel(n_codes = n_codes, d_emb = d_emb, n_layers = n_layers, n_proj = n_proj, d_model = d_model, d_transf = d_transf, dff = dff)
	
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	cce        = tf.keras.losses.CategoricalCrossentropy(from_logits=False)#, reduction='none')	

	return temp_mm, train_loss, cce

def rebuild_model():
	global ARGS
	# Defining model
	n_codes  = ARGS.numberOfInputCodes#846#270#my_transformer['n_codes']#846#847

	d_emb    = ARGS.dimensionEmbedding#960#480#960#my_transformer['d_emb']#120#240#512#200#424
	d_model  = ARGS.dimensionEmbedding#960#480#960#my_transformer['d_model']#120#240#512#200#424
	dff      = ARGS.dff#120#60#120#480#my_transformer['dff']#240#480#1200#400#1692

	n_layers = ARGS.numberDecoderLayers#1#my_transformer['n_layers']#6#2#1#6
	n_proj   = ARGS.numberProjections#5#my_transformer['n_proj']#8
	d_transf = ARGS.headDimension#15#my_transformer['d_transf']#15#64

	new_temp_mm = MyModel(n_codes = n_codes, d_emb = d_emb, n_layers = n_layers, n_proj = n_proj, d_model = d_model, d_transf = d_transf, dff = dff)

	path     = ARGS.outFile
	f = ARGS.f	
	f.write('path: ' + path+ '\n')
	new_path = os.path.join(path, 'pesos')
	new_temp_mm.load_weights(filepath = new_path)

	return new_temp_mm

def performEvaluation(model, eval_Set, numberOfInputCodes):
  # parametros de evaluacion
  batchSize = ARGS.batchSize#params['batchSize']
  n_batches_val = int(np.ceil(float(len(eval_Set)) / float(batchSize)))
  #print('n_batches_val: ', n_batches_val)
  crossEntropySum = 0.0
  #anotherCorssEntropySum = [] # Estoy guardando en un numpy para comprobar la forma en la que calcula el cross entropy
  dataCount = 0.0

  # parametros de metricas
  predictedY_list             = []
  predictedProbabilities_list = []
  actualY_list                = []

  # inicializadores de loss  
  eval_loss     = tf.keras.metrics.Mean(name='eval_loss')    
  cce_eval      = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

  eval_loss.reset_states()  

  #idx_b = 0
  #for index in xrange(n_batches):
  for index in range(n_batches_val):
    batchX = eval_Set[index * batchSize:(index + 1) * batchSize]
    #batchY = eval_Set[index * batchSize:(index + 1) * batchSize]
    		
    xf_val, y_val, mask_val, nVisitsOfEachPatient_List, maxNumberOfAdmissions = prepareHotVectors(batchX, numberOfInputCodes)
    xf_val   = tf.convert_to_tensor(xf_val, dtype=tf.float32)
    y_val    = tf.convert_to_tensor(y_val, dtype=tf.float32)
    mask_val = tf.convert_to_tensor(mask_val, dtype=tf.float32)

    #print('VALy: ', y_val.shape)    
    predicted_y_val = model(xf_val, False, mask_val)
    #print('VALpredicted_y: ', predicted_y_val.shape)
    loss        = cce_eval(y_val, predicted_y_val)

    eval_loss(loss)    
    
    crossEntropy = eval_loss.result().numpy() #TEST_MODEL_COMPILED(xf, xb, y, mask, nVisitsOfEachPatient_List)
    #anotherCorssEntropySum.append(crossEntropy)

    #accumulation by simple summation taking the batch size into account
    crossEntropySum += crossEntropy * len(batchX)
    dataCount += float(len(batchX))  
    #At the end, it returns the mean cross entropy considering all the batches

      
  crossEntropy1 = crossEntropySum / dataCount
  #crossEntropy2 = np.mean(anotherCorssEntropySum)

  
  return n_batches_val, crossEntropy1


def performEvaluationToPrint(model, eval_Set, numberOfInputCodes):
	# parametros de evaluacion
	batchSize = ARGS.batchSize#params['batchSize']  
	n_batches_val = int(np.ceil(float(len(eval_Set)) / float(batchSize)))
	####print('n_batches_val: ', n_batches_val)
	crossEntropySum        = 0.0
	#anotherCorssEntropySum = [] # Estoy guardando en un numpy para comprobar la forma en la que calcula el cross entropy
	dataCount              = 0.0

	# parametros de metricas
	predictedY_list             = []
	predictedProbabilities_list = []
	actualY_list                = []

	# inicializadores de loss  
	eval_loss     = tf.keras.metrics.Mean(name='eval_loss')    
	cce_eval      = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

	eval_loss.reset_states()
#	global ARGS
	f = ARGS.f	

	for index in range(n_batches_val):
		print('index: ', index)
		batchX = eval_Set[index * batchSize:(index + 1) * batchSize]
		#batchY = eval_Set[index * batchSize:(index + 1) * batchSize]
				
		xf_val, y_val, mask_val, nVisitsOfEachPatient_List, maxNumberOfAdmissions = prepareHotVectors(batchX, numberOfInputCodes)
		xf_val   = tf.convert_to_tensor(xf_val, dtype=tf.float32)
		y_val    = tf.convert_to_tensor(y_val, dtype=tf.float32)
		mask_val = tf.convert_to_tensor(mask_val, dtype=tf.float32)
		
		predicted_y_val = model(xf_val, False, mask_val)		
		loss        = cce_eval(y_val, predicted_y_val)

		eval_loss(loss)    

		crossEntropy = eval_loss.result().numpy() #TEST_MODEL_COMPILED(xf, xb, y, mask, nVisitsOfEachPatient_List)
		#anotherCorssEntropySum.append(crossEntropy)

		#accumulation by simple summation taking the batch size into account
		crossEntropySum += crossEntropy * len(batchX)
		dataCount += float(len(batchX))  
		#At the end, it returns the mean cross entropy considering all the batches

		for ith_patient in range(predicted_y_val.shape[1]):
			predictedPatientSlice = predicted_y_val[:, ith_patient, :]	# obtiene todas las admisiones de un paciente
			#retrieve actual y from batch tensor -> actual codes, not the hotvector
			actual_y = batchX[ith_patient][1:]	# obtiene las verdaderas y
			#for each admission of the ith-patient			
			for ith_admission in range(nVisitsOfEachPatient_List[ith_patient]):
				#convert array of actual answers to list
				actualY_list.append(actual_y[ith_admission])
				#retrieves ith-admission of ths ith-patient
				ithPrediction = predictedPatientSlice[ith_admission]
				#since ithPrediction is a vector of probabilties with the same dimensionality of the hotvectors
				#enumerate is enough to retrieve the original codes
				enumeratedPrediction = [codeProbability_pair for codeProbability_pair in enumerate(ithPrediction)]
				#sort everything
				sortedPredictionsAll = sorted(enumeratedPrediction, key=lambda x: x[1],reverse=True)
				#creates trimmed list up to max(maxNumberOfAdmissions,30) elements
				sortedTopPredictions = sortedPredictionsAll[0:max(maxNumberOfAdmissions,30)]
				#here we simply toss off the probability and keep only the sorted codes
				sortedTopPredictions_indexes = [codeProbability_pair[0] for codeProbability_pair in sortedTopPredictions]
				#stores results in a list of lists - after processing all batches, predictedY_list stores all the prediction results
				predictedY_list.append(sortedTopPredictions_indexes)
				predictedProbabilities_list.append(sortedPredictionsAll)
    
	f.write('==> computation of prediction results with constant k'+ '\n')
	recall_sum = [0.0, 0.0, 0.0]

	k_list = [10,20,30]

	for ith_admission in range(len(predictedY_list)):
		ithActualYSet = set(actualY_list[ith_admission])
		for ithK, k in enumerate(k_list):
			ithPredictedY = set(predictedY_list[ith_admission][:k])
			intersection_set = ithActualYSet.intersection(ithPredictedY)
			recall_sum[ithK] += len(intersection_set) / float(len(ithActualYSet)) # this is recall because the numerator is len(ithActualYSet)

	precision_sum = [0.0, 0.0, 0.0]
	k_listForPrecision = [1,2,3]
	for ith_admission in range(len(predictedY_list)):
		ithActualYSet = set(actualY_list[ith_admission])
		for ithK, k in enumerate(k_listForPrecision):
			ithPredictedY = set(predictedY_list[ith_admission][:k])
			intersection_set = ithActualYSet.intersection(ithPredictedY)
			precision_sum[ithK] += len(intersection_set) / float(k) # this is precision because the numerator is k \in [10,20,30]

	finalRecalls = []
	finalPrecisions = []
	for ithK, k in enumerate(k_list):
		finalRecalls.append(recall_sum[ithK] / float(len(predictedY_list)))
		finalPrecisions.append(precision_sum[ithK] / float(len(predictedY_list)))

	f.write('Results for Recall@' + str(k_list)+ '\n')
	f.write(str(finalRecalls[0])+ '\n')
	f.write(str(finalRecalls[1])+ '\n')
	f.write(str(finalRecalls[2])+ '\n')

	f.write('Results for Precision@' + str(k_listForPrecision))
	f.write(str(finalPrecisions[0])+ '\n')
	f.write(str(finalPrecisions[1])+ '\n')
	f.write(str(finalPrecisions[2])+ '\n')

	#### AUC-ROC
	fullListOfTrueYOutcomeForAUCROCAndPR_list = []
	fullListOfPredictedYProbsForAUCROC_list = []
	fullListOfPredictedYForPrecisionRecall_list = []
	for ith_admission in range(len(predictedY_list)):
		ithActualY = actualY_list[ith_admission]
		nActualCodes = len(ithActualY)
		ithPredictedProbabilities = predictedProbabilities_list[ith_admission]#[0:nActualCodes]
		ithPrediction = 0
		for predicted_code, predicted_prob in ithPredictedProbabilities:
			fullListOfPredictedYProbsForAUCROC_list.append(predicted_prob)
			#for precision-recall purposes, the nActual first codes correspond to what was estimated as correct answers
			if ithPrediction < nActualCodes:
				fullListOfPredictedYForPrecisionRecall_list.append(1)
			else:
				fullListOfPredictedYForPrecisionRecall_list.append(0)

			#the list fullListOfTrueYOutcomeForAUCROCAndPR_list corresponds to the true answer, either positive or negative
			#it is used for both Precision Recall and for AUCROC
			if predicted_code in ithActualY:
				fullListOfTrueYOutcomeForAUCROCAndPR_list.append(1)
				#file.write("1 " + str(predicted_prob) + '\n')
			else:
				fullListOfTrueYOutcomeForAUCROCAndPR_list.append(0)
				#file.write("0 " + str(predicted_prob) + '\n')
			ithPrediction += 1
	#file.close()
  
	#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
	#print("Weighted AUC-ROC score: " + str(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,fullListOfPredictedYProbsForAUCROC_list,average = 'weighted')))
	f.write("Weighted AUC-ROC score: " + str(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
														fullListOfPredictedYProbsForAUCROC_list,
														average = 'weighted')) + '\n')
 

	#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
	#PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,fullListOfPredictedYForPrecisionRecall_list,average = 'binary')
	PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,
														fullListOfPredictedYForPrecisionRecall_list,
														average = 'binary'+ '\n')

	#print('Precision: ' + str(PRResults[0]))
	#print('Recall: ' + str(PRResults[1]))
	f.write('Binary F1 Score: ' + str(PRResults[2])+ '\n') #FBeta score with beta = 1.0
	#print('Support: ' + str(PRResults[3]))

    
	# Calculate cross entropy value  
	crossEntropy1 = crossEntropySum / dataCount
	#crossEntropy2 = np.mean(anotherCorssEntropySum)

	return n_batches_val, crossEntropy1#, crossEntropy2

def test_model():
	global ARGS

	f = ARGS.f	
	f.write('==> data loading'+ '\n')
	trainSet, testSet, numberOfInputCodes = load_data()
	ARGS.numberOfInputCodes = numberOfInputCodes

	f.write('==> model re-building'+ '\n')
	new_temp_mm = rebuild_model()

	f.write('==> performing evaluation'+ '\n')
	n_batches_test, crossEntropyTest1 = performEvaluationToPrint(new_temp_mm, testSet, numberOfInputCodes)
	f.write('--->>> Cross entropy considering %d TESTING batches: %f' % (n_batches_test, crossEntropyTest1)+ '\n')


def main_test(JSON_param, path_input, path_model):
	global ARGS
	
	out_file = path_model + '/' + 'out.txt'
	ARGS.out_file = out_file

	print('out_file: ', out_file)
	f = open(out_file, "w")
	ARGS.f = f
	#f.write('JSON_param: ' + JSON_param)
	#f.write('path_model? ' + path_model)

	
	load_json(JSON_param, path_input, path_model)
	
	f.write('==> TESTING <=='+ '\n')
	test_model()
	f.close()