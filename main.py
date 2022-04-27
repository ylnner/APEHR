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


global ARGS

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
	"""
	parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .train and .test files) with pickled data organized as patient x admission x codes.')
	parser.add_argument('outFile', metavar='out_file', default='model_output', help='Any file name to store the model.')
	parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=10, help='Training wiil run until reaching the maximum number of epochs without improvement before stopping the training.')
	parser.add_argument('--batchSize', type=int, default=32, help='Batch size.')
	parser.add_argument('--nEpochs', type=int, default=50, help='Number of training iterations.')
	parser.add_argument('--dimensionEmbedding', type = int, default = 960, help = 'Dimension of the embedding in decoder.')
	parser.add_argument('--dff', type = int, default = 120, help = 'Dimension of dff in decoder.')
	parser.add_argument('--numberDecoderLayers', type = int, default = 1, help ='The number of decoder layers in the model.')
	parser.add_argument('--numberProjections', type = int, default = 5, help = 'The number of projections in transformer.')
	parser.add_argument('--headDimension', type = int, default = 15, help = 'Dimension of each head in decoder.')
	"""
	ARGStemp = parser.parse_args()	

	return ARGStemp

def load_json():
	global ARGS

	with open(ARGS.jsonFile) as f:
			params = json.load(f)
			
	ARGS.inputFileRadical = params['inputFileRadical']
	ARGS.outFile = params['outFile']
	ARGS.maxConsecutiveNonImprovements = params['maxConsecutiveNonImprovements']
	ARGS.batchSize = params['batchSize']
	ARGS.nEpochs = params['nEpochs']
	ARGS.dimensionEmbedding = params['dimensionEmbedding']
	ARGS.dff = params['dff']
	ARGS.numberDecoderLayers = params['numberDecoderLayers']
	ARGS.numberProjections = params['numberProjections']
	ARGS.headDimension = params['headDimension']

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
	print("-> " + str(len(main_trainSet)) + " patients at dimension 0 for file: .train dimensions ")	
	print("-> " + str(len(main_testSet)) + " patients at dimension 0 for file: .test dimensions ")

	print("Note: these files carry 3D tensor data; the above numbers refer to dimension 0, dimensions 1 and 2 have irregular sizes.")

	numberOfInputCodes = getNumberOfCodes([main_trainSet,main_testSet])
	print('Number of diagnosis input codes: ' + str(numberOfInputCodes))

	train_sorted_index = sorted(range(len(main_trainSet)), key=lambda x: len(main_trainSet[x]))  #lambda x: len(seq[x]) --> f(x) return len(seq[x])
	main_trainSet = [main_trainSet[i] for i in train_sorted_index]  

	test_sorted_index = sorted(range(len(main_testSet)), key=lambda x: len(main_testSet[x]))
	main_testSet = [main_testSet[i] for i in test_sorted_index]

	return main_trainSet, main_testSet, numberOfInputCodes

def train_step(temp_mm, cce, optimizer, train_loss, inp, target, target_mask):
	
	with tf.GradientTape() as tape:
		tape.watch(inp)
		predictions = temp_mm(inp, True, target_mask)    		
		loss        = cce(target, predictions)
    

	gradients = tape.gradient(loss, temp_mm.trainable_variables)        
	optimizer.apply_gradients(zip(gradients, temp_mm.trainable_variables))
	            
	train_loss(loss)    
	
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

def train_model():
	global ARGS
	print('==> data loading')
	trainSet, testSet, numberOfInputCodes = load_data()
	ARGS.numberOfInputCodes = numberOfInputCodes

	print('==> model building')
	temp_mm, train_loss, cce = build_model()

	print('==> creating optimizer')
	learning_rate = CustomSchedule(ARGS.dimensionEmbedding)
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)

	print('==> training and validation')
	batchSize = ARGS.batchSize
	n_batches = int(np.ceil(float(len(trainSet)) / float(batchSize)))
	EPOCHS    = ARGS.nEpochs

	bestValidationCrossEntropy  = 1e20
	bestValidationEpoch         = 0	
	iImprovementEpochs          = 0
	iConsecutiveNonImprovements = 0

	# Creating folder to save model files
	path = ARGS.outFile
	print('path:> ', path)
	
	
	if os.path.exists(path):
		os.rmdir(path)
	os.mkdir(path)

	for epoch in range(EPOCHS):
		start = time.time()

		predictedY_list             = []
		predictedProbabilities_list = []
		actualY_list                = []

		trainCrossEntropyVector = []
		train_loss.reset_states()
		
		for index in random.sample(range(n_batches), n_batches):
			batchX = trainSet[index*batchSize:(index+1)*batchSize]

			xf, y, mask, nVisitsOfEachPatient_List, maxNumberOfAdmissions = prepareHotVectors(batchX, numberOfInputCodes)
			xf          = tf.convert_to_tensor(xf, dtype=tf.float32)
			y           = tf.convert_to_tensor(y, dtype=tf.float32)
			mask        = tf.convert_to_tensor(mask, dtype=tf.float32)                      
			predicted_y = train_step(temp_mm, cce, optimizer, train_loss, xf, y, mask)

			trainCrossEntropyVector.append(train_loss.result().numpy())

		print('--->>> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch, n_batches, np.mean(trainCrossEntropyVector)))       
  		# Ejecuta validacion 
		n_batches_val, crossEntropyVal1 = performEvaluation(temp_mm, testSet, numberOfInputCodes)
		print('--->>> Epoch: %d, mean cross entropy considering %d VALIDATION batches: %f' % (epoch, n_batches_val, crossEntropyVal1))

		#save_to_plot(epoch, np.mean(trainCrossEntropyVector), crossEntropyVal1, crossEntropyVal2)
		if crossEntropyVal1 < bestValidationCrossEntropy:
			bestValidationCrossEntropy = crossEntropyVal1
			bestValidationEpoch = epoch
			iConsecutiveNonImprovements = 0
			iImprovementEpochs += 1

			file_1 = os.path.join(path, 'pesos.index') 
			file_2 = os.path.join(path, 'pesos.data-00000-of-00001')
			#if os.path.exists('pesos.index') and os.path.exists('pesos.data-00000-of-00001'):
			if os.path.exists(file_1) and os.path.exists(file_2):
				#os.remove('pesos.index')
				#os.remove('pesos.data-00000-of-00001')				
				os.remove(file_1)
				os.remove(file_2)

			new_path = os.path.join(path, 'pesos')
			temp_mm.save_weights(new_path)
		else:
			print('Epoch ended without improvement.')
			iConsecutiveNonImprovements += 1
			print('iConsecutiveNonImprovements: ', iConsecutiveNonImprovements)

		if iConsecutiveNonImprovements > ARGS.maxConsecutiveNonImprovements: #default is 10
			break
	
	#Best results
	print('--------------SUMMARY--------------')
	print('The best VALIDATION cross entropy occurred at epoch %d, the value was of %f ' % (bestValidationEpoch, bestValidationCrossEntropy))	
	print('Number of improvement epochs: ' + str(iImprovementEpochs))
	print('Note: the smaller the cross entropy, the better.')
	print('-----------------------------------')


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
	new_path = os.path.join(path, 'pesos')
	new_temp_mm.load_weights(filepath = new_path)

	return new_temp_mm

def performEvaluation(model, eval_Set, numberOfInputCodes):
  # parametros de evaluacion
  batchSize = ARGS.batchSize#params['batchSize']
  n_batches_val = int(np.ceil(float(len(eval_Set)) / float(batchSize)))
  #print('n_batches_val: ', n_batches_val)
  crossEntropySum = 0.0
  dataCount = 0.0

  # parametros de metricas
  predictedY_list             = []
  predictedProbabilities_list = []
  actualY_list                = []

  # inicializadores de loss  
  eval_loss     = tf.keras.metrics.Mean(name='eval_loss')    
  cce_eval      = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

  eval_loss.reset_states()  

  for index in range(n_batches_val):
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

      
  crossEntropy1 = crossEntropySum / dataCount
  #crossEntropy2 = np.mean(anotherCorssEntropySum)

  
  return n_batches_val, crossEntropy1


def performEvaluationToPrint(model, eval_Set, numberOfInputCodes):
	# parametros de evaluacion
	batchSize = ARGS.batchSize#params['batchSize']  
	n_batches_val = int(np.ceil(float(len(eval_Set)) / float(batchSize)))
	
	crossEntropySum        = 0.0
	
	dataCount              = 0.0

	# parametros de metricas
	predictedY_list             = []
	predictedProbabilities_list = []
	actualY_list                = []

	# inicializadores de loss  
	eval_loss     = tf.keras.metrics.Mean(name='eval_loss')    
	cce_eval      = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

	eval_loss.reset_states()


	for index in range(n_batches_val):
		#print('index: ', index)
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
    
	print('==> computation of prediction results with constant k')
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

	print('Results for Recall@' + str(k_list))
	print(str(finalRecalls[0]))
	print(str(finalRecalls[1]))
	print(str(finalRecalls[2]))

	print('Results for Precision@' + str(k_listForPrecision))
	print(str(finalPrecisions[0]))
	print(str(finalPrecisions[1]))
	print(str(finalPrecisions[2]))

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
	print("Weighted AUC-ROC score: " + str(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
														fullListOfPredictedYProbsForAUCROC_list,
														average = 'weighted')))
 

	#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
	#PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,fullListOfPredictedYForPrecisionRecall_list,average = 'binary')
	PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,
														fullListOfPredictedYForPrecisionRecall_list,
														average = 'binary')

	#print('Precision: ' + str(PRResults[0]))
	#print('Recall: ' + str(PRResults[1]))
	print('Binary F1 Score: ' + str(PRResults[2])) #FBeta score with beta = 1.0
	#print('Support: ' + str(PRResults[3]))

    
	# Calculate cross entropy value  
	crossEntropy1 = crossEntropySum / dataCount
	#crossEntropy2 = np.mean(anotherCorssEntropySum)

	return n_batches_val, crossEntropy1#, crossEntropy2

def test_model():
	global ARGS
	print('==> data loading')
	trainSet, testSet, numberOfInputCodes = load_data()
	ARGS.numberOfInputCodes = numberOfInputCodes

	print('==> model re-building')
	new_temp_mm = rebuild_model()

	print('==> performing evaluation')
	n_batches_test, crossEntropyTest1 = performEvaluationToPrint(new_temp_mm, testSet, numberOfInputCodes)
	print('--->>> Cross entropy considering %d TESTING batches: %f' % (n_batches_test, crossEntropyTest1))

if __name__ == '__main__':	

	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print(e)



	global ARGS
	ARGS = parse_arguments()

	load_json()
	print('ARGS: ', ARGS)
		
	print('==> TRAINING <==')
	train_model()
	
	with tf.device('/cpu:0'):
		print('==> TESTING <==')
		test_model()
