Collin Gros
Zac Holt
Nha Quynh Nguyen
Phillip Sauers


Our code uses the bank dataset. You will need to have the bank.csv
file in the current directory prior to running it.

--- Data Analysis 
To run the code, do
	python3 data_analysis.py [data_path]


--- Classification Models
To run the code, do
	python3 main.py [classifier] -[optional args]

Optional args include
	-dataset [string]
	-neighbors [int]
		number of neighbors to use in KNN.
	-p_num [int]
		P argument for KNN.
	-metric [string]
		metric for KNN.
	-epochs [int]
		epochs for perceptron/AdaBoost
	-eta [float]
		eta for perceptron/Adaboost
	-random_state [int]
		random state
	-criterion [string]
		criterion for DT or RF (such as "gini")
	-max_depth [int]
		max depth for DT, Adaboost, RF
	-kernel [string]
		SVM kernel
	-c_num [float]
		SVM c value
	-gamma [float]
		SVM gamma value
	-n_jobs [int]
		number of bagging jobs 
	-n_est [int]
		estimators for Random Forst

To see all possible arguments, do
	python3 main.py -h

For example, to run Perceptron with default arguments, do
	python3 main.py pcpn