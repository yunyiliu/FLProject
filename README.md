# FLProject

1.Implement Generated Algorithm to choose next generation
2.Using Federated Learning to calculated two objects(f1: test error, f2: communication round), 
which are input in generated learning

<p>Step 1: GenAlgrithm choose R population by random</p>
<p>Step 2: Do t times communication in Federated Learning</p>
<p>Step 3: Calculate R numbers of f(f1, f2)</p>
<p>Step 4: Choose M numbers of Perato Optimal solution, can keep record their hyperparameters</p>
<p>Step 5: Reapet the previous process till the result we find final M results</p>
<p>Step 6: Choose the best performanced point and using their hyperparameter as our model's hyperparameter.</p>
