# L1_adversarial_examples_attack
The L1 adversarial examples attack base on EAD and C&amp;W.
This project contains the following L1 attack methods:
1. EAD EN rule. Natural FISTA method.
2. EAD L1 rule. Natural FISTA method.
3. EN rule but using COV method.
4. L1 rule but using COV method.
5. Only L1. The object fuction is composed of classification error and L1 distance.

1 and 2 are proposed by EAD.
3 and 4 are extended from EAD using COV method. 
The reason why beta is insensitive is that the beta is too small compared with 1, the coefficent of L2 distance.
Increasing the beta can be work. 
5 can be work too.
