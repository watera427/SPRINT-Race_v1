clear;
clc;
% Preparation work
% add the class in default path
STARTUP % remember to change the path name in the .m file to where you put the SPRINT_Race class

M = 5; % the number of initial models
D = 2;  % the number of objectives
Gamma = 0.1; % the overall error probability of SPRINT-Race
delta = 0.1; % the parameter of indifference zone

% generate M multi-variate Gaussian distributions for sampling
DistributionGeneration(M, D);

% Initial a SPRINT_Race object
obj = SPRINT_Race(M,D,Gamma,delta);
% start racing
Racing(obj);
returned = obj.models; % indices of the final set of non-dominated models