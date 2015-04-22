classdef SPRINT_Race < handle
% SPRINT_Race class

% COPYRIGHT
%  (C) 2015 Tiantian Zhang, Machine Learning Lab, University of Central Florida
% zhangtt@knights.ucf.edu
% Reference
% T. Zhang, M. Georgiopoulos, G. C. Anagnostopoulos, "SPRINT Multi-Objective Model Racing", GECCO 2015
    
    %% Public Properties
    properties
        M               % number of initial models
        D               % number of objectives
        toDel           % index of the models have been removed so far
        alpha           % alpha values used in each dual-SPRT of SPRINT-Race
        Gamma           % the overall error probability of SPRINT-Race
        delta           % the parameter of indifference zone
        models          % the final set of non-dominated models returned by SPRINT-Race
    end

    %% Protected Properties
    properties(GetAccess = protected, SetAccess = protected)
        stop            % the termination matrix, stop(i,j,1) indicates if the SPRT_1
                        % between the i-th model and j-th model is active
                        % or not; stop(i,j,2) indicates if the SPRT_2 is
                        % active or not. (0 - active; 1 - terminated)
        dom             % the dominance matrix, dom(i,j) is the number that the i-th
                        % model dominates the j-th model
        t               % the step index of SPRINT-Race
        data            % store the performance vectors needed at the current step 
    end
    
    %% Public Methods
    methods
        
        % Constructor
        function obj = SPRINT_Race(M, D, Gamma, delta)
            if nargin == 4
                obj.M = M;
                obj.D = D;
                obj.Gamma = Gamma;
                obj.delta = delta;
                obj.models = 1:obj.M;
                obj.toDel = [];
                obj.alpha = Gamma / nchoosek(M,2) / 2;
                obj.stop = zeros(M, M, 2);
                obj.stop(:,:,1) = eye(M);
                obj.stop(:,:,2) = eye(M);
                obj.dom = zeros(M, M);
                obj.t = 1;
                obj.data = cell(1,M);
            else
                error('Too few/many arguments');
            end
        end % SPRINT_Race
        
        function dataAcquisition(obj)
            % obtain the performance vectors needed for the current step
            % this function is user-specified
            % in this illustration, we simply generate performance vectors
            % from a set of predefined multivariate Gaussian distributions
            obj.data = cell(1,obj.M);
            load('gaussians.mat'); % load the matrix containing the means and variances of M 
            for i = 1:obj.M
                for j = i + 1:obj.M
                    if(min(obj.stop(i,j,:)) == 0) % meaning that performance vectors of the i-th and j-th model are needed
                        if isempty(obj.data{i})
                            obj.data{i} = mvnrnd(gaussians{i,1},gaussians{i,2},1);
                        end
                        if isempty(obj.data{j})
                            obj.data{j} = mvnrnd(gaussians{j,1},gaussians{j,2},1);
                        end                        
                    end
                end
            end            
        end % dataAcquisition
        
        
        % SPRINT-Race
        function Racing(obj)
            while(min(min(min(obj.stop)) == 0)) % at least one test is active
                dataAcquisition(obj);
                for i = 1:obj.M
                    for j = i + 1:obj.M
                        if(min(obj.stop(i,j,:)) == 0) % start the test
                            % data acquisition
                            if obj.t == 1
                                [w1, w2] = obj.dominates(obj.data{i}, obj.data{j});
                                obj.dom(i,j) = w1;
                                obj.dom(j,i) = w2;
                            else
                                [w1, w2] = obj.dominates(obj.data{i}, obj.data{j});
                                obj.dom(i,j) = obj.dom(i,j) + w1;
                                obj.dom(j,i) = obj.dom(j,i) + w2;
                            end
                            w1 = obj.dom(i,j);
                            w2 = obj.dom(j,i);
                            % compute boundary values
                            A = (1 - obj.alpha) / obj.alpha; % assume alpha = beta
                            B = 1 / A;
                            b1 = log(B)/log((1 + 2 * obj.delta)/(1 - 2 * obj.delta)) + (w1 + w2) * log(1 + 2 * obj.delta)/log((1 + 2 * obj.delta)/(1 - 2 * obj.delta));
                            b2 = log(A)/log((1 + 2 * obj.delta)/(1 - 2 * obj.delta)) + (w1 + w2) * log(1 + 2 * obj.delta)/log((1 + 2 * obj.delta)/(1 - 2 * obj.delta));
                            b3 = log(B)/log((1 + 2 * obj.delta)/(1 - 2 * obj.delta)) + (w1 + w2) * log(1 / (1 - 2 * obj.delta))/log((1 + 2 * obj.delta)/(1 - 2 * obj.delta));
                            b4 = log(A)/log((1 + 2 * obj.delta)/(1 - 2 * obj.delta)) + (w1 + w2) * log(1 / (1 - 2 * obj.delta))/log((1 + 2 * obj.delta)/(1 - 2 * obj.delta));

                            if w1 <= b1 || w1 >= b4 % a dominance relation is detected
                                obj.stop(i,j,1) = 1;
                                obj.stop(j,i,1) = 1;
                                obj.stop(i,j,2) = 1;
                                obj.stop(j,i,2) = 1;
                                if w1 <= b1 % i is removed from racing
                                    obj.toDel = [obj.toDel,i]; 
                                    obj.stop(i,:,1) = ones(1,obj.M); % stop all the tests involving model i
                                    obj.stop(i,:,2) = ones(1,obj.M); 
                                    obj.stop(:,i,1) = ones(obj.M,1);
                                    obj.stop(:,i,2) = ones(obj.M,1);
                                elseif w1 >= b4 % j is removed from racing
                                    obj.toDel = [obj.toDel,j];
                                    obj.stop(j,:,1) = ones(1,obj.M); % stop all the tests involving model j
                                    obj.stop(j,:,2) = ones(1,obj.M); 
                                    obj.stop(:,j,1) = ones(obj.M,1);
                                    obj.stop(:,j,2) = ones(obj.M,1);
                                end 
                            end
                            if w1 >= b2 % SPRT_1 is stopped
                                obj.stop(i,j,1) = 1;
                                obj.stop(j,i,1) = 1;
                            end
                            if w1 <= b3 % SPRT_2 is stopped
                                obj.stop(i,j,2) = 1;
                                obj.stop(j,i,2) = 1;
                            end
                        end
                    end
                end
                obj.t = obj.t + 1;                
            end
            % find out the models that are returned by sequential racing
            obj.toDel = unique(obj.toDel);
            obj.models(obj.toDel) = [];    
        end % Racing
                
    end
    
    methods (Access = private)
        % find the number of dominance 
        function [w1, w2] = dominates(~,data1, data2)
            w1 = 0;
            w2 = 0;
            tmp = data1 - data2;
            % for minimization problem
            if min(tmp) >=0 && max(tmp) > 0
                w2 = w2 + 1;
            elseif max(tmp) <=0 && min(tmp) < 0
                w1 = w1 + 1;
            end
        end % dominates        
       
    end
    
end