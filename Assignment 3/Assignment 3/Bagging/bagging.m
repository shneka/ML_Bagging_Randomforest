function [error_train_ans,error_test_ans] = bagging(training_set,test_set,column,M)

[r_train,c_train] = size(training_set);
[r_test,~] = size(test_set);

% Seperating the label in training set and test set
x_old = training_set(:,1:column-1);
y_old = training_set(:,column);
x_test = test_set(:,1:column-1);
y_test = test_set(:,column);

final_feature = zeros(M,3);
final_split = zeros(M,3);
final_decision = zeros(M,4);

for no = 1:M
    %To create random sample with repetition
    sample = randsample(r_train,r_train,'true');
    flag = 1;
    for val = 1:r_train
        if(flag==1)
            x_train = x_old(sample(val),:);
            y_train = y_old(sample(val),:);
            flag = 0;
        else
            x_train = vertcat(x_train,x_old(sample(val),:));
            y_train = vertcat(y_train,y_old(sample(val),:));
        end
    end
    
    for feat_count=1:2
        if feat_count == 1    
%             feature_1 = zeros(1,c_train);
%             split_1 = zeros(1,c_train);
        
            [feature_1,split_1] = feature_select(x_train,y_train,c_train-1,final_feature(no,1));
            [~,ind_1] = max(feature_1(1,:));
            final_feature(no,1) = ind_1;
            final_split(no,1) = split_1(ind_1);
        else
%             feature_1 = zeros(1,c_train);
%             split_1 = zeros(1,c_train);
            flag_1 = 1;
            flag_2 = 1;
            p_1 = 0;
            p_2 = 0;
            n_1 = 0;
            n_2 = 0;
            
            for j=1:r_train 
                if(x_train(j,final_feature(no,1))>= final_split(no,1))
                    if(flag_1 == 1)
                        x_train_1 = x_train(j,:);
                        y_train_1 = y_train(j,:);
                        flag_1 = 0;
                    else
                        x_train_1 = vertcat(x_train_1,x_train(j,:));
                        y_train_1 = vertcat(y_train_1,y_train(j,:));
                    end
                else
                    if(flag_2==1)
                        x_train_2 = x_train(j,:);
                        y_train_2 = y_train(j,:);
                        flag_2 = 0;
                    else
                        x_train_2 = vertcat(x_train_2,x_train(j,:));
                        y_train_2 = vertcat(y_train_2,y_train(j,:));
                    end
                end
            end
                
            [feature_1,split_1] = feature_select(x_train_1,y_train_1,c_train-1,final_feature(no,1));
            [~,ind_1] = max(feature_1(1,:));
            final_feature(no,2) = ind_1;
            final_split(no,2) = split_1(ind_1);          
            
            [feature_1,split_1] = feature_select(x_train_2,y_train_2,c_train-1,final_feature(no,1));
            [~,ind_1] = max(feature_1(1,:));
            final_feature(no,3) = ind_1;
            final_split(no,3) = split_1(ind_1);
            % To get the solution for the second layer of the decision tree
            fin_pos_1 = 0;
            fin_neg_1 = 0;
            fin_pos_2 = 0;
            fin_neg_2 = 0;
            for s_1=1:size(x_train_1,1)
                if(x_train_1(s_1,final_feature(no,2))>=final_split(no,2))
                    if(y_train_1(s_1,1)==1)
                        fin_pos_1 = fin_pos_1+1;
                    else
                        fin_neg_1 = fin_neg_1+1;
                    end
                else
                    if(y_train_1(s_1,1)==1)
                        fin_pos_2 = fin_pos_2+1;
                    else
                        fin_neg_2 = fin_neg_2+1;
                    end
                end
            end
            if fin_pos_1>fin_neg_1
                final_decision(no,1) = +1;
            else
                final_decision(no,1) = -1;
            end
            if fin_pos_2>fin_neg_2
                final_decision(no,2) = +1;
            else
                final_decision(no,2) = -1;
            end
         % To get the solution for the right side second layer
            fin_pos_3 = 0;
            fin_neg_3 = 0;
            fin_pos_4 = 0;
            fin_neg_4 = 0;
            for s_2=1:size(x_train_2,1)
                if(x_train_2(s_2,final_feature(no,3))>=final_split(no,3))
                    if(y_train_2(s_2,1)==1)
                        fin_pos_3 = fin_pos_3+1;
                    else
                        fin_neg_3 = fin_neg_3+1;
                    end
                else
                    if(y_train_2(s_2,1)==1)
                        fin_pos_4 = fin_pos_4+1;
                    else
                        fin_neg_4 = fin_neg_4+1;
                    end
                end
            end
            if fin_pos_3>fin_neg_3
                final_decision(no,3) = +1;
            else
                final_decision(no,3) = -1;
            end
            if fin_pos_4>fin_neg_4
                final_decision(no,4) = +1;
            else
                final_decision(no,4) = -1;
            end
        end
    end
end

%to find the error in the training set
error_train = 0;
 for t_1 = 1:r_train
     sum = 0;
     for e_1 = 1:M
        if(x_train(t_1,final_feature(e_1,1))>= final_split(e_1,1))
            if (x_train(t_1,final_feature(e_1,2))>= final_split(e_1,2))
                sum = sum+final_decision(e_1,1);
            else
                sum = sum+final_decision(e_1,2);
            end
        else
            if (x_train(t_1,final_feature(e_1,3))>= final_split(e_1,3))
                sum = sum+final_decision(e_1,3);
            else
                sum = sum+final_decision(e_1,4);
            end
        end
     end
     if sum>0
         decision = +1;
     else
         decision = -1;
      end
      if decision ~= y_train(t_1,1)
          error_train= error_train+1;
      end
  end
 error_train_ans =  error_train/r_train;

% to find training set errors 
 error_test = 0;
 for t_2 = 1:r_test
     sum = 0;
     for e_2 = 1:M
        if(x_test(t_2,final_feature(e_2,1))>= final_split(e_2,1))
            if (x_test(t_2,final_feature(e_2,2))>= final_split(e_2,2))
                sum = sum+final_decision(e_2,1);
            else
                sum = sum+final_decision(e_2,2);
            end
        else
            if (x_test(t_2,final_feature(e_2,3))>= final_split(e_2,3))
                sum = sum+final_decision(e_2,3);
            else
                sum = sum+final_decision(e_2,4);
            end
        end
     end
     if sum>0
         decision = +1;
     else
         decision = -1;
     end
     if decision ~= y_test(t_2,1)
         error_test= error_test+1;
     end
 end   
 error_test_ans =  error_test/r_test;



    

