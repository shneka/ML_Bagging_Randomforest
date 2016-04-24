function myRForest2(filename,M,k)

[num_old,~]=importdata(filename);
[row,column] = size(num_old);

% for k- fold cross validation

for m=1:size(M,2)
    
    row_value = int64(row)/k;
    test_row = row - (k-1)*row_value;

    test_set = zeros(test_row,column);
    error= zeros(2,k);
    for i=1:k
        j = 1;
        val = 0;
        flag = 1;
        num = num_old(randperm(size(num_old,1)),:);
        while j<row
            if j == i || int64(j/row_value)+1 == i
                test_set = num(j:j+test_row,:);
                j= val+test_row;
                val = j;
            else
                if flag == 1;
                    training_set = num(j:j+row_value,:);
                    j = val+row_value;
                    val = j;
                    flag = 0;
                else
                    training_set = vertcat(training_set,num(j:j+row_value,:));
                    j = val+row_value;
                    val = j;
                end
            end
        end
        [error(1,i), error(2,i)]= random_forest(training_set,test_set,column,M(m));
        fprintf('Train Error for fold %d with 100 base class: %f\n ',i,error(1,i));
        fprintf('Test Error for fold %d with 100 base class: %f\n ',i,error(2,i));
    end
fprintf('\n');
fprintf('Mean Train error 10 fold 100 base class is %f\n',mean(error(1,:))); 
fprintf('Std Train error 10 fold 100 base class is %f\n',std(error(1,:)));
fprintf('\n');
fprintf('Mean Test error 10 fold 100 base class is %f\n',mean(error(2,:)));
fprintf('Std Test error 10 fold 100 base class is %f\n',std(error(2,:)));
fprintf('__________________________________________________________\n');
end
  