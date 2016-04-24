function [feature_1,split_1] = feature_select(x_train,y_train,c_train,index)
feature_1 = zeros(1,c_train);
split_1 = zeros(1,c_train);

 for i=1:c_train
     if i == index
         feature_1(1,i) = 0;
         split_1(1,i) = 0;
     else
        unique_values = unique(x_train(:,i));
        size_val = size(unique_values,1);
        if size_val >1
            if size_val == 2
                [split_1(1,i),feature_1(1,i)] = classify_2(x_train(:,i),y_train,unique_values);
            else
                [split_1(1,i),feature_1(1,i)] = classify_multi(x_train(:,i),y_train,unique_values,size_val);
            end
        end
    end
 end
