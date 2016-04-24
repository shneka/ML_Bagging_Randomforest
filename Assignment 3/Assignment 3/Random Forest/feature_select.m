function [feature_1,split_1] = feature_select(x_train,y_train,c_train,index,N)
feature_1 = zeros(1,N);
split_1 = zeros(1,N);
sample = zeros(1,N);

if N <10
for j=1:N
    sample_set = randi([1,c_train],1,1);
    x_new = x_train(:,sample_set);
    unique_values = unique(x_new);
    
    while sample_set == index || size(unique_values,1) ==1 || ismember(sample_set,sample)==1
        sample_set = randi([1,c_train],1,1);
        x_new = x_train(:,sample_set);
        unique_values = unique(x_new);
    end
    sample(1,j) = sample_set;
    if size((unique_values),1) == 2
        [split_2(1,j),feature_2(1,j)] = classify_2(x_train(:,j),y_train,unique_values);
    else
        [split_2(1,j),feature_2(1,j)] = classify_multi(x_train(:,j),y_train,unique_values,size((unique_values),1));
   end
end

else
    random_array = randsample(c_train,N);
    sample = random_array;
    size(random_array);
    for j=1:N
        if j == index
            feature_2(1,j) = 0;
            split_2(1,j) = 0;
        else
         unique_values = unique(x_train(:,random_array(j,1)));
         size_val =  size(unique_values,1);
         if size_val>1
             if size_val == 2
                 [split_2(1,j),feature_2(1,j)] = classify_2(x_train(:,random_array(j,1)),y_train,unique_values);
             else
                 [split_2(1,j),feature_2(1,j)] = classify_multi(x_train(:,random_array(j,1)),y_train,unique_values,size_val);
             end
         end
     end
    end
end

[~,ind] = max(feature_2);
feature_1 = sample(ind);
split_1 = split_2(ind);