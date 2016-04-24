function [split,result] = classify_multi(x_train,y_train,unique_values,size_val)
entr = zeros(1,size_val);
for i=1:size_val
    pos = zeros(1,2);
    neg = zeros(1,2);
    
    for j =1:size(x_train,1);
        if x_train(j,1) >= unique_values(i)
            if(y_train(j,1) == 1)
                pos(1,1) = pos(1,1)+1;
            else
                neg(1,1) = neg(1,1)+1;
            end
        else
            if(y_train(j,1) == 1)
                pos(1,2) = pos(1,2)+1;
            else
                neg(1,2) = neg(1,2)+1;
            end
        end
    end
    entr(1,i) = entropy(pos,neg);
end 
[result,index] = max(entr(1,:));
 split = unique_values(index);  