function [split,value] = classify_2(x_train,y_train,unique_values)
pos = zeros(1,2);
neg = zeros(1,2);
for i = 1:size(x_train,1)
    if(x_train(i,1)== unique_values(1))
        if(y_train(i,:)== 1)
            pos(1,1) = pos(1,1)+1;
        else
            neg(1,1) = neg(1,1)+1;
        end
    else
        if(y_train(i,:)==1)
            pos(1,2) = pos(1,2)+1;
        else
            neg(1,2) = neg(1,2)+1;
        end
    end
end
value = entropy(pos,neg);
split = max(unique_values);
    