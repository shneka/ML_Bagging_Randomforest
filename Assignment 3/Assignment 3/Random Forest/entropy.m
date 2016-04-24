function [result] = entropy(pos,neg)
    
    left_pos_prob = pos(1,1)/(pos(1,1)+neg(1,1));
    right_pos_prob = pos(1,2)/(pos(1,2)+neg(1,2));
    left_neg_prob =  neg(1,1)/(pos(1,1)+neg(1,1));
    right_neg_prob = neg(1,2)/(pos(1,2)+neg(1,2));
    
     % To calculate entropy
    p = pos(1,1)+pos(1,2);
    n = neg(1,1)+neg(1,2);
    p_prob = p/(p+n);
    n_prob = n/(p+n);
    base_entropy = - p_prob*log2(p_prob)-n_prob*log2(n_prob);
    result = 0;
    if pos(1,1)~=0 && neg(1,1)~=0
        result = result-left_pos_prob*log2(left_pos_prob)-left_neg_prob*log2(left_neg_prob);
    end
    if pos(1,2)~=0 && neg(1,2)~=0
        result = result-right_pos_prob*log2(right_pos_prob)-right_neg_prob*log2(right_neg_prob);
    end
    result = base_entropy-result;
end


        
    