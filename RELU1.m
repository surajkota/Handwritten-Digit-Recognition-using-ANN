function [ y ] = RELU1( x )
[r c]=size(x);
for ti=1:r
    for tj=1:c
        if(x(ti,tj)>0)
            y(ti,tj)=x(ti,tj);
        else
            y(ti,tj)=0.01*x(ti,tj);
            %y(i,j)=0.01*x(i,j);
        end
    end
end
end
