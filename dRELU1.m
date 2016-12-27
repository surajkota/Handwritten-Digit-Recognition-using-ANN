function [ y ] = dRELU1( x )
[r c]=size(x);
for di=1:r
    for dj=1:c
        if(x(di,dj)>0)
            y(di,dj)=1;
        else
            y(di,dj)=0.01;
        end
    end
end
end