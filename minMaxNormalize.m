function imNor = minMaxNormalize(im)
% Normalize im into the range of [0, 1] using min-max normalization

sz=size(im);
imc=single(im(:));

valMin = min(imc);
valMax = max(imc);
imcn=(imc-valMin)./(valMax-valMin);

imNor=reshape(imcn, sz);