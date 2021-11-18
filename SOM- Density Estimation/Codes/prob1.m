close all
clear
clc

Labels=["Dove" "Hen" "Duck" "Goose" "Owl" "Hawk" "Eagle" "Fox" "Dog" "Wolf" "Cat" "Tiger" "Lion" "Horse" "Zebra" "Cow" "Bear"]; 
Traits=["small" "medium" "large" "2 legs" "4 legs" "hair" "hooves" "mane" "feathers" "hunt" "run" "fly" "swim"];

L=numel(Labels);
Features=[1 1 1 1 0 1 0 0 0 0 1 0 0 0 0 0 0 
          0 0 0 0 1 0 1 1 1 1 0 0 0 0 0 0 0
          0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1
          1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
          0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0
          0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0
          1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
          0 0 0 0 1 1 1 1 0 1 1 1 1 0 0 0 1
          0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 0 1
          1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0
          0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1];
HotCode=diag(ones(1,L));
X=[HotCode;Features]';

Data=som_data_struct(X,'labels',cellstr(Labels'),'comp_names',cellstr([Labels Traits]'));
sMap=som_randinit(Data,'munits',169,'msize',[13 13],'lattice','hexa');
sMap=som_seqtrain(sMap,Data,'radius',3,'neigh','bubble','trainlen',1000,'alpha',0.3);

som_show(sMap,'umat','all','size',10)
figure,
som_show(sMap,'comp',1:17)
u=som_umat(sMap);
u=u(1:2:end,1:2:end);
figure,
imagesc(u)
csvwrite('out.csv',[u(:) sMap.codebook])

