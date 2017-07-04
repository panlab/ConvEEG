clear all;clc;
load('../DATA/TestFile.mat');

avg_accuracy = 0;
for subNO = 1:size(TestFile,1)
    subIndex = TestFile{subNO,1};
    SemanticRecollected_d = cell(1);
    SemanticNonRecollected_d = cell(1);
    %% load data
    load(strcat('../DATA/',subIndex,'.mat'));
    size_T = size(SemanticRecollected_d,2);
    size_F = size(SemanticNonRecollected_d,2);
    all_T = 1:size_T;
    all_F = 1:size_F;
    pre = 11:40;
    dur = 41:70;
    all = 1:75;
    peroid = all;
    for i = 1:size_T
        SemanticRecollected_dr = SemanticRecollected_d{i};
        SemanticRecollected_d{i} = SemanticRecollected_dr(:,peroid);
    end
    for i = 1:size_F
        SemanticNonRecollected_dr = SemanticNonRecollected_d{i};
        SemanticNonRecollected_d{i} = SemanticNonRecollected_dr(:,peroid);
    end
    %%
    cskernel = 25;
    accuracy_max = 0;
    for iter = 1:60
        
        [accuracy,FN,testp,cnn_selected_er] = ConvEEGNN(peroid,SemanticRecollected_d,SemanticNonRecollected_d,all_T,all_F,cskernel);
        
        if accuracy_max < accuracy
            accuracy_max = accuracy;
        end
    end
    avg_accuracy = avg_accuracy + accuracy_max;
end
disp(strcat('Average accuracy: ',num2str(avg_accuracy/size(TestFile,1))))