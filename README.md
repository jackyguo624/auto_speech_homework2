# Auto Speech Tech Project2

## Baseline

### Extract Feature
1. `mkdir features` in the baseline
2. specify the wav and the label dir in the `feature_extraction.m`
3. run `feature_extraction.m`, u will get the cqcc and mfcc feature

### Run GMM
1. specify the wav and the label dir in the `baseline_cqcc.m`
2. /xxx/matlab -nodisplay -nodesktop -nosplash -r baseline_cqcc.m > cqcc.log
3. /xxx/matlab -nodisplay -nodesktop -nosplash -r baseline_mfcc.m > mfcc.log
4. check the log file



|    system    | feature |EER(Dev) | EER(Eval) |
| :---------- | :---: |:---: | :---: |
| Baseline | cqcc | |  |
| DNN(4096x4096x4096x2048x512X2) | cqcc | |  |