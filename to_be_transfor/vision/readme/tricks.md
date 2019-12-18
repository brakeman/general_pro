## VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

## 印象

- 在 Alex_net & NIN 之后；
- 可以通过小的 3*3 conv 的形式 增加深度（alex net 基础上）
- A stack of convolutional layers (which has a different depth in different architectures) is followed by three Fully-Connected (FC) layers
- 证明了 alex net 中 LRN的 无用性： We note that none of our networks (except for one) contain Local Response Normalisation (LRN) normalisation 
- 使用了1*1：  is a way to increase the non- linearity of the decision function without affecting the receptive fields of the conv. layers.
- 证明了越深越好；而小的3*3可以使之更深； 