## 对nlp 的历史和框架梳理

### A.nlp encoder 历史
- 能力对比 
    - 远距离 trfm>lstm>cnn
    - 语义 trfm>cnn=lstm
    - 综合 trfm>cnn=lstm
    
- cnn
    - 必看论文 kim; convs2s; tcn;
    - 所设计全部技术点：
        - pooling 层； 
        - dilated conv; 
        - causal conv; 
        - glu; 
        - skip connection;
        - pos emb(加不加都可)
        - 寄居到 transformer 中；
        
    - kim
    ![kim](../pics/kim_2014_1.png)
        - 特点
            - 前面正常的cnn能不用普通的pooling，会破坏掉语序
            - 这里尽最后用了 channel pooling；
        - 缺点
            - 单层感受野 长距离依赖差
        - 解决方式：
            - dilated cnn
            - deep layer
                - 缺点：三层就精度上不去了；
                - 解决方式： 
                    - 加入 norm， skip connection
       
- rnn
    - rnn -- lstm -- attention -- Encoder-Decoder


- trfm


### B.框架 - 4任务（分类，推断，生成，标注）
- 几个重要比赛
- CoQA (Conversational Question Answering)
    - [CoQA](https://blog.csdn.net/cindy_1102/article/details/88560048#_44)