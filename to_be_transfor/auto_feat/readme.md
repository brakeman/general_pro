## 流程
    - 关键词： 测试集合 transform； 预处理，后处理； autoFeat;  业务逻辑特征
    
    
## 文件
    |--------       __init__.pu
    |--------       PrePostProcess.py        
    |--------       AutoFeat.py
    |--------       TransForm.py
    |--------       HumanLook.py #看列null#看行null 分布一致性#看列分布一致性
    |--------       test
        |----       auto_combine_test.py
        |----       to_bins_test.py
        |----       k_order_comb_test.py
        |----       auto_topk_test.py
        
## 通用衍生核心:
    - 类别变量的组合 --> dummy lgb topk --> 小微类别变量
    - 小微类别变量的组合（加 减 乘 除）

## 除了本项目主要的通用特征衍生，还有业务逻辑特征:
    - 1。 多个地址列，可能其中一列是 621785 这种身份证前6位地址，也可能是用户自己填的，可以衍生一个DIFF(dist1, dist2)用来判断用户是否虚报地址；
    
## finished
    - 分箱及其transform; 完成；
    - _auto_combine(transfrom空置率控制) 完成；
    - lgb_topk features; 半完成——no grid search;
    - PPP.py 半完成； 删除的操作都没做；
        -连续变量处理 排序分箱； 完成；
        -null 处理 数一下行空率； 完成；
        -列处理：相关性，方差，分布一致性 # 要删列的操作都先不做；
        -行处理：异常点  # 要删除行都操作都先不做；
    - HumanLook 完成；
        -看数字
            - 看列方差，列空置，列类别个数，列分布一致性
    - auto_transform 完成；
        自动做k阶 加减乘除特征组合；
    - dummy_lgb_topk 完成；
        自动降维；
    - merge_box 完成；
        自动合箱；
    - 手工1阶特征 提交一下看看 完成
    - chimerge_woe 完成
    
    
    
    
    
