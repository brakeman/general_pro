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
## 今天to be done;
    - woe , target-enco [zhihu](https://zhuanlan.zhihu.com/p/91098922)
    - 知道怎么做pipeline 吗？ [22]<https://github.com/scikit-learn-contrib/categorical-encoding>
        - TargetEncoder(cols=['CHAS', 'RAD']).fit(X_train, y_train)
        - 把 cols 放在init 里；

    - 重构代码
    - 采样增加随机性 
    - 阅读文章  
    - 一个是feat importance 的参数选择，双选！ 完成
    - 一个是dummy_top_k 的收集方式，是滚动式为好，改！ 完成
    - bug: 某种情况下会使得两个train ,test cols 不一致；
            - dummy！ train col 中有，比如 123 这个数， 但是test中没有；
            - 只需要 tra_x[val_x.columns]
            train:(70000, 15245)   valid:(62029, 13915)
    - PPP 一致性功能
    
# handcraft pipeline
    # id 类 certId, dist, bankCard,
    # 时间类  certValidBegin, certValidStop，weekday，setupHour
        - 分箱子
        - 是否周末 weekday
        - 是否白天
    
    # rank类 age, lmt 
        - 严格分箱，弃旧不用；
        
    # 相似类 
        - certId, dist, residentAddr
        - certValidStop, certValidBegin
        - highestEdu, edu
        # DIFF, DIV, SUM, MULTIPLY
    

## to be done:
    - 时序形表格特征提取/transform
    - 不平衡数据
    
    
    
    
    
