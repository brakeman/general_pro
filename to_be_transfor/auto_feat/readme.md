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
        
## 除了本项目主要的通用特征衍生，还有业务逻辑特征
    - 1。 多个地址列，可能其中一列是 621785 这种身份证前6位地址，也可能是用户自己填的，可以衍生一个DIFF(dist1, dist2)用来判断用户是否虚报地址；
## finished
    - 分箱及其transform; 完成；
    - _auto_combine (transfrom空置率控制) 完成；
    - lgb_topk features; 半完成——no grid search;
    - PPP.py 半完成； 删除的操作都没做；
        -连续变量处理 排序分箱； 完成；
        -null 处理 数一下行空率； 完成；
        -列处理：相关性，方差，分布一致性 # 要删列的操作都先不做；
        -行处理：异常点  # 要删除行都操作都先不做；
        - #更多的transform 类型 先等下把；
            - DIFF
            - DIV
    - K阶trans； 完成
    
    - HumanLook 完成；
        -看数字
            - 看列方差，列空置，列类别个数，列分布一致性
    
## 今天to be done;
    - 主函数框架
    -     def main():
        # 0. feature_dfs = [ori]
        # 1. 1阶变形
          --# 0.1 rank形， diff/div
          --# 0.2 相似形， diff/div 例如地址有4个
            # 1. cont形，rank_cut
          --# 1.1 超多类的列做 dummpy lgb topk
            # 2. concat cats
            # 3. 对 cats 做null_ratio
            # 4. lgb_topk
            # 5. feature_dfs. append(new_cats)

        # 2. 2阶变形
            # 1. k_order_comb(order=2)
            # 2. DIFF/DIV(order型) + null_ratio
            # 3. lgb_topk
            # 4. feature_dfs.append()
            
        # 3. 3阶变形
            
        # 4.concat all;
        
        # 5.根据分布一致性删除列；
        
    - transform 框架
        - 找到 rank 类 并完成DIFF/DIV
        
    - 类别列转换；
        - dummy 1-hot - lgb top_k
    

## to be done:
    - 时序形表格特征提取/transform
    - 不平衡数据
    
    
    
    
    
